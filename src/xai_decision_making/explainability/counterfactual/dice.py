"""
DiCE counterfactual explainer for survival models.

Wraps dice-ml's genetic/KD-tree CF search with plausibility snapping,
saturation gating, permitted range bounds from PlausibilityConfig,
biological direction priors, and a SIGALRM timeout guard.
"""
from __future__ import annotations

import signal
import traceback

import dice_ml
import numpy as np
import pandas as pd

from xai_decision_making.explainability.base import (
    AggregatedSanityResult, CounterfactualExplainer, ExplainerModality,
    ExplainerRegistry, Explanation, ModelTarget,
)
from xai_decision_making.explainability.counterfactual._base import (
    _run_sanity_checks, apply_direction_priors, clip_to_ranges,
    compute_cf_confidence, compute_content_deltas, enforce_immutables,
    extract_scalar_prediction, resolve_direction_priors, risk_predict_fn,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config, condor_target_t_norm, snap_and_verify,
)

registry = ExplainerRegistry()

ORDINAL_BIN_FEATURES = [
    "cstadium", "stadium", "pstadium", "ct", "cn", "cm", "pt", "pn", "pm",
    "yct", "ycn", "ycm", "perf_stat", "asa", "diffgrad", "cci_cat",
]

def _legal_integer_values_for(feature, ranges):
    if feature not in ranges:
        return np.array([], dtype=np.float64)
    lo, hi = ranges[feature]
    return np.asarray([float(v) for v in range(int(np.floor(lo)), int(np.ceil(hi)) + 1)], dtype=np.float64)

class _SklearnLikeModel:
    def __init__(self, predict_fn):
        self._predict_fn = predict_fn

    def predict(self, X):
        arr = X.values.astype(np.float64) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self._predict_fn(arr)


class _DiCETimeout(Exception):
    pass

@registry.register
class DiCEExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_counterfactuals=1, method="genetic",
                 desired_class="opposite", immutable=None, feature_ranges=None,
                 diversity_weight=0.0, proximity_weight=1.0, time_horizon=None,
                 horizon_saturation_threshold=99.0, direction_priors=None, max_time=120):
        super().__init__(model_adapter)
        self._n_cf = n_counterfactuals
        self._method = method
        self._desired_class = desired_class
        self._diversity_weight = diversity_weight
        self._proximity_weight = proximity_weight
        self._time_horizon = time_horizon
        self._max_time = max_time
        bounds = getattr(model_adapter, "prediction_saturation_bounds", None)
        if bounds is not None:
            self._sat_floor, self._sat_ceiling = float(bounds[0]), float(bounds[1])
        else:
            t = float(horizon_saturation_threshold)
            self._sat_ceiling = t
            self._sat_floor = 100.0 - t
        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(model_adapter, extra_immutable=immutable, feature_ranges=feature_ranges)
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values

        feat_names = model_adapter.feature_names()
        for feat in ORDINAL_BIN_FEATURES:
            if feat in feat_names:
                legal = _legal_integer_values_for(feat, self._ranges)
                if legal.size:
                    self._categorical_values[feat] = legal

        self._direction_priors = resolve_direction_priors(model_adapter, override=direction_priors)
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)
        self._dice_data = self._dice_model = self._dice_exp = None
        self._train_risk_min = self._train_risk_max = None
        self._train_risk_q25 = self._train_risk_median = self._train_risk_q75 = None

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def _is_saturated_at_horizon(self, patient_features):
        if self._time_horizon is None:
            return False, None
        try:
            pred = extract_scalar_prediction(self._adapter.predict(patient_features.head(1), time_horizon=self._time_horizon))
        except Exception:
            return False, None
        if self._higher_is_better and pred >= self._sat_ceiling:
            return True, pred
        if (not self._higher_is_better) and pred <= self._sat_floor:
            return True, pred
        return False, pred

    def _build_dice(self):
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        risks = self._predict_fn(x_train[feat_names].fillna(0).values.astype(np.float64))

        train_with_outcome = x_train[feat_names].fillna(0).copy()
        train_with_outcome["_outcome"] = risks
        continuous_features = x_train[feat_names].select_dtypes(include=[np.number]).columns.tolist()

        permitted_range = {}
        for feat in continuous_features:
            b = self._cfg.cf_bounds_for(feat)
            if b is not None:
                permitted_range[feat] = [float(b[0]), float(b[1])]

        self._dice_data = dice_ml.Data(dataframe=train_with_outcome, continuous_features=continuous_features,
                                        outcome_name="_outcome", permitted_range=permitted_range or None)
        self._dice_model = dice_ml.Model(model=_SklearnLikeModel(self._predict_fn), backend="sklearn", model_type="regressor")
        self._dice_exp = dice_ml.Dice(self._dice_data, self._dice_model, method=self._method)

        self._train_risk_min, self._train_risk_max = float(risks.min()), float(risks.max())
        self._train_risk_q25 = float(np.percentile(risks, 25))
        self._train_risk_median = float(np.median(risks))
        self._train_risk_q75 = float(np.percentile(risks, 75))

    def generate_counterfactual(self, patient_features, target_outcome=None):
        if self._dice_exp is None:
            self._build_dice()
        feat_names = self._adapter.feature_names()

        saturated, _ = self._is_saturated_at_horizon(patient_features)
        if saturated:
            return None

        query = patient_features[feat_names].fillna(0).head(1)
        raw_current = float(self._predict_fn(query.values.astype(np.float64))[0])

        if target_outcome is not None:
            desired_range = target_outcome
        else:
            iqr = self._train_risk_q75 - self._train_risk_q25
            if self._higher_is_better:
                if raw_current >= self._train_risk_max - 0.25 * iqr:
                    return None
                lower = max(raw_current + 0.5 * iqr, self._train_risk_median)
                upper = min(max(raw_current + 1.0 * iqr, self._train_risk_q75), self._train_risk_max)
                desired_range = [float(min(lower, upper - 1e-6)), float(upper)]
            else:
                if raw_current <= self._train_risk_min + 0.25 * iqr:
                    return None
                upper = min(raw_current - 0.5 * iqr, self._train_risk_median)
                lower = max(min(raw_current - 1.0 * iqr, self._train_risk_q25), self._train_risk_min)
                desired_range = [float(lower), float(max(upper, lower + 1e-6))]

        features_to_vary = [f for f in feat_names if f not in self._immutable]

        try:
            def _timeout_handler(signum, frame):
                raise _DiCETimeout(f"DiCE exceeded {self._max_time}s timeout")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self._max_time)
            try:
                exp = self._dice_exp.generate_counterfactuals(query, total_CFs=self._n_cf, desired_range=desired_range, features_to_vary=features_to_vary)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            cf_df = exp.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                return None
            if "_outcome" in cf_df.columns:
                cf_df = cf_df.drop(columns=["_outcome"])
            cf_df = cf_df[[c for c in feat_names if c in cf_df.columns]]
            cf_df = clip_to_ranges(cf_df, self._ranges)

            cf_df, report = snap_and_verify(cf_df.values[0], feat_names, self._categorical_values, self._predict_fn,
                target=None, higher_is_better=self._higher_is_better, query=query.values[0].astype(np.float64),
                constraints=self._cfg.constraints, target_t_norm=condor_target_t_norm(self._adapter, self._time_horizon))
            cf_df = enforce_immutables(query, cf_df, self._immutable)

            if self._direction_priors:
                cf_corrected, n_reset = apply_direction_priors(cf_df.values[0], query.values[0].astype(np.float64), feat_names, self._direction_priors)
                if n_reset > 0:
                    cf_df = pd.DataFrame([cf_corrected], columns=feat_names)
                    if np.all(np.abs(cf_corrected - query.values[0].astype(np.float64)) < 1e-8):
                        return None

            return cf_df

        except _DiCETimeout:
            return None
        except Exception:
            return None

    def explain(self, patient_features, patient_id=None):
        feat_names = self._adapter.feature_names()
        query = patient_features[feat_names].fillna(0).head(1)
        cf = self.generate_counterfactual(patient_features)
        if cf is not None:
            content, raw_query, raw_cf = compute_content_deltas(self._adapter, query, cf, feat_names)
        else:
            content, raw_query, raw_cf = {}, None, None
        orig_pred = extract_scalar_prediction(self._adapter.predict(query, time_horizon=self._time_horizon))
        cf_pred = extract_scalar_prediction(self._adapter.predict(cf.head(1), time_horizon=self._time_horizon)) if cf is not None else None
        confidence_metrics = compute_cf_confidence(raw_query, raw_cf, orig_pred, cf_pred, higher_is_better=self._higher_is_better)
        return self._make_explanation(patient_id, content=content, method="DiCE", n_counterfactuals=self._n_cf,
            counterfactual_df=cf, original_prediction=orig_pred, cf_prediction=cf_pred,
            raw_query=raw_query, raw_cf=raw_cf, higher_is_better=self._higher_is_better, confidence_metrics=confidence_metrics)

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)