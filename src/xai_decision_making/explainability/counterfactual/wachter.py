"""
Wachter counterfactual explainer via L-BFGS-B optimisation.

Minimises a prediction loss + proximity + sparsity objective over
mutable features only, with per-feature bounds from PlausibilityConfig,
multi-restart L-BFGS-B, saturation gating, plausibility snapping, and
biological direction priors.
"""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    CounterfactualExplainer,
    ExplainerModality,
    ExplainerRegistry,
    Explanation,
    ModelTarget,
)
from xai_decision_making.explainability.counterfactual._base import (
    _run_sanity_checks,
    apply_direction_priors,
    clip_to_ranges,
    compute_cf_confidence,
    compute_content_deltas,
    enforce_immutables,
    extract_scalar_prediction,
    resolve_direction_priors,
    risk_predict_fn,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config,
    condor_target_t_norm,
    snap_and_verify,
)

registry = ExplainerRegistry()


@registry.register
class WachterExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        lambda_=10.0,
        sparsity_weight=0.1,
        max_iter=500,
        n_restarts=5,
        immutable=None,
        feature_ranges=None,
        time_horizon=None,
        direction_priors=None,
        horizon_saturation_threshold=99.0,
    ):
        super().__init__(model_adapter)
        self._lambda = lambda_
        self._sparsity = sparsity_weight
        self._max_iter = max_iter
        self._n_restarts = n_restarts
        self._time_horizon = time_horizon
        self._horizon_saturation_threshold = horizon_saturation_threshold

        bounds = getattr(model_adapter, "prediction_saturation_bounds", None)
        if bounds is not None:
            self._sat_floor, self._sat_ceiling = float(bounds[0]), float(bounds[1])
        else:
            t = float(horizon_saturation_threshold)
            self._sat_ceiling = t
            self._sat_floor = 100.0 - t

        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(
            model_adapter,
            extra_immutable=immutable,
            feature_ranges=feature_ranges,
        )
        self._direction_priors = resolve_direction_priors(
            model_adapter, override=direction_priors,
        )
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)

        self._train_risk_min = None
        self._train_risk_max = None
        self._train_risk_q25 = None
        self._train_risk_median = None
        self._train_risk_q75 = None

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def _is_saturated_at_horizon(self, patient_features):
        if self._time_horizon is None:
            return False, None
        try:
            pred = extract_scalar_prediction(
                self._adapter.predict(
                    patient_features.head(1),
                    time_horizon=self._time_horizon,
                )
            )
        except Exception:
            return False, None
        if self._higher_is_better and pred >= self._sat_ceiling:
            return True, pred
        if (not self._higher_is_better) and pred <= self._sat_floor:
            return True, pred
        return False, pred

    def _ensure_risk_stats(self):
        if self._train_risk_median is not None:
            return
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        risks = self._predict_fn(
            x_train[feat_names].fillna(0).values
        )
        self._train_risk_min = float(risks.min())
        self._train_risk_max = float(risks.max())
        self._train_risk_q25 = float(np.percentile(risks, 25))
        self._train_risk_median = float(np.median(risks))
        self._train_risk_q75 = float(np.percentile(risks, 75))

    def _get_target_risk(self, current_risk):
        self._ensure_risk_stats()
        iqr = self._train_risk_q75 - self._train_risk_q25

        if self._higher_is_better:
            if current_risk >= self._train_risk_max - 0.25 * iqr:
                return None
            target = max(current_risk + 0.75 * iqr, self._train_risk_q75)
            target = min(target, self._train_risk_max)
            return float(target)
        else:
            if current_risk <= self._train_risk_min + 0.25 * iqr:
                return None
            target = min(current_risk - 0.75 * iqr, self._train_risk_q25)
            target = max(target, self._train_risk_min)
            return float(target)

    def generate_counterfactual(self, patient_features, target_outcome=None):
        feat_names = self._adapter.feature_names()

        saturated, horizon_pred = self._is_saturated_at_horizon(patient_features)
        if saturated:
            return None

        x_orig = patient_features[feat_names].fillna(0).values[0].astype(np.float64)
        d = len(x_orig)

        current_risk = float(self._predict_fn(x_orig.reshape(1, -1))[0])

        if target_outcome is not None:
            y_target = float(target_outcome)
        else:
            y_target = self._get_target_risk(current_risk)
            if y_target is None:
                return None

        mutable_mask = np.array([f not in self._immutable for f in feat_names])
        mutable_idx = np.where(mutable_mask)[0]
        if len(mutable_idx) == 0:
            return None

        range_scales = np.ones(d)
        for i, feat in enumerate(feat_names):
            if feat in self._ranges:
                lo, hi = self._ranges[feat]
                span = hi - lo
                if span > 0:
                    range_scales[i] = span
        mutable_scales = range_scales[mutable_idx]

        bounds = []
        for i in mutable_idx:
            feat = feat_names[i]
            b = self._cfg.cf_bounds_for(feat)
            bounds.append(b if b is not None else (None, None))

        def objective(x_mutable):
            x_prime = x_orig.copy()
            x_prime[mutable_idx] = x_mutable
            pred = float(self._predict_fn(x_prime.reshape(1, -1))[0])

            prediction_loss = self._lambda * (pred - y_target) ** 2
            delta = (x_prime[mutable_idx] - x_orig[mutable_idx]) / mutable_scales
            proximity_loss = np.sum(delta ** 2)
            sparsity_loss = self._sparsity * np.sum(np.abs(delta))

            return prediction_loss + proximity_loss + sparsity_loss

        best_result = None
        best_loss = np.inf

        for restart in range(self._n_restarts):
            if restart == 0:
                x0 = x_orig[mutable_idx].copy()
            else:
                noise = np.random.normal(0, 0.05, size=len(mutable_idx)) * mutable_scales
                x0 = x_orig[mutable_idx] + noise

            try:
                result = minimize(
                    objective,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self._max_iter, "ftol": 1e-8},
                )
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None:
            return None

        x_cf = x_orig.copy()
        x_cf[mutable_idx] = best_result.x

        tiny = np.abs(x_cf - x_orig) / range_scales < 1e-4
        x_cf[tiny] = x_orig[tiny]

        cf_df = pd.DataFrame([x_cf], columns=feat_names)
        cf_df = clip_to_ranges(cf_df, self._ranges)

        cf_df, report = snap_and_verify(
            cf_df.values[0],
            feat_names,
            self._categorical_values,
            self._predict_fn,
            target=y_target,
            higher_is_better=self._higher_is_better,
            query=x_orig,
            constraints=self._cfg.constraints,
            target_t_norm=condor_target_t_norm(self._adapter, self._time_horizon),
        )

        cf_df = enforce_immutables(
            patient_features[feat_names].head(1),
            cf_df,
            self._immutable,
        )

        if self._direction_priors:
            cf_corrected, n_reset = apply_direction_priors(
                cf_df.values[0],
                x_orig,
                feat_names,
                self._direction_priors,
            )
            if n_reset > 0:
                if np.all(np.abs(cf_corrected - x_orig) < 1e-8):
                    return None
                cf_df = pd.DataFrame([cf_corrected], columns=feat_names)

        cf_risk = float(self._predict_fn(cf_df.values[0].reshape(1, -1))[0])
        if abs(cf_risk - current_risk) < 1e-6:
            return None

        return cf_df

    def explain(self, patient_features, patient_id=None):
        feat_names = self._adapter.feature_names()
        query = patient_features[feat_names].fillna(0).head(1)
        cf = self.generate_counterfactual(patient_features)

        if cf is not None:
            content, raw_query, raw_cf = compute_content_deltas(
                self._adapter, query, cf, feat_names
            )
        else:
            content, raw_query, raw_cf = {}, None, None

        orig_pred = extract_scalar_prediction(
            self._adapter.predict(query, time_horizon=self._time_horizon)
        )
        cf_pred = (
            extract_scalar_prediction(
                self._adapter.predict(cf.head(1), time_horizon=self._time_horizon)
            )
            if cf is not None
            else None
        )

        confidence_metrics = compute_cf_confidence(
            raw_query, raw_cf, orig_pred, cf_pred,
            higher_is_better=self._higher_is_better,
        )

        return self._make_explanation(
            patient_id,
            content=content,
            method="Wachter",
            counterfactual_df=cf,
            original_prediction=orig_pred,
            cf_prediction=cf_pred,
            raw_query=raw_query,
            raw_cf=raw_cf,
            higher_is_better=self._higher_is_better,
            confidence_metrics=confidence_metrics,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)