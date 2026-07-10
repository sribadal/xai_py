"""
Wachter counterfactual explainer via L-BFGS-B optimisation.

Minimises a prediction loss + proximity + sparsity objective in the
model's feature space with per-feature bounds from PlausibilityConfig,
multi-restart L-BFGS-B, plausibility snapping, and biological direction
priors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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

@registry.register
class WachterExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, lambda_distance=0.1, lambda_sparsity=0.01,
                 maxiter=200, n_restarts=3, immutable=None, direction_priors=None,
                 time_horizon=None):
        super().__init__(model_adapter)
        self._lambda_distance = lambda_distance
        self._lambda_sparsity = lambda_sparsity
        self._maxiter = maxiter
        self._n_restarts = n_restarts
        self._time_horizon = time_horizon
        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(model_adapter, extra_immutable=immutable)
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._direction_priors = resolve_direction_priors(model_adapter, override=direction_priors)
        self._risk_median = None
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def _ensure_risk_median(self):
        if self._risk_median is not None:
            return
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        risks = self._predict_fn(x_train[feat_names].fillna(0).values.astype(np.float64))
        self._risk_median = float(np.median(risks))

    def generate_counterfactual(self, patient_features, target_outcome=None):
        self._ensure_risk_median()
        feat_names = self._adapter.feature_names()
        target_t_norm = condor_target_t_norm(self._adapter, self._time_horizon)
        x_orig = patient_features[feat_names].fillna(0).values[0].astype(np.float64)
        threshold = float(target_outcome) if target_outcome is not None else self._risk_median
        go_lower = not self._higher_is_better
        y_target = threshold * 0.8 if go_lower else threshold * 1.2
        immutable_idx = np.array([i for i, f in enumerate(feat_names) if f in self._immutable], dtype=int)

        bounds = []
        for i, feat in enumerate(feat_names):
            if feat in self._immutable:
                bounds.append((x_orig[i], x_orig[i]))
            else:
                b = self._cfg.cf_bounds_for(feat)
                if b is not None:
                    bounds.append((float(b[0]), float(b[1])))
                elif feat in self._ranges:
                    bounds.append(self._ranges[feat])
                else:
                    bounds.append((None, None))

        range_denom = np.array([max(self._ranges.get(f, (0, 1))[1] - self._ranges.get(f, (0, 1))[0], 1e-12) for f in feat_names])

        def objective(x_cf):
            risk = self._predict_fn(x_cf.reshape(1, -1))[0]
            pred_loss = (risk - y_target) ** 2
            proximity = np.sum(((x_cf - x_orig) / range_denom) ** 2)
            sparsity = np.sum(np.abs((x_cf - x_orig) / range_denom))
            return pred_loss + self._lambda_distance * proximity + self._lambda_sparsity * sparsity

        best_cf, best_cost = None, float("inf")
        rng = np.random.default_rng(42)

        for restart in range(self._n_restarts):
            if restart == 0:
                x0 = x_orig.copy()
            else:
                noise = rng.normal(0, 0.1, size=len(x_orig)) * range_denom
                x0 = x_orig + noise
                for i, (lo, hi) in enumerate(bounds):
                    if lo is not None and hi is not None:
                        x0[i] = np.clip(x0[i], lo, hi)
                if len(immutable_idx) > 0:
                    x0[immutable_idx] = x_orig[immutable_idx]

            try:
                result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                                  options={"maxiter": self._maxiter, "ftol": 1e-12})
            except Exception:
                continue

            cf_candidate = result.x.copy()
            if len(immutable_idx) > 0:
                cf_candidate[immutable_idx] = x_orig[immutable_idx]

            risk_cf = float(self._predict_fn(cf_candidate.reshape(1, -1))[0])
            crossed = (go_lower and risk_cf < threshold) or (not go_lower and risk_cf >= threshold)
            if crossed and result.fun < best_cost:
                best_cost = result.fun
                best_cf = cf_candidate

        if best_cf is None:
            return None

        tiny = np.abs(best_cf - x_orig) / range_denom < 1e-4
        best_cf[tiny] = x_orig[tiny]

        cf_df = pd.DataFrame([best_cf], columns=feat_names)
        cf_df = clip_to_ranges(cf_df, self._ranges)
        cf_df, _ = snap_and_verify(cf_df.values[0], feat_names, self._categorical_values, self._predict_fn,
            target=threshold, higher_is_better=self._higher_is_better, query=x_orig,
            constraints=self._cfg.constraints, target_t_norm=target_t_norm)
        cf_df = enforce_immutables(patient_features[feat_names].head(1), cf_df, self._immutable)

        if self._direction_priors:
            cf_corrected, n_reset = apply_direction_priors(cf_df.values[0], x_orig, feat_names, self._direction_priors)
            if n_reset > 0:
                if np.all(np.abs(cf_corrected - x_orig) < 1e-8):
                    return None
                cf_df = pd.DataFrame([cf_corrected], columns=feat_names)

        return cf_df

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
        return self._make_explanation(patient_id, content=content, method="Wachter",
            counterfactual_df=cf, original_prediction=orig_pred, cf_prediction=cf_pred,
            raw_query=raw_query, raw_cf=raw_cf, higher_is_better=self._higher_is_better, confidence_metrics=confidence_metrics)

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)