"""
LORE counterfactual explainer via local surrogate tree.

Generates a neighbourhood around the query patient, fits a local
decision tree, and extracts a counterfactual by flipping the
minimal tree split. Supports plausibility constraints, biological
direction priors, and treatment flag derivation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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

def _generate_neighbourhood(x_orig, n_samples, feature_ranges, feat_names, immutable):
    d = len(x_orig)
    scales = np.ones(d)
    for i, feat in enumerate(feat_names):
        if feat in feature_ranges:
            lo, hi = feature_ranges[feat]
            if hi - lo > 0:
                scales[i] = hi - lo
    neighbourhood = x_orig + np.random.normal(0, 0.1, size=(n_samples, d)) * scales
    for i, feat in enumerate(feat_names):
        if feat in immutable:
            neighbourhood[:, i] = x_orig[i]
        elif feat in feature_ranges:
            lo, hi = feature_ranges[feat]
            neighbourhood[:, i] = np.clip(neighbourhood[:, i], lo, hi)
    return neighbourhood

def _extract_counterfactual_from_tree(tree, x_orig, orig_label, feat_names, feature_ranges, immutable):
    tree_ = tree.tree_
    path_nodes = tree.decision_path(x_orig.reshape(1, -1)).indices
    best_cf, best_delta = None, float("inf")
    for node_id in path_nodes:
        if tree_.feature[node_id] < 0:
            continue
        feat_idx = tree_.feature[node_id]
        feat = feat_names[feat_idx]
        if feat in immutable:
            continue
        threshold = tree_.threshold[node_id]
        x_flip = x_orig.copy()
        x_flip[feat_idx] = threshold + 1e-6 if x_orig[feat_idx] <= threshold else threshold - 1e-6
        if feat in feature_ranges:
            lo, hi = feature_ranges[feat]
            x_flip[feat_idx] = np.clip(x_flip[feat_idx], lo, hi)
        if tree.predict(x_flip.reshape(1, -1))[0] != orig_label:
            rng = max(feature_ranges.get(feat, (0, 1))[1] - feature_ranges.get(feat, (0, 1))[0], 1e-12)
            normalised_delta = abs(x_flip[feat_idx] - x_orig[feat_idx]) / rng
            if normalised_delta < best_delta:
                best_delta = normalised_delta
                best_cf = x_flip.copy()
    return best_cf

@registry.register
class LOREExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_samples=1000, tree_max_depth=5,
                 immutable=None, direction_priors=None, time_horizon=None):
        super().__init__(model_adapter)
        self._n_samples = n_samples
        self._tree_max_depth = tree_max_depth
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
        x_orig = patient_features[feat_names].fillna(0).values[0].astype(np.float64)
        threshold = float(target_outcome) if target_outcome is not None else self._risk_median

        cf_bounds = {}
        for f in feat_names:
            b = self._cfg.cf_bounds_for(f)
            if b is not None:
                cf_bounds[f] = b

        immutable_set = set(self._immutable)
        neighbourhood = _generate_neighbourhood(x_orig, self._n_samples, cf_bounds, feat_names, immutable_set)
        risks = self._predict_fn(neighbourhood)
        labels = (risks < threshold).astype(int) if self._higher_is_better else (risks >= threshold).astype(int)
        if len(np.unique(labels)) < 2:
            return None

        tree = DecisionTreeClassifier(max_depth=self._tree_max_depth, random_state=42)
        tree.fit(neighbourhood, labels)
        cf_values = _extract_counterfactual_from_tree(tree, x_orig, 1, feat_names, cf_bounds, immutable_set)
        if cf_values is None:
            return None

        cf_df = pd.DataFrame([cf_values], columns=feat_names)
        cf_df = clip_to_ranges(cf_df, self._ranges)
        cf_df, _ = snap_and_verify(cf_df.values[0], feat_names, self._categorical_values, self._predict_fn,
            target=threshold, higher_is_better=self._higher_is_better, query=x_orig,
            constraints=self._cfg.constraints, target_t_norm=condor_target_t_norm(self._adapter, self._time_horizon))
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
        return self._make_explanation(patient_id, content=content, method="LORE",
            counterfactual_df=cf, original_prediction=orig_pred, cf_prediction=cf_pred,
            raw_query=raw_query, raw_cf=raw_cf, higher_is_better=self._higher_is_better, confidence_metrics=confidence_metrics)

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)