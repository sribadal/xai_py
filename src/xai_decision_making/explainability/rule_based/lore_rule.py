"""
LORE factual rule explainer via local surrogate decision tree.

Returns humanised clinician-readable conditions alongside the raw
encoded-space conditions for audit and debugging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from xai_decision_making.explainability.base import (
    ExplainerModality,
    ExplainerRegistry,
    ModelTarget,
    RuleBasedExplainer,
)
from xai_decision_making.explainability.rule_based._base import (
    _run_sanity_checks,
    binarise_risk,
    format_condition,
    format_rule,
    humanize_rule_conditions,
    outcome_label,
    risk_predict_fn,
)

registry = ExplainerRegistry()

def _get_feature_ranges(adapter):
    x_train = adapter.training_data()
    ranges = {}
    for col in x_train.columns:
        if pd.api.types.is_numeric_dtype(x_train[col]):
            ranges[col] = (float(x_train[col].min()), float(x_train[col].max()))
    return ranges


def _generate_neighbourhood(x_orig, n_samples, feature_ranges, feat_names,
                            noise_scale, seed):
    rng = np.random.RandomState(seed)
    d = len(x_orig)
    scales = np.ones(d)
    for i, feat in enumerate(feat_names):
        if feat in feature_ranges:
            lo, hi = feature_ranges[feat]
            span = hi - lo
            if span > 0:
                scales[i] = span

    noise = rng.normal(0, noise_scale, size=(n_samples, d)) * scales
    neighbourhood = x_orig + noise

    for i, feat in enumerate(feat_names):
        if feat in feature_ranges:
            lo, hi = feature_ranges[feat]
            neighbourhood[:, i] = np.clip(neighbourhood[:, i], lo, hi)

    return neighbourhood

def _extract_path_rule(tree, x_orig, feat_names):
    tree_ = tree.tree_
    path_nodes = tree.decision_path(x_orig.reshape(1, -1)).indices

    lower = {}
    upper = {}

    for node_id in path_nodes:
        if tree_.feature[node_id] < 0:
            continue
        feat_idx = tree_.feature[node_id]
        feat = feat_names[feat_idx]
        threshold = tree_.threshold[node_id]
        if x_orig[feat_idx] <= threshold:
            if feat not in upper or threshold < upper[feat]:
                upper[feat] = threshold
        else:
            if feat not in lower or threshold > lower[feat]:
                lower[feat] = threshold

    conditions = []
    for feat in sorted(set(list(lower.keys()) + list(upper.keys()))):
        if feat in lower:
            conditions.append(format_condition(feat, ">", lower[feat]))
        if feat in upper:
            conditions.append(format_condition(feat, "≤", upper[feat]))

    return conditions

@registry.register
class LORERuleExplainer(RuleBasedExplainer):

    _modality = ExplainerModality.RULE_BASED
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_samples=1000, tree_max_depth=5,
                 noise_scale=0.25, verbose=True, time_horizon=None):
        super().__init__(model_adapter)
        self._n_samples = n_samples
        self._tree_max_depth = tree_max_depth
        self._noise_scale = noise_scale
        self._verbose = verbose
        self._time_horizon = time_horizon
        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._ranges = _get_feature_ranges(model_adapter)
        self._risk_median = None

    @property
    def modality(self):
        return self._modality

    def _ensure_risk_median(self):
        if self._risk_median is not None:
            return
        _, threshold, _ = binarise_risk(
            self._adapter, time_horizon=self._time_horizon
        )
        self._risk_median = threshold
        if self._verbose:
            print(f"  [LORE-Rule] Training score median={self._risk_median:.1f}")

    def extract_rule(self, patient_features):
        self._ensure_risk_median()
        feat_names = self._adapter.feature_names()
        x_orig = patient_features[feat_names].fillna(0).values[0].astype(np.float64)

        orig_risk = float(self._predict_fn(x_orig.reshape(1, -1))[0])
        orig_label = int(orig_risk >= self._risk_median)
        outcome = outcome_label(self._adapter, bool(orig_label))

        patient_seed = hash(tuple(np.round(x_orig, 6).tolist())) % (2**31)

        neighbourhood = None
        labels = None
        chosen_scale = None
        for noise_scale in (self._noise_scale, self._noise_scale * 2,
                            self._noise_scale * 4, self._noise_scale * 8):
            candidate = _generate_neighbourhood(
                x_orig, self._n_samples, self._ranges, feat_names,
                noise_scale=noise_scale, seed=patient_seed,
            )
            candidate_risks = self._predict_fn(candidate)
            candidate_labels = (candidate_risks >= self._risk_median).astype(int)
            if len(np.unique(candidate_labels)) >= 2:
                neighbourhood = candidate
                labels = candidate_labels
                chosen_scale = noise_scale
                break

        if neighbourhood is None:
            return {
                "conditions": [],
                "conditions_raw": [],
                "outcome": outcome,
                "rule_text": format_rule([], outcome),
                "fidelity": float("nan"),
                "note": "single-class neighbourhood at all noise scales",
            }

        tree = DecisionTreeClassifier(max_depth=self._tree_max_depth, random_state=42)
        tree.fit(neighbourhood, labels)
        fidelity = float(tree.score(neighbourhood, labels))

        conditions_raw = _extract_path_rule(tree, x_orig, feat_names)
        patient_row = dict(zip(feat_names, x_orig.tolist()))
        conditions = humanize_rule_conditions(
            conditions_raw, adapter=self._adapter, patient_row=patient_row,
        )

        if self._verbose:
            print(f"  [LORE-Rule] local fidelity={fidelity:.3f}, "
                  f"depth={tree.get_depth()}, n_conditions={len(conditions)} "
                  f"(raw={len(conditions_raw)}), noise={chosen_scale}")

        return {
            "conditions": conditions,
            "conditions_raw": conditions_raw,
            "outcome": outcome,
            "rule_text": format_rule(conditions, outcome),
            "fidelity": fidelity,
            "noise_scale": chosen_scale,
        }

    def global_rules(self):
        return None

    def fidelity(self, held_out_df):
        feat_names = self._adapter.feature_names()
        fidelities = []
        for i in range(len(held_out_df)):
            patient = held_out_df.iloc[i:i+1][feat_names]
            rule = self.extract_rule(patient)
            f = rule.get("fidelity", float("nan"))
            if not np.isnan(f):
                fidelities.append(f)
        return float(np.mean(fidelities)) if fidelities else 0.0

    def explain(self, patient_features, patient_id=None):
        rule = self.extract_rule(patient_features)
        return self._make_explanation(
            patient_id,
            content=rule,
            method="LORE-Rule",
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)