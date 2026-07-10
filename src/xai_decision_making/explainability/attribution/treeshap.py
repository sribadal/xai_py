"""
TreeSHAP attribution for masurvival's RandomSurvivalForest.

masurvival's SurvivalTree exposes standard sklearn tree arrays but
stores survival curves at leaves instead of scalars. Shap's
TreeExplainer expects scalar leaf values, so we work around this by
walking the training set to find leaf assignments, querying each
tree's predict path for scalar values, and building shap.SingleTree
dicts directly. Each time horizon requires a fresh build because
leaf scalars change.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    AttributionExplainer,
    Explanation,
    ExplainerModality,
    ExplainerRegistry,
    ModelTarget,
)
from xai_decision_making.explainability.sanity_checks import ATTRIBUTION_CHECKS
from xai_decision_making.explainability.attribution._base import (
    _attrs_from_shap_values,
    _run_sanity_checks,
)

registry = ExplainerRegistry()

def _tree_survival_at(survival_tree, x_f32, horizon_days):
    sf = survival_tree.predict_survival_function(x_f32)[0]
    try:
        return float(sf(horizon_days))
    except TypeError:
        x_arr = np.asarray(sf.x)
        y_arr = np.asarray(sf.y)
        idx = int(np.searchsorted(x_arr, horizon_days, side="right") - 1)
        idx = max(0, min(idx, len(y_arr) - 1))
        return float(y_arr[idx])

def _tree_dict_from_src(src_tree, scalar_values):
    return {
        "children_left": src_tree.children_left.astype(np.int32),
        "children_right": src_tree.children_right.astype(np.int32),
        "children_default": src_tree.children_left.astype(np.int32),
        "features": src_tree.feature.astype(np.int32),
        "thresholds": src_tree.threshold.astype(np.float64),
        "values": scalar_values.reshape(-1, 1).astype(np.float64),
        "node_sample_weight": src_tree.weighted_n_node_samples.astype(np.float64),
    }

def _leaf_scalars(survival_tree, X_for_apply, time_horizon):
    tree = survival_tree.tree_
    n_nodes = tree.node_count
    values = np.zeros(n_nodes, dtype=np.float64)

    leaf_idx_per_sample = survival_tree.apply(X_for_apply)
    unique_leaves = np.unique(leaf_idx_per_sample)

    rep_samples = np.empty(len(unique_leaves), dtype=np.int64)
    for i, leaf in enumerate(unique_leaves):
        rep_samples[i] = np.where(leaf_idx_per_sample == leaf)[0][0]

    X_reps = X_for_apply[rep_samples]

    if time_horizon is None:
        preds = survival_tree.predict(X_reps)
        for leaf, pred in zip(unique_leaves, preds):
            values[leaf] = float(pred)
        return values

    sf_list = survival_tree.predict_survival_function(X_reps)

    for leaf, sf in zip(unique_leaves, sf_list):
        try:
            s_at_t = float(sf(time_horizon))
        except TypeError:
            x_arr = np.asarray(sf.x)
            y_arr = np.asarray(sf.y)
            idx = int(np.searchsorted(x_arr, time_horizon, side="right") - 1)
            idx = max(0, min(idx, len(y_arr) - 1))
            s_at_t = float(y_arr[idx])
        values[leaf] = 1.0 - s_at_t

    return values

@registry.register
class TreeSHAPExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        top_k=40,
        time_horizon=None,
        background_size=200,
        random_state=None,
    ):
        super().__init__(model_adapter)
        self._top_k = top_k
        self._time_horizon = time_horizon
        self._background_size = background_size
        self._random_state = random_state
        self._explainer = None
        self._base_value = None
        self._last_completeness_gap = None
        self._last_output_range = None

    def _build(self):
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        X_train_np = x_train[feat_names].fillna(0).values.astype(np.float64)
        X_train_f32 = X_train_np.astype(np.float32, copy=False)

        raw_forest = self._adapter.raw_model

        tree_dicts = []
        for est in raw_forest.estimators_:
            leaf_vals = _leaf_scalars(est, X_train_f32, self._time_horizon)
            tree_dicts.append(_tree_dict_from_src(est.tree_, leaf_vals))

        rng = np.random.default_rng(self._random_state)
        bg_idx = rng.choice(
            len(X_train_np),
            size=min(self._background_size, len(X_train_np)),
            replace=False,
        )
        background = X_train_np[bg_idx]

        model_dict = {
            "trees": tree_dicts,
            "base_offset": 0.0,
            "objective": "reg_squarederror",
            "tree_output": "raw_value",
            "input_dtype": np.float64,
            "internal_dtype": np.float64,
        }

        try:
            self._explainer = shap.TreeExplainer(
                model_dict,
                data=background,
                feature_perturbation="interventional",
            )
        except Exception as exc:
            self._explainer = shap.TreeExplainer(
                model_dict,
                feature_perturbation="tree_path_dependent",
            )

        self._base_value = float(
            np.asarray(self._explainer.expected_value).ravel()[0]
        )

    @property
    def modality(self):
        return self._modality

    def feature_importances(self, patient_features):
        if self._explainer is None:
            self._build()

        feat_names = self._adapter.feature_names()
        x_raw = patient_features[feat_names].fillna(0).values
        x_f64 = x_raw.astype(np.float64)
        x_f32 = x_raw.astype(np.float32).reshape(1, -1)

        shap_values = self._explainer.shap_values(x_f64)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 2:
            shap_values = shap_values[0]

        f_x_shap_native = None
        try:
            pred = self._explainer.model.predict(x_f64)
            f_x_shap_native = float(np.asarray(pred).ravel()[0])
        except (AttributeError, TypeError, NotImplementedError):
            pass

        if f_x_shap_native is not None:
            f_x = f_x_shap_native
        else:
            raw_forest = self._adapter.raw_model
            if self._time_horizon is None:
                tree_preds = np.array([
                    float(est.predict(x_f32)[0])
                    for est in raw_forest.estimators_
                ])
            else:
                tree_preds = np.array([
                    1.0 - _tree_survival_at(est, x_f32, self._time_horizon)
                    for est in raw_forest.estimators_
                ])
            f_x = float(tree_preds.mean())

        sum_shap = float(np.sum(shap_values))
        expected_diff = f_x - self._base_value
        self._last_completeness_gap = abs(sum_shap - expected_diff)
        self._last_output_range = abs(expected_diff)

        attrs = dict(zip(feat_names, shap_values))
        ranked = sorted(attrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return dict(ranked[: self._top_k])

    def explain(self, patient_features, patient_id=None):
        importances = self.feature_importances(patient_features)
        gap = self._last_completeness_gap
        rng = self._last_output_range or 0.0
        ratio = (gap / rng) if (gap is not None and rng > 1e-9) else None

        confidence = {
            "completeness_gap": round(gap, 6) if gap is not None else None,
            "completeness_ratio": round(ratio, 4) if ratio is not None else None,
            "output_range": round(rng, 6),
            "base_value": round(self._base_value, 4) if self._base_value is not None else None,
            "source": "tier_c_native",
        }
        return self._make_explanation(
            patient_id,
            content=importances,
            method="TreeSHAP",
            time_horizon=self._time_horizon,
            confidence_metrics=confidence,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df, ATTRIBUTION_CHECKS)