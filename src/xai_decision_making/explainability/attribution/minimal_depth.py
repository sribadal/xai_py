"""
Minimal depth feature importance for Random Survival Forests.

Computes the mean minimal split depth per feature across all trees
via BFS traversal. Features split closer to the root are considered
more important.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    AttributionExplainer,
    Explanation,
    ExplainerModality,
    ExplainerRegistry,
    ModelTarget,
    SurvivalModelAdapter,
)
from xai_decision_making.explainability.sanity_checks import ATTRIBUTION_CHECKS
from xai_decision_making.explainability.attribution._base import _run_sanity_checks

registry = ExplainerRegistry()

LEAF_SENTINEL = -2

def _compute_minimal_depth(rsf):
    n_features = rsf.n_features_in_
    all_min_depths = []

    for estimator in rsf.estimators_:
        t = estimator.tree_
        features = np.array(t.feature)
        left = np.array(t.children_left)
        right = np.array(t.children_right)
        n_nodes = len(features)

        depths = np.full(n_nodes, -1, dtype=int)
        depths[0] = 0
        queue = [0]
        while queue:
            node = queue.pop(0)
            d = depths[node]
            if left[node] != -1:
                depths[left[node]] = d + 1
                queue.append(left[node])
            if right[node] != -1:
                depths[right[node]] = d + 1
                queue.append(right[node])

        tree_min_depths = np.full(n_features, np.inf)
        for node in range(n_nodes):
            f = features[node]
            if f != LEAF_SENTINEL and depths[node] < tree_min_depths[f]:
                tree_min_depths[f] = depths[node]
        all_min_depths.append(tree_min_depths)

    mean_min_depths = np.mean(all_min_depths, axis=0)
    return {i: float(mean_min_depths[i]) for i in range(n_features)}


@registry.register
class MinimalDepthExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter):
        super().__init__(model_adapter)
        self._global_importances = None

    @property
    def modality(self):
        return self._modality

    def _compute(self):
        rsf = self._adapter.raw_model
        feat_names = self._adapter.feature_names()
        depth_by_idx = _compute_minimal_depth(rsf)
        max_depth = max(
            (v for v in depth_by_idx.values() if v != np.inf),
            default=1.0,
        )
        self._global_importances = {
            feat_names[i]: (
                (max_depth - depth_by_idx[i])
                if depth_by_idx[i] != np.inf
                else 0.0
            )
            for i in range(len(feat_names))
        }

    def feature_importances(self, patient_features):
        if self._global_importances is None:
            self._compute()
        return self._global_importances

    def explain(self, patient_features, patient_id=None):
        return self._make_explanation(
            patient_id,
            content=self.feature_importances(patient_features),
            method="MinimalDepth",
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df, ATTRIBUTION_CHECKS)