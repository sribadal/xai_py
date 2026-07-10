"""
FACE (Feasible Actionable Counterfactual Explanations) explainer.

Retrieves counterfactuals via shortest-path search on a density-weighted
k-NN graph of the training set. CFs are real training patients, so no
plausibility snapping is needed beyond immutability enforcement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from xai_decision_making.explainability.base import (
    AggregatedSanityResult, CounterfactualExplainer, ExplainerModality,
    ExplainerRegistry, Explanation, ModelTarget,
)
from xai_decision_making.explainability.counterfactual._base import (
    _run_sanity_checks, compute_cf_confidence, compute_content_deltas,
    direction_compatible_mask, enforce_immutables, extract_scalar_prediction,
    resolve_direction_priors, risk_predict_fn, snap_categoricals,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config,
)

registry = ExplainerRegistry()

@registry.register
class FACEExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_neighbours=10, density_bandwidth=1.0,
                 distance_metric="euclidean", density_threshold=0.0, immutable=None,
                 max_graph_size=2000, time_horizon=None, direction_priors=None):
        super().__init__(model_adapter)
        self._k = n_neighbours
        self._bandwidth = density_bandwidth
        self._metric = distance_metric
        self._density_threshold = density_threshold
        self._max_graph_size = max_graph_size
        self._time_horizon = time_horizon
        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(model_adapter, extra_immutable=immutable)
        self._direction_priors = resolve_direction_priors(model_adapter, override=direction_priors)
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)
        self._graph = self._graph_data = self._graph_data_raw = None
        self._graph_risks = self._risk_median = self._scaler = None

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def _build_graph(self):
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        data_raw = x_train[feat_names].fillna(0).values.astype(np.float64)
        if len(data_raw) > self._max_graph_size:
            data_raw = data_raw[np.random.choice(len(data_raw), self._max_graph_size, replace=False)]
        n = len(data_raw)
        self._scaler = MinMaxScaler()
        data_norm = self._scaler.fit_transform(data_raw)
        dist_matrix = cdist(data_norm, data_norm, metric=self._metric)
        kde = KernelDensity(bandwidth=self._bandwidth, kernel="gaussian")
        kde.fit(data_norm)
        densities = np.exp(kde.score_samples(data_norm))

        rows, cols, weights = [], [], []
        for i in range(n):
            neighbours = np.argsort(dist_matrix[i])[1:self._k + 1]
            for j in neighbours:
                min_density = max(min(densities[i], densities[j]), 1e-12)
                if min_density >= self._density_threshold:
                    w = dist_matrix[i, j] / min_density
                    rows.extend([i, j])
                    cols.extend([j, i])
                    weights.extend([w, w])

        self._graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
        self._graph_data = data_norm
        self._graph_data_raw = data_raw
        self._graph_risks = self._predict_fn(data_raw)
        self._risk_median = float(np.median(self._graph_risks))

    def generate_counterfactual(self, patient_features, target_outcome=None):
        if self._graph is None:
            self._build_graph()
        feat_names = self._adapter.feature_names()
        x_query_raw = patient_features[feat_names].fillna(0).values[0].astype(np.float64)
        x_query_norm = self._scaler.transform(x_query_raw.reshape(1, -1))[0]
        threshold = float(target_outcome) if target_outcome is not None else self._risk_median

        target_mask = self._graph_risks > threshold if self._higher_is_better else self._graph_risks < threshold
        if self._direction_priors:
            target_mask = target_mask & direction_compatible_mask(self._graph_data_raw, x_query_raw, feat_names, self._direction_priors)
        if not target_mask.any():
            return None

        entry_node = int(np.argmin(cdist(x_query_norm.reshape(1, -1), self._graph_data, metric=self._metric)[0]))
        sp_dists = shortest_path(self._graph, directed=False, indices=entry_node)
        reachable_targets = np.where(target_mask & np.isfinite(sp_dists))[0]
        if len(reachable_targets) == 0:
            return None

        best_target = reachable_targets[np.argmin(sp_dists[reachable_targets])]
        cf_df = pd.DataFrame([self._graph_data_raw[best_target]], columns=feat_names)
        cf_df = snap_categoricals(cf_df, self._categorical_values)
        cf_df = enforce_immutables(patient_features[feat_names].head(1), cf_df, self._immutable)
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
        return self._make_explanation(patient_id, content=content, method="FACE",
            counterfactual_df=cf, original_prediction=orig_pred, cf_prediction=cf_pred,
            raw_query=raw_query, raw_cf=raw_cf, higher_is_better=self._higher_is_better, confidence_metrics=confidence_metrics)

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)