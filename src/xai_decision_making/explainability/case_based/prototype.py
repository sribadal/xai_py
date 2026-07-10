"""
Prototype-based case explanations via k-medoids clustering.

Learns a small set of prototype patients from the training cohort
using k-medoids on pre-treatment features so that every prototype
is a real patient. Explains a new case by its similarity to the
nearest prototype(s), including per-feature differences, shared
features, outcome context, and predicted trajectories.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    CaseBasedExplainer,
    Explanation,
    ExplainerModality,
    ExplainerRegistry,
    ModelTarget,
    SurvivalModelAdapter,
)
from xai_decision_making.explainability.sanity_checks import (
    CaseBasedDistanceCheck,
    CaseBasedPretreatmentCheck,
)
from xai_decision_making.explainability.case_based._base import (
    _column_ranges,
    _euclidean_distance_row,
    _gower_distance_row,
    _run_sanity_checks,
    compute_case_confidence,
    enrich_with_context,
)
from xai_decision_making.explainability.case_based.knn import (
    _default_pretreatment_features,
    _compute_trajectories,
)

registry = ExplainerRegistry()

def _pairwise_distances(X, numeric_cols, categorical_cols, col_ranges, metric="gower"):
    n = len(X)
    D = np.zeros((n, n))
    rows = [X.iloc[i : i + 1].reset_index(drop=True) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "gower":
                d = _gower_distance_row(
                    rows[i], rows[j], numeric_cols, categorical_cols, col_ranges
                )
            else:
                d = _euclidean_distance_row(
                    rows[i], rows[j], numeric_cols, col_ranges
                )
            D[i, j] = D[j, i] = d
    return D

def _kmedoids(D, n_prototypes, max_iter=100, seed=42):
    rng = np.random.RandomState(seed)
    n = D.shape[0]
    n_prototypes = min(n_prototypes, n)

    medoids = [int(np.argmin(D.sum(axis=1)))]
    for _ in range(1, n_prototypes):
        dists_to_nearest = D[:, medoids].min(axis=1)
        candidate = int(np.argmax(dists_to_nearest))
        medoids.append(candidate)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        changed = False
        for m_idx in range(n_prototypes):
            cluster_members = np.where(labels == m_idx)[0]
            if len(cluster_members) <= 1:
                continue
            current_cost = D[cluster_members][:, medoids[m_idx]].sum()
            best_candidate, best_cost = medoids[m_idx], current_cost
            for c in cluster_members:
                if c == medoids[m_idx]:
                    continue
                cost = D[cluster_members][:, c].sum()
                if cost < best_cost:
                    best_cost, best_candidate = cost, c
            if best_candidate != medoids[m_idx]:
                medoids[m_idx] = best_candidate
                changed = True
        if not changed:
            break

    labels = np.argmin(D[:, medoids], axis=1)
    return np.array(medoids), labels

@registry.register
class PrototypeExplainer(CaseBasedExplainer):

    _modality = ExplainerModality.CASE_BASED
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        time_horizon=None,
        time_horizons=None,
        horizon_labels=None,
        n_prototypes=12,
        metric="gower",
        pretreatment_cols=None,
        max_train_for_clustering=2000,
    ):
        super().__init__(model_adapter)
        if metric not in ("gower", "euclidean"):
            raise ValueError(f"Unsupported metric: {metric!r}")
        self._metric = metric
        self._n_prototypes = n_prototypes
        self._max_train = max_train_for_clustering
        self._time_horizon = time_horizon
        self._time_horizons = time_horizons
        self._horizon_labels = horizon_labels or {}

        all_feats = self._adapter.feature_names()
        self._pretreatment_cols = (
            pretreatment_cols if pretreatment_cols is not None
            else _default_pretreatment_features(all_feats)
        )

        x_train = self._adapter.training_data()
        self._numeric_cols = [
            c for c in self._pretreatment_cols
            if c in x_train.columns and pd.api.types.is_numeric_dtype(x_train[c])
        ]
        self._categorical_cols = [
            c for c in self._pretreatment_cols
            if c in x_train.columns and not pd.api.types.is_numeric_dtype(x_train[c])
        ]
        self._col_ranges = _column_ranges(x_train, self._numeric_cols)

        self._prototypes = None
        self._prototype_predictions = None
        self._prototype_train_indices = None
        self._cluster_labels = None
        self._cluster_sizes = None
        self._build_prototypes()

    def _build_prototypes(self):
        x_train = self._adapter.training_data()
        pre_cols = [c for c in self._pretreatment_cols if c in x_train.columns]

        if len(x_train) > self._max_train:
            subset_idx = np.random.RandomState(42).choice(
                len(x_train), self._max_train, replace=False
            )
            x_sub = x_train.iloc[subset_idx].reset_index(drop=True)
        else:
            x_sub = x_train.reset_index(drop=True)

        D = _pairwise_distances(
            x_sub[pre_cols].fillna(0),
            self._numeric_cols,
            self._categorical_cols,
            self._col_ranges,
            metric=self._metric,
        )

        medoid_idx, labels = _kmedoids(D, self._n_prototypes)

        if len(x_train) > self._max_train:
            global_idx = [int(subset_idx[m]) for m in medoid_idx]
        else:
            global_idx = [int(m) for m in medoid_idx]

        self._prototype_train_indices = list(global_idx)
        self._prototypes = x_train.iloc[global_idx][pre_cols].reset_index(drop=True)
        self._cluster_labels = labels
        self._cluster_sizes = {
            i: int(np.sum(labels == i)) for i in range(len(medoid_idx))
        }

        self._prototype_predictions = []
        for i, gidx in enumerate(global_idx):
            proto_row = x_train.iloc[gidx : gidx + 1].reset_index(drop=True)
            try:
                pred = self._adapter.predict(proto_row)
                self._prototype_predictions.append(pred)
            except Exception:
                self._prototype_predictions.append({})

    @property
    def modality(self):
        return self._modality

    def pretreatment_features(self):
        return list(self._pretreatment_cols)

    def retrieve_similar(self, patient_features, k=3):
        k = min(k, len(self._prototypes))
        distances = []

        for i in range(len(self._prototypes)):
            proto = self._prototypes.iloc[i : i + 1].reset_index(drop=True)
            if self._metric == "gower":
                d = _gower_distance_row(
                    patient_features, proto,
                    self._numeric_cols, self._categorical_cols,
                    self._col_ranges,
                )
            else:
                d = _euclidean_distance_row(
                    patient_features, proto,
                    self._numeric_cols, self._col_ranges,
                )
            distances.append(d)

        order = np.argsort(distances)[:k]
        result = self._prototypes.iloc[order].copy().reset_index(drop=True)
        result["_distance"] = [distances[i] for i in order]
        result["_cluster_size"] = [self._cluster_sizes.get(i, 0) for i in order]
        result["_risk_score"] = [
            self._prototype_predictions[i].get("risk_score", np.nan)
            for i in order
        ]

        if hasattr(self._adapter, "inverse_transform_numeric"):
            result = self._adapter.inverse_transform_numeric(result)

        if self._prototype_train_indices is not None:
            selected_train_indices = [
                self._prototype_train_indices[i] for i in order
            ]
            result = enrich_with_context(
                result, self._adapter, selected_train_indices,
            )

        if self._time_horizons and self._prototype_train_indices is not None:
            x_train = self._adapter.training_data()
            train_indices = [self._prototype_train_indices[i] for i in order]
            trajectories = _compute_trajectories(
                self._adapter, x_train, train_indices,
                self._time_horizons, self._horizon_labels,
            )
            if trajectories is not None:
                result["_trajectory"] = trajectories

        return result

    def explain(self, patient_features, patient_id=None):
        similar = self.retrieve_similar(patient_features)

        nearest = similar.iloc[0:1].drop(
            columns=["_distance", "_cluster_size", "_risk_score", "_trajectory"],
            errors="ignore",
        )

        if hasattr(self._adapter, "inverse_transform_numeric"):
            nearest = self._adapter.inverse_transform_numeric(nearest)
            patient_features_clin = self._adapter.inverse_transform_numeric(patient_features.copy())
        else:
            patient_features_clin = patient_features

        pre_cols = [c for c in self._pretreatment_cols if c in nearest.columns]

        diffs = {}
        shared = {}
        for c in pre_cols:
            q_val = patient_features_clin[c].iloc[0]
            p_val = nearest[c].iloc[0]
            try:
                diff = float(q_val) - float(p_val)
                if diff == 0.0:
                    shared[c] = float(q_val)
                else:
                    diffs[c] = diff
            except (TypeError, ValueError):
                if q_val == p_val:
                    shared[c] = q_val
                else:
                    diffs[c] = "mismatch"

        confidence_metrics = compute_case_confidence(similar)

        return self._make_explanation(
            patient_id,
            content={
                "similar_prototypes": similar,
                "feature_diffs_to_nearest": diffs,
                "shared_features_with_nearest": shared,
            },
            method="Prototype",
            metric=self._metric,
            n_prototypes=self._n_prototypes,
            confidence_metrics=confidence_metrics,
        )

    def sanity_check(self, held_out_df):
        prototype_checks = [
            CaseBasedPretreatmentCheck(),
            CaseBasedDistanceCheck(threshold=3.0),
        ]
        return _run_sanity_checks(self, held_out_df, prototype_checks)

    @property
    def prototypes(self):
        return self._prototypes.copy()

    @property
    def cluster_sizes(self):
        return dict(self._cluster_sizes)