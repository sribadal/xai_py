"""
K-nearest-neighbor retrieval for case-based survival explanations.

Retrieves the k closest training patients by Gower or Euclidean
distance on pre-treatment features, enriched with outcome/treatment
context and predicted trajectories at configurable time horizons.
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
from xai_decision_making.explainability.sanity_checks import CASE_BASED_CHECKS
from xai_decision_making.explainability.case_based._base import (
    _column_ranges,
    _euclidean_distance_row,
    _gower_distance_row,
    _run_sanity_checks,
    compute_case_confidence,
    enrich_with_context,
)

registry = ExplainerRegistry()

_POST_TREATMENT_SUBSTRINGS = [
    "adjuvant", "chemo_after", "radiation_after", "recurrence",
    "follow_up", "postop", "post_op", "response",
    "ts_surg", "had_surg", "impulse_surg",
    "ts_syst", "had_syst", "impulse_syst",
]

_POST_TREATMENT_EXACT = {
    "tumgericht_ther", "chir", "destr", "rt", "chemo",
}

def _default_pretreatment_features(all_features):
    return [
        f for f in all_features
        if f not in _POST_TREATMENT_EXACT
        and not any(sub in f.lower() for sub in _POST_TREATMENT_SUBSTRINGS)
    ]

def _compute_trajectories(adapter, x_train, order, time_horizons, horizon_labels=None):
    if not time_horizons:
        return None
    trajectories = []
    for idx in order:
        case_row = x_train.iloc[[idx]].reset_index(drop=True)
        traj = {}
        for h in time_horizons:
            label = horizon_labels.get(h, f"{h}") if horizon_labels else f"{h}"
            try:
                pred = adapter.predict(case_row, time_horizon=h)
                score = pred.get("risk_score")
                if score is not None:
                    traj[label] = float(score)
                else:
                    traj[label] = None
            except Exception:
                traj[label] = None
        trajectories.append(traj)
    return trajectories

@registry.register
class KNNRetriever(CaseBasedExplainer):
    _modality = ExplainerModality.CASE_BASED
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        time_horizon=None,
        time_horizons=None,
        horizon_labels=None,
        metric="gower",
        pretreatment_cols=None,
    ):
        super().__init__(model_adapter)
        if metric not in ("gower", "euclidean"):
            raise ValueError(f"Unsupported metric: {metric!r}")
        self._metric = metric
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

    @property
    def modality(self):
        return self._modality

    def pretreatment_features(self):
        return list(self._pretreatment_cols)

    def retrieve_similar(self, patient_features, k=3):
        x_train = self._adapter.training_data()
        pre_cols = self._pretreatment_cols

        distances = []
        for _, train_row in x_train.iterrows():
            train_df = train_row.to_frame().T.reset_index(drop=True)
            if self._metric == "gower":
                d = _gower_distance_row(
                    patient_features, train_df,
                    self._numeric_cols, self._categorical_cols,
                    self._col_ranges,
                )
            else:
                d = _euclidean_distance_row(
                    patient_features, train_df,
                    self._numeric_cols, self._col_ranges,
                )
            distances.append(d)

        order = np.argsort(distances)[:k]
        result = x_train.iloc[order][pre_cols].copy().reset_index(drop=True)
        result["_distance"] = [distances[i] for i in order]

        if hasattr(self._adapter, "inverse_transform_numeric"):
            result = self._adapter.inverse_transform_numeric(result)

        result = enrich_with_context(result, self._adapter, order.tolist())

        trajectories = _compute_trajectories(
            self._adapter, x_train, order,
            self._time_horizons, self._horizon_labels,
        )
        if trajectories is not None:
            result["_trajectory"] = trajectories

        return result

    def explain(self, patient_features, patient_id=None):
        similar = self.retrieve_similar(patient_features)
        confidence_metrics = compute_case_confidence(similar)
        return self._make_explanation(
            patient_id,
            content=similar,
            method="KNN",
            metric=self._metric,
            confidence_metrics=confidence_metrics,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df, CASE_BASED_CHECKS)