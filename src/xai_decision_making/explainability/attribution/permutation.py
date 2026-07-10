"""
Permutation importance for survival models.

Estimates local feature importance by repeatedly replacing each
feature with a random training value and measuring the change in
predicted risk. Reports per-feature standard deviation across
repeats as a stability confidence metric.
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
from xai_decision_making.explainability.sanity_checks import DeletionFaithfulnessCheck
from xai_decision_making.explainability.attribution._base import _run_sanity_checks

registry = ExplainerRegistry()

@registry.register
class PermutationImportanceExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_repeats=10, n_background=100, time_horizon=None):
        super().__init__(model_adapter)
        self._n_repeats = n_repeats
        self._n_background = n_background
        self._time_horizon = time_horizon
        self._last_stds = None

    @property
    def modality(self):
        return self._modality

    def _get_risk(self, patient_df):
        if self._time_horizon is not None:
            return self._adapter.predict_at_horizon(patient_df, self._time_horizon)[0]
        return self._adapter.predict(patient_df)["risk_score"]

    def feature_importances(self, patient_features):
        feat_names = self._adapter.feature_names()
        x_train = self._adapter.training_data()

        background = x_train[feat_names].fillna(0).sample(
            n=min(self._n_background, len(x_train)),
            replace=False,
        ).reset_index(drop=True)

        patient = patient_features[feat_names].fillna(0).reset_index(drop=True)
        base_pred = self._get_risk(patient)

        importances = {}
        stds = {}
        for feat in feat_names:
            deltas = []
            for _ in range(self._n_repeats):
                perturbed = patient.copy()
                perturbed[feat] = background[feat].sample(1).values[0]
                try:
                    deltas.append(abs(base_pred - self._get_risk(perturbed)))
                except Exception:
                    pass
            importances[feat] = float(np.mean(deltas)) if deltas else 0.0
            stds[feat] = float(np.std(deltas)) if deltas else 0.0

        self._last_stds = stds
        return importances

    def explain(self, patient_features, patient_id=None):
        importances = self.feature_importances(patient_features)

        stds = self._last_stds or {}
        cvs = []
        for feat, std_val in stds.items():
            mean_val = importances.get(feat, 0.0)
            if mean_val > 1e-9:
                cvs.append(std_val / mean_val)
        median_cv = float(np.median(cvs)) if cvs else float("nan")
        mean_cv = float(np.mean(cvs)) if cvs else float("nan")

        confidence = {
            "median_relative_std": round(median_cv, 4) if cvs else None,
            "mean_relative_std": round(mean_cv, 4) if cvs else None,
            "n_repeats": self._n_repeats,
            "n_features_with_signal": len(cvs),
            "source": "tier_c_native",
        }
        return self._make_explanation(
            patient_id,
            content=importances,
            method="PermutationImportance",
            confidence_metrics=confidence,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(
            self,
            held_out_df,
            [DeletionFaithfulnessCheck()],
        )