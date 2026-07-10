"""
SAGE (Shapley Additive Global importancE) for survival models.

Computes global feature importance using the marginal imputer and
permutation estimator from the sage-importance library. Returns
dataset-level Shapley values rather than per-patient attributions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sage

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

@registry.register
class SAGEExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_samples=512, threshold=0.025, time_horizon=None):
        super().__init__(model_adapter)
        self._n_samples = n_samples
        self._threshold = threshold
        self._time_horizon = time_horizon
        self._global_importances = None

    @property
    def modality(self):
        return self._modality

    def _make_predict_fn(self, feat_names):
        adapter = self._adapter
        time_horizon = self._time_horizon

        def batch_predict(X):
            if time_horizon is not None:
                df = pd.DataFrame(X, columns=feat_names)
                return adapter.predict_at_horizon(df, time_horizon)
            scores = []
            for i in range(len(X)):
                row = pd.DataFrame(X[i : i + 1], columns=feat_names)
                try:
                    scores.append(adapter.predict(row)["risk_score"])
                except Exception:
                    scores.append(np.nan)
            return np.array(scores)
        return batch_predict

    def _compute(self):
        feat_names = self._adapter.feature_names()
        x_train = self._adapter.training_data()
        predict_fn = self._make_predict_fn(feat_names)

        X_bg = x_train[feat_names].fillna(0).values
        X_sample = X_bg[: min(self._n_samples, len(X_bg))]

        print(
            f"  SAGE: computing background predictions for "
            f"{len(X_sample)} samples...",
            flush=True,
        )
        Y_sample = predict_fn(X_sample)

        print(
            f"  SAGE: running permutation estimator "
            f"(thresh={self._threshold})...",
            flush=True,
        )
        imputer = sage.MarginalImputer(predict_fn, X_bg)
        estimator = sage.PermutationEstimator(imputer, "mse")
        sage_values = estimator(
            X_sample,
            Y_sample,
            thresh=self._threshold,
            verbose=True,
        )

        self._global_importances = dict(zip(feat_names, sage_values.values))

    def feature_importances(self, patient_features):
        if self._global_importances is None:
            self._compute()
        return self._global_importances

    def explain(self, patient_features, patient_id=None):
        return self._make_explanation(
            patient_id,
            content=self.feature_importances(patient_features),
            method="SAGE",
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df, ATTRIBUTION_CHECKS)