"""
GradientSHAP attribution for survival torch models.

Wraps Captum's GradientShap to compute gradient-based Shapley
value estimates against a sampled background set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import GradientShap

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
from xai_decision_making.explainability.attribution._base import (
    _attrs_from_shap_values,
    _run_sanity_checks,
)

registry = ExplainerRegistry()

@registry.register
class GradientSHAPExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_samples=50, stdevs=0.0):
        super().__init__(model_adapter)
        self._n_samples = n_samples
        self._stdevs = stdevs
        self._gs = None

    def _build(self):
        model = self._adapter.raw_model
        if not isinstance(model, nn.Module):
            raise TypeError("GradientSHAP only works with torch models.")
        model.eval()
        self._gs = GradientShap(model)

    @property
    def modality(self):
        return self._modality

    def feature_importances(self, patient_features):
        if self._gs is None:
            self._build()

        feat_names = self._adapter.feature_names()
        x_train = self._adapter.training_data()

        x = torch.tensor(
            patient_features[feat_names].fillna(0).values,
            dtype=torch.float32,
            requires_grad=True,
        )

        bg_np = x_train[feat_names].fillna(0).values
        bg_idx = np.random.choice(
            len(bg_np),
            size=min(self._n_samples, len(bg_np)),
            replace=False,
        )
        baselines = torch.tensor(bg_np[bg_idx], dtype=torch.float32)

        attrs = self._gs.attribute(
            x,
            baselines=baselines,
            stdevs=self._stdevs,
        )
        return _attrs_from_shap_values(attrs.detach().numpy()[0], feat_names)

    def explain(self, patient_features, patient_id=None):
        return self._make_explanation(
            patient_id,
            content=self.feature_importances(patient_features),
            method="GradientSHAP",
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df, ATTRIBUTION_CHECKS)