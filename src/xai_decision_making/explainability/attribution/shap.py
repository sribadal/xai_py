"""
SHAP attribution for survival models.

Provides SurvSHAPExplainer (auto-selects TreeExplainer or
KernelExplainer based on model type and time horizon) and
MeanAbsSHAPExplainer (aggregated global importance via mean
absolute SHAP values across a patient sample).
"""
from __future__ import annotations

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
    SurvivalModelAdapter,
)
from xai_decision_making.explainability.sanity_checks import (
    ATTRIBUTION_CHECKS,
    DeletionFaithfulnessCheck,
)
from xai_decision_making.explainability.attribution._base import (
    _attrs_from_shap_values,
    _risk_predict_fn,
    _run_sanity_checks,
    _try_tree_explainer,
)

registry = ExplainerRegistry()

@registry.register
class SurvSHAPExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        n_background=100,
        use_tree_explainer=None,
        time_horizon=None,
    ):
        super().__init__(model_adapter)
        self._n_background = n_background
        self._use_tree_explainer = use_tree_explainer
        self._time_horizon = time_horizon
        self._shap_explainer = None
        self._use_kernel = False

    def _build_shap_explainer(self):
        model = self._adapter.raw_model
        x_train = self._adapter.training_data()

        if self._use_tree_explainer is not False and self._time_horizon is None:
            tree_exp = _try_tree_explainer(model)
            if tree_exp is not None:
                self._shap_explainer = tree_exp
                self._use_kernel = False
                return

        if self._use_tree_explainer is True and self._time_horizon is not None:
            raise RuntimeError(
                "TreeExplainer does not support time_horizon. "
                "Use KernelExplainer (use_tree_explainer=False)."
            )

        if self._use_tree_explainer is True:
            raise RuntimeError(
                "TreeExplainer failed and use_tree_explainer=True."
            )

        background = shap.kmeans(
            x_train.fillna(x_train.median(numeric_only=True)),
            min(self._n_background, len(x_train)),
        )
        self._shap_explainer = shap.KernelExplainer(
            _risk_predict_fn(self._adapter, self._time_horizon),
            background,
        )
        self._use_kernel = True

    @property
    def modality(self):
        return self._modality

    def feature_importances(self, patient_features):
        if self._shap_explainer is None:
            self._build_shap_explainer()

        feat_names = self._adapter.feature_names()
        x = patient_features[feat_names].fillna(0).values

        shap_vals = (
            self._shap_explainer.shap_values(x, nsamples=50)
            if self._use_kernel
            else self._shap_explainer.shap_values(x)
        )

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[-1]
        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim == 2:
            shap_vals = shap_vals[0]

        return _attrs_from_shap_values(shap_vals, feat_names)

    def explain(self, patient_features, patient_id=None):
        return self._make_explanation(
            patient_id,
            content=self.feature_importances(patient_features),
            method="SHAP",
            use_kernel=self._use_kernel,
        )

    def sanity_check(self, held_out_df):
        if self._shap_explainer is None:
            self._build_shap_explainer()
        checks = (
            ATTRIBUTION_CHECKS
            if not self._use_kernel
            else [DeletionFaithfulnessCheck()]
        )
        return _run_sanity_checks(self, held_out_df, checks)


@registry.register
class MeanAbsSHAPExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_background=100, time_horizon=None):
        super().__init__(model_adapter)
        self._shap_exp = SurvSHAPExplainer(
            model_adapter,
            n_background=n_background,
            time_horizon=time_horizon,
        )
        self._global_attrs = None

    @property
    def modality(self):
        return self._modality

    def _compute_global(self):
        x_train = self._adapter.training_data()
        sample = x_train.sample(
            n=min(200, len(x_train)),
            replace=False,
        ).reset_index(drop=True)
        all_attrs = []
        for _, row in sample.iterrows():
            try:
                all_attrs.append(
                    self._shap_exp.feature_importances(
                        row.to_frame().T.reset_index(drop=True)
                    )
                )
            except Exception:
                pass
        if all_attrs:
            self._global_attrs = {
                f: np.mean([abs(a.get(f, 0)) for a in all_attrs])
                for f in self._adapter.feature_names()
            }

    def feature_importances(self, patient_features):
        return self._shap_exp.feature_importances(patient_features)

    def global_importances(self):
        if self._global_attrs is None:
            self._compute_global()
        return self._global_attrs or {}

    def explain(self, patient_features, patient_id=None):
        return self._make_explanation(
            patient_id,
            content=self.feature_importances(patient_features),
            method="MeanAbsSHAP",
            global_importances=self.global_importances(),
        )

    def sanity_check(self, held_out_df):
        if self._shap_exp._shap_explainer is None:
            self._shap_exp._build_shap_explainer()
        checks = (
            ATTRIBUTION_CHECKS
            if not self._shap_exp._use_kernel
            else [DeletionFaithfulnessCheck()]
        )
        return _run_sanity_checks(self, held_out_df, checks)