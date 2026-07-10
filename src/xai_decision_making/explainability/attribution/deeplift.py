"""
DeepLift and DeepLiftSHAP attribution for survival torch models.

Wraps Captum's DeepLift and DeepLiftShap with a risk-output wrapper,
computes completeness-axiom confidence via convergence delta, and
supports time-horizon-specific explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import DeepLift, DeepLiftShap

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

def _completeness_from_delta(delta):
    try:
        return float(abs(delta.item()))
    except (AttributeError, RuntimeError):
        return float(np.abs(delta.detach().cpu().numpy()).mean())


class _RiskWrapper(nn.Module):
    def __init__(self, model, adapter, time_horizon):
        super().__init__()
        self.model = model
        self.adapter = adapter
        self.time_horizon = time_horizon

    def forward(self, x):
        if self.time_horizon is None:
            out = self.model(x)
        else:
            t_norm = float(self.time_horizon) / 60.0
            out = self.model(x, target_t_norm=t_norm)

        if out.ndim == 1:
            return out.unsqueeze(1)
        return out


@registry.register
class DeepLiftExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        baseline="mean",
        time_horizon=None,
        random_state=None,
    ):
        super().__init__(model_adapter)
        self._baseline_type = baseline
        self._dl = None
        self._fwd = None
        self._time_horizon = time_horizon
        self._random_state = random_state
        self._last_completeness_gap = None
        self._last_output_range = None

    def _build(self):
        model = self._adapter.raw_model
        if not isinstance(model, nn.Module):
            raise TypeError("DeepLift only works with torch models.")
        model.eval()

        self._fwd = _RiskWrapper(model, self._adapter, self._time_horizon)
        self._dl = DeepLift(self._fwd)

    @property
    def modality(self):
        return self._modality

    def feature_importances(self, patient_features):
        if self._dl is None:
            self._build()

        feat_names = self._adapter.feature_names()
        x_train = self._adapter.training_data()

        x = torch.tensor(
            patient_features[feat_names].fillna(0).values,
            dtype=torch.float32,
            requires_grad=True,
        )

        if self._baseline_type == "mean":
            bl = torch.tensor(
                x_train[feat_names].fillna(0).mean().values[None, :],
                dtype=torch.float32,
            )
        else:
            bl = torch.zeros_like(x)

        attrs, delta = self._dl.attribute(
            x, baselines=bl, return_convergence_delta=True,
        )
        self._last_completeness_gap = _completeness_from_delta(delta)

        with torch.no_grad():
            f_x = float(self._fwd(x).item())
            f_bl = float(self._fwd(bl).item())
        self._last_output_range = abs(f_x - f_bl)

        return _attrs_from_shap_values(attrs.detach().numpy()[0], feat_names)

    def explain(self, patient_features, patient_id=None):
        importances = self.feature_importances(patient_features)
        gap = self._last_completeness_gap
        rng = self._last_output_range or 0.0
        ratio = (gap / rng) if (gap is not None and rng > 1e-9) else None

        confidence = {
            "completeness_gap": round(gap, 6) if gap is not None else None,
            "completeness_ratio": round(ratio, 4) if ratio is not None else None,
            "output_range": round(rng, 6),
            "source": "tier_c_native",
        }
        return self._make_explanation(
            patient_id,
            content=importances,
            method="DeepLift",
            baseline=self._baseline_type,
            confidence_metrics=confidence,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df, ATTRIBUTION_CHECKS)


@registry.register
class DeepLiftSHAPExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, n_samples=50, time_horizon=None, random_state=None):
        super().__init__(model_adapter)
        self._n_samples = n_samples
        self._time_horizon = time_horizon
        self._random_state = random_state
        self._dls = None
        self._fwd = None
        self._last_completeness_gap = None
        self._last_output_range = None

    def _build(self):
        model = self._adapter.raw_model
        if not isinstance(model, nn.Module):
            raise TypeError("DeepLiftSHAP only works with torch models.")
        model.eval()

        self._fwd = _RiskWrapper(model, self._adapter, self._time_horizon)
        self._dls = DeepLiftShap(self._fwd)

    @property
    def modality(self):
        return self._modality

    def feature_importances(self, patient_features):
        if self._dls is None:
            self._build()

        feat_names = self._adapter.feature_names()
        x_train = self._adapter.training_data()

        x = torch.tensor(
            patient_features[feat_names].fillna(0).values,
            dtype=torch.float32,
            requires_grad=True,
        )

        bg_np = x_train[feat_names].fillna(0).values
        rng = np.random.default_rng(self._random_state)
        bg_idx = rng.choice(
            len(bg_np),
            size=min(self._n_samples, len(bg_np)),
            replace=False,
        )
        baselines = torch.tensor(bg_np[bg_idx], dtype=torch.float32)

        attrs, delta = self._dls.attribute(
            x, baselines=baselines, return_convergence_delta=True,
        )
        self._last_completeness_gap = _completeness_from_delta(delta)

        with torch.no_grad():
            f_x = float(self._fwd(x).item())
            f_bls = self._fwd(baselines).detach().cpu().numpy().ravel()
            f_bl_mean = float(np.mean(f_bls))
        self._last_output_range = abs(f_x - f_bl_mean)

        return _attrs_from_shap_values(attrs.detach().numpy()[0], feat_names)

    def explain(self, patient_features, patient_id=None):
        importances = self.feature_importances(patient_features)
        gap = self._last_completeness_gap
        rng = self._last_output_range or 0.0
        ratio = (gap / rng) if (gap is not None and rng > 1e-9) else None

        confidence = {
            "completeness_gap": round(gap, 6) if gap is not None else None,
            "completeness_ratio": round(ratio, 4) if ratio is not None else None,
            "output_range": round(rng, 6),
            "n_baseline_samples": self._n_samples,
            "source": "tier_c_native",
        }
        return self._make_explanation(
            patient_id,
            content=importances,
            method="DeepLiftSHAP",
            confidence_metrics=confidence,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df, ATTRIBUTION_CHECKS)