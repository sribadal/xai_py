"""
Treatment-scenario counterfactual explainer.

Clinically grounded CF method that constructs treatment-flip scenarios
(surgery vs. no surgery, chemotherapy regimen swaps) by toggling
treatment indicators and recomputing derived features (time-since-
treatment, impulse flags). No black-box optimisation: the CF is an
explicit clinical "what if" with full plausibility enforcement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xai_decision_making.explainability.base import (
    AggregatedSanityResult, CounterfactualExplainer, ExplainerModality,
    ExplainerRegistry, Explanation, ModelTarget,
)
from xai_decision_making.explainability.counterfactual._base import (
    _run_sanity_checks, compute_cf_confidence, compute_content_deltas,
    enforce_immutables, extract_scalar_prediction,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config, condor_target_t_norm, derive_treatment_started_flags,
    snap_and_verify,
)

registry = ExplainerRegistry()

TREATMENT_TOGGLE_FEATURES = [
    "tumgericht_ther",
    "chir",
    "destr",
    "rt",
    "chemo",
    "target",
]

TREATMENT_INTERVAL_FEATURES = [
    "chir_int1", "chir_int2",
    "destr_int1",
    "rt_start_int1",
    "meta_ther_int1", "meta_ther_int2", "meta_ther_int3",
    "syst_start_int1", "syst_stop_int1",
    "syst_start_int2", "syst_stop_int2",
    "syst_start_int3", "syst_stop_int3",
    "syst_start_int4", "syst_stop_int4",
    "syst_start_int5", "syst_stop_int5",
    "syst_start_int6", "syst_stop_int6",
]

TREATMENT_DETAIL_FEATURES = [
    "chir_type1", "chir_type2",
    "chir_rad1", "chir_rad2",
    "chir_tech1", "chir_tech2",
    "destr_type1",
    "rt_type1",
    "syst_schemanum1", "syst_schemanum2", "syst_schemanum3",
    "syst_schemanum4", "syst_schemanum5", "syst_schemanum6",
    "syst_kuren1", "syst_kuren2", "syst_kuren3",
    "syst_kuren4", "syst_kuren5", "syst_kuren6",
    "syst_aanpassing1", "syst_aanpassing2", "syst_aanpassing3",
    "syst_aanpassing4", "syst_aanpassing5", "syst_aanpassing6",
    "syst_prepost1", "syst_prepost2", "syst_prepost3",
    "syst_prepost4", "syst_prepost5", "syst_prepost6",
    "syst_chemort1", "syst_chemort2", "syst_chemort3",
    "syst_chemort4", "syst_chemort5", "syst_chemort6",
    "syst_code1", "syst_code2", "syst_code3",
    "syst_code4", "syst_code5", "syst_code6",
    "syst_compl1", "syst_compl2",
    "geen_ther_reden", "geen_ther_type",
    "meta_rt", "meta_chir", "lok_chemo_immuno",
]

ALL_TREATMENT_FEATURES = sorted(
    set(TREATMENT_TOGGLE_FEATURES + TREATMENT_INTERVAL_FEATURES + TREATMENT_DETAIL_FEATURES)
)


def _make_no_treatment_scenario(x_orig, feat_names, immutable):
    cf = x_orig.copy()
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    for feat in ALL_TREATMENT_FEATURES:
        if feat in name_to_idx and feat not in immutable:
            cf[name_to_idx[feat]] = 0.0
    return cf


def _make_toggle_scenario(x_orig, feat_names, toggle_features, immutable):
    cf = x_orig.copy()
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    for feat in toggle_features:
        if feat in name_to_idx and feat not in immutable:
            cf[name_to_idx[feat]] = 1.0 - cf[name_to_idx[feat]]
    return cf


def _make_surgery_scenario(x_orig, feat_names, immutable, *, add_surgery=True):
    cf = x_orig.copy()
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    surgery_feat = "chir"
    if surgery_feat in name_to_idx and surgery_feat not in immutable:
        cf[name_to_idx[surgery_feat]] = 1.0 if add_surgery else 0.0
    return cf


SCENARIO_REGISTRY = {
    "no_treatment": lambda x, fn, imm: _make_no_treatment_scenario(x, fn, imm),
    "toggle_chemo": lambda x, fn, imm: _make_toggle_scenario(x, fn, ["chemo"], imm),
    "toggle_surgery": lambda x, fn, imm: _make_toggle_scenario(x, fn, ["chir"], imm),
    "add_surgery": lambda x, fn, imm: _make_surgery_scenario(x, fn, imm, add_surgery=True),
    "remove_surgery": lambda x, fn, imm: _make_surgery_scenario(x, fn, imm, add_surgery=False),
}


@registry.register
class TreatmentScenarioExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, scenario="toggle_chemo", immutable=None,
                 time_horizon=None, custom_scenario_fn=None):
        super().__init__(model_adapter)
        self._scenario = scenario
        self._time_horizon = time_horizon
        self._custom_scenario_fn = custom_scenario_fn
        self._cfg = build_plausibility_config(model_adapter, extra_immutable=immutable)
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def generate_counterfactual(self, patient_features, target_outcome=None):
        feat_names = self._adapter.feature_names()
        target_t_norm = condor_target_t_norm(self._adapter, self._time_horizon)
        x_orig = patient_features[feat_names].fillna(0).values[0].astype(np.float64)
        immutable_set = set(self._immutable)

        if self._custom_scenario_fn is not None:
            cf_values = self._custom_scenario_fn(x_orig, feat_names, immutable_set)
        else:
            scenario_fn = SCENARIO_REGISTRY.get(self._scenario)
            if scenario_fn is None:
                raise ValueError(f"Unknown scenario: {self._scenario!r}. "
                                 f"Available: {list(SCENARIO_REGISTRY)}")
            cf_values = scenario_fn(x_orig, feat_names, immutable_set)

        if target_t_norm is not None:
            cf_values, _ = derive_treatment_started_flags(cf_values, feat_names, target_t_norm)

        cf_df, _ = snap_and_verify(cf_values, feat_names, self._categorical_values,
            lambda x: self._adapter.predict(pd.DataFrame(x, columns=feat_names))["risk_score"]
                if isinstance(self._adapter.predict(pd.DataFrame(x, columns=feat_names)), dict)
                else np.array([0.0]),
            target=None, higher_is_better=self._higher_is_better, query=x_orig,
            constraints=self._cfg.constraints, target_t_norm=target_t_norm)
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
        return self._make_explanation(patient_id, content=content, method="TreatmentScenario",
            scenario=self._scenario, counterfactual_df=cf, original_prediction=orig_pred,
            cf_prediction=cf_pred, raw_query=raw_query, raw_cf=raw_cf,
            higher_is_better=self._higher_is_better, confidence_metrics=confidence_metrics)

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)