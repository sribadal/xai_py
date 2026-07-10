"""
Treatment-scenario counterfactual explainer.

Generates counterfactuals by comparing clinically valid treatment
scenarios rather than perturbing pretreatment features. Holds the
patient's RDM baseline fixed and varies only the TDM treatment
conditioning vector, producing sparse, actionable "what if" comparisons
across BSC, surgery-only, chemo-only, and surgery+chemo regimens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    CounterfactualExplainer,
    ExplainerModality,
    ExplainerRegistry,
    Explanation,
    ModelTarget,
)
from xai_decision_making.explainability.counterfactual._base import (
    _run_sanity_checks,
    compute_cf_confidence,
    compute_content_deltas,
    enforce_immutables,
    extract_scalar_prediction,
    risk_predict_fn,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config,
    condor_target_t_norm,
    derive_treatment_started_flags,
)

registry = ExplainerRegistry()

_TREAT_FEATURES = ["ts_surg", "had_surg", "impulse_surg",
                    "ts_syst", "had_syst", "impulse_syst"]

DEFAULT_SURGERY_DAY = 42
DEFAULT_CHEMO_DAY = 60


def _compute_ts_at_horizon(event_day, horizon_months):
    horizon_days = horizon_months * 30.44
    if event_day > horizon_days:
        return 0.0
    return (horizon_days - event_day) / 30.44 / 60.0


class TreatmentScenario:
    __slots__ = ("name", "label", "has_surgery", "has_chemo",
                 "surgery_day", "chemo_day")

    def __init__(self, name, label, *, has_surgery, has_chemo,
                 surgery_day=DEFAULT_SURGERY_DAY, chemo_day=DEFAULT_CHEMO_DAY):
        self.name = name
        self.label = label
        self.has_surgery = has_surgery
        self.has_chemo = has_chemo
        self.surgery_day = surgery_day
        self.chemo_day = chemo_day

    def to_treatment_vector(self, horizon_months):
        if self.has_surgery:
            ts_surg = _compute_ts_at_horizon(self.surgery_day, horizon_months)
            had_surg = 1.0 if ts_surg > 0 else 0.0
        else:
            ts_surg = 0.0
            had_surg = 0.0

        if self.has_chemo:
            ts_syst = _compute_ts_at_horizon(self.chemo_day, horizon_months)
            had_syst = 1.0 if ts_syst > 0 else 0.0
        else:
            ts_syst = 0.0
            had_syst = 0.0

        return {
            "ts_surg": ts_surg,
            "had_surg": had_surg,
            "impulse_surg": 0.0,
            "ts_syst": ts_syst,
            "had_syst": had_syst,
            "impulse_syst": 0.0,
        }


DEFAULT_SCENARIOS = [
    TreatmentScenario(
        "bsc", "Best Supportive Care",
        has_surgery=False, has_chemo=False,
    ),
    TreatmentScenario(
        "surgery_only", "Surgery Only",
        has_surgery=True, has_chemo=False,
    ),
    TreatmentScenario(
        "chemo_only", "Chemotherapy Only",
        has_surgery=False, has_chemo=True,
    ),
    TreatmentScenario(
        "surgery_chemo", "Surgery + Chemotherapy",
        has_surgery=True, has_chemo=True,
    ),
]


@registry.register
class TreatmentScenarioCFExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        scenarios=None,
        surgery_day=DEFAULT_SURGERY_DAY,
        chemo_day=DEFAULT_CHEMO_DAY,
        immutable=None,
        time_horizon=None,
    ):
        super().__init__(model_adapter)
        self._time_horizon = time_horizon
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)

        if scenarios is not None:
            self._scenarios = list(scenarios)
        else:
            self._scenarios = [
                TreatmentScenario(
                    s.name, s.label,
                    has_surgery=s.has_surgery,
                    has_chemo=s.has_chemo,
                    surgery_day=surgery_day,
                    chemo_day=chemo_day,
                )
                for s in DEFAULT_SCENARIOS
            ]

        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(
            model_adapter, extra_immutable=immutable,
        )

        feat_names = model_adapter.feature_names()
        self._treat_indices = {
            f: feat_names.index(f)
            for f in _TREAT_FEATURES
            if f in feat_names
        }
        self._static_indices = [
            i for i, f in enumerate(feat_names)
            if f not in self._treat_indices
        ]
        self._immutable = [
            f for f in feat_names
            if f not in self._treat_indices
        ]
        self._ranges = self._cfg.ranges
        self._categorical_values = self._cfg.categorical_values
        self._last_scenario_results = None

        if not self._treat_indices:
            raise ValueError(
                "TreatmentScenarioCF requires an adapter with treatment "
                "features in its feature space. The adapter's feature_names() "
                f"contains no treatment dims from {_TREAT_FEATURES}. "
                "Use a checkpoint trained with use_treatment=True."
            )

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def _inject_treatment(self, patient_row, scenario, horizon_months):
        cf = patient_row.copy()
        treat_dict = scenario.to_treatment_vector(horizon_months)
        for feat, val in treat_dict.items():
            if feat in self._treat_indices:
                cf[self._treat_indices[feat]] = val
        return cf

    def _detect_current_scenario(self, patient_row):
        has_surg = False
        has_syst = False
        if "had_surg" in self._treat_indices:
            has_surg = patient_row[self._treat_indices["had_surg"]] > 0.5
        elif "ts_surg" in self._treat_indices:
            has_surg = patient_row[self._treat_indices["ts_surg"]] > 0
        if "had_syst" in self._treat_indices:
            has_syst = patient_row[self._treat_indices["had_syst"]] > 0.5
        elif "ts_syst" in self._treat_indices:
            has_syst = patient_row[self._treat_indices["ts_syst"]] > 0

        if has_surg and has_syst:
            return "surgery_chemo"
        if has_surg:
            return "surgery_only"
        if has_syst:
            return "chemo_only"
        return "bsc"

    def generate_counterfactual(self, patient_features, target_outcome=None):
        feat_names = self._adapter.feature_names()
        x_orig = patient_features[feat_names].fillna(0).values[0].astype(np.float64)

        current_pred = float(self._predict_fn(x_orig.reshape(1, -1))[0])
        current_scenario = self._detect_current_scenario(x_orig)

        horizon_months = float(self._time_horizon) if self._time_horizon else 6.0

        scenario_results = []
        for scenario in self._scenarios:
            cf_row = self._inject_treatment(x_orig, scenario, horizon_months)
            pred = float(self._predict_fn(cf_row.reshape(1, -1))[0])

            if self._higher_is_better:
                improvement = pred - current_pred
            else:
                improvement = current_pred - pred

            scenario_results.append({
                "scenario": scenario,
                "cf_row": cf_row,
                "prediction": pred,
                "improvement": improvement,
                "is_current": (scenario.name == current_scenario),
            })

        self._last_scenario_results = scenario_results

        candidates = [r for r in scenario_results if not r["is_current"]]

        if not candidates:
            return None

        best = max(candidates, key=lambda r: r["improvement"])
        return pd.DataFrame([best["cf_row"]], columns=feat_names)

    def explain(self, patient_features, patient_id=None):
        feat_names = self._adapter.feature_names()
        query = patient_features[feat_names].fillna(0).head(1)
        cf = self.generate_counterfactual(patient_features)

        if cf is not None:
            content, raw_query, raw_cf = compute_content_deltas(
                self._adapter, query, cf, feat_names
            )
        else:
            content, raw_query, raw_cf = {}, None, None

        orig_pred = extract_scalar_prediction(
            self._adapter.predict(query, time_horizon=self._time_horizon)
        )
        cf_pred = (
            extract_scalar_prediction(
                self._adapter.predict(cf.head(1), time_horizon=self._time_horizon)
            )
            if cf is not None
            else None
        )

        confidence_metrics = compute_cf_confidence(
            raw_query, raw_cf, orig_pred, cf_pred,
            higher_is_better=self._higher_is_better,
        )

        if cf is not None:
            x_orig = query.values[0].astype(np.float64)
            current_scenario = self._detect_current_scenario(x_orig)
            x_cf = cf.values[0].astype(np.float64)
            cf_scenario = self._detect_current_scenario(x_cf)

            current_label = current_scenario
            cf_label = cf_scenario
            for s in self._scenarios:
                if s.name == current_scenario:
                    current_label = s.label
                if s.name == cf_scenario:
                    cf_label = s.label

            content["_treatment_scenario_current"] = current_label
            content["_treatment_scenario_cf"] = cf_label
            content["_treatment_scenario_improvement"] = (
                round(float(cf_pred - orig_pred), 2)
                if cf_pred is not None and orig_pred is not None
                else None
            )

            if self._last_scenario_results is not None:
                content["_all_scenarios"] = [
                    {
                        "label": r["scenario"].label,
                        "name": r["scenario"].name,
                        "prediction": round(float(r["prediction"]), 2),
                        "improvement": round(float(r["improvement"]), 2),
                        "is_current": bool(r["is_current"]),
                        "is_chosen_cf": bool(
                            r["scenario"].name == cf_scenario
                            and not r["is_current"]
                        ),
                    }
                    for r in self._last_scenario_results
                ]

        return self._make_explanation(
            patient_id,
            content=content,
            method="TreatmentScenarioCF",
            counterfactual_df=cf,
            original_prediction=orig_pred,
            cf_prediction=cf_pred,
            raw_query=raw_query,
            raw_cf=raw_cf,
            higher_is_better=self._higher_is_better,
            confidence_metrics=confidence_metrics,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)