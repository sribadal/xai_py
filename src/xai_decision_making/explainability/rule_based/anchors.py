"""
Anchors explainer for per-patient high-precision rule explanations.

Returns humanised clinician-readable conditions alongside the raw
anchor strings for audit and debugging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from alibi.explainers import AnchorTabular

from xai_decision_making.explainability.base import (
    ExplainerModality,
    ExplainerRegistry,
    ModelTarget,
    RuleBasedExplainer,
)
from xai_decision_making.explainability.sanity_checks import (
    RuleFidelityCheck,
    RuleLengthCheck,
    AnchorPrecisionCheck,
)
from xai_decision_making.explainability.rule_based._base import (
    _run_sanity_checks,
    binarise_risk,
    format_rule,
    humanize_rule_conditions,
    outcome_label,
)

registry = ExplainerRegistry()

@registry.register
class AnchorsExplainer(RuleBasedExplainer):

    _modality = ExplainerModality.RULE_BASED
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, precision_threshold=0.95, verbose=True,
                 time_horizon=None):
        super().__init__(model_adapter)
        self._precision_threshold = precision_threshold
        self._verbose = verbose
        self._time_horizon = time_horizon
        self._anchor_explainer = None
        self._risk_threshold = None
        self._build()

    def _build(self):
        _, threshold, _ = binarise_risk(
            self._adapter, time_horizon=self._time_horizon
        )
        self._risk_threshold = threshold

        feat_names = self._adapter.feature_names()
        x_train = self._adapter.training_data()

        adapter = self._adapter
        time_horizon = self._time_horizon

        def predict_fn(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            df = pd.DataFrame(x, columns=feat_names)

            if time_horizon is not None:
                risks = adapter.predict_at_horizon(df, time_horizon)
                return (risks >= threshold).astype(int)

            labels = []
            for i in range(len(df)):
                pred = adapter.predict(df.iloc[i:i+1])
                labels.append(int(pred.get("risk_score", 0) >= threshold))
            return np.asarray(labels)

        self._anchor_explainer = AnchorTabular(
            predictor=predict_fn,
            feature_names=feat_names,
        )

        X_train = x_train[feat_names].fillna(0).values.astype(np.float64)
        self._anchor_explainer.fit(X_train)

        if self._verbose:
            print(f"  [Anchors] Fitted on {len(X_train)} patients, "
                  f"threshold={threshold:.1f}, "
                  f"precision_threshold={self._precision_threshold}")

    @property
    def modality(self):
        return self._modality

    def extract_rule(self, patient_features):
        feat_names = self._adapter.feature_names()
        x = patient_features[feat_names].fillna(0).values[0].astype(np.float64)

        explanation = self._anchor_explainer.explain(
            x, threshold=self._precision_threshold,
        )

        conditions_raw = list(explanation.anchor) if explanation.anchor else []
        precision = (
            float(explanation.precision)
            if explanation.precision is not None else float("nan")
        )
        coverage = (
            float(explanation.coverage)
            if explanation.coverage is not None else float("nan")
        )

        if self._time_horizon is not None:
            df = patient_features[feat_names].iloc[0:1]
            risk = self._adapter.predict_at_horizon(df, self._time_horizon)[0]
        else:
            pred = self._adapter.predict(patient_features[feat_names].iloc[0:1])
            risk = pred.get("risk_score", 0)

        is_above = risk >= self._risk_threshold
        outcome = outcome_label(self._adapter, is_above)

        patient_row = patient_features[feat_names].fillna(0).iloc[0].to_dict()
        conditions = humanize_rule_conditions(
            conditions_raw, adapter=self._adapter, patient_row=patient_row,
        )

        if self._verbose:
            print(f"  [Anchors] n_conditions={len(conditions)} "
                  f"(raw={len(conditions_raw)}), "
                  f"precision={precision:.3f}, coverage={coverage:.3f}")

        return {
            "conditions": conditions,
            "conditions_raw": conditions_raw,
            "outcome": outcome,
            "rule_text": format_rule(conditions, outcome)
                         + f"  [precision={precision:.2f}, "
                         f"coverage={coverage:.2f}]",
            "precision": precision,
            "coverage": coverage,
        }

    def global_rules(self):
        return None

    def fidelity(self, held_out_df):
        feat_names = self._adapter.feature_names()
        precisions = []
        for i in range(len(held_out_df)):
            patient = held_out_df.iloc[i:i+1][feat_names]
            try:
                rule = self.extract_rule(patient)
                if not np.isnan(rule["precision"]):
                    precisions.append(rule["precision"])
            except Exception:
                continue
        return float(np.mean(precisions)) if precisions else 0.0

    def explain(self, patient_features, patient_id=None):
        rule = self.extract_rule(patient_features)
        return self._make_explanation(
            patient_id,
            content=rule,
            method="Anchors",
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df, [
            RuleFidelityCheck(),
            RuleLengthCheck(),
            AnchorPrecisionCheck(),
        ])