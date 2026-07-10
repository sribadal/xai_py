"""
RuleFit explainer for global sparse rule lists.

Fits Friedman & Popescu's RuleFit against binary risk labels and
returns humanised firing rules with coefficients and support.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imodels import RuleFitClassifier

from xai_decision_making.explainability.base import (
    ExplainerModality,
    ExplainerRegistry,
    ModelTarget,
    RuleBasedExplainer,
)
from xai_decision_making.explainability.rule_based._base import (
    _run_sanity_checks,
    binarise_risk,
    format_rule,
    humanize_rule_conditions,
    outcome_label,
)

registry = ExplainerRegistry()

def _coef_of(rule_obj):
    args = getattr(rule_obj, "args", None)
    if args is None or len(args) == 0:
        return 0.0
    try:
        return float(args[0])
    except (TypeError, ValueError):
        return 0.0

def _extract_rules(rulefit_model):
    raw_rules = getattr(rulefit_model, "rules_", None)
    if not raw_rules:
        return []

    rules = []
    for r in raw_rules:
        text = getattr(r, "rule", None)
        if not text:
            continue
        coef = _coef_of(r)
        if coef == 0.0:
            continue
        support = float(getattr(r, "support", 0.0) or 0.0)
        rules.append({"rule": text, "coef": coef, "support": support})

    rules.sort(key=lambda x: abs(x["coef"]), reverse=True)
    return rules


def _evaluate_rule(rule_text, feature_dict):
    try:
        return bool(eval(rule_text, {"__builtins__": {}}, feature_dict))
    except Exception:
        return False

@registry.register
class RuleFitExplainer(RuleBasedExplainer):
    _modality = ExplainerModality.RULE_BASED
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, max_rules=30, tree_size=4, verbose=True,
                 time_horizon=None):
        super().__init__(model_adapter)
        self._max_rules = max_rules
        self._tree_size = tree_size
        self._verbose = verbose
        self._time_horizon = time_horizon
        self._rulefit = None
        self._risk_threshold = None
        self._rules = []
        self._fit()

    def _fit(self):
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        X = x_train[feat_names].fillna(0).values.astype(np.float64)

        y_binary, threshold, _ = binarise_risk(
            self._adapter, time_horizon=self._time_horizon
        )
        self._risk_threshold = threshold

        if self._verbose:
            print(f"  [RuleFit] Fitting on {len(X)} patients, "
                  f"threshold={threshold:.1f}, max_rules={self._max_rules}")

        self._rulefit = RuleFitClassifier(
            max_rules=self._max_rules,
            tree_size=self._tree_size,
            random_state=42,
        )
        self._rulefit.fit(X, y_binary, feature_names=feat_names)

        self._rules = _extract_rules(self._rulefit)

        if self._verbose:
            total = len(getattr(self._rulefit, "rules_", []) or [])
            print(f"  [RuleFit] {len(self._rules)} active rules "
                  f"(out of {total} candidate rules)")

    @property
    def modality(self):
        return self._modality

    def _patient_prediction(self, patient_features):
        feat_names = self._adapter.feature_names()
        X = patient_features[feat_names].fillna(0).values.astype(np.float64)
        pred = self._rulefit.predict(X)
        return int(pred[0])

    def _firing_rules(self, patient_features):
        if not self._rules:
            return []

        feat_names = self._adapter.feature_names()
        x = patient_features[feat_names].fillna(0).iloc[0]
        feature_dict = {f: float(x[f]) for f in feat_names}

        firing = []
        for rule in self._rules:
            if _evaluate_rule(rule["rule"], feature_dict):
                firing.append(rule)
        return firing

    def extract_rule(self, patient_features):
        prediction = self._patient_prediction(patient_features)
        outcome = outcome_label(self._adapter, prediction == 1)

        firing = self._firing_rules(patient_features)
        conditions_raw = [r["rule"] for r in firing]

        feat_names = self._adapter.feature_names()
        patient_row = patient_features[feat_names].fillna(0).iloc[0].to_dict()
        conditions = humanize_rule_conditions(
            conditions_raw, adapter=self._adapter, patient_row=patient_row,
        )

        if not conditions:
            rule_text = f"No rules fired; default {outcome}"
        else:
            rule_text = format_rule(conditions, outcome)

        if self._verbose:
            print(f"  [RuleFit] n_firing={len(firing)}, "
                  f"n_conditions={len(conditions)} (raw={len(conditions_raw)})")

        return {
            "conditions": conditions,
            "conditions_raw": conditions_raw,
            "outcome": outcome,
            "rule_text": rule_text,
            "firing_rules": firing,
            "n_firing": len(firing),
        }

    def global_rules(self):
        return list(self._rules)

    def fidelity(self, held_out_df):
        feat_names = self._adapter.feature_names()
        X = held_out_df[feat_names].fillna(0).values.astype(np.float64)
        y_true = []
        for i in range(len(held_out_df)):
            pred = self._adapter.predict(held_out_df.iloc[i:i+1][feat_names])
            y_true.append(int(pred.get("risk_score", 0) >= self._risk_threshold))
        y_true = np.asarray(y_true)
        y_pred = self._rulefit.predict(X)
        return float(np.mean(y_true == y_pred))

    def explain(self, patient_features, patient_id=None):
        rule = self.extract_rule(patient_features)
        return self._make_explanation(
            patient_id,
            content=rule,
            method="RuleFit",
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)