"""
Skope-Rules explainer for precision-bounded rule extraction.

Mines rules from gradient-boosted trees and filters by minimum
precision and recall. Returns humanised firing rules with per-rule
precision and recall guarantees.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from imodels import SkopeRulesClassifier

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

def _normalise_rule_text(rule_text, feat_names):
    def _sub(match):
        idx = int(match.group(1))
        if 0 <= idx < len(feat_names):
            return feat_names[idx]
        return match.group(0)

    return re.sub(r"feature_(\d+)", _sub, str(rule_text))

def _extract_rules(skope_model, feat_names):
    raw_rules = (
        getattr(skope_model, "rules_", None)
        or getattr(skope_model, "rules_without_feature_names_", None)
        or []
    )
    if not raw_rules:
        return []

    rules = []
    for r in raw_rules:
        if isinstance(r, tuple):
            text = r[0] if len(r) >= 1 else None
            metrics = r[1] if len(r) >= 2 else ()
            precision = float(metrics[0]) if len(metrics) >= 1 else float("nan")
            recall = float(metrics[1]) if len(metrics) >= 2 else float("nan")
            support = float(metrics[2]) if len(metrics) >= 3 else 0.0
        else:
            text = getattr(r, "rule", None)

            agg = getattr(r, "agg_dict", None) or {}
            precision = (
                agg.get("precision")
                or agg.get("OOB_precision")
                or getattr(r, "precision", None)
            )
            recall = (
                agg.get("recall")
                or agg.get("OOB_recall")
                or getattr(r, "recall", None)
            )
            support = (
                agg.get("support")
                or getattr(r, "support", 0.0)
                or 0.0
            )

            if precision is None or recall is None:
                args = getattr(r, "args", None)
                if isinstance(args, (tuple, list)) and len(args) >= 2:
                    if precision is None:
                        precision = args[0]
                    if recall is None:
                        recall = args[1]

            try:
                precision = float(precision) if precision is not None else float("nan")
            except (TypeError, ValueError):
                precision = float("nan")
            try:
                recall = float(recall) if recall is not None else float("nan")
            except (TypeError, ValueError):
                recall = float("nan")
            try:
                support = float(support) if support is not None else 0.0
            except (TypeError, ValueError):
                support = 0.0

        if not text:
            continue

        text = _normalise_rule_text(text, feat_names)
        rules.append({
            "rule": text,
            "precision": precision,
            "recall": recall,
            "support": support,
        })

    rules.sort(
        key=lambda x: (
            x["precision"] if not np.isnan(x["precision"]) else -1.0,
            x["recall"] if not np.isnan(x["recall"]) else -1.0,
        ),
        reverse=True,
    )
    return rules

def _evaluate_rule(rule_text, feature_dict):
    try:
        return bool(eval(rule_text, {"__builtins__": {}}, feature_dict))
    except Exception:
        return False

@registry.register
class SkopeRulesExplainer(RuleBasedExplainer):
    _modality = ExplainerModality.RULE_BASED
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, precision_min=0.5, recall_min=0.01,
                 n_estimators=30, max_depth=3, max_rules=30, verbose=True,
                 time_horizon=None, random_state=42):
        super().__init__(model_adapter)
        self._precision_min = precision_min
        self._recall_min = recall_min
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._max_rules = max_rules
        self._verbose = verbose
        self._time_horizon = time_horizon
        self._random_state = random_state
        self._skope = None
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
            print(f"  [Skope-Rules] Fitting on {len(X)} patients, "
                  f"threshold={threshold:.1f}, "
                  f"precision_min={self._precision_min}, "
                  f"recall_min={self._recall_min}")

        kwargs = dict(
            precision_min=self._precision_min,
            recall_min=self._recall_min,
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
        )

        try:
            self._skope = SkopeRulesClassifier(max_depth_duplication=2, **kwargs)
        except TypeError:
            self._skope = SkopeRulesClassifier(**kwargs)

        try:
            self._skope.fit(X, y_binary, feature_names=feat_names)
        except TypeError:
            self._skope = SkopeRulesClassifier(feature_names=feat_names, **kwargs)
            self._skope.fit(X, y_binary)

        self._rules = _extract_rules(self._skope, feat_names)

        if self._max_rules and len(self._rules) > self._max_rules:
            self._rules = self._rules[: self._max_rules]

        if self._verbose:
            print(f"  [Skope-Rules] {len(self._rules)} rules retained "
                  f"(precision_min={self._precision_min}, "
                  f"recall_min={self._recall_min})")

    @property
    def modality(self):
        return self._modality

    def _patient_prediction(self, patient_features):
        feat_names = self._adapter.feature_names()
        X = patient_features[feat_names].fillna(0).values.astype(np.float64)
        try:
            pred = self._skope.predict(X)
            return int(pred[0])
        except Exception:
            return int(bool(self._firing_rules(patient_features)))

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
            best_precision = float("nan")
            best_recall = float("nan")
        else:
            rule_text = format_rule(conditions, outcome)
            best_precision = firing[0]["precision"]
            best_recall = firing[0]["recall"]

        if self._verbose:
            print(f"  [Skope-Rules] n_firing={len(firing)}, "
                  f"n_conditions={len(conditions)} (raw={len(conditions_raw)}), "
                  f"best_precision={best_precision}, "
                  f"best_recall={best_recall}")

        return {
            "conditions": conditions,
            "conditions_raw": conditions_raw,
            "outcome": outcome,
            "rule_text": rule_text,
            "firing_rules": firing,
            "n_firing": len(firing),
            "precision": best_precision,
            "recall": best_recall,
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

        try:
            y_pred = self._skope.predict(X)
        except Exception:
            y_pred = []
            for i in range(len(held_out_df)):
                firing = self._firing_rules(held_out_df.iloc[i:i+1][feat_names])
                y_pred.append(int(bool(firing)))
            y_pred = np.asarray(y_pred)

        return float(np.mean(y_true == y_pred))

    def explain(self, patient_features, patient_id=None):
        rule = self.extract_rule(patient_features)
        return self._make_explanation(
            patient_id,
            content=rule,
            method="Skope-Rules",
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)