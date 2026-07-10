"""
Shared utilities for rule-based explainers.

Provides risk prediction wrappers, binary label construction, rule
formatting, fidelity computation, sanity check runner, and the rule
humanizer that converts encoded-space conditions (z-scored thresholds,
one-hot dummies) into clinician-readable form with sentinel-zone
filtering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    RuleBasedExplainer,
)
from xai_decision_making.explainability.sanity_checks import RULE_BASED_CHECKS

def risk_predict_fn(adapter, time_horizon=None):
    feat_names = adapter.feature_names()
    if time_horizon is not None:
        def _fn(x):
            if isinstance(x, pd.DataFrame):
                x = x.values
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            df = pd.DataFrame(x, columns=feat_names)
            return adapter.predict_at_horizon(df, time_horizon)
        return _fn

    def _fn(x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        df = pd.DataFrame(x, columns=feat_names)
        risks = []
        for i in range(len(df)):
            pred = adapter.predict(df.iloc[i:i+1])
            risks.append(pred.get("risk_score", np.nan))
        return np.asarray(risks, dtype=np.float64)
    return _fn


def binarise_risk(adapter, threshold=None, time_horizon=None):
    x_train = adapter.training_data()

    if time_horizon is not None:
        risks = adapter.predict_at_horizon(x_train, time_horizon)
    else:
        risks = []
        for i in range(len(x_train)):
            pred = adapter.predict(x_train.iloc[i:i+1])
            risks.append(pred.get("risk_score", np.nan))
        risks = np.asarray(risks, dtype=np.float64)

    if threshold is None:
        threshold = float(np.median(risks))
    y_binary = (risks >= threshold).astype(int)
    return y_binary, threshold, risks

def outcome_label(adapter, is_above_threshold):
    if adapter.higher_is_better:
        return "good outcome" if is_above_threshold else "poor outcome"
    return "high risk" if is_above_threshold else "low risk"

def format_condition(feature, op, value):
    if isinstance(value, float):
        return f"{feature} {op} {value:.3f}"
    return f"{feature} {op} {value}"

def format_rule(conditions, outcome):
    if not conditions:
        return f"THEN {outcome}"
    return "IF " + " AND ".join(conditions) + f" THEN {outcome}"

def compute_fidelity(rule_predict_fn, rsf_binary_labels, x_eval):
    preds = rule_predict_fn(x_eval)
    preds = np.asarray(preds).ravel()
    return float(np.mean(preds == rsf_binary_labels))

def _run_sanity_checks(explainer, held_out_df, checks=None):
    if checks is None:
        checks = RULE_BASED_CHECKS
    results = []
    for check in checks:
        result = check.run(explainer, held_out_df)
        result.modality = explainer._modality
        results.append(result)
    agg = AggregatedSanityResult(
        modality=explainer._modality,
        target=explainer._adapter.target,
        explainer_name=type(explainer).__name__,
        check_results=results,
    )
    agg.print_summary()
    return agg

@dataclass(frozen=True)
class Atom:
    feature: str
    op: str
    threshold: float

_ATOM_RE = re.compile(
    r"^\s*(?P<feat>[^\s<>=!]+)\s*"
    r"(?P<op>(?:<=|>=|≤|≥|<|>|==|!=))\s*"
    r"(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)

_COMPOUND_SPLIT_RE = re.compile(r"\s+(?:and|AND|&)\s+")

_DUMMY_RE = re.compile(r"^(?P<base>[A-Za-z][A-Za-z0-9_]*?)_(?P<idx>\d+)$")

def _normalize_op(op):
    return {"≤": "<=", "≥": ">="}.get(op, op)

def parse_condition(cond):
    parts = _COMPOUND_SPLIT_RE.split(cond.strip())
    atoms = []
    for part in parts:
        m = _ATOM_RE.match(part)
        if not m:
            continue
        atoms.append(Atom(
            feature=m.group("feat"),
            op=_normalize_op(m.group("op")),
            threshold=float(m.group("val")),
        ))
    return atoms


def detect_dummy(feat, all_features):
    m = _DUMMY_RE.match(feat)
    if not m:
        return None
    base, level = m.group("base"), int(m.group("idx"))
    siblings = [f for f in all_features
                if f.startswith(f"{base}_") and _DUMMY_RE.match(f)]
    return (base, level) if len(siblings) >= 2 else None

CATEGORICAL_LABELS = {
    "leeft_cat": {
        6: "30-34", 7: "35-39", 8: "40-44", 9: "45-49", 10: "50-54",
        11: "55-59", 12: "60-64", 13: "65-69", 14: "70-74", 15: "75-79",
        16: "80-84", 17: "85-89", 18: "≥90",
    },
    "gesl": {1: "male", 2: "female"},
    "seizoen": {1: "winter", 2: "spring", 3: "summer", 4: "autumn"},
}

FEATURE_DISPLAY = {
    "ca199": "CA19-9",
    "ldh": "LDH",
    "crp": "CRP",
    "albumine": "albumin",
    "bilirubine": "bilirubin",
    "cci": "Charlson index",
    "cci_cat": "Charlson category",
    "perf_stat": "performance status",
    "klin_tum_afm": "tumour size",
    "cstadium": "clinical stage",
    "ct": "clinical T-stage",
    "cm": "clinical M-stage",
    "diffgrad": "differentiation grade",
    "morf": "morphology code",
    "meta_lever": "liver metastases",
    "meta_perit": "peritoneal metastases",
    "meta_long": "lung metastases",
    "meta_hersenen": "brain metastases",
    "meta_bot": "bone metastases",
    "meta_lymf": "lymph node metastases",
    "meta_bijnier": "adrenal metastases",
    "meta_overig": "other metastases",
    "tumgericht_ther": "tumour-directed therapy",
    "chir": "surgery",
    "chemo": "chemotherapy",
    "rt": "radiotherapy",
    "destr": "destructive therapy",
    "ts_surg": "time since surgery",
    "ts_syst": "time since systemic therapy",
    "had_surg": "received surgery",
    "had_syst": "received systemic therapy",
    "impulse_surg": "ongoing surgery effect",
    "impulse_syst": "ongoing systemic-therapy effect",
    "leeft_cat": "age band",
    "gesl": "sex",
    "seizoen": "diagnosis season",
    "diag_basis": "basis of diagnosis",
    "veneus_vaatbetr": "venous involvement",
    "arterieel_vaatbetr": "arterial involvement",
}

BINARY_FEATURES = {
    "meta_lever", "meta_bijnier", "meta_bot", "meta_hersenen", "meta_long",
    "meta_lymf", "meta_perit", "meta_overig",
    "chir", "chemo", "rt", "destr", "tumgericht_ther",
    "had_surg", "had_syst", "impulse_surg", "impulse_syst",
}

def _display_name(feat):
    return FEATURE_DISPLAY.get(feat, feat.replace("_", " "))

def _is_trivially_true(atom, patient_value, eps=1e-6):
    op, t, v = atom.op, atom.threshold, patient_value
    if op == "<=": return v <= t + eps
    if op == "<":  return v <  t - eps
    if op == ">=": return v >= t - eps
    if op == ">":  return v >  t + eps
    if op == "==": return abs(v - t) <= eps
    if op == "!=": return abs(v - t) > eps
    return False

def _is_sentinel_threshold(feature, threshold, adapter, *, multiplier=20.0,
                           quantile=0.99, min_train_rows=10):
    if not hasattr(adapter, "training_data"):
        return False

    try:
        train = adapter.training_data()
    except Exception:
        return False

    if feature not in train.columns:
        return False

    col = pd.to_numeric(train[feature], errors="coerce").dropna()
    if len(col) < min_train_rows:
        return False

    p_high = col.quantile(quantile)
    if not (p_high > 0):
        return False

    return threshold > p_high * multiplier

def humanize_rule_conditions(conditions, adapter, patient_row=None, *, decimals=1):
    feat_names = list(adapter.feature_names()) if hasattr(adapter, "feature_names") else []
    inv_df = getattr(adapter, "inverse_transform_numeric", None)

    def _inverse_z(feature, z):
        if inv_df is None:
            return float(z)
        try:
            inverted = inv_df(pd.DataFrame({feature: [float(z)]}))
            return float(inverted[feature].iloc[0])
        except Exception:
            return float(z)

    out = []
    dummy_buckets = {}
    sentinel_skipped = []

    for cond in conditions:
        atoms = parse_condition(cond)
        if not atoms:
            out.append(cond)
            continue

        for atom in atoms:
            dummy = detect_dummy(atom.feature, feat_names)
            if dummy is not None:
                dummy_buckets.setdefault(dummy[0], []).append(atom)
                continue

            if atom.feature in BINARY_FEATURES:
                indicates_yes = atom.op in (">", ">=") and atom.threshold < 1.0
                indicates_no  = atom.op in ("<", "<=") and atom.threshold < 1.0
                label = _display_name(atom.feature)
                if indicates_yes:
                    out.append(f"{label}: yes")
                    continue
                if indicates_no:
                    out.append(f"{label}: no")
                    continue

            threshold = _inverse_z(atom.feature, atom.threshold)

            if _is_sentinel_threshold(atom.feature, threshold, adapter):
                sentinel_skipped.append(atom.feature)
                continue

            out.append(f"{_display_name(atom.feature)} {atom.op} "
                       f"{_format_threshold(threshold, decimals)}")

    for base, atoms in dummy_buckets.items():
        active_level = None
        if patient_row is not None:
            for f, v in patient_row.items():
                m = _DUMMY_RE.match(f)
                if m and m.group("base") == base and float(v) == 1.0:
                    active_level = int(m.group("idx"))
                    break

        positive_levels = []
        for a in atoms:
            indicates_in = a.op in (">", ">=") and a.threshold < 1.0
            if indicates_in:
                m = _DUMMY_RE.match(a.feature)
                if m:
                    positive_levels.append(int(m.group("idx")))

        labels = CATEGORICAL_LABELS.get(base, {})
        display_base = _display_name(base)

        if positive_levels:
            lvl = positive_levels[0]
            out.append(f"{display_base} = {labels.get(lvl, lvl)}")
        elif active_level is not None:
            out.append(f"{display_base} = {labels.get(active_level, active_level)}")

    if not out and sentinel_skipped:
        dropped = sorted({_display_name(f) for f in sentinel_skipped})
        out.append(
            f"(no informative local rule: patient deep within predicted "
            f"class; sentinel-zone conditions dropped on {', '.join(dropped)})"
        )

    return out

def _format_threshold(value, decimals):
    if abs(value) >= 100 or (value == int(value)):
        return f"{value:.0f}"
    formatted = f"{value:.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
        if "." not in formatted:
            formatted += ".0"
    return formatted