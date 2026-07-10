"""
Co-12 property audit for the explainability framework. Scores every
(model_type, modality, method, horizon) combination against the eleven
measurable Nauta CO-12 properties plus an architecture-availability gate.

Properties and how each is scored:

  1. cross_model
     Checks whether the method appears in both the RSF and ConDOR
     registries. PASS if present in both, PARTIAL if absent from one
     due to architectural necessity (tree-only or neural-only methods),
     FAIL otherwise.

  2. compactness
     Attribution: effective_k at 90-percent cumulative absolute mass
     (median across patients). Other modalities: median raw explanation
     size (number of changed features, retrieved cases, or rule
     conditions).

  3. correctness
     Attribution: live deletion-faithfulness delta — prediction change
     when masking features ranked by importance vs random order.
     Counterfactual: flip rate from exports — fraction of generated
     counterfactuals that actually reverse the prediction direction.
     Case-based: post-treatment leakage check from exports — whether
     retrieved cases expose treatment columns.
     Rule-based: median fidelity from exports — agreement between the
     surrogate rule and the black-box model.

  4. contrastivity
     Attribution: cohort-divergence — top-K feature overlap between
     high-risk and low-risk patient groups (lower overlap = more
     contrastive). Falls back to signed-rate when cohort data is
     unavailable. Counterfactual: PASS by construction. Case-based:
     PASS if prototypes carry risk scores enabling contrast. Rule-based:
     PASS if precision is reported per rule.

  5. covariate_complexity
     Measures whether explanations reference raw clinical features or
     z-scored latent dimensions. Scores raw-feature overlap with training
     columns, flags counterfactual values outside plausible ranges, and
     checks whether values fall in z-score space.

  6. composition
     Combined structural completeness and size consistency. Checks
     modality-specific required fields (non-empty content, predictions,
     similarity scores, conditions with precision) and coefficient of
     variation of explanation size across patients.

  7. continuity
     Live re-run: top-K Jaccard similarity between the original
     explanation and explanations for slightly perturbed inputs (Gaussian
     noise with std 0.01). Computed once per method and reused across
     horizons.

  8. consistency
     Live re-run: top-K Jaccard similarity between two independently
     instantiated explainers on the same input. Deterministic methods
     (TreeSHAP, IG, DeepLift, MinimalDepth) are floored to PASS since
     any non-1.0 Jaccard is tie-breaking noise, not real inconsistency.

  9. completeness
     Attribution-specific axiomatic check. LIME: median local_r2.
     Permutation: median relative_std. IG/DeepLift/TreeSHAP: median
     completeness_ratio (sum-of-attributions vs prediction delta). Other
     modalities: N/A.

 10. confidence
     Data-driven per-modality from metadata.confidence_metrics. LIME:
     local_r2 as surrogate quality. Permutation/DeepLiftSHAP:
     relative_std as stability. Counterfactual: flip_rate. Case-based:
     median retrieval distance. Rule-based: precision or fidelity from
     content. Deterministic attribution methods: FAIL (no uncertainty
     signal exists). Falls back to per-method categorical when data is
     absent.

 11. controllability
     Per-modality categorical. Counterfactual and case-based: PASS
     (user can specify targets, constraints, k). Attribution and
     rule-based: PARTIAL (limited user steering).

 12. coherence
     Attribution only. Median fraction of absolute attribution mass
     landing on PDAC clinical prognostic factors (NCCN-derived feature
     set). Other modalities: N/A (evaluated qualitatively in the
     clinician vignette study).

The twelfth Nauta property (Context) is delegated to the clinician
vignette study and not scored here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")

SCHEMA_VERSION = "nauta_11"

PASS, PARTIAL, FAIL, NA = "PASS", "PARTIAL", "FAIL", "N/A"

VERDICT_COLS = [
    "cross_model",
    "compactness",
    "correctness",
    "contrastivity",
    "covariate_complexity",
    "composition",
    "continuity",
    "consistency",
    "completeness",
    "confidence",
    "controllability",
    "coherence",
]

ARCH_NATIVE = {
    "tree_shap", "minimal_depth",
    "integrated_gradients", "deep_lift", "deep_lift_shap",
}

_UNSIGNED_ATTRIBUTION_METHODS = frozenset({
    "permutation", "permutation_importance", "ablation", "minimal_depth",
})

DETERMINISTIC_METHODS = frozenset({
    "tree_shap",
    "minimal_depth",
    "integrated_gradients",
    "deep_lift",
})

MASS_THRESHOLD = 0.90
COMPACTNESS_PASS_AT = 10
COMPACTNESS_PARTIAL_AT = 15

CONTRASTIVITY_TOP_K = 5
CONTRASTIVITY_PASS_AT = 0.60
CONTRASTIVITY_PARTIAL_AT = 0.80

COHERENCE_PASS_AT = 0.50
COHERENCE_PARTIAL_AT = 0.30

CONTROLLABILITY_BY_MODALITY = {
    "attribution": PARTIAL,
    "case_based": PASS,
    "counterfactual": PASS,
    "rule_based": PARTIAL,
}

CONFIDENCE_BY_METHOD = {
    "tree_shap": FAIL,
    "minimal_depth": FAIL,
    "integrated_gradients": FAIL,
    "deep_lift": FAIL,
    "lime": PARTIAL,
    "permutation": PARTIAL,
    "deep_lift_shap": PARTIAL,
    "knn": PASS,
    "embedding": PASS,
    "prototype": PASS,
    "face": PASS,
    "wachter": PASS,
    "revise": PASS,
    "dice": PASS,
    "cchvae": PASS,
    "counternet": PASS,
    "lore_counterfactual": PASS,
    "anchors": PASS,
    "skope_rules": PASS,
    "lore": PARTIAL,
    "rulefit": PARTIAL,
}

PDAC_PROGNOSTIC_FACTORS = {
    "cstadium", "stadium", "ctnm", "ptnm", "ct", "cn", "cm",
    "pt", "pn", "pm", "diffgrad", "grading", "locatie", "location",
    "meta_lever", "meta_bot", "meta_perit", "meta_long", "meta_other",
    "meta_lymf", "meta_aant",
    "chir", "rt", "chemo", "tumgericht_ther", "destr",
    "perf_stat", "ecog",
    "ca19_9", "ca199", "bilirubin",
    "age", "lftyd", "geslacht", "sex",
    "albumine", "bilirubine", "cci", "cci_cat", "crp", "ldh",
    "leeft_cat", "leeftijd", "gesl", "seizoen",
    "ts_surg", "had_surg", "impulse_surg",
    "ts_syst", "had_syst", "impulse_syst",
    "chir_type1", "chir_int1", "chir_rad1", "chir_tech1",
    "syst_kuren1", "syst_start_int1", "syst_stop_int1",
}

N_LIVE_DEFAULT = 10

_OP_PATTERN = re.compile(
    r"\s*(?:<=|>=|==|!=|<|>|≤|≥|=)\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*"
)
_LOGICAL_SPLIT = re.compile(r"\s+(?:and|or|AND|OR|&|\|)\s+")


def _strip_threshold(label):
    label = str(label).strip()
    if " < " in label and " <= " in label:
        return label.split(" < ")[1].split(" <= ")[0].strip()
    stripped = _OP_PATTERN.sub(" ", label).strip()
    parts = _LOGICAL_SPLIT.split(stripped)
    return parts[0].strip() if parts else stripped


def _verdict_from_thresholds(value, *, pass_at, partial_at,
                             higher_is_better=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return NA
    if higher_is_better:
        if value >= pass_at:
            return PASS
        if value >= partial_at:
            return PARTIAL
        return FAIL
    if value <= pass_at:
        return PASS
    if value <= partial_at:
        return PARTIAL
    return FAIL


def _extract_all_features(condition):
    stripped = _OP_PATTERN.sub(" ", str(condition)).strip()
    return {p.strip() for p in _LOGICAL_SPLIT.split(stripped) if p.strip()}


def _sample_patients(adapter, n, *, seed=42):
    train = adapter.training_data()
    if len(train) == 0:
        return []
    n = min(n, len(train))
    sampled = train.sample(n=n, random_state=seed).reset_index(drop=True)
    return [sampled.iloc[[i]].reset_index(drop=True) for i in range(n)]


def _scalar_prediction(adapter, patient_df):
    try:
        pred = adapter.predict(patient_df)
        if isinstance(pred, dict):
            if "risk_score" in pred:
                return float(pred["risk_score"])
            for v in pred.values():
                arr = np.asarray(v).ravel()
                if arr.size:
                    return float(arr[-1])
        return float(np.asarray(pred).ravel()[-1])
    except Exception:
        return None


def _extract_size(modality, explanation):
    if explanation is None:
        return None
    content = getattr(explanation, "content", explanation)
    try:
        if modality == "attribution" and isinstance(content, dict):
            return sum(1 for v in content.values()
                       if v is not None and abs(float(v)) > 1e-9)
        if modality == "counterfactual" and isinstance(content, dict):
            return len(content)
        if modality == "case_based":
            if isinstance(content, list):
                return len(content)
            if isinstance(content, dict) and "similar_prototypes" in content:
                return len(content["similar_prototypes"])
        if modality == "rule_based" and isinstance(content, dict):
            return len(content.get("conditions") or [])
    except Exception:
        return None
    return None


def _top_k_features(modality, explanation, k=5):
    if explanation is None:
        return set()
    content = getattr(explanation, "content", explanation)
    try:
        if modality == "attribution" and isinstance(content, dict):
            ranked = sorted(content.items(),
                            key=lambda kv: abs(float(kv[1] or 0)),
                            reverse=True)
            return {_strip_threshold(label) for label, _ in ranked[:k]}
        if modality == "counterfactual" and isinstance(content, dict):
            return set(list(content.keys())[:k])
        if modality == "rule_based" and isinstance(content, dict):
            conds = content.get("conditions") or []
            feats = set()
            for c in conds[:k]:
                feats.update(_extract_all_features(c))
            return feats
        if modality == "case_based" and isinstance(content, list):
            ids = []
            for case in content[:k]:
                if isinstance(case, dict):
                    for key in ("patient_id", "key_eid", "key_nkr"):
                        if key in case:
                            ids.append(str(case[key]))
                            break
            return set(ids)
    except Exception:
        return set()
    return set()


def _extract_attribution_dict(content):
    if content is None:
        return None
    if isinstance(content, dict):
        for key in ("content", "feature_importances", "attributions",
                    "importances"):
            inner = content.get(key)
            if isinstance(inner, dict) and inner:
                return {k: float(v) for k, v in inner.items()
                        if isinstance(v, (int, float)) and not np.isnan(v)}
        only_numeric = {k: float(v) for k, v in content.items()
                        if isinstance(v, (int, float)) and not np.isnan(v)}
        if only_numeric:
            return only_numeric
    return None


def _effective_k(attribution, mass_threshold=MASS_THRESHOLD):
    abs_vals = sorted((abs(v) for v in attribution.values()), reverse=True)
    if not abs_vals:
        return 0
    total = sum(abs_vals)
    if total == 0:
        return len(abs_vals)
    cumsum = 0.0
    for i, v in enumerate(abs_vals, 1):
        cumsum += v
        if cumsum / total >= mass_threshold:
            return i
    return len(abs_vals)


class _ExportEntry:
    __slots__ = ("content", "metadata")

    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata or {}


def _load_export_patients(exports_dir, model_type):
    model_dir = exports_dir / model_type
    if not model_dir.is_dir():
        return []
    out = []
    for fp in sorted(model_dir.glob("patient_*.json")):
        try:
            with open(fp) as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    return out


def _explanations_from_exports(patients, modality, method, horizon=None):
    out = []
    for p in patients:
        explanations = p.get("explanations") or {}
        if horizon is None or horizon == "base":
            block = explanations.get(modality, {}).get(method)
        else:
            th = explanations.get("time_horizons", {})
            if not isinstance(th, dict):
                continue
            block = (th.get(horizon, {}) or {}).get(modality, {}).get(method)
        if not isinstance(block, dict):
            continue
        if "error" in block:
            continue
        out.append(_ExportEntry(block.get("content"), block.get("metadata")))
    return out


def _discover_horizons(patients):
    horizons = set()
    for p in patients:
        th = (p.get("explanations") or {}).get("time_horizons")
        if isinstance(th, dict):
            horizons.update(th.keys())

    def sort_key(h):
        m = re.match(r"^(\d+)", h)
        return (0, int(m.group(1)), h) if m else (1, 0, h)
    return sorted(horizons, key=sort_key)


def score_correctness_counterfactual_from_exports(explanations, higher_is_better):
    n_total = n_generated = n_flipped = 0
    for e in explanations:
        n_total += 1
        meta = e.metadata or {}
        cf_df = meta.get("counterfactual_df")
        if not cf_df:
            continue
        n_generated += 1
        orig = meta.get("original_prediction")
        cf = meta.get("cf_prediction")
        if orig is None or cf is None:
            continue
        try:
            orig_f, cf_f = float(orig), float(cf)
        except (TypeError, ValueError):
            continue
        if higher_is_better and cf_f > orig_f:
            n_flipped += 1
        elif (not higher_is_better) and cf_f < orig_f:
            n_flipped += 1

    if n_total == 0:
        return NA, {"n": 0}
    success_rate = n_flipped / n_total
    return _verdict_from_thresholds(
        success_rate, pass_at=0.7, partial_at=0.4,
    ), {
        "n_total": n_total,
        "n_generated": n_generated,
        "n_flipped": n_flipped,
        "success_rate": round(success_rate, 3),
        "source": "exports",
    }


_POST_TREATMENT_BY_MODEL = {
    "rsf": set(),
    "condor": set(),
}


def score_correctness_case_based_from_exports(
    explanations, pretreatment, train_cols, *, model_type=None,
):
    post_cols = train_cols - pretreatment
    if not post_cols and model_type in _POST_TREATMENT_BY_MODEL:
        post_cols = _POST_TREATMENT_BY_MODEL[model_type] & train_cols
        source_note = f"hardcoded list for {model_type}"
    else:
        source_note = "explainer.pretreatment_features() diff"
    if not post_cols:
        return PASS, {"reason": "no post-treatment columns to test",
                      "source": "exports",
                      "post_cols_source": source_note}

    total_checks = violations = 0
    for e in explanations:
        content = e.content
        cases = content if isinstance(content, list) else (
            content.get("similar_prototypes")
            if isinstance(content, dict) else None
        )
        if not cases:
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            for col in post_cols:
                if col in case:
                    total_checks += 1
                    val = case[col]
                    if val not in (None, 0, 0.0) and not (
                        isinstance(val, float) and np.isnan(val)
                    ):
                        violations += 1

    if total_checks == 0:
        return PASS, {"violations": 0, "total_checks": 0,
                      "reason": "post-treatment columns absent from cases",
                      "source": "exports"}
    rate = violations / total_checks
    return _verdict_from_thresholds(
        rate, pass_at=0.0, partial_at=0.05, higher_is_better=False,
    ), {"violations": violations, "total_checks": total_checks,
        "violation_rate": round(rate, 4), "source": "exports"}


def score_correctness_rule_based_from_exports(explanations):
    fidelities = []
    for e in explanations:
        content = e.content
        meta = e.metadata or {}
        f = None
        if isinstance(content, dict):
            f = content.get("fidelity")
        if f is None:
            f = meta.get("fidelity")
        try:
            f = float(f)
            if not np.isnan(f):
                fidelities.append(f)
        except (TypeError, ValueError):
            continue

    if not fidelities:
        return NA, {"reason": "no fidelity field in exports", "source": "exports"}

    median_fidelity = float(np.median(fidelities))
    return _verdict_from_thresholds(
        median_fidelity, pass_at=0.85, partial_at=0.7,
    ), {
        "median_fidelity": round(median_fidelity, 4),
        "n": len(fidelities),
        "source": "exports",
    }


def score_compactness(modality, explanations, *, method=None):
    if modality == "attribution":
        ks = []
        for e in explanations:
            attr = _extract_attribution_dict(getattr(e, "content", e))
            if attr is None:
                continue
            ks.append(_effective_k(attr))
        if not ks:
            return NA, {"n": 0}
        median_k = float(np.median(ks))
        if median_k <= COMPACTNESS_PASS_AT:
            verdict = PASS
        elif median_k <= COMPACTNESS_PARTIAL_AT:
            verdict = PARTIAL
        else:
            verdict = FAIL
        return verdict, {
            "median_effective_k": round(median_k, 2),
            "n_patients_scored": len(ks),
            "mass_threshold": MASS_THRESHOLD,
        }

    sizes = [_extract_size(modality, e) for e in explanations]
    sizes = [s for s in sizes if s is not None]
    if not sizes:
        return NA, {"n": 0}
    median = float(np.median(sizes))
    thresholds = {
        "counterfactual": (5, 15),
        "case_based": (5, 10),
        "rule_based": (5, 10),
    }
    pass_at, partial_at = thresholds.get(modality, (10, 25))
    return _verdict_from_thresholds(
        median, pass_at=pass_at, partial_at=partial_at,
        higher_is_better=False,
    ), {"median_size": median, "n": len(sizes)}


def score_correctness_attribution(explainer, patients, adapter):
    fill_values = adapter.training_data().select_dtypes(
        include=[np.number]
    ).mean().to_dict()
    fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    attr_changes, rand_changes = [], []
    rng = np.random.default_rng(42)

    for patient in patients:
        base = _scalar_prediction(adapter, patient)
        if base is None:
            continue
        try:
            attrs = explainer.feature_importances(patient)
        except Exception:
            continue
        if not attrs:
            continue

        ranked = sorted(attrs, key=lambda f: abs(attrs[f]), reverse=True)
        n = len(ranked)

        atc = []
        for frac in fractions:
            n_mask = max(1, int(frac * n))
            masked = patient.copy()
            for feat in ranked[:n_mask]:
                if feat in masked.columns:
                    masked[feat] = fill_values.get(feat, 0.0)
            scored = _scalar_prediction(adapter, masked)
            if scored is not None:
                atc.append(abs(base - scored))
        if atc:
            attr_changes.append(float(np.mean(atc)))

        shuffled = list(ranked)
        rng.shuffle(shuffled)
        rtc = []
        for frac in fractions:
            n_mask = max(1, int(frac * n))
            masked = patient.copy()
            for feat in shuffled[:n_mask]:
                if feat in masked.columns:
                    masked[feat] = fill_values.get(feat, 0.0)
            scored = _scalar_prediction(adapter, masked)
            if scored is not None:
                rtc.append(abs(base - scored))
        if rtc:
            rand_changes.append(float(np.mean(rtc)))

    if not attr_changes:
        return NA, {"n": 0}
    attr_mean = float(np.mean(attr_changes))
    rand_mean = float(np.mean(rand_changes)) if rand_changes else 0.0
    delta = attr_mean - rand_mean
    return _verdict_from_thresholds(
        delta, pass_at=0.05, partial_at=0.0,
    ), {
        "attr_mean_change": round(attr_mean, 4),
        "rand_mean_change": round(rand_mean, 4),
        "delta": round(delta, 4),
        "n": len(attr_changes),
        "source": "live",
    }


def score_contrastivity_attribution_cohort(patients, modality, method, horizon):
    samples = []
    for p in patients:
        explanations = p.get("explanations") or {}
        if horizon == "base":
            block = explanations.get(modality, {}).get(method)
        else:
            th = explanations.get("time_horizons") or {}
            block = ((th.get(horizon) or {}).get(modality) or {}).get(method)
        if not isinstance(block, dict):
            continue
        if "error" in block:
            continue
        attr = _extract_attribution_dict(block.get("content"))
        if not attr:
            continue
        meta = block.get("metadata") or {}
        pred = meta.get("original_prediction")
        if pred is None:
            continue
        try:
            samples.append((float(pred), attr))
        except (TypeError, ValueError):
            continue

    if len(samples) < 20:
        return NA, {
            "reason": f"not enough patients (need >= 20, have {len(samples)})",
            "n_samples": len(samples),
        }

    predictions = [s[0] for s in samples]
    median = float(np.median(predictions))
    high = [s[1] for s in samples if s[0] >= median]
    low = [s[1] for s in samples if s[0] < median]
    if not high or not low:
        return NA, {"reason": "median split yielded empty cohort",
                    "n_high": len(high), "n_low": len(low)}

    def mean_abs_attribution(group):
        sums = {}
        for attr in group:
            for f, v in attr.items():
                sums[f] = sums.get(f, 0.0) + abs(v)
        n = len(group)
        return {f: sums[f] / n for f in sums}

    high_mean = mean_abs_attribution(high)
    low_mean = mean_abs_attribution(low)
    high_rank = sorted(high_mean, key=lambda f: high_mean[f], reverse=True)
    low_rank = sorted(low_mean, key=lambda f: low_mean[f], reverse=True)

    k = CONTRASTIVITY_TOP_K
    high_top = set(high_rank[:k])
    low_top = set(low_rank[:k])
    overlap = len(high_top & low_top) / k

    if overlap < CONTRASTIVITY_PASS_AT:
        verdict = PASS
    elif overlap < CONTRASTIVITY_PARTIAL_AT:
        verdict = PARTIAL
    else:
        verdict = FAIL

    return verdict, {
        "top_k_overlap": round(overlap, 3),
        "k": k,
        "n_high_risk": len(high),
        "n_low_risk": len(low),
        "median_prediction": round(median, 4),
        "high_risk_top_features": list(high_rank[:k]),
        "low_risk_top_features": list(low_rank[:k]),
        "thresholds": {"pass_below": CONTRASTIVITY_PASS_AT,
                       "partial_below": CONTRASTIVITY_PARTIAL_AT},
        "source": "exports",
    }


def score_contrastivity(modality, explanations, method,
                        *, raw_patients=None, horizon=None):
    if modality == "counterfactual":
        return PASS, {"reason": "inherent (counterfactual semantics)"}

    if modality == "rule_based":
        for e in explanations:
            content = getattr(e, "content", None)
            if isinstance(content, dict) and content.get("precision") is not None:
                return PASS, {"reason": "explicit precision per rule"}
        return PARTIAL, {"reason": "rule structure but no precision/coverage guarantee"}

    if modality == "case_based":
        for e in explanations:
            content = getattr(e, "content", None)
            if isinstance(content, dict) and "similar_prototypes" in content:
                return PASS, {"reason": "prototype risk scores enable contrast"}
        return PARTIAL, {"reason": "similarity-only retrieval; contrast is implicit"}

    if modality == "attribution":
        if raw_patients and horizon is not None:
            v, m = score_contrastivity_attribution_cohort(
                raw_patients, modality, method, horizon,
            )
            if v != NA:
                return v, m

        method_lower = (method or "").lower()
        if any(m in method_lower for m in _UNSIGNED_ATTRIBUTION_METHODS):
            return PARTIAL, {
                "reason": "magnitude-only attribution; sign not defined by method",
                "method_class": "unsigned_by_construction",
            }
        n_signed = n_total = 0
        for e in explanations:
            content = getattr(e, "content", None)
            if not isinstance(content, dict):
                continue
            n_total += 1
            try:
                vals = [float(v) for v in content.values() if v is not None]
                if any(v < 0 for v in vals) and any(v > 0 for v in vals):
                    n_signed += 1
            except (TypeError, ValueError):
                pass
        if n_total == 0:
            return NA, {"n": 0}
        rate = n_signed / n_total
        return _verdict_from_thresholds(
            rate, pass_at=0.8, partial_at=0.4,
        ), {"signed_rate": round(rate, 3), "n": n_total}

    return NA, {}


def score_covariate_complexity(modality, method, explanations, feature_ranges,
                               *, model_type=None):
    referenced = set()
    sample_values = []
    implausible_count = 0
    n_examined = 0

    for e in explanations:
        content = getattr(e, "content", None)

        if modality == "case_based":
            if isinstance(content, pd.DataFrame):
                for col in content.columns:
                    if not str(col).startswith("_"):
                        referenced.add(str(col))
            elif isinstance(content, list):
                for case in content:
                    if isinstance(case, dict):
                        for k in case:
                            if not k.startswith("_"):
                                referenced.add(k)
            elif isinstance(content, dict):
                protos = content.get("similar_prototypes")
                if protos is not None:
                    if isinstance(protos, pd.DataFrame):
                        for col in protos.columns:
                            if not str(col).startswith("_"):
                                referenced.add(str(col))
                    else:
                        for proto in protos:
                            if isinstance(proto, dict):
                                for k in proto:
                                    if not k.startswith("_"):
                                        referenced.add(k)
                for key in ("feature_diffs_to_nearest",
                            "shared_features_with_nearest"):
                    diffs = content.get(key)
                    if diffs is not None:
                        if isinstance(diffs, pd.DataFrame):
                            for col in diffs.columns:
                                referenced.add(str(col))
                        else:
                            for k in diffs:
                                referenced.add(k)
            continue

        if not isinstance(content, dict):
            continue

        if modality == "rule_based":
            for cond in (content.get("conditions") or []):
                for feat in _extract_all_features(cond):
                    referenced.add(feat)
            continue

        for k, v in content.items():
            bare = _strip_threshold(k)
            referenced.add(bare)
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            sample_values.append(val)
            n_examined += 1
            if modality == "counterfactual" and bare in feature_ranges:
                lo, hi = feature_ranges[bare]
                span = max(hi - lo, 1.0)
                if val < lo - 5 * span or val > hi + 5 * span:
                    implausible_count += 1

    metrics = {"n_features_referenced": len(referenced),
               "n_values": n_examined}
    if not referenced:
        return NA, metrics

    raw_clinical = set(feature_ranges.keys())
    overlap = len(referenced & raw_clinical) / len(referenced)
    metrics["raw_feature_overlap"] = round(overlap, 3)

    if feature_ranges:
        raw_spans = [hi - lo for lo, hi in feature_ranges.values()]
        max_raw_span = float(max(raw_spans)) if raw_spans else 0.0
        metrics["max_raw_feature_span"] = round(max_raw_span, 1)
    else:
        max_raw_span = 0.0

    if sample_values:
        in_zscore = sum(1 for v in sample_values if -3 <= v <= 3) / len(sample_values)
        metrics["frac_in_zscore_range"] = round(in_zscore, 3)

    if modality == "counterfactual" and implausible_count > 0 and n_examined > 0:
        impl_rate = implausible_count / n_examined
        metrics["implausibility_rate"] = round(impl_rate, 3)
        if impl_rate > 0.05:
            return FAIL, metrics

    if model_type == "rsf":
        if overlap >= 0.8 and max_raw_span > 100:
            return PASS, metrics
        if overlap >= 0.5:
            return PARTIAL, metrics
        return FAIL, metrics

    if overlap >= 0.8:
        return PASS, metrics
    if overlap >= 0.5:
        return PARTIAL, metrics
    return FAIL, metrics


def _normalize_cases(content):
    if content is None:
        return []
    if isinstance(content, pd.DataFrame):
        return [] if content.empty else content.to_dict("records")
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        for key in ("similar_prototypes", "cases"):
            inner = content.get(key)
            if inner is None:
                continue
            if isinstance(inner, pd.DataFrame):
                return [] if inner.empty else inner.to_dict("records")
            if isinstance(inner, list):
                return inner
    return []


def _check_structural_completeness(modality, explanation):
    content = getattr(explanation, "content", None)
    metadata = getattr(explanation, "metadata", {}) or {}

    fields = {}
    required = []
    optional = []

    if modality == "attribution":
        required = ["content_nonempty"]
        optional = ["confidence_metrics", "original_prediction"]
        fields["content_nonempty"] = (
            isinstance(content, dict) and any(
                v is not None and abs(float(v)) > 1e-12
                for v in content.values()
                if isinstance(v, (int, float))
            )
        )
        fields["confidence_metrics"] = bool(metadata.get("confidence_metrics"))
        fields["original_prediction"] = metadata.get("original_prediction") is not None

    elif modality == "counterfactual":
        required = ["content_nonempty", "counterfactual_df",
                    "original_prediction", "cf_prediction"]
        optional = ["confidence_metrics"]
        fields["content_nonempty"] = (
            isinstance(content, dict) and bool(content)
        )
        fields["counterfactual_df"] = metadata.get("counterfactual_df") is not None
        fields["original_prediction"] = metadata.get("original_prediction") is not None
        fields["cf_prediction"] = metadata.get("cf_prediction") is not None
        fields["confidence_metrics"] = bool(metadata.get("confidence_metrics"))

    elif modality == "case_based":
        required = ["cases_nonempty", "similarity_or_distance"]
        optional = ["case_outcomes"]
        cases = _normalize_cases(content)
        fields["cases_nonempty"] = len(cases) > 0
        sim_keys = ("similarity", "distance", "_distance", "score", "weight")
        outcome_keys = ("outcome", "risk", "label", "key_eid", "key_nkr",
                        "patient_id", "risk_score")
        fields["similarity_or_distance"] = any(
            isinstance(c, dict) and any(k in c for k in sim_keys)
            for c in cases
        )
        fields["case_outcomes"] = any(
            isinstance(c, dict) and any(k in c for k in outcome_keys)
            for c in cases
        )

    elif modality == "rule_based":
        required = ["conditions_nonempty", "precision_or_fidelity"]
        optional = ["coverage"]
        if isinstance(content, dict):
            conditions = content.get("conditions") or []
            fields["conditions_nonempty"] = len(conditions) > 0
            fields["precision_or_fidelity"] = (
                content.get("precision") is not None or
                content.get("fidelity") is not None
            )
            fields["coverage"] = content.get("coverage") is not None
        else:
            fields["conditions_nonempty"] = False
            fields["precision_or_fidelity"] = False
            fields["coverage"] = False

    req_present = sum(1 for f in required if fields.get(f, False))
    opt_present = sum(1 for f in optional if fields.get(f, False))
    return req_present, len(required), opt_present, len(optional)


def score_composition(modality, explanations):
    req_fracs = []
    opt_fracs = []
    sizes = []
    for e in explanations:
        nr, tr, no, to = _check_structural_completeness(modality, e)
        if tr > 0:
            req_fracs.append(nr / tr)
        if to > 0:
            opt_fracs.append(no / to)
        size = _extract_size(modality, e)
        if size is not None:
            sizes.append(size)

    if not req_fracs:
        return NA, {"n": 0, "reason": "no explanations to check"}

    median_req = float(np.median(req_fracs))
    median_opt = float(np.median(opt_fracs)) if opt_fracs else 0.0

    if sizes:
        mean_size = float(np.mean(sizes))
        std_size = float(np.std(sizes))
        cv = std_size / mean_size if mean_size > 0 else float("inf")
    else:
        mean_size = std_size = 0.0
        cv = float("inf")

    if median_req < 0.9:
        verdict = FAIL
    elif median_req >= 0.95 and cv <= 0.3:
        verdict = PASS
    elif median_req >= 0.9 and cv <= 0.7:
        verdict = PARTIAL
    elif median_req >= 0.95:
        verdict = PARTIAL
    else:
        verdict = FAIL

    return verdict, {
        "median_required_fields_present": round(median_req, 3),
        "median_optional_fields_present": round(median_opt, 3),
        "size_cv": round(cv, 3) if cv != float("inf") else None,
        "mean_size": round(mean_size, 2),
        "n_explanations": len(req_fracs),
    }


def score_continuity_consistency(explainer_a, explainer_b, modality, patients,
                                 *, method=None):
    out = {"continuity": (NA, {"n": 0}), "consistency": (NA, {"n": 0})}
    if not patients:
        return out
    cont_jaccards, cons_jaccards = [], []
    rng = np.random.default_rng(123)

    for patient in patients:
        try:
            base = explainer_a.explain(patient)
            base_top = _top_k_features(modality, base, k=5)
        except Exception:
            continue

        noisy = patient.copy()
        for col in noisy.select_dtypes(include=[np.number]).columns:
            noisy[col] = noisy[col] + rng.normal(0, 0.01)
        try:
            noisy_explanation = explainer_a.explain(noisy)
            noisy_top = _top_k_features(modality, noisy_explanation, k=5)
            if base_top or noisy_top:
                cont_jaccards.append(
                    len(base_top & noisy_top) /
                    max(len(base_top | noisy_top), 1)
                )
        except Exception:
            pass

        try:
            reseeded = explainer_b.explain(patient)
            reseeded_top = _top_k_features(modality, reseeded, k=5)
            if base_top or reseeded_top:
                cons_jaccards.append(
                    len(base_top & reseeded_top) /
                    max(len(base_top | reseeded_top), 1)
                )
        except Exception:
            pass

    if cont_jaccards:
        cont_mean = float(np.mean(cont_jaccards))
        out["continuity"] = (
            _verdict_from_thresholds(cont_mean, pass_at=0.7, partial_at=0.4),
            {"mean_jaccard": round(cont_mean, 3), "n": len(cont_jaccards),
             "source": "live"},
        )
    if cons_jaccards:
        cons_mean = float(np.mean(cons_jaccards))
        measured_verdict = _verdict_from_thresholds(
            cons_mean, pass_at=0.8, partial_at=0.5)
        if method in DETERMINISTIC_METHODS:
            out["consistency"] = (PASS, {
                "mean_jaccard": round(cons_mean, 3),
                "n": len(cons_jaccards),
                "measured_verdict": measured_verdict,
                "method_class": "deterministic_by_construction",
                "source": "live",
            })
        else:
            out["consistency"] = (
                measured_verdict,
                {"mean_jaccard": round(cons_mean, 3), "n": len(cons_jaccards),
                 "source": "live"},
            )
    return out


def _collect_cm(explanations):
    out = []
    for e in explanations:
        cm = (e.metadata or {}).get("confidence_metrics")
        if isinstance(cm, dict):
            out.append(cm)
    return out


def _score_completeness_attribution(method, cms):
    m = (method or "").lower()
    if "lime" in m:
        vals = [cm.get("local_r2") for cm in cms]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return NA, {"reason": "no local_r2"}
        med = float(np.median(vals))
        return _verdict_from_thresholds(
            med, pass_at=0.7, partial_at=0.4,
        ), {"median_local_r2": round(med, 4), "n": len(vals)}

    if "permutation" in m:
        vals = [cm.get("median_relative_std") for cm in cms]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return NA, {"reason": "no median_relative_std"}
        med = float(np.median(vals))
        return _verdict_from_thresholds(
            med, pass_at=0.5, partial_at=1.0, higher_is_better=False,
        ), {"median_relative_std": round(med, 4), "n": len(vals)}

    if "integrated_gradients" in m or "deep_lift" in m or "tree_shap" in m:
        vals = [cm.get("completeness_ratio") for cm in cms]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return NA, {"reason": "no completeness_ratio"}
        med = float(np.median(vals))
        return _verdict_from_thresholds(
            med, pass_at=0.05, partial_at=0.15, higher_is_better=False,
        ), {"median_completeness_ratio": round(med, 4), "n": len(vals)}

    return NA, {"reason": f"no completeness schema for {method!r}",
                "n_with_cm": len(cms)}


def score_completeness(modality, method, explanations):
    if modality != "attribution":
        return NA, {"reason": "completeness is attribution-specific in Nauta CO12"}

    cms = _collect_cm(explanations)
    if not cms:
        return NA, {"reason": "no metadata.confidence_metrics in exports",
                    "n_explanations": len(explanations)}
    return _score_completeness_attribution(method, cms)


def _score_confidence_attribution_data_driven(method, cms):
    m = (method or "").lower()

    if any(d in m for d in DETERMINISTIC_METHODS):
        return FAIL, {
            "method_class": "deterministic_by_construction",
            "reason": "no uncertainty signal exists",
            "verdict_source": "structural",
        }

    if "lime" in m:
        vals = [cm.get("local_r2") for cm in cms]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            med = float(np.median(vals))
            return _verdict_from_thresholds(
                med, pass_at=0.7, partial_at=0.4,
            ), {
                "median_local_r2": round(med, 4),
                "n": len(vals),
                "verdict_source": "data_driven",
            }

    if "permutation" in m or "deep_lift_shap" in m:
        vals = [cm.get("median_relative_std") for cm in cms]
        vals = [float(v) for v in vals if v is not None]
        if vals:
            med = float(np.median(vals))
            return _verdict_from_thresholds(
                med, pass_at=0.5, partial_at=1.0, higher_is_better=False,
            ), {
                "median_relative_std": round(med, 4),
                "n": len(vals),
                "verdict_source": "data_driven",
            }

    return CONFIDENCE_BY_METHOD.get(method, NA), {
        "verdict_source": "categorical_fallback",
        "reason": "no data-driven confidence signal in exports",
    }


def _score_confidence_counterfactual_data_driven(cms):
    if not cms:
        return NA, {"reason": "no confidence_metrics across cohort"}

    flipped = [bool(cm.get("flipped")) for cm in cms if "flipped" in cm]
    if not flipped:
        return NA, {"reason": "no 'flipped' field in confidence_metrics"}

    flip_rate = sum(flipped) / len(flipped)
    metrics = {
        "flip_rate": round(flip_rate, 3),
        "n_flipped": sum(flipped),
        "n_total": len(flipped),
        "verdict_source": "data_driven",
    }
    proxs = [cm.get("proximity_l1_raw") for cm in cms]
    proxs = [float(p) for p in proxs if p is not None]
    if proxs:
        metrics["median_proximity_l1_raw"] = round(float(np.median(proxs)), 3)
    sparsities = [cm.get("sparsity") for cm in cms]
    sparsities = [int(s) for s in sparsities if s is not None]
    if sparsities:
        metrics["median_sparsity"] = float(np.median(sparsities))

    return _verdict_from_thresholds(
        flip_rate, pass_at=0.8, partial_at=0.4,
    ), metrics


def _score_confidence_case_based_data_driven(method, cms):
    if not cms:
        return NA, {"reason": "no confidence_metrics across cohort"}

    vals = [cm.get("median_distance") for cm in cms]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return NA, {"reason": "no median_distance signal"}

    med = float(np.median(vals))
    is_embedding = "embedding" in (method or "").lower()
    if is_embedding:
        pass_at, partial_at, metric_label = 0.05, 0.15, "cosine"
    else:
        pass_at, partial_at, metric_label = 0.30, 0.60, "gower"

    return _verdict_from_thresholds(
        med, pass_at=pass_at, partial_at=partial_at, higher_is_better=False,
    ), {
        "median_distance": round(med, 4),
        "n": len(vals),
        "distance_metric": metric_label,
        "thresholds": {"pass_at": pass_at, "partial_at": partial_at},
        "verdict_source": "data_driven",
    }


def _score_confidence_rule_based_data_driven(explanations):
    fids, precs, covs = [], [], []
    for e in explanations:
        content = getattr(e, "content", None)
        if not isinstance(content, dict):
            continue
        for src, dst in (("fidelity", fids), ("precision", precs),
                         ("coverage", covs)):
            v = content.get(src)
            if v is not None:
                try:
                    fv = float(v)
                    if not np.isnan(fv):
                        dst.append(fv)
                except (TypeError, ValueError):
                    continue

    primary = fids if fids else precs
    if not primary:
        return NA, {"reason": "no fidelity or precision in content"}

    med = float(np.median(primary))
    metric_name = "fidelity" if fids else "precision"
    metrics = {
        f"median_{metric_name}": round(med, 4),
        "n": len(primary),
        "verdict_source": "data_driven",
    }
    if covs:
        metrics["median_coverage"] = round(float(np.median(covs)), 4)

    return _verdict_from_thresholds(
        med, pass_at=0.85, partial_at=0.7,
    ), metrics


def score_confidence(modality, method, explanations=None):
    if explanations is None:
        verdict = CONFIDENCE_BY_METHOD.get(method, NA)
        return verdict, {
            "verdict_source": "categorical_fallback",
            "reason": "no explanations provided",
        }

    if modality == "rule_based":
        v, m = _score_confidence_rule_based_data_driven(explanations)
        if v == NA:
            v = CONFIDENCE_BY_METHOD.get(method, NA)
            m["verdict_source"] = "categorical_fallback"
        return v, m

    cms = _collect_cm(explanations)

    if modality == "attribution":
        return _score_confidence_attribution_data_driven(method, cms)
    if modality == "counterfactual":
        v, m = _score_confidence_counterfactual_data_driven(cms)
        if v == NA:
            v = CONFIDENCE_BY_METHOD.get(method, NA)
            m["verdict_source"] = "categorical_fallback"
        return v, m
    if modality == "case_based":
        v, m = _score_confidence_case_based_data_driven(method, cms)
        if v == NA:
            v = CONFIDENCE_BY_METHOD.get(method, NA)
            m["verdict_source"] = "categorical_fallback"
        return v, m

    return CONFIDENCE_BY_METHOD.get(method, NA), {
        "verdict_source": "categorical_fallback",
        "reason": f"unknown modality {modality!r}",
    }


def score_controllability(modality):
    verdict = CONTROLLABILITY_BY_MODALITY.get(modality, NA)
    return verdict, {
        "verdict_source": "per-modality categorical",
        "rationale": f"modality={modality} -> {verdict}",
    }


def _coherence_mass_fraction(attribution, clinical_factors):
    if not attribution:
        return 0.0
    total = sum(abs(v) for v in attribution.values())
    if total == 0:
        return 0.0
    clinical_mass = sum(
        abs(v) for f, v in attribution.items()
        if _strip_threshold(f) in clinical_factors
    )
    return clinical_mass / total


def score_coherence(modality, method, explanations,
                    clinical_factors=PDAC_PROGNOSTIC_FACTORS):
    if modality != "attribution":
        return NA, {
            "reason": "coherence is attribution-specific; other modalities "
                      "evaluated qualitatively in vignette study",
        }

    fracs = []
    for e in explanations:
        attr = _extract_attribution_dict(getattr(e, "content", e))
        if attr is None:
            continue
        fracs.append(_coherence_mass_fraction(attr, clinical_factors))

    if not fracs:
        return NA, {"reason": "no attribution dicts in explanations",
                    "n": len(explanations)}

    median_frac = float(np.median(fracs))
    if median_frac >= COHERENCE_PASS_AT:
        verdict = PASS
    elif median_frac >= COHERENCE_PARTIAL_AT:
        verdict = PARTIAL
    else:
        verdict = FAIL
    return verdict, {
        "median_clinical_mass_fraction": round(median_frac, 3),
        "n_patients_scored": len(fracs),
        "n_clinical_factors": len(clinical_factors),
        "thresholds": {"pass_at": COHERENCE_PASS_AT,
                       "partial_at": COHERENCE_PARTIAL_AT},
        "source": "exports",
    }


def _load_adapters_and_registries(*, rsf_artifact, condor_checkpoint,
                                  n_patients, train_data, test_data):
    from xai_decision_making.explainability.adapters import load_rsf_adapter, load_condor_adapter
    from xai_decision_making.explainability.run_export import build_registry

    out = {}
    for model_type, artifact in (
        ("rsf", rsf_artifact),
        ("condor", condor_checkpoint),
    ):
        try:
            if model_type == "rsf":
                adapter, _, _, _ = load_rsf_adapter(
                    train_data, test_data, n_patients, artifact,
                )
            else:
                adapter, _, _, _ = load_condor_adapter(
                    train_data, test_data, n_patients, artifact,
                )
        except Exception:
            continue
        out[model_type] = {
            "adapter": adapter,
            "registry": build_registry(model_type),
        }
    return out


def _build_feature_ranges(adapter):
    train = adapter.training_data()
    ranges = {}
    for col in train.select_dtypes(include=[np.number]).columns:
        col_vals = train[col].dropna()
        if len(col_vals) >= 5:
            ranges[col] = (float(col_vals.min()), float(col_vals.max()))
    return ranges


def _instantiate_with_seed(cls, adapter, seed):
    np.random.seed(seed)
    try:
        return cls(adapter, random_state=seed)
    except TypeError:
        return cls(adapter)


def _empty_verdicts(cross_model_verdict=NA):
    return {col: (cross_model_verdict if col == "cross_model" else NA)
            for col in VERDICT_COLS}


def _row_from_verdicts(model_type, modality, method, horizon, verdicts,
                       elapsed_s):
    n_pass = sum(1 for v in verdicts.values() if v == PASS)
    n_fail = sum(1 for v in verdicts.values() if v == FAIL)
    n_partial = sum(1 for v in verdicts.values() if v == PARTIAL)

    if verdicts["cross_model"] == FAIL:
        decision = "DROP (cross-model)"
    elif n_fail >= 2:
        decision = "DROP (>=2 fails)"
    else:
        decision = "KEEP"

    return {
        "model_type": model_type,
        "modality": modality,
        "method": method,
        "horizon": horizon,
        **verdicts,
        "aggregate_score": n_pass - n_fail,
        "n_pass": n_pass,
        "n_partial": n_partial,
        "n_fail": n_fail,
        "decision": decision,
        "elapsed_s": round(elapsed_s, 1),
    }


def _detail_path(out_dir, model_type, modality, method, horizon="base"):
    return out_dir / f"co12_{model_type}_{modality}_{method}_{horizon}.json"


def _load_existing_audit_row(out_dir, model_type, modality, method,
                             horizon="base"):
    fp = _detail_path(out_dir, model_type, modality, method, horizon)
    if not fp.is_file():
        return None
    try:
        with open(fp) as f:
            detail = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if detail.get("schema_version") != SCHEMA_VERSION:
        return None

    verdicts = detail.get("verdicts") or {}
    decision = detail.get("decision") or "KEEP"

    full_verdicts = {col: verdicts.get(col, NA) for col in VERDICT_COLS}
    return _row_from_verdicts(
        model_type, modality, method, horizon, full_verdicts, 0.0,
    ) | {"decision": decision}


def _write_failed_stub(out_dir, model_type, modality, method, cross_model,
                       reason, horizon="base"):
    verdicts = _empty_verdicts(cross_model)
    detail = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "modality": modality,
        "method": method,
        "horizon": horizon,
        "n_patients": 0,
        "verdicts": verdicts,
        "metrics": {col: {"reason": reason} for col in VERDICT_COLS},
        "decision": f"DROP (skip: {reason})",
        "stub": True,
    }
    fp = _detail_path(out_dir, model_type, modality, method, horizon)
    with open(fp, "w") as f:
        json.dump(detail, f, indent=2)


def _failed_row(model_type, modality, method, cross_model, reason,
                horizon="base"):
    verdicts = _empty_verdicts(cross_model)
    return {
        "model_type": model_type,
        "modality": modality,
        "method": method,
        "horizon": horizon,
        **verdicts,
        "aggregate_score": 0,
        "n_pass": 0,
        "n_partial": 0,
        "n_fail": 0,
        "decision": f"DROP (skip: {reason})",
        "elapsed_s": 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsf-artifact", required=True)
    parser.add_argument("--condor-checkpoint", required=True)
    parser.add_argument("--exports-dir", default=None)
    parser.add_argument("--n-patients", type=int, default=30)
    parser.add_argument("--n-live", type=int, default=N_LIVE_DEFAULT)
    parser.add_argument("--skip-method", action="append", default=[])
    parser.add_argument("--only-method", default=None)
    parser.add_argument("--only-modality", default=None)
    parser.add_argument("--only-model", default=None)
    parser.add_argument("--out-dir",
                        default=os.path.join(ROOT, "exports", "co12"))
    parser.add_argument("--train-data",
                        default=os.path.join(ROOT, "data", "dev_set_processed.csv"))
    parser.add_argument("--test-data",
                        default=os.path.join(ROOT, "data", "test_set_processed.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_set = set(args.skip_method or [])

    bundles = _load_adapters_and_registries(
        rsf_artifact=args.rsf_artifact,
        condor_checkpoint=args.condor_checkpoint,
        n_patients=args.n_patients,
        train_data=args.train_data,
        test_data=args.test_data,
    )
    if args.only_model:
        bundles = {k: v for k, v in bundles.items() if k == args.only_model}
    if not bundles:
        sys.exit(1)

    exports_by_model = {}
    if args.exports_dir:
        exports_dir = Path(args.exports_dir)
        if not exports_dir.is_dir():
            sys.exit(1)
        for model_type in bundles:
            exports_by_model[model_type] = _load_export_patients(
                exports_dir, model_type)

    rsf_methods = set()
    if "rsf" in bundles:
        for modality, methods in bundles["rsf"]["registry"].items():
            for method in methods:
                if method in skip_set:
                    continue
                rsf_methods.add((modality, method))
    condor_methods = set()
    if "condor" in bundles:
        for modality, methods in bundles["condor"]["registry"].items():
            for method in methods:
                if method in skip_set:
                    continue
                condor_methods.add((modality, method))
    all_methods = rsf_methods | condor_methods

    cross_model_verdicts = {}
    for (modality, method) in all_methods:
        if (modality, method) in rsf_methods and (modality, method) in condor_methods:
            cross_model_verdicts[(modality, method)] = PASS
        elif method in ARCH_NATIVE:
            cross_model_verdicts[(modality, method)] = PARTIAL
        else:
            cross_model_verdicts[(modality, method)] = FAIL

    samples = {
        model_type: _sample_patients(b["adapter"], args.n_patients,
                                     seed=args.seed)
        for model_type, b in bundles.items()
    }
    feature_ranges = {
        model_type: _build_feature_ranges(b["adapter"])
        for model_type, b in bundles.items()
    }

    rows = []
    n_reused_cells = 0
    n_computed_cells = 0
    n_live_runs = 0
    t_start = time.time()

    for model_type, bundle in bundles.items():
        adapter = bundle["adapter"]
        registry = bundle["registry"]
        patients = samples[model_type]
        ranges = feature_ranges[model_type]
        export_patients = exports_by_model.get(model_type, [])

        horizons_to_audit = ["base"] + _discover_horizons(export_patients)

        for modality, methods in registry.items():
            if args.only_modality and modality != args.only_modality:
                continue
            for method, cls in methods.items():
                if method in skip_set:
                    continue
                if args.only_method and method != args.only_method:
                    continue

                cm_verdict = cross_model_verdicts.get((modality, method), NA)

                if modality == "case_based":
                    scoring_horizons = ["base"]
                    mirror_horizons = [h for h in horizons_to_audit if h != "base"]
                else:
                    scoring_horizons = horizons_to_audit
                    mirror_horizons = []

                if not args.refresh:
                    all_cached = True
                    cached_rows = []
                    for h in horizons_to_audit:
                        cached = _load_existing_audit_row(
                            out_dir, model_type, modality, method, h,
                        )
                        if cached is None:
                            all_cached = False
                            break
                        cached_rows.append(cached)
                    if all_cached:
                        rows.extend(cached_rows)
                        n_reused_cells += len(cached_rows)
                        continue

                t_method_start = time.time()
                try:
                    explainer_a = _instantiate_with_seed(cls, adapter, args.seed)
                except Exception as exc:
                    reason = f"instantiation failed: {exc}"
                    traceback.print_exc(limit=2)
                    for h in horizons_to_audit:
                        rows.append(_failed_row(
                            model_type, modality, method, cm_verdict, reason, h,
                        ))
                        _write_failed_stub(
                            out_dir, model_type, modality, method, cm_verdict,
                            reason, h,
                        )
                    continue

                try:
                    explainer_b = _instantiate_with_seed(
                        cls, adapter, args.seed + 1)
                except Exception:
                    explainer_b = None

                live_sample = patients[:args.n_live]

                live_explanations = []
                for patient in live_sample:
                    try:
                        live_explanations.append(explainer_a.explain(patient))
                    except Exception:
                        pass

                if not live_explanations and not export_patients:
                    reason = "all explain() failed; no exports either"
                    for h in horizons_to_audit:
                        rows.append(_failed_row(
                            model_type, modality, method, cm_verdict, reason, h,
                        ))
                        _write_failed_stub(
                            out_dir, model_type, modality, method, cm_verdict,
                            reason, h,
                        )
                    continue

                if modality == "attribution":
                    live_cor_v, live_cor_m = score_correctness_attribution(
                        explainer_a, live_sample, adapter)
                else:
                    live_cor_v, live_cor_m = None, None

                if explainer_b is not None and live_explanations:
                    rerun = score_continuity_consistency(
                        explainer_a, explainer_b, modality, live_sample,
                        method=method,
                    )
                else:
                    reason = ("explainer_b instantiation failed"
                              if explainer_b is None
                              else "no live explanations")
                    rerun = {
                        "continuity": (NA, {"reason": reason}),
                        "consistency": (NA, {"reason": reason}),
                    }
                cont_v, cont_m = rerun["continuity"]
                cons_v, cons_m = rerun["consistency"]

                n_live_runs += 1

                base_row = None
                base_detail = None
                for horizon in scoring_horizons:
                    if not args.refresh:
                        cached = _load_existing_audit_row(
                            out_dir, model_type, modality, method, horizon,
                        )
                        if cached is not None:
                            rows.append(cached)
                            n_reused_cells += 1
                            if horizon == "base" and mirror_horizons:
                                bp = _detail_path(out_dir, model_type, modality,
                                                  method, "base")
                                if bp.is_file():
                                    try:
                                        base_detail = json.loads(bp.read_text())
                                    except Exception:
                                        base_detail = None
                                base_row = cached
                            continue

                    t0 = time.time()
                    n_computed_cells += 1

                    export_explanations = (
                        _explanations_from_exports(
                            export_patients, modality, method, horizon=horizon,
                        ) if export_patients else []
                    )
                    use_exports = bool(export_explanations)
                    source_explanations = (
                        export_explanations if use_exports else live_explanations
                    )

                    if modality == "attribution":
                        cor_v, cor_m = live_cor_v, dict(live_cor_m)
                    elif modality == "counterfactual":
                        if use_exports:
                            higher_better = getattr(
                                adapter, "higher_is_better", False)
                            cor_v, cor_m = (
                                score_correctness_counterfactual_from_exports(
                                    export_explanations,
                                    higher_is_better=higher_better,
                                )
                            )
                        else:
                            cor_v, cor_m = NA, {"reason": "no exports"}
                    elif modality == "case_based":
                        if use_exports:
                            try:
                                pretreatment = set(
                                    explainer_a.pretreatment_features())
                            except Exception:
                                pretreatment = set()
                            train_cols = set(adapter.training_data().columns)
                            cor_v, cor_m = (
                                score_correctness_case_based_from_exports(
                                    export_explanations, pretreatment,
                                    train_cols, model_type=model_type,
                                )
                            )
                        else:
                            cor_v, cor_m = NA, {"reason": "no exports"}
                    elif modality == "rule_based":
                        if use_exports:
                            cor_v, cor_m = (
                                score_correctness_rule_based_from_exports(
                                    export_explanations,
                                )
                            )
                        else:
                            cor_v, cor_m = NA, {"reason": "no exports"}
                    else:
                        cor_v, cor_m = NA, {}

                    comp_v, comp_m = score_compactness(
                        modality, source_explanations, method=method)
                    con_v, con_m = score_contrastivity(
                        modality, source_explanations, method,
                        raw_patients=export_patients,
                        horizon=horizon,
                    )
                    cov_v, cov_m = score_covariate_complexity(
                        modality, method, source_explanations, ranges,
                        model_type=model_type)
                    cmp_v, cmp_m = score_composition(
                        modality, source_explanations)
                    completeness_v, completeness_m = score_completeness(
                        modality, method, source_explanations)
                    conf_v, conf_m = score_confidence(
                        modality, method, source_explanations)
                    ctrl_v, ctrl_m = score_controllability(modality)
                    cohe_v, cohe_m = score_coherence(
                        modality, method, source_explanations)

                    for d in (comp_m, con_m, cov_m, cmp_m, cohe_m,
                              completeness_m):
                        if isinstance(d, dict) and "source" not in d:
                            d["source"] = "exports" if use_exports else "live"
                        if isinstance(d, dict) and "horizon" not in d:
                            d["horizon"] = horizon

                    verdicts = {
                        "cross_model": cm_verdict,
                        "compactness": comp_v,
                        "correctness": cor_v,
                        "contrastivity": con_v,
                        "covariate_complexity": cov_v,
                        "composition": cmp_v,
                        "continuity": cont_v,
                        "consistency": cons_v,
                        "completeness": completeness_v,
                        "confidence": conf_v,
                        "controllability": ctrl_v,
                        "coherence": cohe_v,
                    }
                    row = _row_from_verdicts(
                        model_type, modality, method, horizon, verdicts,
                        time.time() - t0,
                    )
                    rows.append(row)

                    detail = {
                        "schema_version": SCHEMA_VERSION,
                        "model_type": model_type,
                        "modality": modality,
                        "method": method,
                        "horizon": horizon,
                        "n_patients": len(patients),
                        "n_live": len(live_sample),
                        "n_exports": len(export_explanations),
                        "verdicts": verdicts,
                        "metrics": {
                            "cross_model": {"verdict": cm_verdict},
                            "compactness": comp_m,
                            "correctness": cor_m,
                            "contrastivity": con_m,
                            "covariate_complexity": cov_m,
                            "composition": cmp_m,
                            "continuity": cont_m,
                            "consistency": cons_m,
                            "completeness": completeness_m,
                            "confidence": conf_m,
                            "controllability": ctrl_m,
                            "coherence": cohe_m,
                        },
                        "decision": row["decision"],
                    }
                    detail_path = _detail_path(
                        out_dir, model_type, modality, method, horizon)
                    with open(detail_path, "w") as f:
                        json.dump(detail, f, indent=2, default=str)

                    if horizon == "base" and mirror_horizons:
                        base_row = row
                        base_detail = detail

                if mirror_horizons and base_row is not None:
                    for mh in mirror_horizons:
                        if not args.refresh:
                            cached = _load_existing_audit_row(
                                out_dir, model_type, modality, method, mh,
                            )
                            if cached is not None:
                                rows.append(cached)
                                n_reused_cells += 1
                                continue

                        mirrored_row = dict(base_row)
                        mirrored_row["horizon"] = mh
                        mirrored_row["elapsed_s"] = 0.0
                        rows.append(mirrored_row)
                        n_computed_cells += 1

                        if base_detail is not None:
                            mirrored_detail = json.loads(
                                json.dumps(base_detail, default=str))
                            mirrored_detail["horizon"] = mh
                            mirrored_detail["mirrored_from_horizon"] = "base"
                            for col, metric_dict in mirrored_detail.get(
                                    "metrics", {}).items():
                                if isinstance(metric_dict, dict):
                                    metric_dict["horizon"] = mh
                                    metric_dict["mirrored_from"] = "base"
                            fp = _detail_path(
                                out_dir, model_type, modality, method, mh)
                            with open(fp, "w") as f:
                                json.dump(mirrored_detail, f, indent=2,
                                          default=str)

    df = pd.DataFrame(rows).sort_values(
        ["modality", "model_type", "method", "horizon", "aggregate_score"],
        ascending=[True, True, True, True, False],
    )
    csv_path = out_dir / "co12_audit.csv"
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()