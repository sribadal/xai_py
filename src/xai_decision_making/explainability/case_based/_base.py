"""
Shared utilities for case-based explainers.

Provides distance functions (Gower, Euclidean, cosine), outcome and
treatment enrichment for retrieved cases, sanity check runner, and
confidence metric computation from retrieval distances.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    CaseBasedExplainer,
    SanityCheckResult,
)
from xai_decision_making.explainability.sanity_checks import CASE_BASED_CHECKS, SanityCheck

def _column_ranges(df, cols):
    return {c: max(df[c].max() - df[c].min(), 1.0) for c in cols}

def _gower_distance_row(a, b, numeric_cols, categorical_cols, col_ranges):
    scores = []
    for c in numeric_cols:
        if c in a.columns and c in b.columns:
            scores.append(
                abs(float(a[c].iloc[0]) - float(b[c].iloc[0]))
                / col_ranges.get(c, 1.0)
            )
    for c in categorical_cols:
        if c in a.columns and c in b.columns:
            scores.append(0.0 if a[c].iloc[0] == b[c].iloc[0] else 1.0)
    return float(np.mean(scores)) if scores else 0.0

def _euclidean_distance_row(a, b, cols, col_ranges):
    diffs = []
    for c in cols:
        if c in a.columns and c in b.columns:
            diffs.append(
                (float(a[c].iloc[0]) - float(b[c].iloc[0]))
                / col_ranges.get(c, 1.0)
            )
    return float(np.sqrt(np.sum(np.square(diffs)))) if diffs else 0.0

def _cosine_distance_row(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)

CASE_OUTCOME_COLS = [
    "vit_stat",
    "vit_stat_int",
    "rfs_event",
    "rfs_int",
    "pfs_event",
    "pfs_int",
    "dead_with_6months",
    "qol_summary_event",
    "qol_summary_time",
    "qol_physical_event",
    "qol_physical_time",
    "qol_emotional_event",
    "qol_emotional_time",
    "qol_social_event",
    "qol_social_time",
    "qol_cognitive_event",
    "qol_cognitive_time",
    "qol_role_event",
    "qol_role_time",
]

CASE_TREATMENT_COLS = [
    "tumgericht_ther",
    "geen_ther_reden",
    "geen_ther_type",
    "chir",
    "chir_type1",
    "chir_tech1",
    "chir_rad1",
    "chir_int1",
    "rt",
    "rt_type1",
    "rt_start_int1",
    "chemo",
    "target",
    "syst_code1",
    "syst_schemanum1",
    "syst_start_int1",
    "syst_kuren1",
]

CASE_CONTEXT_COLS = CASE_OUTCOME_COLS + CASE_TREATMENT_COLS

def enrich_with_context(result, adapter, train_row_indices):
    if not hasattr(adapter, "training_data_raw"):
        return result

    try:
        x_raw = adapter.training_data_raw()
    except Exception:
        return result

    if x_raw is None or len(x_raw) == 0:
        return result

    cols = [c for c in CASE_CONTEXT_COLS if c in x_raw.columns]
    if not cols:
        return result

    indices = list(train_row_indices)
    enrichment = x_raw.iloc[indices][cols].reset_index(drop=True)

    for c in cols:
        result[c] = enrichment[c].values

    return result

def _run_sanity_checks(explainer, held_out_df, checks=None):
    if checks is None:
        checks = CASE_BASED_CHECKS
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

def compute_case_confidence(content):
    distances = []

    if isinstance(content, pd.DataFrame):
        if "_distance" in content.columns:
            for v in content["_distance"].tolist():
                try:
                    df = float(v)
                    if not (np.isnan(df) or np.isinf(df)):
                        distances.append(df)
                except (TypeError, ValueError):
                    continue
    elif isinstance(content, list):
        for case in content:
            if isinstance(case, dict) and "_distance" in case:
                try:
                    df = float(case["_distance"])
                    if not (np.isnan(df) or np.isinf(df)):
                        distances.append(df)
                except (TypeError, ValueError):
                    continue
    elif isinstance(content, dict):
        protos = content.get("similar_prototypes")
        if isinstance(protos, list):
            for case in protos:
                if isinstance(case, dict) and "_distance" in case:
                    try:
                        df = float(case["_distance"])
                        if not (np.isnan(df) or np.isinf(df)):
                            distances.append(df)
                    except (TypeError, ValueError):
                        continue

    if not distances:
        return None

    return {
        "median_distance": round(float(np.median(distances)), 4),
        "max_distance": round(float(np.max(distances)), 4),
        "min_distance": round(float(np.min(distances)), 4),
        "n_cases": len(distances),
        "source": "tier_b_native",
    }