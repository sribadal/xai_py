"""
Shared utilities for counterfactual explainers.

Provides risk prediction wrappers, feature range/categorical detection,
immutable enforcement, outcome flipping, content delta computation,
confidence metrics, biological direction priors, and the direction
filter applied post-hoc to all CF methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    CounterfactualExplainer,
    ExplainerModality,
)
from xai_decision_making.explainability.sanity_checks import (
    CounterfactualConstraintCheck,
    CounterfactualProximityCheck,
    CounterfactualSuccessRateCheck,
)

COUNTERFACTUAL_CHECKS = [
    CounterfactualSuccessRateCheck(),
    CounterfactualProximityCheck(),
    CounterfactualConstraintCheck(),
]

DEFAULT_IMMUTABLE = [
    # Coding systems (cannot be modified)
    "morf", "morf_cat", "topo_sublok", "diag_basis", "seizoen",
    # Demographics (not actionable)
    "gesl", "leeft_cat", "incjr", "geboorteland",
    # Identifiers
    "key_nkr", "key_eid",
    # Disease stage (derived clinical summary, not independently modifiable)
    "stadium", "cstadium", "pstadium",
]

ORDINAL_BIN_FEATURES = [
    # Staging (TNM 7/8 ordinal codes)
    "cstadium", "stadium", "pstadium",
    "ct", "cn", "cm", "pt", "pn", "pm",
    "yct", "ycn", "ycm",
    # Performance scoring (ordinal scales)
    "perf_stat", "asa", "diffgrad",
    # Comorbidity bins
    "cci_cat",
]

def _run_sanity_checks(explainer, held_out_df, checks=None):
    checks = checks or COUNTERFACTUAL_CHECKS
    results = []

    for check in checks:
        result = check.run(explainer, held_out_df)
        result.modality = ExplainerModality.COUNTERFACTUAL
        results.append(result)

    return AggregatedSanityResult(
        modality=ExplainerModality.COUNTERFACTUAL,
        target=explainer._adapter.target,
        explainer_name=type(explainer).__name__,
        check_results=results,
    )


def risk_predict_fn(adapter, time_horizon=None):
    raw = getattr(adapter, "raw_model", None)
    feat_names = adapter.feature_names()

    if time_horizon is not None:
        def _predict_at_t(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            df = pd.DataFrame(X, columns=feat_names)
            return adapter.predict_at_horizon(df, time_horizon)
        return _predict_at_t

    def _predict_batch(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return raw.predict(X.astype(np.float64))

    def _predict_fallback(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        df = pd.DataFrame(X, columns=feat_names)
        preds = []
        for i in range(len(df)):
            out = adapter.predict(df.iloc[[i]])
            if isinstance(out, dict):
                preds.append(out.get("risk_score", 0.0))
            else:
                preds.append(float(np.asarray(out).ravel()[0]))
        return np.array(preds)

    if raw is not None and hasattr(raw, "predict"):
        try:
            test = np.zeros((2, len(feat_names)), dtype=np.float64)
            out = raw.predict(test)
            if isinstance(out, np.ndarray) and out.shape == (2,):
                return _predict_batch
        except Exception:
            pass

    return _predict_fallback


def get_feature_ranges(adapter):
    x_train = adapter.training_data()
    numeric = x_train.select_dtypes(include=[np.number])

    return {
        c: (float(numeric[c].min()), float(numeric[c].max()))
        for c in numeric.columns
    }


def get_categorical_values(adapter, max_unique=20):
    x_train = adapter.training_data()
    feat_names = adapter.feature_names()
    categoricals = {}

    for col in feat_names:
        if col not in x_train.columns:
            continue
        series = x_train[col].dropna()
        unique = series.unique()
        if len(unique) <= max_unique:
            categoricals[col] = np.sort(unique.astype(np.float64))

    return categoricals


def snap_categoricals(cf_df, categorical_values):
    if not categorical_values:
        return cf_df

    cf = cf_df.copy()
    for col, valid_values in categorical_values.items():
        if col not in cf.columns or len(valid_values) == 0:
            continue
        raw = cf[col].values[0]
        if pd.isna(raw):
            continue
        idx = np.argmin(np.abs(valid_values - float(raw)))
        cf[col] = valid_values[idx]

    return cf

def clip_to_ranges(cf_df, ranges):
    cf = cf_df.copy()
    for col, (lo, hi) in ranges.items():
        if col in cf.columns:
            cf[col] = cf[col].clip(lo, hi)
    return cf


def enforce_immutables(original, counterfactual, immutable_features):
    cf = counterfactual.copy()
    for feat in immutable_features:
        if feat in cf.columns and feat in original.columns:
            cf[feat] = original[feat].values[0]
    return cf


def outcome_flipped(
    adapter,
    original,
    counterfactual,
    *,
    threshold=None,
    direction=None,
):
    orig_pred = adapter.predict(original)
    cf_pred = adapter.predict(counterfactual)
    orig_risk = (
        orig_pred["risk_score"]
        if isinstance(orig_pred, dict)
        else float(orig_pred)
    )
    cf_risk = (
        cf_pred["risk_score"]
        if isinstance(cf_pred, dict)
        else float(cf_pred)
    )

    if direction is None:
        hib = getattr(adapter, "higher_is_better", False)
        direction = "higher" if hib else "lower"

    if threshold is not None:
        if direction == "lower":
            return orig_risk >= threshold and cf_risk < threshold
        return orig_risk <= threshold and cf_risk > threshold

    if direction == "lower":
        return cf_risk < orig_risk
    return cf_risk > orig_risk

def extract_scalar_prediction(pred):
    if isinstance(pred, dict):
        if "risk_score" in pred:
            return float(pred["risk_score"])
        if "trajectory" in pred:
            traj = pred["trajectory"]
            if hasattr(traj, '__len__') and len(traj) > 0:
                return float(traj[-1])
        for v in pred.values():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return float(np.asarray(pred).ravel()[0])

def compute_content_deltas(adapter, query_df, cf_df, feat_names):
    inverse_fn = getattr(adapter, "inverse_transform_numeric", None)

    if inverse_fn is not None:
        raw_query_df = inverse_fn(query_df[feat_names])
        raw_cf_df = inverse_fn(cf_df[feat_names])

        diffs = (raw_cf_df.iloc[0] - raw_query_df.iloc[0]).to_dict()
        content = {k: v for k, v in diffs.items() if abs(v) > 1e-8}

        return content, raw_query_df.iloc[0].to_dict(), raw_cf_df.iloc[0].to_dict()

    diffs = (cf_df[feat_names].iloc[0] - query_df[feat_names].iloc[0]).to_dict()
    content = {k: v for k, v in diffs.items() if abs(v) > 1e-8}

    return content, None, None


def desired_direction(adapter, current_pred, median_pred):
    hib = getattr(adapter, "higher_is_better", False)
    return not hib

def compute_cf_confidence(
    raw_query,
    raw_cf,
    orig_pred,
    cf_pred,
    *,
    higher_is_better,
    sparsity_eps: float = 1e-6,
):
    if not raw_query or not raw_cf:
        return None

    deltas = {}
    for feat, q_val in raw_query.items():
        if feat not in raw_cf:
            continue
        try:
            qf, cf_f = float(q_val), float(raw_cf[feat])
        except (TypeError, ValueError):
            continue
        deltas[feat] = abs(cf_f - qf)

    if not deltas:
        return None

    proximity_l1_raw = float(sum(deltas.values()))
    sparsity = int(sum(1 for d in deltas.values() if d > sparsity_eps))

    flipped = None
    if orig_pred is not None and cf_pred is not None:
        try:
            of, cfp = float(orig_pred), float(cf_pred)
            flipped = bool(
                (higher_is_better and cfp > of)
                or ((not higher_is_better) and cfp < of)
            )
        except (TypeError, ValueError):
            pass

    return {
        "proximity_l1_raw":    round(proximity_l1_raw, 4),
        "sparsity":            sparsity,
        "n_features_compared": len(deltas),
        "flipped":             flipped,
        "sparsity_eps":        sparsity_eps,
        "normalisation":       "none_raw_units",
        "source":              "tier_b_native",
    }

BIOLOGICAL_DIRECTION_PRIORS = {
    "ca199":      "decreasing",
    "bilirubine": "decreasing",
    "ldh":        "decreasing",
    "crp":        "decreasing",
    "ts_surg":    "increasing",
    "ts_syst":    "increasing",
}


def resolve_direction_priors(adapter, override=None):
    adapter_feats = set(adapter.feature_names())
    priors = dict(BIOLOGICAL_DIRECTION_PRIORS) if override is None else dict(override)
    return {f: d for f, d in priors.items() if f in adapter_feats}


def apply_direction_priors(cf_values, query_values, feat_names, priors):
    cf_corrected = cf_values.copy()
    n_reset = 0
    for i, feat in enumerate(feat_names):
        if feat not in priors:
            continue
        delta = cf_corrected[i] - query_values[i]
        if abs(delta) < 1e-8:
            continue
        direction = priors[feat]
        if direction == "decreasing" and delta > 0:
            cf_corrected[i] = query_values[i]
            n_reset += 1
        elif direction == "increasing" and delta < 0:
            cf_corrected[i] = query_values[i]
            n_reset += 1
    return cf_corrected, n_reset


def direction_compatible_mask(candidates, query, feat_names, priors):
    import numpy as np
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    mask = np.ones(len(candidates), dtype=bool)
    for feat, direction in priors.items():
        i = name_to_idx.get(feat)
        if i is None:
            continue
        q = query[i]
        col = candidates[:, i]
        if direction == "decreasing":
            mask &= col <= q + 1e-8
        elif direction == "increasing":
            mask &= col >= q - 1e-8
    return mask