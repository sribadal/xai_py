"""
Clinical plausibility constraints for counterfactual generation.

Defines per-feature type constraints (sentinel-continuous, ordinal,
binary, nominal), z-score range bounds, empirical training-distribution
percentile tightening, treatment flag derivation, and the snap pipeline
that enforces all constraints on generated counterfactuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from xai_decision_making.explainability.counterfactual.feature_policy import (
    IMMUTABLE_FEATURES,
    partition_features,
)
from xai_decision_making.explainability.counterfactual._base import (
    get_categorical_values,
    get_feature_ranges,
    snap_categoricals,
)

DEFAULT_IMMUTABLE = sorted(IMMUTABLE_FEATURES)

THERPREPOST_VALID      = {0, 1, 2, 3, 4}
SYST_KUREN_REAL_COUNTS = set(range(1, 28)) | {31, 32, 35, 38}
SYST_AANPASSING_VALID  = {0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14}
CHIR_TYPE_HPB_VALID    = {21, 22, 23, 24, 25, 26, 27, 29, 98}
CHIR_RAD_VALID         = {0, 1, 2}
CHIR_TECH_VALID        = {0, 1, 2, 3, 4}
DESTR_TYPE_HPB_VALID   = {1, 3, 8}
RT_TYPE_HPB_VALID      = {1, 3, 7, 8, 9}
MAX_TREATMENT_LINES    = 12

_TREATMENT_DERIVATION_PAIRS = [
    ("had_surg", "ts_surg"),
    ("had_syst", "ts_syst"),
]

_TREATMENT_IMPULSE_ZERO = ["impulse_surg", "impulse_syst"]

FEATURE_CONSTRAINTS = {
    "ca199": {
        "type": "sentinel_continuous",
        "range": (0.0, 9000.0),
        "zscore_range": (-3.0, 3.0),
        "sentinels": {9000.0: "ceiling", 9999.0: "unknown"},
        "sentinel_policy": "flagged",
    },
    "bilirubine": {
        "type": "sentinel_continuous",
        "range": (0.0, 500.0),
        "zscore_range": (-3.0, 3.0),
        "sentinels": {9999.0: "unknown"},
        "sentinel_policy": "strict",
    },
    "kreatinine": {
        "type": "sentinel_continuous",
        "range": (0.0, 1500.0),
        "sentinels": {9999.0: "unknown"},
        "sentinel_policy": "strict",
    },
    "ldh": {
        "type": "sentinel_continuous",
        "range": (0.0, 5000.0),
        "zscore_range": (-3.0, 3.0),
        "sentinels": {9999.0: "unknown"},
        "sentinel_policy": "strict",
    },
    "trombocyten": {
        "type": "sentinel_continuous",
        "range": (0.0, 1500.0),
        "sentinels": {9999.0: "unknown"},
        "sentinel_policy": "strict",
    },
    "hb": {
        "type": "sentinel_continuous",
        "range": (0.0, 15.0),
        "sentinels": {99.9: "unknown"},
        "sentinel_policy": "strict",
    },
    "albumine": {
        "type": "sentinel_continuous",
        "range": (0.0, 60.0),
        "zscore_range": (-3.0, 3.0),
        "sentinels": {999.9: "unknown"},
        "sentinel_policy": "strict",
    },
    "neutrofielen": {
        "type": "sentinel_continuous",
        "range": (0.0, 50.0),
        "sentinels": {999.9: "unknown"},
        "sentinel_policy": "strict",
    },
    "crp": {
        "type": "sentinel_continuous",
        "range": (0.0, 500.0),
        "zscore_range": (-3.0, 3.0),
        "sentinels": {9999.9: "unknown"},
        "sentinel_policy": "strict",
    },
    "cea": {
        "type": "sentinel_continuous",
        "range": (0.0, 10000.0),
        "sentinels": {9999.9: "unknown"},
        "sentinel_policy": "strict",
    },
    "lengte": {
        "type": "sentinel_continuous",
        "range": (50.0, 230.0),
        "sentinels": {999.0: "unknown"},
        "sentinel_policy": "strict",
    },
    "diag_gewicht": {
        "type": "sentinel_continuous",
        "range": (20.0, 250.0),
        "sentinels": {999.0: "unknown"},
        "sentinel_policy": "strict",
    },
    "gebr_gewicht": {
        "type": "sentinel_continuous",
        "range": (20.0, 250.0),
        "sentinels": {999.0: "unknown"},
        "sentinel_policy": "strict",
    },
    "klin_tum_afm": {
        "type": "sentinel_continuous",
        "range": (0.0, 300.0),
        "sentinels": {999.0: "unknown"},
        "sentinel_policy": "strict",
    },
    "pa_tum_afm": {
        "type": "sentinel_continuous",
        "range": (0.0, 300.0),
        "sentinels": {999.0: "unknown"},
        "sentinel_policy": "strict",
    },
    "perf_stat": {
        "type": "sentinel_ordinal",
        "valid": {0, 1, 2, 3, 4},
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    },
    "asa": {
        "type": "sentinel_ordinal",
        "valid": {1, 2, 3, 4, 5, 6},
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    },
    "ct": {
        "type": "sentinel_ordinal",
        "valid": {1, 2, 3, 4},
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    },
    "cn": {
        "type": "sentinel_ordinal",
        "valid": {0, 1, 2, 3},
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    },
    "diffgrad": {
        "type": "sentinel_ordinal",
        "valid": {1, 2, 3, 4},
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    },
    "veneus_vaatbetr": {
        "type": "sentinel_ordinal",
        "valid": {0, 1, 2, 3, 4, 7},
        "sentinels": {8: "n/a", 9: "unknown"},
        "sentinel_policy": "strict",
    },
    "arterieel_vaatbetr": {
        "type": "sentinel_ordinal",
        "valid": {0, 1, 2, 3, 4, 7},
        "sentinels": {8: "n/a", 9: "unknown"},
        "sentinel_policy": "strict",
    },
    "cci_cat":   {"type": "ordinal", "valid": {0, 1, 2}},
    "stadium":   {"type": "ordinal", "valid": {1, 2, 3, 4, 5}},
    "cstadium":  {"type": "ordinal", "valid": {1, 2, 3, 4, 5, 6, 7, 8, 9}},
    "pstadium":  {"type": "ordinal", "valid": {1, 2, 3, 4, 5, 6, 7, 8, 9}},
    "leeft_cat": {"type": "ordinal", "valid": set(range(4, 22)) - {5}},
    "cci": {
        "type": "continuous",
        "range": (0.0, 30.0),
        "zscore_range": (-3.0, 3.0),
    },
    "cm":             {"type": "binary"},
    "pm":             {"type": "binary"},
    "meta_lever":     {"type": "binary"},
    "meta_bijnier":   {"type": "binary"},
    "meta_bot":       {"type": "binary"},
    "meta_hersenen":  {"type": "binary"},
    "meta_long":      {"type": "binary"},
    "meta_lymf":      {"type": "binary"},
    "meta_perit":     {"type": "binary"},
    "meta_overig":    {"type": "binary"},
    "meta_lok_aantal": {
        "type": "sentinel_ordinal",
        "valid": {1, 2, 3, 4, 5, 6, 7},
        "sentinels": {77: "more_than_3_locations"},
        "sentinel_policy": "strict",
    },
    "ond_lymf": {
        "type": "sentinel_ordinal",
        "valid": set(range(0, 100)),
        "sentinels": {98: "unknown_count", 99: "unknown_any"},
        "sentinel_policy": "strict",
    },
    "pos_lymf": {
        "type": "sentinel_ordinal",
        "valid": set(range(0, 100)),
        "sentinels": {98: "unknown_count", 99: "unknown_any"},
        "sentinel_policy": "strict",
    },
    "gesl":       {"type": "nominal", "valid": {1, 2}},
    "seizoen":    {"type": "nominal", "valid": {1, 2, 3, 4}},
    "diag_basis": {"type": "nominal", "valid": {0, 1, 2, 4, 5, 6, 7}},
    "morf_cat":   {"type": "nominal", "valid": {10, 11, 12, 13, 14}},
    "vit_stat":   {"type": "binary"},
    "rook_stat": {
        "type": "sentinel_ordinal",
        "valid": {1, 2, 3, 4},
        "sentinels": {7: "not_in_epd", 9: "unknown"},
        "sentinel_policy": "strict",
    },
    "alcohol_stat": {
        "type": "sentinel_ordinal",
        "valid": {11, 12, 22, 33, 44},
        "sentinels": {77: "not_in_epd", 99: "unknown"},
        "sentinel_policy": "strict",
    },
    "ycm": {"type": "binary"},
    "tumgericht_ther":  {"type": "binary"},
    "expl_chir":        {"type": "binary"},
    "chir":             {"type": "binary"},
    "destr":            {"type": "binary"},
    "rt":               {"type": "binary"},
    "meta_rt":          {"type": "binary"},
    "meta_chir":        {"type": "binary"},
    "lok_chemo_immuno": {"type": "binary"},
    "chemo":            {"type": "binary"},
    "target":           {"type": "binary"},
    "geen_ther_reden": {
        "type": "sentinel_ordinal",
        "valid": {11, 12, 13, 14},
        "sentinels": {8: "anders", 9: "unknown"},
        "sentinel_policy": "strict",
    },
    "geen_ther_type": {
        "type": "nominal",
        "valid": {1, 2, 3, 4, 5},
    },
    "destr_type1": {"type": "nominal", "valid": DESTR_TYPE_HPB_VALID},
    "rt_type1":    {"type": "nominal", "valid": RT_TYPE_HPB_VALID},
    "syst_compl1": {
        "type": "sentinel_ordinal",
        "valid": {1, 2, 3},
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    },
    "syst_compl2": {
        "type": "sentinel_ordinal",
        "valid": {1, 2, 3},
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    },
    "fup_diagnostiek_reden": {
        "type": "sentinel_ordinal",
        "valid": {1, 2, 3, 4, 7},
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    },
    "fup_event_type": {
        "type": "sentinel_ordinal",
        "valid": {1, 2, 3},
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    },
    "rfs_event": {"type": "nominal", "valid": {0, 1, 2}},
    "pfs_event": {"type": "nominal", "valid": {0, 1, 2}},
    "rfs_1jaar_afgerond": {"type": "binary"},
    "rfs_3jaar_afgerond": {"type": "binary"},
    "rfs_5jaar_afgerond": {"type": "binary"},
    "pfs_1jaar_afgerond": {"type": "binary"},
    "pfs_3jaar_afgerond": {"type": "binary"},
    "pfs_5jaar_afgerond": {"type": "binary"},
    "ts_surg":      {"type": "continuous", "range": (0.0, 1.0)},
    "ts_syst":      {"type": "continuous", "range": (0.0, 1.0)},
    "had_surg":     {"type": "binary"},
    "had_syst":     {"type": "binary"},
    "impulse_surg": {"type": "binary"},
    "impulse_syst": {"type": "binary"},
}

for _i in range(1, MAX_TREATMENT_LINES + 1):
    FEATURE_CONSTRAINTS[f"chir_type{_i}"] = {
        "type": "nominal",
        "valid": CHIR_TYPE_HPB_VALID,
    }
    FEATURE_CONSTRAINTS[f"chir_rad{_i}"] = {
        "type": "sentinel_ordinal",
        "valid": CHIR_RAD_VALID,
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    }
    FEATURE_CONSTRAINTS[f"chir_tech{_i}"] = {
        "type": "sentinel_ordinal",
        "valid": CHIR_TECH_VALID,
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    }
    FEATURE_CONSTRAINTS[f"syst_prepost{_i}"] = {
        "type": "nominal",
        "valid": THERPREPOST_VALID,
    }
    FEATURE_CONSTRAINTS[f"syst_chemort{_i}"] = {
        "type": "nominal",
        "valid": THERPREPOST_VALID,
    }
    FEATURE_CONSTRAINTS[f"syst_kuren{_i}"] = {
        "type": "sentinel_ordinal",
        "valid": SYST_KUREN_REAL_COUNTS,
        "sentinels": {
            66: "sensitizer",
            77: "maintenance",
            98: "ongoing",
            99: "unknown",
        },
        "sentinel_policy": "strict",
    }
    FEATURE_CONSTRAINTS[f"syst_aanpassing{_i}"] = {
        "type": "sentinel_ordinal",
        "valid": SYST_AANPASSING_VALID,
        "sentinels": {9: "unknown"},
        "sentinel_policy": "strict",
    }
del _i


def compute_empirical_zscore_ranges(adapter, constraints, *, lower_percentile=1.0,
                                     upper_percentile=99.0):
    try:
        x_train = adapter.training_data()
    except Exception:
        return {}

    if x_train is None or len(x_train) == 0:
        return {}

    inverse_fn = getattr(adapter, "inverse_transform_numeric", None)
    feat_names = list(adapter.feature_names())
    out = {}

    if inverse_fn is None:
        return {}
    try:
        raw_train = inverse_fn(x_train[feat_names])
    except Exception:
        return {}

    for feat in feat_names:
        rule = constraints.get(feat)
        if rule is None:
            continue
        if rule.get("type") not in ("continuous", "sentinel_continuous"):
            continue
        zr = rule.get("zscore_range")
        if zr is None:
            continue
        if feat not in x_train.columns or feat not in raw_train.columns:
            continue

        z_vals = pd.to_numeric(x_train[feat], errors="coerce")
        raw_vals = pd.to_numeric(raw_train[feat], errors="coerce")

        real = z_vals.notna() & raw_vals.notna()

        sentinels = rule.get("sentinels", {}) or {}
        for s in sentinels:
            real &= ~np.isclose(raw_vals, float(s), rtol=1e-4, atol=1e-3)

        clin = rule.get("range")
        if clin is not None:
            lo_c, hi_c = clin
            real &= (raw_vals >= lo_c) & (raw_vals <= hi_c)

        survivors = z_vals[real].values
        if len(survivors) < 10:
            continue

        emp_lo = float(np.percentile(survivors, lower_percentile))
        emp_hi = float(np.percentile(survivors, upper_percentile))

        z_lo, z_hi = zr
        emp_lo = max(emp_lo, float(z_lo))
        emp_hi = min(emp_hi, float(z_hi))

        if emp_hi <= emp_lo:
            continue

        out[feat] = (emp_lo, emp_hi)

    return out


@dataclass
class PlausibilityConfig:
    feature_names: list
    ranges: dict
    immutable: list
    continuous: list
    categorical_values: dict
    unknown: list
    constraints: dict = field(default_factory=dict)
    empirical_zscore_ranges: dict = field(default_factory=dict)

    def cf_bounds_for(self, feat):
        emp = self.empirical_zscore_ranges.get(feat)
        if emp is not None:
            return emp

        rule = self.constraints.get(feat)
        if rule and rule.get("type") in ("continuous", "sentinel_continuous"):
            zr = rule.get("zscore_range")
            if zr is not None:
                return tuple(zr)
            return tuple(rule["range"])

        if feat in self.ranges:
            return tuple(self.ranges[feat])

        return None


def build_plausibility_config(adapter, *, extra_immutable=None, feature_ranges=None):
    feat_names = list(adapter.feature_names())
    ranges = feature_ranges or get_feature_ranges(adapter)
    feeds_zscored = bool(getattr(adapter, "feeds_zscored", False))

    partition = partition_features(feat_names, ranges)

    cat_values = {}
    for feat, vals in partition["categorical_values"].items():
        cat_values[feat] = np.asarray(vals, dtype=np.float64)

    adapter_cats = get_categorical_values(adapter) or {}
    for feat, vals in adapter_cats.items():
        if feat in feat_names and feat not in cat_values:
            cat_values[feat] = np.asarray(vals, dtype=np.float64)

    merged_immutable = set(partition["immutable"]) | set(extra_immutable or [])
    immutable = [f for f in merged_immutable if f in feat_names]

    constraints = {}
    for f in feat_names:
        if f not in FEATURE_CONSTRAINTS:
            continue
        rule = FEATURE_CONSTRAINTS[f]
        if not feeds_zscored and "zscore_range" in rule:
            rule = {k: v for k, v in rule.items() if k != "zscore_range"}
        constraints[f] = rule

    empirical_zscore_ranges = (
        compute_empirical_zscore_ranges(adapter, constraints)
        if feeds_zscored
        else {}
    )

    return PlausibilityConfig(
        feature_names=feat_names,
        ranges=ranges,
        immutable=immutable,
        continuous=partition["continuous"],
        categorical_values=cat_values,
        unknown=partition["unknown"],
        constraints=constraints,
        empirical_zscore_ranges=empirical_zscore_ranges,
    )


@dataclass
class ConstraintViolation:
    feature: str
    kind: str
    original: float
    snapped: float
    detail: str = ""


def derive_treatment_started_flags(cf, feat_names, target_t_norm):
    cf = np.asarray(cf, dtype=np.float64).reshape(-1).copy()
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    violations = []

    for started_name, time_name in _TREATMENT_DERIVATION_PAIRS:
        i_s = name_to_idx.get(started_name)
        i_t = name_to_idx.get(time_name)
        if i_s is None or i_t is None:
            continue

        time_val = float(cf[i_t])
        new_flag = 1.0 if time_val > 0.0 else 0.0
        old_flag = float(cf[i_s])

        if abs(new_flag - old_flag) > 1e-9:
            violations.append(ConstraintViolation(
                started_name,
                "treatment_flag_derived",
                old_flag,
                new_flag,
                detail=(
                    f"recomputed from {time_name}={time_val:.4f} at "
                    f"target_t_norm={target_t_norm:.4f}"
                ),
            ))
            cf[i_s] = new_flag

    for impulse_name in _TREATMENT_IMPULSE_ZERO:
        i = name_to_idx.get(impulse_name)
        if i is None:
            continue
        old_val = float(cf[i])
        if abs(old_val) > 1e-9:
            violations.append(ConstraintViolation(
                impulse_name,
                "treatment_flag_derived",
                old_val,
                0.0,
                detail="impulse flags default to 0 in CF (no edge detection)",
            ))
            cf[i] = 0.0

    return cf, violations


def _snap_one(feat, value, query_value, rule):
    t = rule["type"]

    if t == "binary":
        snapped = 1.0 if value >= 0.5 else 0.0
        if snapped != value:
            return snapped, ConstraintViolation(feat, "binary_round", value, snapped), False
        return snapped, None, False

    if t == "ordinal":
        valid = rule.get("valid")
        if valid is not None:
            valid_arr = np.asarray(sorted(valid), dtype=np.float64)
            snapped = float(valid_arr[np.argmin(np.abs(valid_arr - value))])
        else:
            lo, hi = rule["range"]
            snapped = float(np.clip(round(value), lo, hi))
        if snapped != value:
            return snapped, ConstraintViolation(feat, "ordinal_snap", value, snapped), False
        return snapped, None, False

    if t == "integer":
        lo, hi = rule["range"]
        snapped = float(np.clip(round(value), lo, hi))
        if snapped != value:
            return snapped, ConstraintViolation(feat, "integer_snap", value, snapped), False
        return snapped, None, False

    if t == "nominal":
        valid_arr = np.asarray(sorted(rule["valid"]), dtype=np.float64)
        snapped = float(valid_arr[np.argmin(np.abs(valid_arr - value))])
        if snapped != value:
            return snapped, ConstraintViolation(
                feat, "nominal_snap", value, snapped,
                detail="nearest code; semantic distance may not be small",
            ), False
        return snapped, None, False

    if t == "continuous":
        zr = rule.get("zscore_range")
        if zr is not None:
            z_lo, z_hi = zr
            if value < z_lo or value > z_hi:
                snapped = float(np.clip(value, z_lo, z_hi))
                return snapped, ConstraintViolation(
                    feat, "zscore_out_of_range", value, snapped,
                    detail=f"z-scored value {value:.2f} clipped to [{z_lo}, {z_hi}]",
                ), True
            return value, None, False

        lo, hi = rule["range"]
        if value < lo or value > hi:
            snapped = float(np.clip(value, lo, hi))
            return snapped, ConstraintViolation(
                feat, "continuous_out_of_range", value, snapped,
                detail=f"value {value:.2f} clipped to clinical range [{lo}, {hi}]",
            ), True
        return value, None, False

    if t in ("sentinel_continuous", "sentinel_ordinal"):
        sentinels = rule["sentinels"]
        policy = rule.get("sentinel_policy", "flagged")
        query_is_sentinel = float(query_value) in sentinels

        for s in sentinels:
            if abs(value - s) < 1e-6:
                return float(s), None, False

        if t == "sentinel_continuous":
            zr = rule.get("zscore_range")
            if zr is not None:
                z_lo, z_hi = zr
                if value < z_lo or value > z_hi:
                    snapped = float(np.clip(value, z_lo, z_hi))
                    if query_is_sentinel and policy == "strict":
                        return float(query_value), ConstraintViolation(
                            feat, "sentinel_conversion_rejected", query_value, query_value,
                            detail=(
                                f"CF moved out-of-band z-score from sentinel "
                                f"{sentinels[float(query_value)]}; reverted"
                            ),
                        ), False
                    return snapped, ConstraintViolation(
                        feat, "zscore_out_of_range", value, snapped,
                        detail=f"z-scored value {value:.2f} clipped to [{z_lo}, {z_hi}]",
                    ), True
                snapped = value
            else:
                lo, hi = rule["range"]
                in_range = lo <= value <= hi
                if not in_range:
                    snapped = float(np.clip(value, lo, hi))
                    return snapped, ConstraintViolation(
                        feat, "continuous_out_of_range", value, snapped,
                        detail=(
                            f"value {value:.2f} clipped to clinical range "
                            f"[{lo}, {hi}] (not a sentinel)"
                        ),
                    ), True
                snapped = value
        else:
            valid_arr = np.asarray(sorted(rule["valid"]), dtype=np.float64)
            snapped = float(valid_arr[np.argmin(np.abs(valid_arr - value))])

        if query_is_sentinel:
            if policy == "strict":
                return float(query_value), ConstraintViolation(
                    feat, "sentinel_conversion_rejected", query_value, query_value,
                    detail=(
                        f"CF tried to convert {sentinels[float(query_value)]} "
                        f"sentinel to {value:.2f}; reverted"
                    ),
                ), False
            if policy == "flagged":
                return snapped, ConstraintViolation(
                    feat, "sentinel_conversion", query_value, snapped,
                    detail=(
                        f"CF converted {sentinels[float(query_value)]} → "
                        f"measured value {snapped:.2f}"
                    ),
                ), False
            return snapped, None, False

        if snapped != value:
            return snapped, ConstraintViolation(feat, "ordinal_snap", value, snapped), False
        return snapped, None, False

    return value, None, False


def snap_with_constraints(cf, query, feat_names, constraints, *, target_t_norm=None):
    cf = np.asarray(cf, dtype=np.float64).reshape(-1).copy()
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    violations = []
    rejected = False

    for i, feat in enumerate(feat_names):
        rule = constraints.get(feat)
        if rule is None:
            continue
        snapped, vio, was_rejected = _snap_one(feat, float(cf[i]), float(query[i]), rule)
        cf[i] = snapped
        if vio is not None:
            violations.append(vio)
        if was_rejected:
            rejected = True

    if target_t_norm is not None:
        cf, derived_vios = derive_treatment_started_flags(
            cf, feat_names, float(target_t_norm),
        )
        violations.extend(derived_vios)

    return cf, violations, rejected


@dataclass
class SnapReport:
    pre_snap_pred: float
    post_snap_pred: float
    snap_shift: float
    target: float
    crossed_pre: bool
    crossed_post: bool
    violations: list = field(default_factory=list)
    rejected: bool = False


def snap_and_verify(cf_raw, feat_names, categorical_values, predict_fn, *,
                    target=None, higher_is_better=False, query=None,
                    constraints=None, target_t_norm=None):
    cf_raw = np.asarray(cf_raw, dtype=np.float64).reshape(-1)

    pre_pred = float(predict_fn(cf_raw.reshape(1, -1))[0])

    cf_df = pd.DataFrame([cf_raw], columns=feat_names)
    cf_df = snap_categoricals(cf_df, categorical_values)

    violations = []
    rejected = False

    if constraints and query is not None:
        cf_step, violations, rejected = snap_with_constraints(
            cf_df.values[0], query, feat_names, constraints,
            target_t_norm=target_t_norm,
        )
        cf_df = pd.DataFrame([cf_step], columns=feat_names)

    post_pred = float(predict_fn(cf_df.values.astype(np.float64))[0])

    def _crossed(pred):
        if target is None:
            return True
        if higher_is_better:
            return pred >= target
        return pred <= target

    return cf_df, SnapReport(
        pre_snap_pred=pre_pred,
        post_snap_pred=post_pred,
        snap_shift=abs(post_pred - pre_pred),
        target=target,
        crossed_pre=_crossed(pre_pred),
        crossed_post=_crossed(post_pred),
        violations=violations,
        rejected=rejected,
    )


def format_snap_report(report, method_label="CF"):
    target_str = f"{report.target:.4f}" if report.target is not None else "n/a"
    base = (
        f"  [{method_label}] snap: "
        f"pre={report.pre_snap_pred:.4f} → post={report.post_snap_pred:.4f} "
        f"(shift={report.snap_shift:.4f}, target={target_str}, "
        f"crossed pre/post={report.crossed_pre}/{report.crossed_post})"
    )
    if report.rejected:
        base += " [REJECTED]"
    if report.violations:
        kinds = {}
        for v in report.violations:
            kinds[v.kind] = kinds.get(v.kind, 0) + 1
        summary = ", ".join(f"{k}={n}" for k, n in kinds.items())
        base += f"  violations: {summary}"
    return base


def condor_target_t_norm(adapter, time_horizon):
    if time_horizon is None:
        return None
    if "ts_surg" not in adapter.feature_names():
        return None
    return float(time_horizon) / 60.0