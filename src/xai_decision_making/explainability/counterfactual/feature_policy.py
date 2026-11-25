"""
Feature policy registry for counterfactual explainers.

Classifies every NKR/PACAP feature as immutable, binary, categorical,
ordinal integer, or continuous. Used by build_plausibility_config to
determine which features CF methods may perturb and how generated
values should be snapped.
"""
from __future__ import annotations

from enum import Enum

class FeaturePolicy(str, Enum):
    IMMUTABLE = "immutable"
    BINARY = "binary"
    CATEGORICAL_CODED = "categorical_coded"
    ORDINAL_INTEGER = "ordinal_integer"
    CONTINUOUS = "continuous"

_ENCODED_IMMUTABLE_PREFIXES = ("leeft_cat_", "gesl_", "seizoen_")

IMMUTABLE_FEATURES = {
    "key_nkr", "key_eid",
    "gesl", "leeft_cat", "incjr", "geboorteland", "seizoen",
    "lengte", "gebr_gewicht",
    "cci", "cci_cat",
    "rook_stat", "alcohol_stat",
    "morf", "morf_cat", "topo_sublok", "diag_basis",
    "vit_stat", "vit_stat_int",
    "rfs_event", "rfs_int", "pfs_event", "pfs_int",
    "fup_event_type", "fup_event_int",
    "rfs_1jaar_afgerond", "rfs_3jaar_afgerond", "rfs_5jaar_afgerond",
    "pfs_1jaar_afgerond", "pfs_3jaar_afgerond", "pfs_5jaar_afgerond",
    "respons_uitslag", "respons_int",
    "dead_with_6months", "survival_3m", "risk_score",
    "mortality_risk_3_months", "mortality_risk_6_months",
    "mortality_risk_12_months", "survival_curve",
    "in_pacap", "pretreatment_qol", "pretreatment_date",
    "followup_qol_3m", "followup_date_3m",
    "qlq_diff_3m", "qlq_diff_category",
    "qol_summary_event", "qol_summary_time", "qol_summary_pretreatment",
    "qol_physical_event", "qol_physical_time", "qol_physical_pretreatment",
    "qol_emotional_event", "qol_emotional_time", "qol_emotional_pretreatment",
    "qol_social_event", "qol_social_time", "qol_social_pretreatment",
    "qol_cognitive_event", "qol_cognitive_time", "qol_cognitive_pretreatment",
    "qol_role_event", "qol_role_time", "qol_role_pretreatment",
    "diffgrad", "klin_tum_afm",
    "ct", "cn", "cm", "pt", "pn", "pm",
    "cstadium", "pstadium", "stadium",
    "veneus_vaatbetr", "arterieel_vaatbetr",
    "meta_lever", "meta_bijnier", "meta_bot", "meta_hersenen",
    "meta_long", "meta_lymf", "meta_perit", "meta_overig",
    "meta_lok_aantal", "perf_stat",
}

BINARY_FEATURES = {
    "meta_bijnier", "meta_bot", "meta_hersenen", "meta_lever",
    "meta_long", "meta_lymf", "meta_perit", "meta_overig",
    "tumgericht_ther", "expl_chir", "chir", "destr",
    "meta_rt", "meta_chir", "lok_chemo_immuno",
    "had_surg", "had_syst", "impulse_surg", "impulse_syst",
}

CATEGORICAL_CODED_FEATURES = {
    "perf_stat": [0, 1, 2, 3, 4, 9],
    "asa": [1, 2, 3, 4, 5, 6, 9],
    "diffgrad": [1, 2, 3, 4, 9],
    "ct": [0, 1, 2, 3, 4, 9],
    "cn": [0, 1, 2, 3, 9],
    "cm": [0, 1, 9],
    "pt": [0, 1, 2, 3, 4, 9],
    "pn": [0, 1, 2, 3, 9],
    "pm": [0, 1, 9],
    "yct": [0, 1, 2, 3, 4, 9],
    "ycn": [0, 1, 2, 3, 9],
    "ycm": [0, 1, 9],
    "cstadium": [1, 2, 3, 4, 9],
    "pstadium": [1, 2, 3, 4, 9],
    "stadium": [1, 2, 3, 4, 9],
    "veneus_vaatbetr": [0, 1, 2, 3, 4, 8, 9],
    "arterieel_vaatbetr": [0, 1, 2, 3, 4, 8, 9],
    "geen_ther_reden": [8, 9, 11, 12, 13, 14],
    "geen_ther_type": [1, 2, 3, 4, 5],
    "chir_tech1": [0, 1, 2, 3, 4, 9],
    "chir_tech2": [0, 1, 2, 3, 4, 9],
    "chir_rad1": [0, 1, 2, 9],
    "chir_rad2": [0, 1, 2, 9],
    "chir_type1": [21, 22, 23, 24, 25, 26, 27, 29, 98],
    "chir_type2": [21, 22, 23, 24, 25, 26, 27, 29, 98],
    "destr_type1": [1, 3, 8],
    "rt": [0, 1, 2, 4],
    "rt_type1": [1, 3, 7, 8, 9],
    "chemo": [0, 1, 2, 3, 4],
    "target": [0, 1, 2, 4],
    **{f"syst_aanpassing{i}": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14] for i in range(1, 13)},
    **{f"syst_chemort{i}": [0, 1, 2, 3, 4] for i in range(1, 13)},
    **{f"syst_prepost{i}": [1, 2, 3, 4] for i in range(1, 13)},
    **{f"syst_compl{i}": [1, 2, 3, 9] for i in range(1, 3)},
    "fup_diagnostiek_reden": [1, 2, 3, 4, 7, 9],
}

ORDINAL_INTEGER_FEATURES = {
    "ond_lymf", "pos_lymf", "meta_lok_aantal",
    *{f"syst_kuren{i}" for i in range(1, 13)},
    *{f"syst_dosisreductie{i}" for i in range(1, 13)},
    *{f"syst_schemanum{i}" for i in range(1, 13)},
    "chir_int1", "chir_int2", "destr_int1", "rt_start_int1",
    *{f"meta_ther_int{i}" for i in range(1, 4)},
    *{f"syst_start_int{i}" for i in range(1, 13)},
    *{f"syst_stop_int{i}" for i in range(1, 13)},
}

CONTINUOUS_FEATURES = {
    "diag_gewicht", "klin_tum_afm", "pa_tum_afm",
    "ca199", "bilirubine", "albumine", "kreatinine",
    "hb", "ldh", "crp", "cea", "neutrofielen", "trombocyten",
    "ts_surg", "ts_syst",
}

def classify_feature(feature):
    if feature in IMMUTABLE_FEATURES:
        return FeaturePolicy.IMMUTABLE
    if any(feature.startswith(p) for p in _ENCODED_IMMUTABLE_PREFIXES):
        return FeaturePolicy.IMMUTABLE
    if feature in BINARY_FEATURES:
        return FeaturePolicy.BINARY
    if feature in CATEGORICAL_CODED_FEATURES:
        return FeaturePolicy.CATEGORICAL_CODED
    if feature in ORDINAL_INTEGER_FEATURES:
        return FeaturePolicy.ORDINAL_INTEGER
    if feature in CONTINUOUS_FEATURES:
        return FeaturePolicy.CONTINUOUS
    return FeaturePolicy.CONTINUOUS

def allowed_values(feature, training_range=None):
    policy = classify_feature(feature)
    if policy == FeaturePolicy.BINARY:
        return [0.0, 1.0]
    if policy == FeaturePolicy.CATEGORICAL_CODED:
        return [float(v) for v in CATEGORICAL_CODED_FEATURES[feature]]
    if policy == FeaturePolicy.ORDINAL_INTEGER:
        if training_range is None:
            return None
        lo, hi = training_range
        import math
        return [float(v) for v in range(int(math.floor(lo)), int(math.ceil(hi)) + 1)]
    return None

def partition_features(feature_names, ranges):
    immutable = []
    continuous = []
    categorical_values = {}
    unknown = []

    for feat in feature_names:
        policy = classify_feature(feat)

        if policy == FeaturePolicy.IMMUTABLE:
            immutable.append(feat)
        elif policy == FeaturePolicy.BINARY:
            categorical_values[feat] = [0.0, 1.0]
        elif policy == FeaturePolicy.CATEGORICAL_CODED:
            categorical_values[feat] = [float(v) for v in CATEGORICAL_CODED_FEATURES[feat]]
        elif policy == FeaturePolicy.ORDINAL_INTEGER:
            rng = ranges.get(feat)
            if rng is None:
                continuous.append(feat)
            else:
                vals = allowed_values(feat, rng)
                if vals:
                    categorical_values[feat] = vals
                else:
                    continuous.append(feat)
        elif policy == FeaturePolicy.CONTINUOUS:
            continuous.append(feat)

        if (
            feat not in IMMUTABLE_FEATURES
            and not any(feat.startswith(p) for p in _ENCODED_IMMUTABLE_PREFIXES)
            and feat not in BINARY_FEATURES
            and feat not in CATEGORICAL_CODED_FEATURES
            and feat not in ORDINAL_INTEGER_FEATURES
            and feat not in CONTINUOUS_FEATURES
        ):
            unknown.append(feat)

    return {
        "immutable": immutable,
        "continuous": continuous,
        "categorical_values": categorical_values,
        "unknown": unknown,
    }

CF_FREE_RSF = {"tumgericht_ther", "chir", "destr", "rt", "chemo"}
CF_FREE_CONDOR = {"ts_surg", "had_surg", "impulse_surg",
                  "ts_syst", "had_syst", "impulse_syst"}
CF_FREE_FEATURES = CF_FREE_RSF | CF_FREE_CONDOR
CF_FREE_FLAGS = {"tumgericht_ther", "chir", "destr", "rt", "chemo",
                 "had_surg", "had_syst"}

LAB_TOLERANCE = {
    "albumine":   ("abs",   6.0),
    "bilirubine": ("abs",  40.0),
    "crp":        ("abs",  50.0),
    "ldh":        ("abs", 120.0),
    "kreatinine": ("abs",  40.0),
    "hb":         ("abs",   2.0),
    "ca199":      ("log10", 0.4),
    "cea":        ("log10", 0.4),
}

SENTINEL_VALUES = {
    "albumine": {999.9, 9999.0, 9999.9},
    "ca199": {9000.0, 9999.0, 9999.9},
    "crp":   {9000.0, 9999.0, 9999.9},
    "cea":   {9000.0, 9999.0, 9999.9},
    "ldh":   {9999.0},
    "bilirubine": {9999.0},
}

def is_sentinel(feature, value):
    try:
        return float(value) in SENTINEL_VALUES.get(feature, ())
    except (TypeError, ValueError):
        return False