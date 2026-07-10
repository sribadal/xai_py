"""
Configuration for the survival modeling pipeline.

Defines data paths, artifact directories, output locations, survival
target columns, feature set definitions (baseline through extended/DPCG),
and Random Survival Forest hyperparameters. Feature sets are resolved
at runtime via get_feature_set() against available DataFrame columns.
"""
from __future__ import annotations

import os
from datetime import datetime

TRAIN_DATA_PATH = "data/dev_set_processed.csv"
TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "data/test_set_processed.csv")
BLACK_ENVELOP_TEST_PATH = "data/black_envelop_test_set_processed.csv"

SURVIVAL_MODELS_DIR = "artifacts/survival_models"
CONDOR_MODELS_DIR = "artifacts/condor_models"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = "outputs/overall_survival_comparison"
PLOTS_DIR = os.environ.get("PLOTS_DIR", os.path.join(OUTPUT_DIR, "plots"))
SHAP_DIR = os.path.join(OUTPUT_DIR, "shap_plots")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(OUTPUT_DIR, "results"))
RESULTS_FILENAME = os.path.join(
    RESULTS_DIR,
    f"overall_survival_model_results_{TIMESTAMP}.csv",
)

EVENT_COL = "vit_stat"
TIME_COL = "vit_stat_int"

BASE_NUMERICAL_FEATURES = [
    "cci",
    "ca199",
    "bilirubine",
    "albumine",
    "ldh",
    "crp",
]

BASE_CATEGORICAL_FEATURES = [
    "leeft_cat",
    "cci_cat",
    "perf_stat",
    "diffgrad",
    "klin_tum_afm",
    "ct",
    "cm",
    "cstadium",
    "stadium",
    "veneus_vaatbetr",
    "arterieel_vaatbetr",
    "meta_lever",
    "meta_bijnier",
    "meta_bot",
    "meta_hersenen",
    "meta_long",
    "meta_lymf",
    "meta_perit",
    "meta_overig",
]

BASELINE_FEATURES = BASE_NUMERICAL_FEATURES + BASE_CATEGORICAL_FEATURES

PAPER_BASELINE_FEATURES = BASELINE_FEATURES + [
    "gesl",
    "meta_lok_aantal",
]

PAPER_PRETREATMENT_QOL_FEATURES = PAPER_BASELINE_FEATURES + [
    "qol_summary_pretreatment",
    "qol_physical_pretreatment",
    "qol_emotional_pretreatment",
    "qol_social_pretreatment",
    "qol_cognitive_pretreatment",
    "qol_role_pretreatment",
]

PRETREATMENT_QOL_FEATURES = BASELINE_FEATURES + [
    "qol_summary_pretreatment",
    "qol_physical_pretreatment",
    "qol_emotional_pretreatment",
    "qol_social_pretreatment",
    "qol_cognitive_pretreatment",
    "qol_role_pretreatment",
]

PAPER_HEADLINE_FEATURES = PAPER_PRETREATMENT_QOL_FEATURES + [
    "tumgericht_ther",
    "chir",
    "destr",
    "rt",
    "chemo",
]

PAPER_HEADLINE_FEATURES_NO_LEAK = [
    f for f in PAPER_HEADLINE_FEATURES if f != "stadium"
]

PAPER_PRETREATMENT_QOL_NO_LEAK = [
    f for f in PAPER_PRETREATMENT_QOL_FEATURES if f != "stadium"
]

LEGACY_FEATURES = (
    BASE_NUMERICAL_FEATURES
    + [
        "leeft_cat", "cci_cat", "perf_stat", "diffgrad", "klin_tum_afm",
        "ct", "cm", "cstadium", "stadium",
        "veneus_vaatbetr", "arterieel_vaatbetr",
        "meta_lever",
        "meta_bot", "meta_hersenen", "meta_long", "meta_lymf",
        "meta_perit", "meta_overig",
    ]
    + [
        "qol_summary_pretreatment",
        "qol_physical_pretreatment",
        "qol_emotional_pretreatment",
        "qol_social_pretreatment",
        "qol_cognitive_pretreatment",
        "qol_role_pretreatment",
    ]
    + ["tumgericht_ther", "chir", "destr", "rt", "chemo"]
)

PAPER_EXTENDED_FEATURES = PAPER_HEADLINE_FEATURES_NO_LEAK + [
    "asa",
    "cn",
    "hb",
    "kreatinine",
    "neutrofielen",
    "trombocyten",
    "resectability_status",
]

DPCG_TRUE_FEATURES = [
    "leeft_cat",
    "perf_stat",
    "ca199",
    "diffgrad",
    "ct",
]

DPCG_CLINICAL_FEATURES = [
    "leeft_cat",
    "perf_stat",
    "cci",
    "ca199",
    "diffgrad",
    "cstadium",
    "meta_lever",
    "chemo",
    "chir",
]

TREATMENT_FEATURES = [
    "tumgericht_ther",
    "chir",
    "destr",
    "rt",
    "chemo",
    "chir_int1",
    "chir_int2",
    "destr_int1",
    "rt_start_int1",
    "meta_ther_int1",
    "meta_ther_int2",
    "meta_ther_int3",
    "syst_start_int1",
    "syst_stop_int1",
    "syst_start_int2",
    "syst_stop_int2",
    "syst_start_int3",
    "syst_stop_int3",
    "syst_start_int4",
    "syst_stop_int4",
    "syst_start_int5",
    "syst_stop_int5",
    "syst_start_int6",
    "syst_stop_int6",
    "chir_type1",
    "chir_type2",
    "destr_type1",
    "rt_type1",
]

RANDOM_STATE = 42
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "min_samples_split": 10,
    "min_samples_leaf": 15,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def filter_existing(features, df_cols):
    cols = set(df_cols)
    out = []
    for f in features:
        if f in cols and f not in out:
            out.append(f)
    return out


def infer_all_features(df_cols):
    drop = {EVENT_COL, TIME_COL, "in_pacap"}
    return [c for c in df_cols if c not in drop]


def get_feature_set(name, df_cols):
    name = name.lower().strip()

    if name == "baseline":
        return filter_existing(BASELINE_FEATURES, df_cols)

    if name in {"pretreatment_qol", "baseline_plus_qol", "qol"}:
        return filter_existing(PRETREATMENT_QOL_FEATURES, df_cols)

    if name in {"baseline_plus_treatment", "treatment"}:
        feats = BASELINE_FEATURES + TREATMENT_FEATURES
        return filter_existing(feats, df_cols)

    if name in {"paper_baseline", "baseline_paper"}:
        return filter_existing(PAPER_BASELINE_FEATURES, df_cols)

    if name in {"paper_pretreatment_qol", "pretreatment_qol_paper"}:
        return filter_existing(PAPER_PRETREATMENT_QOL_FEATURES, df_cols)

    if name in {"paper_pretreatment_qol_no_leak", "pretreatment_qol_no_leak"}:
        return filter_existing(PAPER_PRETREATMENT_QOL_NO_LEAK, df_cols)

    if name in {"paper_headline", "headline"}:
        return filter_existing(PAPER_HEADLINE_FEATURES, df_cols)

    if name in {"paper_headline_no_leak", "headline_no_leak", "no_leak"}:
        return filter_existing(PAPER_HEADLINE_FEATURES_NO_LEAK, df_cols)

    if name in {"legacy", "legacy_features", "old_paper", "jan29", "original"}:
        return filter_existing(LEGACY_FEATURES, df_cols)

    if name in {"paper_extended", "extended", "new_variables", "paper_plus"}:
        return filter_existing(PAPER_EXTENDED_FEATURES, df_cols)

    if name in {"dpcg_true", "dpcg5"}:
        return filter_existing(DPCG_TRUE_FEATURES, df_cols)

    if name in {"dpcg_clinical", "dpcg", "clinical", "vignette", "dpcg9"}:
        return filter_existing(DPCG_CLINICAL_FEATURES, df_cols)

    if name == "all":
        return infer_all_features(df_cols)

    raise ValueError(
        f"Unknown feature-set '{name}'. Choose from: "
        f"baseline, pretreatment_qol, baseline_plus_treatment, "
        f"paper_baseline, paper_pretreatment_qol, "
        f"paper_headline (with stadium), paper_headline_no_leak (= no_leak), "
        f"legacy (= old_paper), paper_extended (= new_variables), "
        f"dpcg_true (= dpcg5), dpcg_clinical (= dpcg, dpcg9, vignette), all."
    )