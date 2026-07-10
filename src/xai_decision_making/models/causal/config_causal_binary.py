"""
Unified configuration for binary causal inference analysis.

Default settings are for SYSTEMIC treatment comparison
(FOLFIRINOX vs Gemcitabine-Based). Commented alternatives are for
SURGERY comparison (surgery vs no surgery).

Creates data/full_processed.csv (dev+test merged) on import if missing.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parents[3])
DATA_DIR = os.path.join(BASE_DIR, "data")

FULL_PROCESSED_FILEPATH = os.path.join(DATA_DIR, "full_processed.csv")
DEV_SET_PROCESSED_FILEPATH = os.path.join(DATA_DIR, "dev_set_processed.csv")
TEST_SET_PROCESSED_FILEPATH = os.path.join(DATA_DIR, "test_set_processed.csv")


def _ensure_full_processed():
    if not os.path.exists(FULL_PROCESSED_FILEPATH):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required to create the merged full_processed file."
            ) from e

        if not os.path.exists(DEV_SET_PROCESSED_FILEPATH):
            raise FileNotFoundError(
                f"Missing source file: {DEV_SET_PROCESSED_FILEPATH}"
            )
        if not os.path.exists(TEST_SET_PROCESSED_FILEPATH):
            raise FileNotFoundError(
                f"Missing source file: {TEST_SET_PROCESSED_FILEPATH}"
            )

        df1 = pd.read_csv(DEV_SET_PROCESSED_FILEPATH, low_memory=False)
        df2 = pd.read_csv(TEST_SET_PROCESSED_FILEPATH, low_memory=False)
        merged = pd.concat([df1, df2], ignore_index=True)

        os.makedirs(os.path.dirname(FULL_PROCESSED_FILEPATH), exist_ok=True)
        merged.to_csv(FULL_PROCESSED_FILEPATH, index=False)
        print(f"Created merged file: {FULL_PROCESSED_FILEPATH}")


_ensure_full_processed()

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "causal_models_binary")
DEV_SET_PATH = os.path.join(DATA_DIR, "full_processed.csv")
TEST_SET_PATH = os.path.join(DATA_DIR, "black_envelop_test_set_processed.csv")

# OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'causal_pipeline_surgery')
# MODEL_DIR = os.path.join(BASE_DIR, 'causal_models_surgery')
# DEV_SET_PATH = os.path.join(DATA_DIR, 'dev_set_processed.csv')
# TEST_SET_PATH = os.path.join(DATA_DIR, 'black_envelop_test_set_processed.csv')

TREATMENTS_OF_INTEREST = ["FOLFIRINOX", "Gemcitabine-Based"]

TREATMENT_MAPPING = {
    "FOLFIRINOX": "FOLFIRINOX",
    "Gemcitabine Monotherapy": "Gemcitabine-Based",
    "Gemcitabine + Nab-Paclitaxel": "Gemcitabine-Based",
    "Gemcitabine + Capecitabine": "Gemcitabine-Based",
}

TREATMENT_COLUMN_RAW = "treatment_regimen"
TREATMENT_COLUMN_MAPPED = "treatment_group"

SURVIVAL_STATUS_COLUMN = "vit_stat"
SURVIVAL_TIME_COLUMN = "vit_stat_int"
QOL_TARGET_COLUMN = "qlq_diff_3m"
QOL_BASELINE_COLUMN = "pretreatment_qol"

BINARY_FEATURE_COLS = [
    "ond_lymf",
    "meta_lever",
    "meta_bijnier",
    "meta_bot",
    "meta_hersenen",
    "meta_long",
    "meta_lymf",
    "meta_perit",
    "meta_overig",
    "cm",
    "gesl",
]

PROPENSITY_NUMERICAL_FEATURES = [
    "cci",
    "ca199",
    "bilirubine",
    "albumine",
    "ldh",
    "crp",
] + BINARY_FEATURE_COLS

PROPENSITY_CATEGORICAL_FEATURES = [
    "leeft_cat",
    "cci_cat",
    "perf_stat",
    "seizoen",
    "diffgrad",
    "klin_tum_afm",
    "morf",
    "ct",
    "cn",
    "cstadium",
    "stadium",
    "veneus_vaatbetr",
    "arterieel_vaatbetr",
    "geboorteland",
    "diag_basis",
    "topo_sublok",
    "morf_cat",
]

SURVIVAL_NUMERICAL_FEATURES = PROPENSITY_NUMERICAL_FEATURES
SURVIVAL_CATEGORICAL_FEATURES = PROPENSITY_CATEGORICAL_FEATURES + [
    TREATMENT_COLUMN_MAPPED
]

QOL_NUMERICAL_FEATURES = PROPENSITY_NUMERICAL_FEATURES + [QOL_BASELINE_COLUMN]
QOL_CATEGORICAL_FEATURES = PROPENSITY_CATEGORICAL_FEATURES + [
    TREATMENT_COLUMN_MAPPED
]

RANDOM_STATE = 42

PROPENSITY_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "n_jobs": -1,
}

SURVIVAL_MODEL_PARAMS = {
    "n_estimators": 100,
    "min_samples_split": 10,
    "min_samples_leaf": 15,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

QOL_MODEL_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}