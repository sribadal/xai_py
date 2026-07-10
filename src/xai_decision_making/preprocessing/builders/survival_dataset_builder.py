"""
Survival analysis dataset construction.

Builds preprocessed datasets for survival modeling by merging NKR clinical
data with PACAP QoL scores, handling censored observations, classifying
treatment regimens, and engineering derived features. Supports dev, test,
and black-envelope test set variants.
"""
import os
import sys
from xai_decision_making import config
import numpy as np
import pandas as pd

from xai_decision_making.preprocessing.qol.qlq_c30_scoring import calculate_qlq_c30_scores

def create_censored_dataset():
    df_raw = pd.read_csv(config.RAW_NKR_DATA_PATH, sep=";", low_memory=False)
    censored_data = df_raw[df_raw[config.VITAL_STATUS_COL] == 0].copy()

    dev_set = pd.read_csv(config.DEV_SET_PATH, sep=",", low_memory=False)

    merged_df = pd.concat([dev_set, censored_data], ignore_index=True)
    merged_df = merged_df.groupby(
        [config.KEY_NK, config.KEY_EID], as_index=False
    ).first()

    merged_df.to_csv(config.DEV_SET_CENSORED_PATH, index=False)
    return merged_df

def load_base_dataset(dataset_type: str):
    if dataset_type == "dev":
        main_file = config.DEV_SET_CENSORED_PATH
    elif dataset_type == "test":
        main_file = config.TEST_SET_PATH
    elif dataset_type == "black_envelop_test":
        main_file = config.BLACK_ENVELOP_TEST_PATH
    else:
        raise ValueError("Invalid dataset_type")

    df = pd.read_csv(main_file, low_memory=False)

    if os.path.exists(config.UPDATE_FILE_PATH):
        update_df = pd.read_csv(config.UPDATE_FILE_PATH, sep=";", low_memory=False)
        df = pd.merge(
            df, update_df, on=config.KEY_NK, how="left", suffixes=("", "_update")
        )

        for col in [c for c in df.columns if c.endswith("_update")]:
            base_col = col.replace("_update", "")
            if base_col in df.columns:
                df[base_col] = df[base_col].fillna(df[col])
                df = df.drop(columns=[col])

    return df

def merge_qol_features(base_df, pacap_df):
    qol_scores = calculate_qlq_c30_scores(pacap_df)
    pacap_df["qol_summary"] = qol_scores["Summary Score: QLQ-C30 Summary Score"] / 100.0

    final_df = pd.merge(
        base_df, pacap_df[[config.KEY_NK, "qol_summary"]], on=config.KEY_NK, how="left"
    )

    return final_df

def classify_treatment_regimen(row):
    patient_treatments = {
        str(row[f"syst_code{i}"])
        for i in range(1, 7)
        if f"syst_code{i}" in row and pd.notna(row[f"syst_code{i}"])
    }

    if not patient_treatments:
        return "No Systemic Treatment"

    for name, drugs in config.STANDARD_REGIMENS.items():
        if drugs.issubset(patient_treatments):
            return name

    return "Other Treatment"

def add_engineered_features(df):
    df[config.TREATMENT_REGIMEN_COL] = df.apply(classify_treatment_regimen, axis=1)
    return df

def build_survival_dataset(dataset_type="dev"):
    if dataset_type == "dev":
        create_censored_dataset()

    base_df = load_base_dataset(dataset_type)

    if dataset_type == "dev":
        pacap_file = config.DEV_SET_PACAP_PATH
    elif dataset_type == "test":
        pacap_file = config.TEST_SET_PACAP_PATH
    else:
        pacap_file = config.BLACK_ENVELOP_TEST_PACAP_PATH

    pacap_df = pd.read_csv(pacap_file, low_memory=False)

    final_df = merge_qol_features(base_df, pacap_df)
    final_df = add_engineered_features(final_df)

    output_file = config.PREPROCESSED_CAUSAL_OUTPUT_PATH_TPL.format(
        dataset_type=dataset_type
    )

    final_df.to_csv(output_file, index=False)

    return final_df
