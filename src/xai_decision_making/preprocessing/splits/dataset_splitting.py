"""
Patient-level stratified dataset splitting.

Splits NKR clinical data into dev, test, and black-envelope test sets
with stratification on 6-month mortality. Ensures no patient overlap
between splits and produces matched PACAP variants via inner join.
"""
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(file_path, delimiter=';'):
    try:
        df = pd.read_csv(file_path, sep=delimiter)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise


def save_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")


def get_unique_patient_counts(df, identifier_col):
    return df[identifier_col].nunique()


def check_no_overlap(train_set, test_set, identifier_col):
    train_ids = set(train_set[identifier_col])
    test_ids = set(test_set[identifier_col])
    overlap = train_ids.intersection(test_ids)

    if len(overlap) == 0:
        logging.info("No overlap between train and test sets.")
    else:
        logging.error(f"Overlap found! Overlapping IDs: {overlap}")

    return len(overlap) == 0


def remove_overlap_from_test_set(train_set, test_set, identifier_col):
    train_ids = set(train_set[identifier_col])
    overlap = set(test_set[identifier_col]).intersection(train_ids)

    if overlap:
        logging.warning(
            f"Removing {len(overlap)} overlapping patients from test set."
        )
        test_set_clean = test_set[
            ~test_set[identifier_col].isin(overlap)
        ].copy()
        return test_set_clean, overlap

    logging.info("No overlap between train and test sets.")
    return test_set, None


def filter_patients_by_vital_status(df, include_alive=True):
    df = df.copy()

    if include_alive:
        def calculate_flag(row):
            if row.get("vit_stat") != 1:
                return 0
            if pd.isna(row.get("vit_stat_int")):
                return 0
            return 1 if row["vit_stat_int"] <= 365 / 2 else 0

        df["dead_with_6months"] = df.apply(calculate_flag, axis=1)
        logging.info(f"Including all patients: {len(df)} total")
        return df

    dead_df = df[df["vit_stat"] == 1].copy()
    dead_df["dead_with_6months"] = dead_df.apply(
        lambda row: 1 if row["vit_stat_int"] <= 365 / 2 else 0,
        axis=1
    )
    logging.info(f"Filtered to dead patients: {len(dead_df)}")
    return dead_df


def split_data(df, stratify_col, test_size=0.3, random_state=42):
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[stratify_col],
            random_state=random_state
        )
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error during splitting: {e}")
        raise


def create_train_test_splits(
    df1_path,
    df2_path,
    output_dir,
    identifier_col="key_nkr"
):
    ensure_directory_exists(output_dir)

    df1 = load_data(df1_path)
    df2 = load_data(df2_path)

    initial_count_df1 = get_unique_patient_counts(df1, identifier_col)
    initial_count_df2 = get_unique_patient_counts(df2, identifier_col)

    filtered_df = filter_patients_by_vital_status(df1, include_alive=True)

    train_df, test_df = split_data(
        filtered_df,
        stratify_col="dead_with_6months",
        test_size=0.3
    )

    test_df_clean, _ = remove_overlap_from_test_set(
        train_df,
        test_df,
        identifier_col
    )

    test_df_pacap = pd.merge(
        test_df_clean,
        df2,
        how="inner",
        on=[identifier_col, "key_eid"]
    )

    save_data(test_df_clean, os.path.join(output_dir, "black_envelop_test_set.csv"))
    save_data(test_df_pacap, os.path.join(output_dir, "black_envelop_test_set_pacap.csv"))

    dev_df, holdout_df = split_data(
        train_df,
        stratify_col="dead_with_6months",
        test_size=0.2
    )

    holdout_df_clean, _ = remove_overlap_from_test_set(
        dev_df,
        holdout_df,
        identifier_col
    )

    save_data(dev_df, os.path.join(output_dir, "dev_set.csv"))
    save_data(holdout_df_clean, os.path.join(output_dir, "test_set.csv"))

    logging.info("Splitting pipeline completed.")

    return {
        "initial_df1": initial_count_df1,
        "initial_df2": initial_count_df2,
        "dev_size": len(dev_df),
        "test_size": len(holdout_df_clean),
        "black_envelope_size": len(test_df_clean),
    }