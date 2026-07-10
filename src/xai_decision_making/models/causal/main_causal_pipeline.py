"""
Binary causal inference pipeline for treatment effect estimation.

Estimates the Average Treatment Effect (ATE) of FOLFIRINOX vs
Gemcitabine-based regimens on overall survival and quality of life
using inverse probability of treatment weighting (IPTW). Pipeline
stages: propensity scoring, covariate balance checking, IPTW-weighted
outcome modeling (Random Survival Forest + Random Forest regressor),
and evaluation on a held-out black-envelope test set.
"""

from __future__ import annotations

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from xai_decision_making.models.causal import config_causal_binary as cfg

warnings.filterwarnings("ignore")


def _clean_data_types(df):
    for col in cfg.BINARY_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_and_filter_data(filepath):
    print(f"\n--- Loading and filtering data from: {os.path.basename(filepath)} ---")
    df = pd.read_csv(filepath, low_memory=False)
    df = _clean_data_types(df)

    df[cfg.TREATMENT_COLUMN_MAPPED] = df[cfg.TREATMENT_COLUMN_RAW].map(
        cfg.TREATMENT_MAPPING
    )
    df_filtered = df[df[cfg.TREATMENT_COLUMN_MAPPED].isin(cfg.TREATMENTS_OF_INTEREST)].copy()
    df_filtered.dropna(
        subset=[cfg.SURVIVAL_STATUS_COLUMN, cfg.SURVIVAL_TIME_COLUMN], inplace=True
    )

    print(f"Loaded {len(df)} total patients.")
    print(f"Filtered to {len(df_filtered)} patients in the causal comparison groups.")
    return df_filtered


def preprocess_features(df, numerical_features, categorical_features, scaler=None, encoders=None):
    is_training = scaler is None

    all_features = numerical_features + categorical_features
    seen = set()
    features = []
    for f in all_features:
        if f not in seen and f in df.columns:
            features.append(f)
            seen.add(f)

    X = df[features].copy()

    for col in numerical_features:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    if is_training:
        encoders = {}
        for col in categorical_features:
            if col in X.columns:
                col_data = X[col] if isinstance(X[col], pd.Series) else X[col].iloc[:, 0]
                col_data = col_data.fillna("Unknown").astype(str)
                le = LabelEncoder()
                X[col] = le.fit_transform(col_data)
                encoders[col] = le
    else:
        for col, le in encoders.items():
            if col in X.columns:
                col_data = X[col] if isinstance(X[col], pd.Series) else X[col].iloc[:, 0]
                col_data = col_data.fillna("Unknown").astype(str)
                known_classes = set(le.classes_)
                col_data = col_data.apply(lambda s: s if s in known_classes else "Unknown")
                if "Unknown" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "Unknown")
                X[col] = le.transform(col_data)

    num_features_to_scale = [f for f in numerical_features if f in X.columns]
    if is_training:
        scaler = StandardScaler()
        X[num_features_to_scale] = scaler.fit_transform(X[num_features_to_scale])
    else:
        if num_features_to_scale:
            X[num_features_to_scale] = scaler.transform(X[num_features_to_scale])

    return X, scaler, encoders


def train_propensity_model(df):
    print("\n--- Training Propensity Score Model ---")
    X_prop, scaler, encoders = preprocess_features(
        df, cfg.PROPENSITY_NUMERICAL_FEATURES, cfg.PROPENSITY_CATEGORICAL_FEATURES
    )
    y = (df[cfg.TREATMENT_COLUMN_MAPPED] == "FOLFIRINOX").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_prop,
        y,
        test_size=0.2,
        random_state=cfg.RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(**cfg.PROPENSITY_MODEL_PARAMS)
    model.fit(X_train, y_train)

    print("\nPropensity Model Performance (on internal test set):")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    return model, scaler, encoders


def apply_iptw_and_balance_check(df, model, scaler, encoders):
    print("\n--- Applying IPTW and Checking Covariate Balance ---")
    X_prop, _, _ = preprocess_features(
        df,
        cfg.PROPENSITY_NUMERICAL_FEATURES,
        cfg.PROPENSITY_CATEGORICAL_FEATURES,
        scaler,
        encoders,
    )

    propensity_scores = np.clip(model.predict_proba(X_prop)[:, 1], 1e-6, 1 - 1e-6)
    treatment = (df[cfg.TREATMENT_COLUMN_MAPPED] == "FOLFIRINOX").astype(int)

    weights = np.where(treatment == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))
    df["iptw_weight"] = weights

    p99 = np.percentile(weights, 99)
    df["iptw_weight_truncated"] = np.minimum(weights, p99)
    print(f"IPTW weights calculated and truncated at the 99th percentile ({p99:.2f}).")

    balance_df = check_covariate_balance(df)

    if (np.abs(balance_df["SMD (Weighted)"]) > 0.1).any():
        print("WARNING: Not all covariates are well-balanced after weighting (|SMD| > 0.1).")
    else:
        print("SUCCESS: All covariates are well-balanced after weighting (|SMD| < 0.1).")

    return df


def check_covariate_balance(df):
    balance_data = []
    df_for_balance = df.copy()

    for cov in cfg.PROPENSITY_NUMERICAL_FEATURES:
        if cov in df_for_balance.columns:
            df_for_balance[cov] = df_for_balance[cov].fillna(df_for_balance[cov].median())
            unweighted, weighted = calculate_smd(df_for_balance, cov)
            balance_data.append(
                {
                    "Covariate": cov,
                    "SMD (Unweighted)": unweighted,
                    "SMD (Weighted)": weighted,
                }
            )

    balance_df = pd.DataFrame(balance_data).set_index("Covariate")
    print("\nCovariate Balance (Standardized Mean Difference):")
    print(balance_df.to_string(float_format="%.3f"))
    return balance_df


def calculate_smd(df, covariate):
    group1 = df[df[cfg.TREATMENT_COLUMN_MAPPED] == "FOLFIRINOX"]
    group2 = df[df[cfg.TREATMENT_COLUMN_MAPPED] == "Gemcitabine-Based"]

    pooled_std = np.sqrt((group1[covariate].std() ** 2 + group2[covariate].std() ** 2) / 2)
    if pooled_std == 0:
        return 0.0, 0.0

    smd_unweighted = (group1[covariate].mean() - group2[covariate].mean()) / pooled_std

    w_mean1 = np.average(group1[covariate], weights=group1["iptw_weight_truncated"])
    w_mean2 = np.average(group2[covariate], weights=group2["iptw_weight_truncated"])
    smd_weighted = (w_mean1 - w_mean2) / pooled_std

    return smd_unweighted, smd_weighted


def train_outcome_models(df_with_weights):
    print("\n--- Training Final, Causally-Adjusted Outcome Models ---")
    artifacts = {}

    print("\nTraining Survival Model...")
    X_surv, scaler_surv, encoders_surv = preprocess_features(
        df_with_weights, cfg.SURVIVAL_NUMERICAL_FEATURES, cfg.SURVIVAL_CATEGORICAL_FEATURES
    )
    y_surv = Surv.from_arrays(
        event=df_with_weights[cfg.SURVIVAL_STATUS_COLUMN].astype(bool),
        time=df_with_weights[cfg.SURVIVAL_TIME_COLUMN],
    )
    weights_surv = df_with_weights["iptw_weight_truncated"].to_numpy()

    model_surv = RandomSurvivalForest(**cfg.SURVIVAL_MODEL_PARAMS)
    model_surv.fit(X_surv, y_surv, sample_weight=weights_surv)
    artifacts["survival"] = {"model": model_surv, "scaler": scaler_surv, "encoders": encoders_surv}
    print("Survival model trained.")

    print("\nTraining QoL Model...")
    df_qol = df_with_weights.dropna(subset=[cfg.QOL_TARGET_COLUMN, cfg.QOL_BASELINE_COLUMN])
    if not df_qol.empty:
        X_qol, scaler_qol, encoders_qol = preprocess_features(
            df_qol, cfg.QOL_NUMERICAL_FEATURES, cfg.QOL_CATEGORICAL_FEATURES
        )
        y_qol = df_qol[cfg.QOL_TARGET_COLUMN]
        weights_qol = df_qol["iptw_weight_truncated"].to_numpy()

        model_qol = RandomForestRegressor(**cfg.QOL_MODEL_PARAMS)
        model_qol.fit(X_qol, y_qol, sample_weight=weights_qol)
        artifacts["qol"] = {"model": model_qol, "scaler": scaler_qol, "encoders": encoders_qol}
        print(f"QoL model trained on {len(df_qol)} patients.")
    else:
        artifacts["qol"] = None

    return artifacts


def _apply_iptw_to_test_set(test_df, prop_model, prop_scaler, prop_encoders):
    print("Applying IPTW to test set for ATE calculation...")

    X_prop_test, _, _ = preprocess_features(
        test_df,
        cfg.PROPENSITY_NUMERICAL_FEATURES,
        cfg.PROPENSITY_CATEGORICAL_FEATURES,
        scaler=prop_scaler,
        encoders=prop_encoders,
    )

    propensity_scores = np.clip(prop_model.predict_proba(X_prop_test)[:, 1], 1e-6, 1 - 1e-6)
    treatment = (test_df[cfg.TREATMENT_COLUMN_MAPPED] == "FOLFIRINOX").astype(int)

    weights = np.where(treatment == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))

    p99 = np.percentile(weights, 99)
    test_df["iptw_weight_truncated"] = np.minimum(weights, p99)

    return test_df


def evaluate_on_test_set(test_df, artifacts, prop_artifacts):
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON BLACK ENVELOPE TEST SET")
    print("=" * 80)

    test_df_with_weights = _apply_iptw_to_test_set(
        test_df.copy(),
        prop_artifacts["model"],
        prop_artifacts["scaler"],
        prop_artifacts["encoders"],
    )

    if "survival" in artifacts:
        print("\n--- Survival Model Evaluation ---")
        surv_artifacts = artifacts["survival"]
        X_test_surv, _, _ = preprocess_features(
            test_df,
            cfg.SURVIVAL_NUMERICAL_FEATURES,
            cfg.SURVIVAL_CATEGORICAL_FEATURES,
            scaler=surv_artifacts["scaler"],
            encoders=surv_artifacts["encoders"],
        )
        y_test_surv = Surv.from_arrays(
            event=test_df[cfg.SURVIVAL_STATUS_COLUMN].astype(bool),
            time=test_df[cfg.SURVIVAL_TIME_COLUMN],
        )
        risk_scores = surv_artifacts["model"].predict(X_test_surv)
        c_index = concordance_index_censored(y_test_surv["event"], y_test_surv["time"], risk_scores)[0]
        print(f"Test Set C-Index: {c_index:.3f}")

        print("\n--- Causal Effect on Survival (ATE) ---")
        f_mask = test_df_with_weights[cfg.TREATMENT_COLUMN_MAPPED] == "FOLFIRINOX"
        g_mask = test_df_with_weights[cfg.TREATMENT_COLUMN_MAPPED] == "Gemcitabine-Based"

        risk_series = pd.Series(risk_scores, index=test_df.index)
        w_mean_risk_f = np.average(
            risk_series[f_mask],
            weights=test_df_with_weights.loc[f_mask, "iptw_weight_truncated"],
        )
        w_mean_risk_g = np.average(
            risk_series[g_mask],
            weights=test_df_with_weights.loc[g_mask, "iptw_weight_truncated"],
        )

        print(f"FOLFIRINOX weighted mean risk: {w_mean_risk_f:.3f}")
        print(f"Gemcitabine-Based weighted mean risk: {w_mean_risk_g:.3f}")
        print(f"Risk Difference (ATE): {w_mean_risk_f - w_mean_risk_g:+.3f}")

    if artifacts.get("qol"):
        print("\n--- QoL Model Evaluation ---")
        qol_artifacts = artifacts["qol"]
        df_test_qol = test_df_with_weights.dropna(subset=[cfg.QOL_TARGET_COLUMN, cfg.QOL_BASELINE_COLUMN])

        if not df_test_qol.empty:
            X_test_qol, _, _ = preprocess_features(
                df_test_qol,
                cfg.QOL_NUMERICAL_FEATURES,
                cfg.QOL_CATEGORICAL_FEATURES,
                scaler=qol_artifacts["scaler"],
                encoders=qol_artifacts["encoders"],
            )
            y_test_qol = df_test_qol[cfg.QOL_TARGET_COLUMN]
            qol_predictions = qol_artifacts["model"].predict(X_test_qol)

            print(f"Test Set R²: {r2_score(y_test_qol, qol_predictions):.3f}")
            print(f"Test Set MAE: {mean_absolute_error(y_test_qol, qol_predictions):.3f}")

            print("\n--- Causal Effect on QoL (ATE) ---")
            pred_series = pd.Series(qol_predictions, index=X_test_qol.index)
            f_qol_mask = df_test_qol[cfg.TREATMENT_COLUMN_MAPPED] == "FOLFIRINOX"
            g_qol_mask = df_test_qol[cfg.TREATMENT_COLUMN_MAPPED] == "Gemcitabine-Based"

            w_mean_qol_f = np.average(
                pred_series[f_qol_mask],
                weights=df_test_qol.loc[f_qol_mask, "iptw_weight_truncated"],
            )
            w_mean_qol_g = np.average(
                pred_series[g_qol_mask],
                weights=df_test_qol.loc[g_qol_mask, "iptw_weight_truncated"],
            )

            print(f"FOLFIRINOX weighted mean QoL prediction: {w_mean_qol_f:+.3f}")
            print(f"Gemcitabine-Based weighted mean QoL prediction: {w_mean_qol_g:+.3f}")
            print(f"QoL Difference (ATE): {w_mean_qol_f - w_mean_qol_g:+.3f}")
        else:
            print("No QoL data available in test set for evaluation.")


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    train_df = load_and_filter_data(cfg.DEV_SET_PATH)
    prop_model, prop_scaler, prop_encoders = train_propensity_model(train_df)
    train_df_with_weights = apply_iptw_and_balance_check(
        train_df, prop_model, prop_scaler, prop_encoders
    )

    propensity_artifacts = {"model": prop_model, "scaler": prop_scaler, "encoders": prop_encoders}
    joblib.dump(propensity_artifacts, os.path.join(cfg.MODEL_DIR, "propensity_pipeline.pkl"))

    trained_artifacts = train_outcome_models(train_df_with_weights)

    if "survival" in trained_artifacts:
        joblib.dump(
            trained_artifacts["survival"],
            os.path.join(cfg.MODEL_DIR, "causal_survival_pipeline.pkl"),
        )
    if trained_artifacts.get("qol"):
        joblib.dump(
            trained_artifacts["qol"],
            os.path.join(cfg.MODEL_DIR, "causal_qol_pipeline.pkl"),
        )

    if os.path.exists(cfg.TEST_SET_PATH):
        test_df = load_and_filter_data(cfg.TEST_SET_PATH)
        evaluate_on_test_set(test_df, trained_artifacts, propensity_artifacts)
    else:
        print(f"\nTest set not found at {cfg.TEST_SET_PATH}. Skipping final evaluation.")

    print("\n" + "=" * 80)
    print("CAUSAL PIPELINE COMPLETE")
    print(f"All models and artifacts saved in: {cfg.MODEL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()