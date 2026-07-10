"""
Holdout evaluation for missingness handling strategies.

Trains on the full dev set and evaluates on the held-out test set,
reporting C-index for both overall and PACAP-only populations.
Supports single or all imputation strategies across dataset subsets
and feature sets.
"""
from __future__ import annotations

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from xai_decision_making.models.survival import config_survival as config
from xai_decision_making.models.survival.survival_utils import SurvivalModelWrapper, prepare_survival_data

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    concordance_index_censored = None

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


def _coerce_string_dtype_to_object(df):
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.StringDtype):
            out[c] = out[c].astype(object)
    return out


def _dataset_token(train_subset):
    if train_subset == "pacap":
        return "Small_(PACAP)", "Small (PACAP)"
    return "Large_(All)", "Large (All)"


def _strategy_specs(avoid_alpha):
    return [
        {
            "name": "Drop_Missing_Vars",
            "imputation_strategy": "drop_vars",
            "alpha": 0.0,
        },
        {
            "name": "Missingness_Imputed",
            "imputation_strategy": "simple",
            "alpha": 0.0,
        },
        {
            "name": "Missingness_Indicated",
            "imputation_strategy": "constant",
            "alpha": 0.0,
        },
        {
            "name": "Missingness_Avoiding",
            "imputation_strategy": "passthrough",
            "alpha": float(avoid_alpha),
        },
    ]


def _cindex(event, time, risk):
    if risk is None or len(risk) == 0:
        return float("nan")
    return float(concordance_index_censored(event, time, risk)[0])


def main():
    parser = argparse.ArgumentParser(
        description="Train on dev, evaluate on test for one or all strategies."
    )
    parser.add_argument(
        "--train-subsets",
        nargs="+",
        default=["all", "pacap"],
        choices=["all", "pacap"],
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="pretreatment_qol",
    )
    parser.add_argument(
        "--use-intersection",
        action="store_true",
    )
    parser.add_argument("--run-id", type=str, default=config.TIMESTAMP)
    parser.add_argument("--all-strategies", action="store_true")
    parser.add_argument(
        "--strategy",
        type=str,
        default="Missingness_Avoiding",
        choices=[
            "Drop_Missing_Vars",
            "Missingness_Imputed",
            "Missingness_Indicated",
            "Missingness_Avoiding",
        ],
    )
    parser.add_argument("--avoid-alpha", type=float, default=0.1)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    if concordance_index_censored is None:
        raise ImportError("scikit-survival is required for C-index evaluation.")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.SURVIVAL_MODELS_DIR, exist_ok=True)

    dev_all = pd.read_csv(config.TRAIN_DATA_PATH)
    test_all = pd.read_csv(config.TEST_DATA_PATH)

    test_cols = set(test_all.columns)

    if "in_pacap" in test_all.columns:
        test_pacap = test_all[
            test_all["in_pacap"].astype(str).str.lower().eq("yes")
        ].copy()
    else:
        test_pacap = test_all.iloc[0:0].copy()

    strategies = _strategy_specs(args.avoid_alpha)
    if not args.all_strategies:
        strategies = [s for s in strategies if s["name"] == args.strategy]

    results_rows = []

    for train_subset in args.train_subsets:
        ds_token, ds_label = _dataset_token(train_subset)

        if train_subset == "pacap":
            if "in_pacap" not in dev_all.columns:
                raise ValueError("Requested pacap subset but 'in_pacap' not found.")
            dev_df = dev_all[
                dev_all["in_pacap"].astype(str).str.lower().eq("yes")
            ].copy()
        else:
            dev_df = dev_all.copy()

        if args.use_intersection:
            cols_for_features = [c for c in dev_df.columns if c in test_cols]
        else:
            cols_for_features = list(dev_df.columns)

        features = config.get_feature_set(args.feature_set, cols_for_features)
        if len(features) == 0:
            raise ValueError(
                f"No features found for feature-set='{args.feature_set}'."
            )

        X_train, y_train = prepare_survival_data(dev_df, features)
        X_test_all, y_test_all = prepare_survival_data(test_all, features)

        X_train = _coerce_string_dtype_to_object(X_train)
        X_test_all = _coerce_string_dtype_to_object(X_test_all)

        if len(test_pacap) > 0:
            X_test_p, y_test_p = prepare_survival_data(test_pacap, features)
            X_test_p = _coerce_string_dtype_to_object(X_test_p)
        else:
            X_test_p, y_test_p = None, None

        for strat in strategies:
            strat_name = strat["name"]
            imp = strat["imputation_strategy"]
            alpha = float(strat["alpha"])

            print("\n" + "=" * 80)
            print(
                f"HOLDOUT | subset={ds_label} | feature_set={args.feature_set} | "
                f"strategy={strat_name} (imp={imp}, alpha={alpha}) | run={args.run_id}"
            )
            print("=" * 80)

            model = SurvivalModelWrapper(
                imputation_strategy=imp,
                alpha=alpha,
                **config.RANDOM_FOREST_PARAMS,
            )
            model.fit(X_train, y_train)

            risk_all = model.predict(X_test_all)
            c_all = _cindex(y_test_all["event"], y_test_all["time"], risk_all)

            if X_test_p is not None and y_test_p is not None and len(X_test_p) > 0:
                risk_p = model.predict(X_test_p)
                c_p = _cindex(y_test_p["event"], y_test_p["time"], risk_p)
            else:
                c_p = float("nan")

            print(f"C-index (TEST ALL):   {c_all:.4f}")
            print(f"C-index (TEST PACAP): {c_p:.4f}")

            model_file = None
            if args.save_models:
                model_name = (
                    f"{strat_name}_{ds_token}_{args.feature_set}"
                    f"_run{args.run_id}_fold0.pkl"
                )
                model_path = os.path.join(config.SURVIVAL_MODELS_DIR, model_name)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                model_file = model_name
                print(f"Saved model: {model_path}")

            results_rows.append(
                {
                    "Run_ID": args.run_id,
                    "Dataset": ds_label,
                    "Feature_Set": args.feature_set,
                    "Strategy": strat_name,
                    "Imputation_Strategy": imp,
                    "Alpha": alpha,
                    "ModelFile": model_file,
                    "C-Index (ALL)": c_all,
                    "C-Index (PACAP)": c_p,
                }
            )

    out_csv = os.path.join(
        config.RESULTS_DIR,
        f"holdout_missingness_run{args.run_id}_featureset_{args.feature_set}.csv",
    )
    pd.DataFrame(results_rows).to_csv(out_csv, index=False)
    print(f"\nSaved holdout results: {out_csv}")


if __name__ == "__main__":
    main()