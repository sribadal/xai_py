"""
K-fold cross-validation for missingness handling strategies.

Runs stratified K-fold CV on the dev set comparing four imputation
strategies (drop vars, simple, constant, passthrough/masurvival) across
dataset subsets and feature sets. Saves per-fold C-index results and
optionally persists trained model artifacts per fold.
"""
from __future__ import annotations

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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


def main():
    parser = argparse.ArgumentParser(
        description="K-fold CV on dev_set_processed.csv for missingness strategies."
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--retrain", action="store_true")
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
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()

    if concordance_index_censored is None:
        raise ImportError("scikit-survival is required for C-index evaluation.")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.SURVIVAL_MODELS_DIR, exist_ok=True)

    dev_df_all = pd.read_csv(config.TRAIN_DATA_PATH)

    test_cols = None
    if args.use_intersection:
        test_df = pd.read_csv(config.TEST_DATA_PATH)
        test_cols = set(test_df.columns)

    strategies = _strategy_specs(args.avoid_alpha)
    if not args.all_strategies:
        strategies = [s for s in strategies if s["name"] == args.strategy]

    all_fold_rows = []

    for train_subset in args.train_subsets:
        ds_token, ds_label = _dataset_token(train_subset)

        if train_subset == "pacap":
            if "in_pacap" not in dev_df_all.columns:
                raise ValueError("Requested pacap subset but 'in_pacap' not found.")
            dev_df = dev_df_all[
                dev_df_all["in_pacap"].astype(str).str.lower().eq("yes")
            ].copy()
        else:
            dev_df = dev_df_all.copy()

        if args.use_intersection and test_cols is not None:
            cols_for_features = [c for c in dev_df.columns if c in test_cols]
        else:
            cols_for_features = list(dev_df.columns)

        features = config.get_feature_set(args.feature_set, cols_for_features)
        if len(features) == 0:
            raise ValueError(
                f"No features found for feature-set='{args.feature_set}'."
            )

        X, y = prepare_survival_data(dev_df, features)
        X = _coerce_string_dtype_to_object(X)

        kf = KFold(
            n_splits=args.n_folds,
            shuffle=True,
            random_state=config.RANDOM_STATE,
        )

        for strat in strategies:
            strat_name = strat["name"]
            imp = strat["imputation_strategy"]
            alpha = float(strat["alpha"])

            print("\n" + "=" * 80)
            print(
                f"CV | subset={ds_label} | feature_set={args.feature_set} | "
                f"strategy={strat_name} (imp={imp}, alpha={alpha}) | run={args.run_id}"
            )
            print("=" * 80)

            fold_scores = []

            for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
                X_tr = X.iloc[tr_idx].reset_index(drop=True)
                y_tr = y[tr_idx]
                X_te = X.iloc[te_idx].reset_index(drop=True)
                y_te = y[te_idx]

                model_name = (
                    f"{strat_name}_{ds_token}_{args.feature_set}"
                    f"_run{args.run_id}_fold{fold}.pkl"
                )
                model_path = os.path.join(config.SURVIVAL_MODELS_DIR, model_name)

                if (not args.retrain) and os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                else:
                    model = SurvivalModelWrapper(
                        imputation_strategy=imp,
                        alpha=alpha,
                        **config.RANDOM_FOREST_PARAMS,
                    )
                    model.fit(X_tr, y_tr)

                    if not args.no_save_models:
                        with open(model_path, "wb") as f:
                            pickle.dump(model, f)

                risk = model.predict(X_te)
                c_index = float(
                    concordance_index_censored(y_te["event"], y_te["time"], risk)[0]
                )

                fold_scores.append(c_index)
                all_fold_rows.append(
                    {
                        "Run_ID": args.run_id,
                        "Dataset": ds_label,
                        "Feature_Set": args.feature_set,
                        "Strategy": strat_name,
                        "Imputation_Strategy": imp,
                        "Alpha": alpha,
                        "Fold": fold,
                        "C-Index (CV)": c_index,
                        "ModelFile": model_name,
                    }
                )
                print(f"Fold {fold}: C-index (CV) = {c_index:.4f}")

            mean_c = float(np.mean(fold_scores))
            std_c = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            print(f"Mean ± Std (CV): {mean_c:.4f} ± {std_c:.4f}")

    out_csv = os.path.join(
        config.RESULTS_DIR,
        f"cv_missingness_run{args.run_id}_featureset_{args.feature_set}.csv",
    )
    pd.DataFrame(all_fold_rows).to_csv(out_csv, index=False)
    print(f"\nSaved CV results: {out_csv}")


if __name__ == "__main__":
    main()