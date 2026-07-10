"""
SHAP summary plots for trained survival model artifacts.

Loads a saved .pkl model, builds background and evaluation sets from
the PACAP subset, runs KernelExplainer to compute SHAP values, and
saves a beeswarm summary plot. Handles both pipeline-based and
passthrough (LabelEncoder) preprocessing strategies.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from xai_decision_making.models.survival import config_survival as config
from xai_decision_making.models.survival.survival_utils import (
    load_trained_survival_model,
    prepare_survival_data,
)

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

warnings.filterwarnings("ignore")


def _safe_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if hasattr(x, "tolist"):
        return list(x.tolist())
    try:
        return list(x)
    except TypeError:
        return []


def _coerce_string_dtype_to_object(df):
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.StringDtype):
            out[c] = out[c].astype(object)
    return out


def _to_numpy_float(x):
    if hasattr(x, "to_numpy"):
        return x.to_numpy(dtype=np.float64)
    if hasattr(x, "toarray"):
        return x.toarray().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _build_shap_matrices(model, X_bg_df, X_eval_df):
    imputation_strategy = getattr(model, "imputation_strategy", None)

    if imputation_strategy == "passthrough":
        X_bg_df = _coerce_string_dtype_to_object(X_bg_df)
        X_eval_df = _coerce_string_dtype_to_object(X_eval_df)

        X_bg_enc = model._encode_categorical_features(X_bg_df, fit=False)
        X_eval_enc = model._encode_categorical_features(X_eval_df, fit=False)

        X_bg_np = X_bg_enc.to_numpy(dtype=np.float64)
        X_eval_np = X_eval_enc.to_numpy(dtype=np.float64)

        feature_names = list(X_bg_enc.columns)
        return X_bg_np, X_eval_np, feature_names

    pipeline = getattr(model, "pipeline", None)
    if pipeline is None:
        raise RuntimeError(
            "Loaded model does not have a fitted pipeline and is not passthrough."
        )

    X_bg_t = pipeline.transform(_coerce_string_dtype_to_object(X_bg_df))
    X_eval_t = pipeline.transform(_coerce_string_dtype_to_object(X_eval_df))

    X_bg_np = _to_numpy_float(X_bg_t)
    X_eval_np = _to_numpy_float(X_eval_t)

    feature_names = []
    if hasattr(pipeline, "get_feature_names_out"):
        try:
            feature_names = _safe_list(pipeline.get_feature_names_out())
        except Exception:
            feature_names = []

    if not feature_names:
        feature_names = _safe_list(getattr(model, "feature_names_", None))

    if not feature_names or len(feature_names) != X_bg_np.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_bg_np.shape[1])]

    return X_bg_np, X_eval_np, feature_names


def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP plots for a trained survival model artifact."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
    )
    parser.add_argument("--n-background", type=int, default=30)
    parser.add_argument("--n-eval", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsamples", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(config.SHAP_DIR, exist_ok=True)

    artifact = load_trained_survival_model(args.pattern)
    model = artifact.model
    print(f"Loaded model: {artifact.path}")

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    combined = pd.concat([train_df, test_df], ignore_index=True)

    if "in_pacap" in combined.columns:
        pacap_df = combined[combined["in_pacap"] == "yes"].copy()
    else:
        pacap_df = combined.copy()

    if len(pacap_df) == 0:
        raise ValueError("No rows available for SHAP.")

    pacap_train, pacap_test = train_test_split(
        pacap_df,
        test_size=0.2,
        random_state=args.seed,
        stratify=(
            pacap_df[config.EVENT_COL]
            if config.EVENT_COL in pacap_df.columns
            else None
        ),
    )

    features = None
    pipe = getattr(model, "pipeline", None)
    if pipe is not None and hasattr(pipe, "feature_names_in_"):
        features = [str(c) for c in list(pipe.feature_names_in_)]
    else:
        fn = getattr(model, "feature_names_", None)
        fn_list = _safe_list(fn)
        if len(fn_list) > 0:
            features = [str(c) for c in fn_list]

    if not features:
        features = [
            c for c in pacap_train.columns if c not in [config.EVENT_COL, config.TIME_COL]
        ]

    X_train, _ = prepare_survival_data(pacap_train, features)
    X_test, _ = prepare_survival_data(pacap_test, features)

    n_bg = min(args.n_background, len(X_train))
    n_eval = min(args.n_eval, len(X_test))
    if n_bg < 2 or n_eval < 1:
        raise ValueError("Not enough rows for SHAP background/eval sampling.")

    X_bg_df = X_train.sample(n=n_bg, random_state=args.seed)
    X_eval_df = X_test.sample(n=n_eval, random_state=args.seed)

    X_bg_np, X_eval_np, feat_names = _build_shap_matrices(
        model=model,
        X_bg_df=X_bg_df,
        X_eval_df=X_eval_df,
    )

    def predict_fn(x_np):
        return model.model.predict(np.asarray(x_np, dtype=np.float64))

    explainer = shap.KernelExplainer(predict_fn, X_bg_np)
    shap_values = explainer.shap_values(
        X_eval_np,
        nsamples=min(args.nsamples, max(20, n_bg * 2)),
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    out_path = os.path.join(
        config.SHAP_DIR,
        f"shap_summary_{os.path.basename(artifact.path).replace('.pkl', '')}.png",
    )

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_eval_np,
        feature_names=feat_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP plot: {out_path}")


if __name__ == "__main__":
    main()