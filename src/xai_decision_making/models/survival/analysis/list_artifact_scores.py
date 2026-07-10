"""
Quick C-index listing for all saved survival model artifacts.

Loads each .pkl artifact, predicts on the test or dev set, and prints
a compact table of per-fold C-index scores (overall and PACAP-only).
Lighter-weight alternative to evaluate_saved_artifacts for spot checks.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from xai_decision_making.models.survival import config_survival as config
from xai_decision_making.models.survival.survival_utils import load_pickle

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


RUN_RE = re.compile(r"_run(?P<run_id>\d{8}_\d{6})_")
FOLD_RE = re.compile(r"_fold(?P<fold>\d+)$")
MISSING_COLS_RE = re.compile(r"columns are missing:\s*(\{.*\})")


@dataclass(frozen=True)
class ParsedName:
    strategy: str
    dataset: str
    feature_set: str
    fold: int
    run_id: str | None


def parse_artifact_name(filename):
    if not filename.endswith(".pkl"):
        return None

    stem = filename[:-4]
    m_fold = FOLD_RE.search(stem)
    if not m_fold:
        return None
    fold = int(m_fold.group("fold"))
    stem = stem[: m_fold.start()]

    run_id = None
    m_run = RUN_RE.search(stem)
    if m_run:
        run_id = m_run.group("run_id")
        stem = RUN_RE.sub("_", stem)

    if "_Small_(PACAP)_" in stem:
        dataset_label = "Small (PACAP)"
        dataset_token = "Small_(PACAP)"
    elif "_Large_(All)_" in stem:
        dataset_label = "Large (All)"
        dataset_token = "Large_(All)"
    else:
        return None

    strategy, feature_set = stem.split(f"_{dataset_token}_", 1)

    return ParsedName(
        strategy=strategy,
        dataset=dataset_label,
        feature_set=feature_set,
        fold=fold,
        run_id=run_id,
    )


def safe_cindex(event, time, risk):
    if risk is None or len(risk) == 0:
        return float("nan")
    return float(concordance_index_censored(event, time, risk)[0])


def coerce_pandas_string_to_object(df):
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.StringDtype):
            out[c] = out[c].astype(object)
    return out


def ensure_columns(df, cols):
    return df.reindex(columns=cols).copy()


def _as_str_list(x):
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    try:
        return [str(v) for v in list(x)]
    except TypeError:
        return []


def infer_raw_input_columns(model, df_fallback):
    pipe = getattr(model, "pipeline", None)

    if pipe is not None:
        pipe_in = getattr(pipe, "feature_names_in_", None)
        cols = _as_str_list(pipe_in)
        if cols:
            return cols

        named_steps = getattr(pipe, "named_steps", None)
        if isinstance(named_steps, dict) and "cleaner" in named_steps:
            cleaner = named_steps["cleaner"]
            cleaner_in = getattr(cleaner, "feature_names_in_", None)
            cols = _as_str_list(cleaner_in)
            if cols:
                return cols

    imp = getattr(model, "imputation_strategy", None)
    if imp == "passthrough":
        cols = _as_str_list(getattr(model, "feature_names_", None))
        if cols:
            return cols

    return [
        c for c in df_fallback.columns if c not in [config.EVENT_COL, config.TIME_COL]
    ]


def parse_missing_columns_from_error(msg):
    m = MISSING_COLS_RE.search(msg)
    if not m:
        return []
    txt = m.group(1)
    inner = txt.strip().lstrip("{").rstrip("}").strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    cols = []
    for p in parts:
        p = p.strip()
        if p.startswith("'") and p.endswith("'"):
            p = p[1:-1]
        if p.startswith('"') and p.endswith('"'):
            p = p[1:-1]
        if p:
            cols.append(p)
    return cols


def predict_with_missing_column_retry(model, X, max_retries=2):
    X_work = X.copy()

    for attempt in range(max_retries + 1):
        try:
            return np.asarray(model.predict(X_work))
        except Exception as e:
            missing = parse_missing_columns_from_error(str(e))
            if not missing or attempt >= max_retries:
                raise
            for col in missing:
                if col not in X_work.columns:
                    X_work[col] = np.nan
            X_work = X_work.copy()

    return np.asarray([])


def main():
    parser = argparse.ArgumentParser(
        description="List C-index scores for all saved survival model artifacts."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=config.SURVIVAL_MODELS_DIR,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        choices=["test", "dev"],
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    data_path = config.TEST_DATA_PATH if args.data == "test" else config.TRAIN_DATA_PATH
    df = pd.read_csv(data_path)

    if "in_pacap" in df.columns:
        pacap_mask = df["in_pacap"].astype(str).str.lower().eq("yes")
    else:
        pacap_mask = pd.Series(False, index=df.index)

    event_all = df[config.EVENT_COL].astype(bool).to_numpy()
    time_all = df[config.TIME_COL].astype(float).to_numpy()

    df_p = df.loc[pacap_mask].copy()
    event_p = df_p[config.EVENT_COL].astype(bool).to_numpy()
    time_p = df_p[config.TIME_COL].astype(float).to_numpy()

    files = sorted(f for f in os.listdir(args.artifacts_dir) if f.endswith(".pkl"))
    rows = []

    for filename in files:
        parsed = parse_artifact_name(filename)
        if parsed is None:
            continue
        if args.run_id is not None and parsed.run_id != args.run_id:
            continue

        path = os.path.join(args.artifacts_dir, filename)
        try:
            model = load_pickle(path)
        except Exception as e:
            rows.append(
                {
                    "ModelFile": filename,
                    "Run_ID": parsed.run_id,
                    "Dataset": parsed.dataset,
                    "Feature_Set": parsed.feature_set,
                    "Strategy": parsed.strategy,
                    "Fold": parsed.fold,
                    "C-Index (ALL)": np.nan,
                    "C-Index (PACAP)": np.nan,
                    "Error": f"load: {e}",
                }
            )
            continue

        input_cols = infer_raw_input_columns(model, df)

        X_all = coerce_pandas_string_to_object(ensure_columns(df, input_cols))
        X_p = coerce_pandas_string_to_object(ensure_columns(df_p, input_cols))

        try:
            risk_all = predict_with_missing_column_retry(model, X_all)
            risk_p = (
                predict_with_missing_column_retry(model, X_p)
                if len(X_p) > 0
                else np.array([])
            )
            err = ""
        except Exception as e:
            risk_all = np.array([])
            risk_p = np.array([])
            err = f"predict: {e}"

        rows.append(
            {
                "ModelFile": filename,
                "Run_ID": parsed.run_id,
                "Dataset": parsed.dataset,
                "Feature_Set": parsed.feature_set,
                "Strategy": parsed.strategy,
                "Fold": parsed.fold,
                "C-Index (ALL)": safe_cindex(event_all, time_all, risk_all),
                "C-Index (PACAP)": safe_cindex(event_p, time_p, risk_p),
                "Error": err,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["Run_ID", "Dataset", "Feature_Set", "Strategy", "Fold", "ModelFile"],
        na_position="last",
    )

    view_cols = [
        "ModelFile",
        "C-Index (ALL)",
        "C-Index (PACAP)",
        "Error",
    ]
    print(out[view_cols].to_string(index=False))

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        out.to_csv(args.csv_out, index=False)
        print(f"\nSaved CSV: {args.csv_out}")


if __name__ == "__main__":
    main()