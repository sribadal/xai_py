"""
External test-set evaluation for saved survival model artifacts.

Loads all .pkl model artifacts from the artifacts directory, infers
their expected input columns, predicts risk scores on the black-envelope
test set (overall and PACAP-only subsets), and computes C-index. Outputs
both per-fold detailed results and strategy-level summary CSVs.
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


RUN_RE = re.compile(r"_run(?P<run_id>\d{8}_\d{6})$")
FOLD_RE = re.compile(r"_fold(?P<fold>\d+)$")
MISSING_COLS_RE = re.compile(r"columns are missing:\s*(\{.*\})")


@dataclass(frozen=True)
class ParsedName:
    strategy: str
    dataset: str
    dataset_token: str
    feature_set_raw: str
    feature_set_norm: str
    fold: int
    run_id: str | None


def normalize_feature_set(name):
    return str(name).strip().lower()


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
        stem = stem[: m_run.start()]

    if "_Small_(PACAP)_" in stem:
        dataset_token = "Small_(PACAP)"
        dataset_label = "Small (PACAP)"
    elif "_Large_(All)_" in stem:
        dataset_token = "Large_(All)"
        dataset_label = "Large (All)"
    else:
        return None

    try:
        strategy, feature_set_raw = stem.split(f"_{dataset_token}_", 1)
    except ValueError:
        return None

    feature_set_norm = normalize_feature_set(feature_set_raw)

    return ParsedName(
        strategy=strategy,
        dataset=dataset_label,
        dataset_token=dataset_token,
        feature_set_raw=feature_set_raw,
        feature_set_norm=feature_set_norm,
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


def predict_with_missing_column_retry(model, X, filename="", max_retries=2):
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
            print(
                f"Retrying {filename}: added {len(missing)} missing column(s), "
                "retrying..."
            )

    return np.asarray([])


def pick_latest_run_id(parsed_items):
    run_ids = sorted({p.run_id for p in parsed_items if p.run_id is not None})
    if not run_ids:
        return None
    return run_ids[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved survival model artifacts on the external test set."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=config.SURVIVAL_MODELS_DIR,
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
    )
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    test_df = pd.read_csv(config.TEST_DATA_PATH)

    if "in_pacap" in test_df.columns:
        pacap_mask = test_df["in_pacap"].astype(str).str.lower().eq("yes")
    else:
        pacap_mask = pd.Series(False, index=test_df.index)

    event_all = test_df[config.EVENT_COL].astype(bool).to_numpy()
    time_all = test_df[config.TIME_COL].astype(float).to_numpy()

    test_pacap_df = test_df.loc[pacap_mask].copy()
    event_p = test_pacap_df[config.EVENT_COL].astype(bool).to_numpy()
    time_p = test_pacap_df[config.TIME_COL].astype(float).to_numpy()

    filenames = sorted(
        f for f in os.listdir(args.artifacts_dir) if f.endswith(".pkl")
    )
    if not filenames:
        raise FileNotFoundError(f"No .pkl artifacts found in {args.artifacts_dir}")

    parsed = []
    for fn in filenames:
        p = parse_artifact_name(fn)
        if p is not None:
            parsed.append((fn, p))

    if not parsed:
        raise RuntimeError("No artifacts matched the expected naming patterns.")

    feature_set_norm = (
        normalize_feature_set(args.feature_set)
        if args.feature_set is not None
        else None
    )

    if feature_set_norm is not None:
        parsed = [(fn, p) for fn, p in parsed if p.feature_set_norm == feature_set_norm]

    if not parsed:
        raise RuntimeError(
            "No artifacts left after filtering. Check --feature-set and filenames."
        )

    selected_run_id = args.run_id
    if selected_run_id is None and args.latest_run:
        latest = pick_latest_run_id([p for _, p in parsed])
        if latest is not None:
            selected_run_id = latest

    if selected_run_id is not None:
        parsed = [(fn, p) for fn, p in parsed if p.run_id == selected_run_id]

    if not parsed:
        raise RuntimeError(
            "No artifacts left after applying run filter. Check --run-id/--latest-run."
        )

    detailed_rows = []

    for filename, p in parsed:
        path = os.path.join(args.artifacts_dir, filename)

        try:
            model = load_pickle(path)
        except Exception as e:
            print(f"Failed loading {filename}: {e}")
            detailed_rows.append(
                {
                    "Model": f"{p.strategy}_{p.dataset}_{p.feature_set_norm}",
                    "Run_ID": p.run_id,
                    "ModelFile": filename,
                    "Strategy": p.strategy,
                    "Dataset": p.dataset,
                    "Feature_Set": p.feature_set_norm,
                    "Fold": p.fold,
                    "C-Index (ALL)": np.nan,
                    "C-Index (PACAP)": np.nan,
                }
            )
            continue

        input_cols = infer_raw_input_columns(model, test_df)
        X_all = coerce_pandas_string_to_object(ensure_columns(test_df, input_cols))
        X_p = coerce_pandas_string_to_object(
            ensure_columns(test_pacap_df, input_cols)
        )

        try:
            risk_all = predict_with_missing_column_retry(
                model, X_all, filename=filename
            )
            risk_p = (
                predict_with_missing_column_retry(model, X_p, filename=filename)
                if len(X_p) > 0
                else np.array([])
            )
        except Exception as e:
            print(f"Failed predicting for {filename}: {e}")
            risk_all = np.array([])
            risk_p = np.array([])

        detailed_rows.append(
            {
                "Model": f"{p.strategy}_{p.dataset}_{p.feature_set_norm}",
                "Run_ID": p.run_id,
                "ModelFile": filename,
                "Strategy": p.strategy,
                "Dataset": p.dataset,
                "Feature_Set": p.feature_set_norm,
                "Fold": p.fold,
                "C-Index (ALL)": safe_cindex(event_all, time_all, risk_all),
                "C-Index (PACAP)": safe_cindex(event_p, time_p, risk_p),
            }
        )

    detailed_df = pd.DataFrame(detailed_rows).sort_values(
        ["Run_ID", "Dataset", "Feature_Set", "Strategy", "Fold", "ModelFile"]
    )

    ts = config.TIMESTAMP
    fs_tag = feature_set_norm if feature_set_norm is not None else "all_featuresets"
    run_tag = selected_run_id if selected_run_id is not None else "all_runs"

    detailed_path = os.path.join(
        config.RESULTS_DIR,
        f"overall_survival_model_results_{ts}_{fs_tag}_{run_tag}_detailed_folds.csv",
    )
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Saved detailed results: {detailed_path}")

    group_cols = ["Dataset", "Feature_Set", "Strategy", "Run_ID"]
    summary_rows = []

    for (ds, fs, st, run_id), g in detailed_df.groupby(group_cols, dropna=False):
        out = {
            "Dataset": ds,
            "Feature_Set": fs,
            "Strategy": st,
            "Run_ID": run_id,
        }
        for metric in ["C-Index (PACAP)", "C-Index (ALL)"]:
            vals = g[metric].dropna().to_numpy()
            out[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            out[f"{metric}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        summary_rows.append(out)

    summary_df = pd.DataFrame(summary_rows).sort_values(group_cols)

    summary_path = os.path.join(
        config.RESULTS_DIR,
        f"overall_survival_model_results_{ts}_{fs_tag}_{run_tag}.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary results: {summary_path}")


if __name__ == "__main__":
    main()