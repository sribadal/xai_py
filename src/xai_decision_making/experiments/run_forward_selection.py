"""
Forward feature selection for survival models.

Starts from the 5 DPCG-true features and greedily adds the best
pretreatment feature at each step by C-index on a dev split. After
each step, retrains on the full dev set and reports C-index on
test and black-envelope sets. Optionally extends with treatment features.
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings
from typing import List

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


START_FEATURES = ["leeft_cat", "perf_stat", "ca199", "diffgrad", "ct"]

PRETREATMENT_POOL = [
    "cci", "bilirubine", "albumine", "ldh", "crp",
    "cci_cat", "klin_tum_afm", "cm", "cstadium",
    "veneus_vaatbetr", "arterieel_vaatbetr",
    "meta_lever", "meta_bijnier", "meta_bot", "meta_hersenen",
    "meta_long", "meta_lymf", "meta_perit", "meta_overig",
    "gesl", "seizoen", "diag_basis", "morf", "meta_lok_aantal",
    "qol_summary_pretreatment", "qol_physical_pretreatment",
    "qol_emotional_pretreatment", "qol_social_pretreatment",
    "qol_cognitive_pretreatment", "qol_role_pretreatment",
]

TREATMENT_FEATURES = ["tumgericht_ther", "chir", "destr", "rt", "chemo"]


def _coerce(df):
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.StringDtype):
            out[c] = out[c].astype(object)
    return out


def _cindex(event, time, risk):
    if risk is None or len(risk) == 0:
        return float("nan")
    return float(concordance_index_censored(event, time, risk)[0])


def _train_and_eval(X_train, y_train, test_sets, features, alpha):
    model = SurvivalModelWrapper(
        imputation_strategy="passthrough",
        alpha=alpha,
        **config.RANDOM_FOREST_PARAMS,
    )
    model.fit(X_train[features].reset_index(drop=True), y_train)
    results = {}
    for name, (X_te, y_te) in test_sets.items():
        risk = model.predict(X_te[features].reset_index(drop=True))
        results[name] = _cindex(y_te["event"], y_te["time"], risk)
    return results


def _quick_eval(X, y, features, split_indices, alpha):
    tr_idx, te_idx = split_indices
    model = SurvivalModelWrapper(
        imputation_strategy="passthrough",
        alpha=alpha,
        **config.RANDOM_FOREST_PARAMS,
    )
    model.fit(X[features].iloc[tr_idx].reset_index(drop=True), y[tr_idx])
    risk = model.predict(X[features].iloc[te_idx].reset_index(drop=True))
    return _cindex(y[te_idx]["event"], y[te_idx]["time"], risk)


def main():
    parser = argparse.ArgumentParser(description="Forward feature selection")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--single-split", action="store_true",
        help="Use a single 80/20 dev split for selection (much faster).",
    )
    parser.add_argument(
        "--include-treatment", action="store_true",
        help="Also consider treatment features in the candidate pool.",
    )
    parser.add_argument("--run-id", type=str, default=config.TIMESTAMP)
    args = parser.parse_args()

    if concordance_index_censored is None:
        raise ImportError("scikit-survival required.")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    dev_df = pd.read_csv(config.TRAIN_DATA_PATH, low_memory=False)
    test_df = pd.read_csv(config.TEST_DATA_PATH, low_memory=False)

    black_df = None
    if os.path.exists(config.BLACK_ENVELOP_TEST_PATH):
        black_df = pd.read_csv(config.BLACK_ENVELOP_TEST_PATH, low_memory=False)

    available_cols = set(dev_df.columns)

    all_candidate_features = list(START_FEATURES)
    pool = list(PRETREATMENT_POOL)
    prior_results = []

    if args.include_treatment:
        pattern = os.path.join(config.RESULTS_DIR, "forward_selection_pretreatment_only*.csv")
        prior_csvs = sorted(glob.glob(pattern))
        if prior_csvs:
            prior_df = pd.read_csv(prior_csvs[-1])
            last_row = prior_df.iloc[-1]
            all_candidate_features = [f.strip() for f in last_row["features"].split(",")]
            prior_results = prior_df.to_dict("records")
            print("Resuming from: {}".format(prior_csvs[-1]))
            print("Starting with {} features".format(len(all_candidate_features)))
            pool = TREATMENT_FEATURES
        else:
            print("No pretreatment results found, running full selection with treatment.")
            pool += TREATMENT_FEATURES

    pool = [f for f in pool if f in available_cols and f not in all_candidate_features]
    all_possible = all_candidate_features + pool

    X_dev, y_dev = prepare_survival_data(dev_df, all_possible)
    X_dev = _coerce(X_dev)

    X_test, y_test = prepare_survival_data(test_df, all_possible)
    X_test = _coerce(X_test)

    test_sets = {"test": (X_test, y_test)}

    if black_df is not None:
        X_black, y_black = prepare_survival_data(black_df, all_possible)
        X_black = _coerce(X_black)
        test_sets["black_envelop"] = (X_black, y_black)

    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X_dev))
    tr_idx, te_idx = train_test_split(
        indices, test_size=0.2, random_state=config.RANDOM_STATE,
    )
    split_indices = (tr_idx, te_idx)

    current_features = list(all_candidate_features)
    remaining = list(pool)

    dev_c = _quick_eval(X_dev, y_dev, current_features, split_indices, args.alpha)
    test_scores = _train_and_eval(X_dev, y_dev, test_sets, current_features, args.alpha)

    results = list(prior_results) if prior_results else []
    start_step = len(results)

    row = {
        "step": start_step,
        "n_features": len(current_features),
        "added_feature": "(resume)" if prior_results else "(start)",
        "dev_split_cindex": dev_c,
        "test_cindex": test_scores.get("test", float("nan")),
        "black_envelop_cindex": test_scores.get("black_envelop", float("nan")),
        "features": ", ".join(current_features),
    }
    results.append(row)

    has_black = "black_envelop" in test_sets
    if has_black:
        hdr = "{:<6} {:<30} {:>5} {:>12} {:>12} {:>12}".format(
            "Step", "Added feature", "N", "Dev split", "Test", "Black env")
        rfmt = "{:<6} {:<30} {:>5} {:>12.4f} {:>12.4f} {:>12.4f}"
    else:
        hdr = "{:<6} {:<30} {:>5} {:>12} {:>12}".format(
            "Step", "Added feature", "N", "Dev split", "Test")
        rfmt = "{:<6} {:<30} {:>5} {:>12.4f} {:>12.4f}"

    print("\n" + "=" * 80)
    print("FORWARD FEATURE SELECTION (MA, alpha={})".format(args.alpha))
    print("=" * 80)
    print(hdr)
    print("-" * 80)

    label = "(resume)" if prior_results else "(start)"
    if has_black:
        print(rfmt.format(start_step, label, len(current_features),
              dev_c, test_scores["test"], test_scores["black_envelop"]))
    else:
        print(rfmt.format(start_step, label, len(current_features),
              dev_c, test_scores["test"]))

    step = start_step
    while remaining:
        step += 1
        best_feat = None
        best_dev_c = -1.0

        for candidate in remaining:
            trial = current_features + [candidate]
            c = _quick_eval(X_dev, y_dev, trial, split_indices, args.alpha)
            if c > best_dev_c:
                best_dev_c = c
                best_feat = candidate

        current_features.append(best_feat)
        remaining.remove(best_feat)

        test_scores = _train_and_eval(X_dev, y_dev, test_sets, current_features, args.alpha)

        row = {
            "step": step,
            "n_features": len(current_features),
            "added_feature": best_feat,
            "dev_split_cindex": best_dev_c,
            "test_cindex": test_scores.get("test", float("nan")),
            "black_envelop_cindex": test_scores.get("black_envelop", float("nan")),
            "features": ", ".join(current_features),
        }
        results.append(row)

        if has_black:
            print(rfmt.format(step, best_feat, len(current_features),
                  best_dev_c, test_scores["test"], test_scores["black_envelop"]))
        else:
            print(rfmt.format(step, best_feat, len(current_features),
                  best_dev_c, test_scores["test"]))

    print("\n" + "=" * 80)
    print("DONE. {} to {} features.".format(len(all_candidate_features), len(current_features)))

    suffix = "_with_treatment" if args.include_treatment else "_pretreatment_only"
    out_csv = os.path.join(
        config.RESULTS_DIR,
        "forward_selection{}_run{}.csv".format(suffix, args.run_id),
    )
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Saved: {}".format(out_csv))


if __name__ == "__main__":
    main()