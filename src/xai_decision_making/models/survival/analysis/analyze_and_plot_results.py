"""
Comparison plots and statistical significance maps for survival models.

Generates per-dataset, per-feature-set bar charts of mean C-index with
standard deviation, alongside paired t-test significance heatmaps across
imputation strategies. Supports filtering by Run_ID.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from xai_decision_making.models.survival import config_survival as config

sns.set_palette("husl")


def _pick_latest_run_id(summary_df):
    if "Run_ID" not in summary_df.columns:
        return None
    runs = summary_df["Run_ID"].dropna().astype(str).unique().tolist()
    if not runs:
        return None
    return sorted(runs)[-1]


def _filter_run_id(df, run_id):
    if df is None or run_id is None:
        return df
    if "Run_ID" not in df.columns:
        return df
    return df[df["Run_ID"].astype(str) == str(run_id)].copy()


def _paired_scores_by_fold(df, strategy, metric):
    sub = df[df["Strategy"] == strategy].copy()
    if "Fold" not in sub.columns:
        return pd.Series(dtype=float)
    if metric not in sub.columns:
        return pd.Series(dtype=float)

    sub = sub[["Fold", metric]].dropna()
    if len(sub) == 0:
        return pd.Series(dtype=float)

    sub["Fold"] = pd.to_numeric(sub["Fold"], errors="coerce")
    sub = sub.dropna(subset=["Fold"])
    sub["Fold"] = sub["Fold"].astype(int)

    ser = sub.groupby("Fold")[metric].mean()
    ser = ser.sort_index()
    return ser


def plot_comparison(summary_df, detailed_df, metric, output_path, title_suffix=""):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary_df = summary_df.copy()
    detailed_df = detailed_df.copy() if detailed_df is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metric_mean_col = f"{metric}_mean"
    metric_std_col = f"{metric}_std"

    ax1 = axes[0]
    if metric_mean_col in summary_df.columns:
        plot_df = summary_df.sort_values(metric_mean_col, ascending=False)
        strategies = plot_df["Strategy"].astype(str).values
        means = plot_df[metric_mean_col].values.astype(float)
        stds = (
            plot_df[metric_std_col].values.astype(float)
            if metric_std_col in plot_df.columns
            else np.zeros_like(means)
        )

        x_pos = np.arange(len(strategies))
        bars = ax1.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.2,
        )

        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax1.set_xlabel("Strategy", fontsize=12, fontweight="bold")
        ax1.set_ylabel(metric, fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Model Comparison (Mean ± Std){title_suffix}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(strategies, rotation=45, ha="right", fontsize=10)
        ax1.grid(True, alpha=0.3, axis="y")

        if len(means) > 0:
            ax1.set_ylim([0, max(means) * 1.2])

        for i, (mean, std) in enumerate(zip(means, stds)):
            ax1.text(
                i,
                mean + std + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    else:
        ax1.text(
            0.5,
            0.5,
            f"No data for {metric}",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
        )

    ax2 = axes[1]
    if detailed_df is not None and metric in detailed_df.columns:
        strategies_list = detailed_df["Strategy"].astype(str).unique()
        n = len(strategies_list)

        if n > 1:
            pvals = np.ones((n, n), dtype=float)
            labels = np.empty((n, n), dtype=object)
            labels[:] = ""

            fold_series = {
                s: _paired_scores_by_fold(detailed_df, s, metric)
                for s in strategies_list
            }

            for i, s1 in enumerate(strategies_list):
                for j, s2 in enumerate(strategies_list):
                    if i == j:
                        continue

                    a = fold_series.get(s1, pd.Series(dtype=float))
                    b = fold_series.get(s2, pd.Series(dtype=float))

                    if len(a) > 0 and len(b) > 0:
                        common = a.index.intersection(b.index)
                        a_al = a.loc[common].dropna()
                        b_al = b.loc[common].dropna()
                        common2 = a_al.index.intersection(b_al.index)
                        a_al = a_al.loc[common2]
                        b_al = b_al.loc[common2]

                        if len(a_al) > 1:
                            try:
                                _, p = stats.ttest_rel(
                                    a_al.to_numpy(), b_al.to_numpy()
                                )
                                pvals[i, j] = float(p)
                                if p < 0.001:
                                    labels[i, j] = "***"
                                elif p < 0.01:
                                    labels[i, j] = "**"
                                elif p < 0.05:
                                    labels[i, j] = "*"
                                else:
                                    labels[i, j] = "ns"
                            except Exception:
                                labels[i, j] = "N/A"
                        else:
                            labels[i, j] = "N/A"
                    else:
                        labels[i, j] = "N/A"

            mask = np.eye(n, dtype=bool)
            pvals_masked = np.ma.masked_array(pvals, mask=mask)

            im = ax2.imshow(
                pvals_masked,
                cmap="RdYlGn_r",
                aspect="auto",
                vmin=0,
                vmax=0.05,
            )

            for i in range(n):
                for j in range(n):
                    if i != j:
                        ax2.text(
                            j,
                            i,
                            labels[i, j],
                            ha="center",
                            va="center",
                            color="black",
                            fontweight="bold",
                            fontsize=10,
                        )

            ax2.set_xticks(np.arange(n))
            ax2.set_yticks(np.arange(n))
            ax2.set_xticklabels(strategies_list, rotation=45, ha="right", fontsize=9)
            ax2.set_yticklabels(strategies_list, fontsize=9)

            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label("p-value", rotation=270, labelpad=15, fontsize=10)

            ax2.set_title(
                f"Statistical Significance (Pairwise Tests){title_suffix}",
                fontsize=14,
                fontweight="bold",
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "Not enough strategies for comparison",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "No detailed data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from survival model results."
    )
    parser.add_argument("--summary-file", type=str, default=None)
    parser.add_argument("--detailed-file", type=str, default=None)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Only plot a specific Run_ID (if present in CSVs).",
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="If set and Run_ID exists, automatically use the latest Run_ID.",
    )
    args = parser.parse_args()

    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    if args.summary_file is None:
        files = glob.glob(os.path.join(config.RESULTS_DIR, "*.csv"))
        files = [f for f in files if "detailed" not in f.lower()]
        if not files:
            raise FileNotFoundError("No summary result CSV found in RESULTS_DIR.")
        args.summary_file = max(files, key=os.path.getctime)

    summary_df = pd.read_csv(args.summary_file)

    detailed_df = None
    if args.detailed_file is not None:
        detailed_df = pd.read_csv(args.detailed_file)
    else:
        detailed_files = glob.glob(os.path.join(config.RESULTS_DIR, "*detailed*.csv"))
        if detailed_files:
            detailed_df = pd.read_csv(max(detailed_files, key=os.path.getctime))

    run_id = args.run_id
    if run_id is None and args.latest_run:
        run_id = _pick_latest_run_id(summary_df)

    if run_id is not None:
        summary_df = _filter_run_id(summary_df, run_id)
        detailed_df = _filter_run_id(detailed_df, run_id)
        run_suffix = f"\nRun: {run_id}"
    else:
        run_suffix = ""

    for dataset in summary_df["Dataset"].unique():
        ds_sum = summary_df[summary_df["Dataset"] == dataset]
        ds_det = (
            detailed_df[detailed_df["Dataset"] == dataset]
            if detailed_df is not None
            else None
        )

        for feat in ds_sum["Feature_Set"].unique():
            feat_sum = ds_sum[ds_sum["Feature_Set"] == feat]
            feat_det = (
                ds_det[ds_det["Feature_Set"] == feat] if ds_det is not None else None
            )

            suffix = f"\n{dataset} | {feat}{run_suffix}"

            out1 = os.path.join(
                config.PLOTS_DIR,
                f"{dataset.replace(' ', '_')}_{feat.replace(' ', '_')}_PACAP.png",
            )
            out2 = os.path.join(
                config.PLOTS_DIR,
                f"{dataset.replace(' ', '_')}_{feat.replace(' ', '_')}_ALL.png",
            )

            if "C-Index (PACAP)_mean" in feat_sum.columns:
                plot_comparison(feat_sum, feat_det, "C-Index (PACAP)", out1, suffix)

            if "C-Index (ALL)_mean" in feat_sum.columns:
                plot_comparison(feat_sum, feat_det, "C-Index (ALL)", out2, suffix)


if __name__ == "__main__":
    main()