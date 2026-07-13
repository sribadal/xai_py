"""Outlier checks and descriptive group comparisons for Study 2."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xai_decision_making.analysis.common import (load, save, OUT, MODALITY_ORDER, MODEL_ORDER, Q_ORDER,
                    RANK_IDS, PAL, LIKERT_LABELS, DIV_COLORS, HEATMAP_CMAP)


def _profiles(lik, demo):
    pids = demo["pid"].tolist()
    ncols = min(3, len(pids))
    nrows = -(-len(pids) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for i, pid in enumerate(pids):
        ax = axes[i]
        sub = lik[lik["pid"] == pid]
        for mod in MODALITY_ORDER:
            vals = [sub[(sub["modality"] == mod) & (sub["q_short"] == q)]["score"].mean()
                    for q in Q_ORDER]
            ax.plot(Q_ORDER, vals, "o-", color=PAL[mod],
                    label=mod if i == 0 else "", markersize=6)
        ax.set_title(f"P{i+1}", fontsize=10)
        ax.set_ylim(0.5, 5.5)
        ax.axhline(3, color="grey", ls=":", alpha=0.4)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    for j in range(len(pids), len(axes)):
        axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Individual response profiles", fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    save(fig, "fig_profiles.png")


def _groups(lik, demo):
    merged = lik.merge(demo[["pid", "experience", "ai_familiarity"]], on="pid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, col, title in zip(axes, ["experience", "ai_familiarity"],
                              ["By experience", "By AI familiarity"]):
        summary = (merged.groupby([col, "modality"])["score"].mean()
                   .unstack()[MODALITY_ORDER])
        summary.plot(kind="bar", ax=ax, color=[PAL[m] for m in MODALITY_ORDER],
                     edgecolor="white", rot=0)
        ax.set_ylabel("Mean rating")
        ax.set_ylim(1, 5.2)
        ax.axhline(3, color="grey", ls=":", alpha=0.4)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Descriptive group comparisons",
             fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig_groups.png")


def _participant_table(lik, demo):
    rows = []
    for i, pid in enumerate(demo["pid"].tolist()):
        sub = lik[lik["pid"] == pid]
        means = sub.groupby("modality")["score"].mean()
        rows.append({
            "participant": f"P{i+1}",
            "mean": sub["score"].mean(),
            "sd": sub["score"].std(),
            "highest": means.idxmax(),
            "lowest": means.idxmin(),
            "range": means.max() - means.min(),
        })
    df = pd.DataFrame(rows)
    df.round(2).to_csv(OUT / "table_participant_profiles.csv", index=False)
    return df


def build():
    d = load()
    _profiles(d.likert, d.demo)
    _groups(d.likert, d.demo)
    _participant_table(d.likert, d.demo)


if __name__ == "__main__":
    build()