"""Descriptive tables and figures for the survey, and the plain-text report."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from xai_decision_making.analysis.common import (
    QUANTITATIVE_DIR, MODALITY_ORDER, MODEL_ORDER, DIMENSION_ORDER, RANKINGS, PAL,
    LIKERT_LABELS, DIV_COLORS, participant_means, participant_number, rank_positions,
    save_figure)

FIGURES = QUANTITATIVE_DIR / "figures"

MEASURE_ORDER = DIMENSION_ORDER + list(RANKINGS.values())
MEASURE_LABELS = {"Understandability": "Understand-\nability",
                  "Critical Assessment": "Critical\nassessment",
                  "Actionability": "Action-\nability",
                  "Overall preference": "Overall\npreference",
                  "Patient communication": "Patient\ncommunication"}


# --- Tables ------------------------------------------------------------------

def participants_table(data):
    demo = data.demo.reset_index(drop=True)
    table = demo[["pid", "role_clean", "experience", "ai_familiarity",
                  "specialty", "version"]].copy()
    table.columns = ["Participant", "Role", "Experience", "AI familiarity",
                     "Specialty", "Interface"]
    return table


def descriptives(likert):
    """Means at two levels: all ratings, and participant means.

    The tests use participant means, so the SD that belongs next to them is the
    participant-level one. The rating-level SD also contains the difference
    between each participant's two models.
    """
    rating = (likert.groupby(["modality", "q_short"])["score"]
              .agg(mean_rating="mean", median_rating="median",
                   sd_rating="std", n_ratings="count"))
    participant = (participant_means(likert).groupby(["modality", "q_short"])["score"]
                   .agg(mean_participant="mean", median_participant="median",
                        sd_participant="std", n_participants="count"))
    table = (rating.join(participant).round(3)
             .reindex([(m, q) for m in MODALITY_ORDER for q in DIMENSION_ORDER]))
    return table.rename_axis(["modality", "dimension"]).reset_index()


def borda(ranks, prefix):
    """Borda points (3 for first place down to 0) and mean rank per type."""
    positions = rank_positions(ranks, prefix)
    out = {}
    for modality in MODALITY_ORDER:
        p = positions.loc[positions["modality"] == modality, "position"]
        out[modality] = {"borda": int((4 - p).sum()),
                         "mean_rank": p.mean() if len(p) else np.nan, "n": len(p)}
    return out


def rankings_table(ranks):
    rows = []
    for prefix, label in RANKINGS.items():
        for modality, r in borda(ranks, prefix).items():
            rows.append({"ranking": label, "modality": modality, "borda": r["borda"],
                         "mean_rank": round(r["mean_rank"], 2), "n": r["n"]})
    return pd.DataFrame(rows)


def positions_table(likert, ranks):
    """Where each type places on every measure (1 = best, ties averaged)."""
    table = pd.DataFrame(index=MODALITY_ORDER, columns=MEASURE_ORDER, dtype=float)
    means = participant_means(likert)
    for dimension in DIMENSION_ORDER:
        table[dimension] = (means[means["q_short"] == dimension]
                            .groupby("modality")["score"].mean()
                            .rank(ascending=False, method="average"))
    for prefix, label in RANKINGS.items():
        mean_rank = pd.Series({m: r["mean_rank"] for m, r in borda(ranks, prefix).items()})
        table[label] = mean_rank.rank(ascending=True, method="average")
    return table


def participant_profiles(likert, demo):
    rows = []
    for pid in demo["pid"]:
        ratings = likert[likert["pid"] == pid]
        means = ratings.groupby("modality")["score"].mean()
        rows.append({"participant": pid, "mean": ratings["score"].mean(),
                     "sd": ratings["score"].std(), "highest": means.idxmax(),
                     "lowest": means.idxmin(), "range": means.max() - means.min()})
    return pd.DataFrame(rows).round(2)


# --- Figures -----------------------------------------------------------------

def _clean_axes(ax, spines=("top", "right")):
    ax.spines[list(spines)].set_visible(False)


def figure_demographics(data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column, title in zip(
            axes, ["experience", "ai_familiarity", "role"],
            ["Years of experience", "AI / prediction tool familiarity", "Professional role"]):
        counts = data.demo[column].value_counts()
        ax.barh(counts.index, counts.values, color="#2D6A4F", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.invert_yaxis()
        _clean_axes(ax)
        for i, v in enumerate(counts.values):
            ax.text(v + 0.05, i, str(v), va="center", fontweight="bold")
    fig.suptitle(f"Participant background (N={data.N})", fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_demographics")


def figure_diverging(likert):
    """Share of each answer per type, centred on Neutral."""
    fig, axes = plt.subplots(len(DIMENSION_ORDER), 1, figsize=(12, 9), sharex=True)
    for ax, dimension in zip(axes, DIMENSION_ORDER):
        answers = likert[likert["q_short"] == dimension]
        for y, modality in enumerate(MODALITY_ORDER):
            counts = answers[answers["modality"] == modality]["answer_label"].value_counts()
            total = counts.sum() or 1
            shares = [counts.get(label, 0) / total * 100 for label in LIKERT_LABELS]
            left = -(shares[0] + shares[1] + shares[2] / 2)
            for i, (share, colour) in enumerate(zip(shares, DIV_COLORS)):
                ax.barh(y, share, left=left, color=colour, edgecolor="white", height=0.55)
                if share >= 8:
                    ax.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if i in (0, 1, 4) else "#333")
                left += share
        ax.set_yticks(range(len(MODALITY_ORDER)))
        ax.set_yticklabels(MODALITY_ORDER)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(dimension, fontsize=11, fontweight="bold")
        _clean_axes(ax)
    axes[-1].set_xlabel("<- disagree (%)      agree (%) ->")
    legend = [Patch(facecolor=c, label=l) for c, l in zip(DIV_COLORS, LIKERT_LABELS)]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, FIGURES, "fig_ratings_diverging")


def figure_heatmap(likert):
    """Mean rating per type and dimension, one panel per model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for model, ax in zip(MODEL_ORDER, axes):
        grid = (likert[likert["model"] == model]
                .pivot_table(index="modality", columns="q_short", values="score",
                             aggfunc="mean")[DIMENSION_ORDER]
                .reindex(MODALITY_ORDER))
        sns.heatmap(grid, annot=True, fmt=".2f",
                    cmap=sns.light_palette("#1B4332", as_cmap=True),
                    vmin=1, vmax=5, linewidths=1.5, linecolor="white",
                    cbar_kws={"shrink": 0.7, "label": "Mean"}, ax=ax,
                    annot_kws={"fontsize": 12, "fontweight": "bold"})
        ax.set_title(model, pad=10, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(11)
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_ratings_heatmap")


def _type_axis(ax, dimension):
    ax.set_xticks(range(len(MODALITY_ORDER)))
    ax.set_xticklabels(MODALITY_ORDER, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.axhline(3, color="grey", ls=":", alpha=0.4)
    ax.set_title(dimension, fontsize=11, fontweight="bold")
    _clean_axes(ax)


def figure_strip(likert):
    """Each participant's mean per type, with the group mean as a bar."""
    means = participant_means(likert)
    fig, axes = plt.subplots(1, len(DIMENSION_ORDER), figsize=(15, 5), sharey=True)
    rng = np.random.default_rng(42)
    for ax, dimension in zip(axes, DIMENSION_ORDER):
        sub = means[means["q_short"] == dimension]
        for i, modality in enumerate(MODALITY_ORDER):
            values = sub[sub["modality"] == modality]["score"].values
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values,
                       color=PAL[modality], s=50, alpha=0.7, edgecolors="white",
                       linewidth=0.5, zorder=3)
            ax.plot([i - 0.2, i + 0.2], [values.mean()] * 2, color="black", lw=2, zorder=4)
        _type_axis(ax, dimension)
    axes[0].set_ylabel("Mean Likert Score (1-5)")
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_ratings_strip")


def figure_raincloud(likert):
    means = participant_means(likert)
    fig, axes = plt.subplots(1, len(DIMENSION_ORDER), figsize=(16, 5.5), sharey=True)
    for ax, dimension in zip(axes, DIMENSION_ORDER):
        sub = means[means["q_short"] == dimension]
        groups = [sub[sub["modality"] == m]["score"].values for m in MODALITY_ORDER]
        violins = ax.violinplot(groups, positions=range(len(MODALITY_ORDER)),
                                showextrema=False, widths=0.6)
        for i, body in enumerate(violins["bodies"]):
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.clip(vertices[:, 0], -np.inf, i)  # left half only
            body.set_facecolor(PAL[MODALITY_ORDER[i]])
            body.set_alpha(0.4)
        boxes = ax.boxplot(groups, positions=range(len(MODALITY_ORDER)), widths=0.15,
                           patch_artist=True, showfliers=False, zorder=2)
        for i, patch in enumerate(boxes["boxes"]):
            patch.set_facecolor(PAL[MODALITY_ORDER[i]])
            patch.set_alpha(0.8)
        for element in ["whiskers", "caps", "medians"]:
            for line in boxes[element]:
                line.set_color("black")
                line.set_linewidth(1)
        rng = np.random.default_rng(42)
        for i, (modality, values) in enumerate(zip(MODALITY_ORDER, groups)):
            ax.scatter(i + rng.uniform(0.15, 0.3, len(values)), values, color=PAL[modality],
                       s=35, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
        _type_axis(ax, dimension)
    axes[0].set_ylabel("Mean Likert Score (1-5)")
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_ratings_raincloud")


def figure_borda(ranks):
    fig, axes = plt.subplots(1, len(RANKINGS), figsize=(12, 5))
    for ax, (prefix, label) in zip(np.atleast_1d(axes), RANKINGS.items()):
        points = sorted(((m, r["borda"]) for m, r in borda(ranks, prefix).items()),
                        key=lambda kv: kv[1], reverse=True)
        names = [m for m, _ in points]
        values = [v for _, v in points]
        ax.barh(range(len(names)), values, color=[PAL[m] for m in names],
                edgecolor="white", height=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Borda score (higher = preferred)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        _clean_axes(ax)
        for i, v in enumerate(values):
            ax.text(v + 0.3, i, str(v), va="center", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_rankings_borda")


def figure_slopegraph(ranks):
    """Each participant's position on both ranking questions, one panel per type.

    Positions are whole numbers, so a small fixed jitter keeps identical lines
    apart. The black line is the unjittered mean.
    """
    labels, prefixes = list(RANKINGS.values()), list(RANKINGS.keys())
    positions = {label: rank_positions(ranks, prefix).set_index(["pid", "modality"])["position"]
                 for label, prefix in zip(labels, prefixes)}
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, len(MODALITY_ORDER), figsize=(7, 3.2), sharey=True)
    for ax, modality in zip(axes, MODALITY_ORDER):
        pids = sorted({p for p, m in positions[labels[0]].index if m == modality},
                      key=participant_number)
        left, right = [], []
        for pid in pids:
            try:
                a = positions[labels[0]].loc[(pid, modality)]
                b = positions[labels[1]].loc[(pid, modality)]
            except KeyError:
                continue
            left.append(a)
            right.append(b)
            jitter = rng.uniform(-0.12, 0.12, 2)
            ax.plot([0, 1], [a + jitter[0], b + jitter[1]], "-", color=PAL[modality],
                    alpha=0.45, lw=1.2, zorder=2)
            ax.scatter([0, 1], [a + jitter[0], b + jitter[1]], color=PAL[modality], s=14,
                       alpha=0.6, zorder=3, edgecolors="none")
        if left:
            ax.plot([0, 1], [np.mean(left), np.mean(right)], "-", color="black",
                    lw=2.2, zorder=4)
        ax.set_title(modality, fontsize=9, fontweight="bold")
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Overall", "Patient"], fontsize=8)
        ax.set_ylim(4.6, 0.4)
        ax.set_yticks([1, 2, 3, 4])
        ax.tick_params(length=0)
        _clean_axes(ax, ("top", "right", "bottom"))
    axes[0].set_ylabel("Rank position\n(1 = most preferred)", fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_rank_slopegraph")


def figure_positions(table):
    """Bump chart: does the type rated highest also get ranked highest?"""
    columns = np.arange(len(MEASURE_ORDER))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axvspan(-0.5, 2.5, color="#F6F8F7", zorder=0)
    for modality in MODALITY_ORDER:
        ax.plot(columns, table.loc[modality, MEASURE_ORDER].values.astype(float), "-o",
                color=PAL[modality], linewidth=2.0, markersize=7, markeredgecolor="white",
                markeredgewidth=1.2, zorder=3, label=modality)
    ax.set_xticks(columns)
    ax.set_xticklabels([MEASURE_LABELS[m] for m in MEASURE_ORDER], fontsize=8)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(4.6, 0.5)
    ax.set_xlim(-0.5, len(MEASURE_ORDER) - 0.5)
    ax.set_ylabel("Position (1 = highest)", fontsize=8)
    ax.tick_params(labelsize=8, length=0)
    ax.text(1.0, 0.62, "rated", fontsize=8, color="#777777", ha="center")
    ax.text(3.5, 0.62, "ranked", fontsize=8, color="#777777", ha="center")
    _clean_axes(ax, ("top", "right", "bottom"))
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_positions")


def figure_profiles(likert, demo):
    """One small panel per participant: mean per type on each dimension."""
    pids = demo["pid"].tolist()
    ncols = min(3, len(pids))
    nrows = -(-len(pids) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for i, pid in enumerate(pids):
        ax = axes[i]
        ratings = likert[likert["pid"] == pid]
        for modality in MODALITY_ORDER:
            values = [ratings[(ratings["modality"] == modality)
                              & (ratings["q_short"] == d)]["score"].mean()
                      for d in DIMENSION_ORDER]
            ax.plot(DIMENSION_ORDER, values, "o-", color=PAL[modality],
                    label=modality if i == 0 else "", markersize=6)
        ax.set_title(pid, fontsize=10)
        ax.set_ylim(0.5, 5.5)
        ax.axhline(3, color="grey", ls=":", alpha=0.4)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    for ax in axes[len(pids):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Individual response profiles", fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    save_figure(fig, FIGURES, "fig_profiles")


def figure_groups(likert, demo):
    """Mean rating per type by experience band and by AI familiarity."""
    merged = likert.merge(demo[["pid", "experience", "ai_familiarity"]], on="pid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, column, title in zip(axes, ["experience", "ai_familiarity"],
                                 ["By experience", "By AI familiarity"]):
        (merged.groupby([column, "modality"])["score"].mean().unstack()[MODALITY_ORDER]
         .plot(kind="bar", ax=ax, color=[PAL[m] for m in MODALITY_ORDER],
               edgecolor="white", rot=0))
        ax.set_ylabel("Mean rating")
        ax.set_ylim(1, 5.2)
        ax.axhline(3, color="grey", ls=":", alpha=0.4)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.set_xlabel("")
        _clean_axes(ax)
    fig.suptitle("Descriptive group comparisons", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_groups")


# --- Report ------------------------------------------------------------------

def write_report(data, tests):
    """A plain-text summary of the main results, written to report.txt."""
    likert = data.likert
    line, rule = "=" * 70, "-" * 70

    counts = likert.groupby("pid").size()
    counts = counts.reindex(sorted(counts.index, key=participant_number))
    completeness = "\n".join(f"  {pid}: {n}{'  <-- incomplete' if n != 24 else ''}"
                             for pid, n in counts.items())

    summary = (participant_means(likert).groupby(["modality", "q_short"])["score"]
               .agg(["mean", "median", "std", "count"]).round(2)
               .reindex([(m, q) for m in MODALITY_ORDER for q in DIMENSION_ORDER]))

    best_worst = []
    for dimension in DIMENSION_ORDER:
        means = likert[likert["q_short"] == dimension].groupby("modality")["score"].mean()
        best, worst = means.idxmax(), means.idxmin()
        best_worst.append(f"  {dimension}: best {best} ({means[best]:.2f}), "
                          f"worst {worst} ({means[worst]:.2f})")

    rankings = []
    for prefix, label in RANKINGS.items():
        rankings.append(f"\n{label}:")
        for modality, r in sorted(borda(data.ranks, prefix).items(),
                                  key=lambda kv: kv[1]["borda"], reverse=True):
            rankings.append(f"  {modality:16s} Borda={r['borda']:3d}  "
                            f"mean rank={r['mean_rank']:.2f}")

    wilcoxon = tests["wilcoxon"]
    parts = [
        line, "STUDY 2 - RESULTS SUMMARY", line,
        f"\nN = {data.N} participants, {len(likert)} Likert responses "
        f"(expected {data.N * 24} if complete).",
        "\nCompleteness (rows per participant, 24 = complete):",
        completeness,
        f"\n\n{rule}\nDescriptives by explanation type (participant level)\n{rule}",
        summary.to_string(),
        f"\n\n{rule}\nBest / worst type per dimension\n{rule}",
        "\n".join(best_worst),
        f"\n\n{rule}\nPreference rankings (Borda)\n{rule}",
        "\n".join(rankings),
        f"\n\n{rule}\nFriedman omnibus (per dimension, models averaged)\n{rule}",
        tests["friedman"].round(3).to_string(index=False),
        "\nAll tests are exploratory; interpret Kendall's W (effect size) alongside p.",
        f"\n\n{rule}\nPairwise Wilcoxon, Holm-corrected\n{rule}",
        (wilcoxon.round(3).to_string(index=False) if len(wilcoxon)
         else "No dimension reached Friedman significance; no pairwise tests run."),
        "",
    ]
    QUANTITATIVE_DIR.mkdir(parents=True, exist_ok=True)
    (QUANTITATIVE_DIR / "report.txt").write_text("\n".join(parts), encoding="utf-8")


def build(data):
    """Draw the figures and return the descriptive tables."""
    positions = positions_table(data.likert, data.ranks)
    figure_demographics(data)
    figure_diverging(data.likert)
    figure_heatmap(data.likert)
    figure_strip(data.likert)
    figure_raincloud(data.likert)
    figure_borda(data.ranks)
    figure_slopegraph(data.ranks)
    figure_positions(positions)
    figure_profiles(data.likert, data.demo)
    figure_groups(data.likert, data.demo)
    return {
        "participants": participants_table(data),
        "descriptives": descriptives(data.likert),
        "rankings": rankings_table(data.ranks),
        "positions": positions.round(2).rename_axis("modality").reset_index(),
        "participant_profiles": participant_profiles(data.likert, data.demo),
    }
