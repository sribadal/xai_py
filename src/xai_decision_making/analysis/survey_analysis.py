"""Survey analysis (quantitative): ratings (RQ2) and rankings (RQ3)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from xai_decision_making.analysis.common import (load, save, OUT, MODALITY_ORDER, MODEL_ORDER, Q_ORDER,
                    RANK_IDS, PAL, LIKERT_LABELS, DIV_COLORS, HEATMAP_CMAP)


def _descriptives(lik):
    desc = (lik.groupby(["modality", "q_short"])["score"]
            .agg(["mean", "median", "std", "count"]).round(2)
            .reindex([(m, q) for m in MODALITY_ORDER for q in Q_ORDER]))
    desc.to_csv(OUT / "table_descriptives.csv")
    return desc


def _diverging(lik):
    fig, axes = plt.subplots(len(Q_ORDER), 1, figsize=(12, 9), sharex=True)
    for ax, q in zip(axes, Q_ORDER):
        sub_q = lik[lik["q_short"] == q]
        for yi, mod in enumerate(MODALITY_ORDER):
            counts = sub_q[sub_q["modality"] == mod]["answer_label"].value_counts()
            total = counts.sum() or 1
            props = [counts.get(lab, 0) / total * 100 for lab in LIKERT_LABELS]
            left = -(props[0] + props[1] + props[2] / 2)
            for li, (prop, col) in enumerate(zip(props, DIV_COLORS)):
                ax.barh(yi, prop, left=left, color=col, edgecolor="white", height=0.55)
                if prop >= 8:
                    ax.text(left + prop / 2, yi, f"{prop:.0f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if li in (0, 1, 4) else "#333")
                left += prop
        ax.set_yticks(range(len(MODALITY_ORDER)))
        ax.set_yticklabels(MODALITY_ORDER)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(q, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("<- disagree (%)      agree (%) ->")
    legend = [Patch(facecolor=c, label=l) for c, l in zip(DIV_COLORS, LIKERT_LABELS)]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig_ratings_diverging.png")


def _heatmap(lik):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for mdl, ax in zip(MODEL_ORDER, axes):
        sub = lik[lik["model"] == mdl]
        grid = (sub.pivot_table(index="modality", columns="q_short",
                                values="score", aggfunc="mean")[Q_ORDER]
                .reindex(MODALITY_ORDER))
        sns.heatmap(grid, annot=True, fmt=".2f",
                    cmap=sns.light_palette("#1B4332", as_cmap=True),
                    vmin=1, vmax=5, linewidths=1.5, linecolor="white",
                    cbar_kws={"shrink": 0.7, "label": "Mean"}, ax=ax,
                    annot_kws={"fontsize": 12, "fontweight": "bold"})
        ax.set_title(mdl, pad=10, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(11)
    fig.tight_layout()
    save(fig, "fig_ratings_heatmap.png")


def _strip(lik):
    avg = lik.groupby(["pid", "modality", "q_short"])["score"].mean().reset_index()
    fig, axes = plt.subplots(1, len(Q_ORDER), figsize=(15, 5), sharey=True)
    rng = np.random.default_rng(42)
    for ax, q in zip(axes, Q_ORDER):
        sub = avg[avg["q_short"] == q]
        for mi, mod in enumerate(MODALITY_ORDER):
            vals = sub[sub["modality"] == mod]["score"].values
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(mi + jitter, vals, color=PAL[mod], s=50, alpha=0.7,
                       edgecolors="white", linewidth=0.5, zorder=3)
            ax.plot([mi - 0.2, mi + 0.2], [vals.mean(), vals.mean()],
                    color="black", lw=2, zorder=4)
        ax.set_xticks(range(len(MODALITY_ORDER)))
        ax.set_xticklabels(MODALITY_ORDER, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.axhline(3, color="grey", ls=":", alpha=0.4)
        ax.set_title(q, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean Likert Score (1-5)")
    fig.tight_layout()
    save(fig, "fig_ratings_strip.png")


def _raincloud(lik):
    avg = lik.groupby(["pid", "modality", "q_short"])["score"].mean().reset_index()
    avg["modality"] = pd.Categorical(avg["modality"],
                                      categories=MODALITY_ORDER, ordered=True)
    fig, axes = plt.subplots(1, len(Q_ORDER), figsize=(16, 5.5), sharey=True)
    for ax, q in zip(axes, Q_ORDER):
        sub = avg[avg["q_short"] == q]
        vp = ax.violinplot([sub[sub["modality"] == m]["score"].values
                            for m in MODALITY_ORDER],
                           positions=range(len(MODALITY_ORDER)),
                           showextrema=False, widths=0.6)
        for i, body in enumerate(vp["bodies"]):
            m = body.get_paths()[0].vertices
            m[:, 0] = np.clip(m[:, 0], -np.inf, i)
            body.set_facecolor(PAL[MODALITY_ORDER[i]])
            body.set_alpha(0.4)
        bp = ax.boxplot([sub[sub["modality"] == m]["score"].values
                         for m in MODALITY_ORDER],
                        positions=range(len(MODALITY_ORDER)),
                        widths=0.15, patch_artist=True,
                        showfliers=False, zorder=2)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PAL[MODALITY_ORDER[i]])
            patch.set_alpha(0.8)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(1)
        rng = np.random.default_rng(42)
        for mi, mod in enumerate(MODALITY_ORDER):
            vals = sub[sub["modality"] == mod]["score"].values
            jitter = rng.uniform(0.15, 0.3, len(vals))
            ax.scatter(mi + jitter, vals, color=PAL[mod], s=35, alpha=0.7,
                       edgecolors="white", linewidth=0.5, zorder=3)
        ax.set_xticks(range(len(MODALITY_ORDER)))
        ax.set_xticklabels(MODALITY_ORDER, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.axhline(3, color="grey", ls=":", alpha=0.4)
        ax.set_title(q, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean Likert Score (1-5)")
    fig.tight_layout()
    save(fig, "fig_ratings_raincloud.png")


def _parse(ranks, prefix):
    positions = {m: [] for m in MODALITY_ORDER}
    sub = ranks[ranks["question_id"].str.startswith(prefix)]
    for order in sub["ranking_order"].dropna():
        for pos, item in enumerate(x.strip() for x in order.split(">")):
            for m in MODALITY_ORDER:
                if m.lower() == item.lower():
                    positions[m].append(pos + 1)
    return positions


def _borda(positions, k=4):
    return {m: sum(k - p for p in pos) for m, pos in positions.items()}


def _rankings(ranks):
    rows, parsed = [], {}
    for prefix, label in RANK_IDS.items():
        pos = _parse(ranks, prefix)
        parsed[label] = pos
        borda = _borda(pos)
        for m in MODALITY_ORDER:
            rows.append({"ranking": label, "modality": m, "borda": borda[m],
                         "mean_rank": round(np.mean(pos[m]), 2) if pos[m] else np.nan,
                         "n": len(pos[m])})
    pd.DataFrame(rows).to_csv(OUT / "table_rankings.csv", index=False)

    fig, axes = plt.subplots(1, len(RANK_IDS), figsize=(12, 5))
    for ax, (label, pos) in zip(np.atleast_1d(axes), parsed.items()):
        ordered = sorted(_borda(pos).items(), key=lambda kv: kv[1], reverse=True)
        mods = [m for m, _ in ordered]
        pts = [v for _, v in ordered]
        ax.barh(range(len(mods)), pts, color=[PAL[m] for m in mods],
                edgecolor="white", height=0.5)
        ax.set_yticks(range(len(mods)))
        ax.set_yticklabels(mods)
        ax.invert_yaxis()
        ax.set_xlabel("Borda score (higher = preferred)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(pts):
            ax.text(v + 0.3, i, str(v), va="center", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_rankings_borda.png")


def build():
    d = load()
    desc = _descriptives(d.likert)
    _diverging(d.likert)
    _heatmap(d.likert)
    _strip(d.likert)
    _raincloud(d.likert)
    _rankings(d.ranks)
    return desc


if __name__ == "__main__":
    build()