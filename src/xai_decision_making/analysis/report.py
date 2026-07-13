"""Assemble a plain-text summary of the Study 2 results into output/report.txt."""
import numpy as np
import pandas as pd
from xai_decision_making.analysis.common import (load, save, OUT, MODALITY_ORDER, MODEL_ORDER, Q_ORDER,
                    RANK_IDS, PAL, LIKERT_LABELS, DIV_COLORS, HEATMAP_CMAP)
from xai_decision_making.analysis import statistical_analysis as stats
import xai_decision_making.analysis.survey_analysis as survey

LINE = "=" * 70
SUB = "-" * 70


def _completeness(lik):
    counts = lik.groupby("pid").size()
    lines = [f"  {pid}: {n}{'  <-- incomplete' if n != 24 else ''}"
             for pid, n in counts.items()]
    return "\n".join(lines)


def _descriptives(lik):
    desc = (lik.groupby(["modality", "q_short"])["score"]
            .agg(["mean", "median", "std", "count"]).round(2)
            .reindex([(m, q) for m in MODALITY_ORDER for q in Q_ORDER]))
    return desc.to_string()


def _rankings(ranks):
    out = []
    for prefix, label in RANK_IDS.items():
        pos = survey._parse(ranks, prefix)
        borda = survey._borda(pos)
        out.append(f"\n{label}:")
        for m, pts in sorted(borda.items(), key=lambda kv: kv[1], reverse=True):
            mean_rank = np.mean(pos[m]) if pos[m] else float("nan")
            out.append(f"  {m:16s} Borda={pts:3d}  mean rank={mean_rank:.2f}")
    return "\n".join(out)


def _best_worst(lik):
    out = []
    for q in Q_ORDER:
        means = lik[lik["q_short"] == q].groupby("modality")["score"].mean()
        best, worst = means.idxmax(), means.idxmin()
        out.append(f"  {q}: best {best} ({means[best]:.2f}), "
                   f"worst {worst} ({means[worst]:.2f})")
    return "\n".join(out)


def build():
    d = load()
    friedman, wilcoxon, _, _, _, _ = stats.run(d.likert, Q_ORDER)
    parts = [
        LINE, "STUDY 2 — RESULTS SUMMARY", LINE,
        f"\nN = {d.N} participants, {len(d.likert)} Likert responses "
        f"(expected {d.N * 24} if complete).",
        "\nCompleteness (rows per participant, 24 = complete):",
        _completeness(d.likert),
        f"\n\n{SUB}\nDescriptives by explanation type (models pooled)\n{SUB}",
        _descriptives(d.likert),
        f"\n\n{SUB}\nBest / worst type per dimension\n{SUB}",
        _best_worst(d.likert),
        f"\n\n{SUB}\nPreference rankings (Borda)\n{SUB}",
        _rankings(d.ranks),
        f"\n\n{SUB}\nFriedman omnibus (per dimension, models averaged)\n{SUB}",
        friedman.round(3).to_string(index=False),
        "\nAll tests are exploratory; interpret Kendall's W (effect size) "
        "alongside p given the sample size.",
        f"\n\n{SUB}\nPairwise Wilcoxon, Holm-corrected\n{SUB}",
        (wilcoxon.round(3).to_string(index=False) if len(wilcoxon)
         else "No dimension reached Friedman significance; no pairwise tests run."),
        "",
    ]
    (OUT / "report.txt").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    build()