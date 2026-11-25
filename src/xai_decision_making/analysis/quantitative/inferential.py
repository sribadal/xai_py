"""Statistical tests for the survey ratings and rankings.

Friedman across the four explanation types per dimension, Holm-corrected
pairwise Wilcoxon where Friedman is significant, Wilcoxon between the two
models, the same tests on the two rankings, Spearman correlations with
background, and the purpose-shift test between the two rankings.

Every test runs on participant means: each participant's two models are
averaged first, giving one value per participant, type and dimension.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from xai_decision_making.analysis.common import (
    MODALITY_ORDER, DIMENSION_ORDER, RANKINGS, rank_positions)

ALPHA = 0.05

EXPERIENCE_LEVELS = {"0 to 5 years": 1, "6 to 10 years": 2,
                     "11 to 20 years": 3, "More than 20 years": 4}
AI_FAMILIARITY_LEVELS = {"Not at all familiar": 1, "Slightly familiar": 2,
                         "Moderately familiar": 3, "Very familiar": 4,
                         "Extremely familiar": 5}

# Purpose shift: rule-based is the type the analysis was specified around;
# the other three are exploratory and Holm-corrected among themselves.
PURPOSE_SHIFT_PRIMARY = "Rule-based"


def holm(pvalues):
    p = np.asarray(pvalues, float)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def rank_biserial(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    positive, negative = ranks[diff > 0].sum(), ranks[diff < 0].sum()
    return (positive - negative) / (positive + negative)


def _wilcoxon(a, b):
    if np.all(np.asarray(a) - np.asarray(b) == 0):
        return np.nan, 1.0
    return stats.wilcoxon(a, b)


def _by_type(likert, dimension):
    """Participants by explanation type, for one dimension."""
    return (likert[likert["q_short"] == dimension]
            .pivot_table(index="pid", columns="modality", values="score", aggfunc="mean")
            .reindex(columns=MODALITY_ORDER)
            .dropna())


def _pairwise(wide, n_label, label_column, label):
    pairs, pvalues = [], []
    for first, second in combinations(MODALITY_ORDER, 2):
        a, b = wide[first].values, wide[second].values
        stat, p = _wilcoxon(a, b)
        pairs.append((first, second, stat, p, rank_biserial(a, b)))
        pvalues.append(p)
    rows = []
    for (first, second, stat, p, r), p_holm in zip(pairs, holm(pvalues)):
        rows.append({label_column: label, "pair": f"{first} vs {second}",
                     "n": n_label, "W": stat, "p": p, "p_holm": p_holm,
                     "rank_biserial_r": r, "significant": p_holm < ALPHA})
    return rows


def friedman_by_dimension(likert):
    rows = []
    for dimension in DIMENSION_ORDER:
        wide = _by_type(likert, dimension)
        n, k = wide.shape
        chi2, p = stats.friedmanchisquare(*[wide[m].values for m in MODALITY_ORDER])
        rows.append({"dimension": dimension, "n": n, "chi2": chi2, "df": k - 1, "p": p,
                     "kendall_w": chi2 / (n * (k - 1)), "significant": p < ALPHA})
    return pd.DataFrame(rows)


def pairwise_wilcoxon(likert, friedman):
    rows = []
    for dimension in friedman.loc[friedman["significant"], "dimension"]:
        wide = _by_type(likert, dimension)
        rows += _pairwise(wide, len(wide), "dimension", dimension)
    return pd.DataFrame(rows)


def model_comparison(likert):
    """RSF against ConDOR, per dimension, averaging over the four types."""
    rows = []
    for dimension in DIMENSION_ORDER:
        means = (likert[likert["q_short"] == dimension]
                 .groupby(["pid", "model"])["score"].mean()
                 .unstack("model").dropna())
        if "RSF" not in means.columns or "CONDOR" not in means.columns:
            continue
        a, b = means["RSF"].values, means["CONDOR"].values
        stat, p = _wilcoxon(a, b)
        rows.append({"dimension": dimension, "n": len(means), "W": stat, "p": p,
                     "rank_biserial_r": rank_biserial(a, b), "significant": p < ALPHA})
    return pd.DataFrame(rows)


def ranking_friedman(ranks):
    omnibus, pairwise = [], []
    for prefix, label in RANKINGS.items():
        wide = (rank_positions(ranks, prefix)
                .pivot(index="pid", columns="modality", values="position")
                .reindex(columns=MODALITY_ORDER).dropna())
        n, k = wide.shape
        if n < 3:
            continue
        chi2, p = stats.friedmanchisquare(*[wide[m].values for m in MODALITY_ORDER])
        omnibus.append({"ranking": label, "n": n, "chi2": chi2, "df": k - 1, "p": p,
                        "kendall_w": chi2 / (n * (k - 1)), "significant": p < ALPHA})
        if p < ALPHA:
            pairwise += _pairwise(wide, n, "ranking", label)
    return pd.DataFrame(omnibus), pd.DataFrame(pairwise)


def background_correlations(likert, demo):
    """Spearman between experience or AI familiarity and each dimension.

    Holm is applied across all six correlations.
    """
    merged = likert.merge(demo[["pid", "experience", "ai_familiarity"]], on="pid")
    merged["exp_ord"] = merged["experience"].map(EXPERIENCE_LEVELS)
    merged["ai_ord"] = merged["ai_familiarity"].map(AI_FAMILIARITY_LEVELS)
    rows = []
    for dimension in DIMENSION_ORDER:
        means = (merged[merged["q_short"] == dimension]
                 .groupby("pid")[["score", "exp_ord", "ai_ord"]].mean())
        for column, label in [("exp_ord", "Experience"), ("ai_ord", "AI familiarity")]:
            clean = means[["score", column]].dropna()
            if len(clean) < 4:
                continue
            rho, p = stats.spearmanr(clean[column], clean["score"])
            rows.append({"dimension": dimension, "variable": label, "n": len(clean),
                         "rho": rho, "p": p, "significant": p < ALPHA})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"])
        out["significant"] = out["p_holm"] < ALPHA
    return out


def purpose_shift(ranks):
    """Does a type's position change between the two ranking questions?

    Positions sum to 10 for every participant, so the four types cannot move
    independently. A negative median shift means the type moved up for
    patient communication.
    """
    overall = rank_positions(ranks, _ranking_prefix("Overall preference"))
    patient = rank_positions(ranks, _ranking_prefix("Patient communication"))
    paired = (overall.rename(columns={"position": "overall"})
              .merge(patient.rename(columns={"position": "patient"}),
                     on=["pid", "modality"]))
    rows = []
    for modality in MODALITY_ORDER:
        s = paired[paired["modality"] == modality]
        x, y = s["overall"].values.astype(float), s["patient"].values.astype(float)
        stat, p = (np.nan, 1.0) if len(x) < 3 else _wilcoxon(x, y)
        rows.append({
            "modality": modality, "n": len(s),
            "mean_rank_overall": round(x.mean(), 2) if len(x) else np.nan,
            "mean_rank_patient": round(y.mean(), 2) if len(y) else np.nan,
            "median_shift": float(np.median(y - x)) if len(x) else np.nan,
            "n_improved": int((y < x).sum()), "n_worsened": int((y > x).sum()),
            "n_unchanged": int((y == x).sum()), "W": stat, "p": p,
            "role": "primary" if modality == PURPOSE_SHIFT_PRIMARY else "exploratory",
        })
    out = pd.DataFrame(rows)
    exploratory = out["role"] == "exploratory"
    out["p_holm"] = np.nan
    out.loc[exploratory, "p_holm"] = holm(out.loc[exploratory, "p"].values)
    out["significant"] = np.where(out["role"] == "primary",
                                  out["p"] < ALPHA, out["p_holm"] < ALPHA)
    return out


def _ranking_prefix(label):
    return next(prefix for prefix, name in RANKINGS.items() if name == label)


def all_tests(data):
    """Every test on one sample, keyed by output table name."""
    friedman = friedman_by_dimension(data.likert)
    rank_omnibus, rank_pairwise = ranking_friedman(data.ranks)
    return {
        "background_correlations": background_correlations(data.likert, data.demo),
        "friedman": friedman,
        "wilcoxon": pairwise_wilcoxon(data.likert, friedman),
        "model_comparison": model_comparison(data.likert),
        "rank_friedman": rank_omnibus,
        "rank_wilcoxon": rank_pairwise,
    }
