"""Nonparametric tests for the XAI preference study: Friedman, pairwise Wilcoxon (Holm), and Spearman correlations."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

MODALITIES = ["Attribution", "Case-based", "Counterfactual", "Rule-based"]
ALPHA = 0.05


def _wide(likert, dimension):
    wide = (likert[likert["q_short"] == dimension]
            .pivot_table(index="pid", columns="modality", values="score", aggfunc="mean")
            .reindex(columns=MODALITIES)
            .dropna())
    return wide


def _rank_biserial(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    total = w_pos + w_neg
    return (w_pos - w_neg) / total if total else 0.0


def _holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def friedman_by_dimension(likert, dimensions):
    rows = []
    for dim in dimensions:
        wide = _wide(likert, dim)
        n, k = wide.shape
        chi2, p = stats.friedmanchisquare(*[wide[m].values for m in MODALITIES])
        rows.append({
            "dimension": dim,
            "n": n,
            "chi2": chi2,
            "df": k - 1,
            "p": p,
            "kendall_w": chi2 / (n * (k - 1)),
            "significant": p < ALPHA,
        })
    return pd.DataFrame(rows)


def pairwise_wilcoxon(likert, dimensions, friedman):
    gated = set(friedman.loc[friedman["significant"], "dimension"])
    rows = []
    for dim in dimensions:
        if dim not in gated:
            continue
        wide = _wide(likert, dim)
        pairs, pvals = [], []
        for m1, m2 in combinations(MODALITIES, 2):
            a, b = wide[m1].values, wide[m2].values
            if np.all(a - b == 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = stats.wilcoxon(a, b)
            pairs.append((m1, m2, stat, p, _rank_biserial(a, b)))
            pvals.append(p)
        for (m1, m2, stat, p, r), p_holm in zip(pairs, _holm(pvals)):
            rows.append({
                "dimension": dim,
                "pair": f"{m1} vs {m2}",
                "n": len(wide),
                "W": stat,
                "p": p,
                "p_holm": p_holm,
                "rank_biserial_r": r,
                "significant": p_holm < ALPHA,
            })
    return pd.DataFrame(rows)

def model_comparison(likert_by_model, dimensions):
    rows = []
    for dim in dimensions:
        sub = likert_by_model[likert_by_model["q_short"] == dim]
        means = (sub.groupby(["pid", "model"])["score"]
                 .mean()
                 .unstack("model")
                 .dropna())
        if "RSF" not in means.columns or "CONDOR" not in means.columns:
            continue
        a, b = means["RSF"].values, means["CONDOR"].values
        if np.all(a - b == 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = stats.wilcoxon(a, b)
        rows.append({
            "dimension": dim,
            "n": len(means),
            "W": stat,
            "p": p,
            "rank_biserial_r": _rank_biserial(a, b),
            "significant": p < ALPHA,
        })
    return pd.DataFrame(rows)

def ranking_friedman(ranks, rank_ids):
    fix = {"Case-Based": "Case-based", "Rule-Based": "Rule-based"}
    omnibus, pairwise = [], []
    for prefix, label in rank_ids.items():
        sub = ranks[ranks["question_id"].str.startswith(prefix)]
        parsed = []
        for _, r in sub.iterrows():
            order = [fix.get(s.strip(), s.strip()) for s in r["ranking_order"].split(">")]
            row = {"pid": r["pid"]}
            for pos, name in enumerate(order, 1):
                row[name] = pos
            parsed.append(row)
        wide = pd.DataFrame(parsed).set_index("pid").reindex(columns=MODALITIES).dropna()
        n, k = wide.shape
        if n < 3:
            continue
        chi2, p = stats.friedmanchisquare(*[wide[m].values for m in MODALITIES])
        omnibus.append({"ranking": label, "n": n, "chi2": chi2, "df": k - 1,
                        "p": p, "kendall_w": chi2 / (n * (k - 1)),
                        "significant": p < ALPHA})
        if p >= ALPHA:
            continue
        pairs, pvals = [], []
        for m1, m2 in combinations(MODALITIES, 2):
            a, b = wide[m1].values, wide[m2].values
            if np.all(a - b == 0):
                stat, pv = np.nan, 1.0
            else:
                stat, pv = stats.wilcoxon(a, b)
            pairs.append((m1, m2, stat, pv, _rank_biserial(a, b)))
            pvals.append(pv)
        for (m1, m2, stat, pv, r), ph in zip(pairs, _holm(pvals)):
            pairwise.append({"ranking": label, "pair": f"{m1} vs {m2}",
                             "n": n, "W": stat, "p": pv, "p_holm": ph,
                             "rank_biserial_r": r, "significant": ph < ALPHA})
    return pd.DataFrame(omnibus), pd.DataFrame(pairwise)


def background_correlations(likert, demo, dimensions):
    exp_map = {"0 to 5 years": 1, "6 to 10 years": 2,
               "11 to 20 years": 3, "More than 20 years": 4}
    ai_map = {"Not at all familiar": 1, "Slightly familiar": 2,
              "Moderately familiar": 3, "Very familiar": 4,
              "Extremely familiar": 5}
    merged = likert.merge(demo[["pid", "experience", "ai_familiarity"]], on="pid")
    merged["exp_ord"] = merged["experience"].map(exp_map)
    merged["ai_ord"] = merged["ai_familiarity"].map(ai_map)
    rows = []
    for dim in dimensions:
        sub = merged[merged["q_short"] == dim]
        means = sub.groupby("pid")[["score", "exp_ord", "ai_ord"]].mean()
        for var, label in [("exp_ord", "Experience"), ("ai_ord", "AI familiarity")]:
            clean = means[["score", var]].dropna()
            if len(clean) < 4:
                continue
            rho, p = stats.spearmanr(clean[var], clean["score"])
            rows.append({
                "dimension": dim,
                "variable": label,
                "n": len(clean),
                "rho": rho,
                "p": p,
                "significant": p < ALPHA,
            })
    return pd.DataFrame(rows)


def run(likert, dimensions, likert_by_model=None, ranks=None, rank_ids=None, demo=None):
    """Run the full nonparametric pipeline and return all result DataFrames."""
    friedman = friedman_by_dimension(likert, dimensions)
    wilcoxon = pairwise_wilcoxon(likert, dimensions, friedman)
    model = (model_comparison(likert_by_model, dimensions)
             if likert_by_model is not None else pd.DataFrame())
    rank_f, rank_w = (ranking_friedman(ranks, rank_ids)
                      if ranks is not None else (pd.DataFrame(), pd.DataFrame()))
    bg = (background_correlations(likert, demo, dimensions)
          if demo is not None else pd.DataFrame())
    return friedman, wilcoxon, model, rank_f, rank_w, bg