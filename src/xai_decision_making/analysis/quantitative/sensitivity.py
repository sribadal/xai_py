"""Sensitivity checks around the N=11 results.

P1-P9 saw interface v1 and P10-P11 saw v2. N=11 is the primary analysis.
Two participants cannot be compared with nine, so nothing here tests v1
against v2. Instead it shows where P10 and P11 fall within the v1 ratings,
and whether the main results hold without them (v1 only, N=9) and without
P10, who gave the same rating to every vignette.

Two further checks: Kendall's W for raters with no spread, and each
background correlation recomputed with one participant left out.
"""
import numpy as np
import pandas as pd

from xai_decision_making.analysis.common import (
    QUANTITATIVE_DIR, MODALITY_ORDER, DIMENSION_ORDER, V1_PARTICIPANTS, V2_PARTICIPANTS,
    CHANGED_MODALITIES, participant_means, participant_number, subset, v1_only)
from xai_decision_making.analysis.quantitative import inferential

NO_SPREAD_PARTICIPANTS = ["P10"]


def placement(likert):
    """Where each v2 participant falls in the v1 distribution. No test."""
    means = participant_means(likert)
    rows = []
    for modality in MODALITY_ORDER:
        for dimension in DIMENSION_ORDER:
            cell = means[(means["modality"] == modality) & (means["q_short"] == dimension)]
            v1 = cell[cell["pid"].isin(V1_PARTICIPANTS)]["score"].dropna()
            if v1.empty:
                continue
            row = {"modality": modality, "dimension": dimension,
                   "changed_in_v2": modality in CHANGED_MODALITIES,
                   "v1_n": len(v1), "v1_min": round(v1.min(), 2),
                   "v1_q1": round(v1.quantile(.25), 2), "v1_median": round(v1.median(), 2),
                   "v1_q3": round(v1.quantile(.75), 2), "v1_max": round(v1.max(), 2),
                   "v1_mean": round(v1.mean(), 3)}
            for pid in V2_PARTICIPANTS:
                value = cell[cell["pid"] == pid]["score"]
                if value.empty or pd.isna(value.iloc[0]):
                    row[pid], row[f"{pid}_position"] = np.nan, "missing"
                    continue
                value = float(value.iloc[0])
                row[pid] = round(value, 2)
                row[f"{pid}_pctile_in_v1"] = round(
                    float((v1 < value).mean() + 0.5 * (v1 == value).mean()), 2)
                row[f"{pid}_position"] = ("below v1 range" if value < v1.min()
                                          else "above v1 range" if value > v1.max()
                                          else "inside v1 range")
            rows.append(row)
    return pd.DataFrame(rows)


def tied_raters(likert):
    """Participants whose four type means are identical on a dimension.

    Such a rater leaves Friedman's chi2 and p unchanged, but Kendall's
    W = chi2 / (n(k-1)) still counts them in n, so W drops by about
    n / (n + 1) per tied rater. Report W with this factor.
    """
    means = participant_means(likert)
    rows = []
    for dimension in DIMENSION_ORDER:
        sub = means[means["q_short"] == dimension]
        spread = sub.groupby("pid")["score"].agg(lambda s: s.max() - s.min())
        tied = sorted(spread[spread == 0].index, key=participant_number)
        n = sub["pid"].nunique()
        rows.append({"dimension": dimension, "n": n, "n_tied": len(tied),
                     "tied_participants": ", ".join(tied),
                     "w_deflation_factor": round((n - len(tied)) / n, 3) if n else np.nan})
    return pd.DataFrame(rows)


def ordering(likert, label):
    """Position of each type per dimension, by mean rating."""
    means = participant_means(likert)
    rows = []
    for dimension in DIMENSION_ORDER:
        by_type = (means[means["q_short"] == dimension].groupby("modality")["score"].mean()
                   .reindex(MODALITY_ORDER).sort_values(ascending=False))
        for position, (modality, mean) in enumerate(by_type.items(), 1):
            rows.append({"subset": label, "dimension": dimension, "position": position,
                         "modality": modality, "mean": round(mean, 3)})
    return pd.DataFrame(rows)


def ordering_stability(orderings, reference="N=11"):
    """Per dimension: is the order of types the same in every subset?"""
    rows = []
    for dimension in orderings["dimension"].unique():
        sub = orderings[orderings["dimension"] == dimension]
        sequences = {label: list(g.sort_values("position")["modality"])
                     for label, g in sub.groupby("subset")}
        ref = sequences.get(reference) or list(sequences.values())[0]
        moved = []
        for label, sequence in sequences.items():
            if sequence == ref:
                continue
            for modality in ref:
                shift = sequence.index(modality) - ref.index(modality)
                if shift:
                    a = sub[(sub["subset"] == reference) & (sub["modality"] == modality)]["mean"]
                    b = sub[(sub["subset"] == label) & (sub["modality"] == modality)]["mean"]
                    delta = (b.iloc[0] - a.iloc[0]) if len(a) and len(b) else np.nan
                    moved.append(f"{modality} {shift:+d} in {label} (mean {delta:+.3f})")
        rows.append({"dimension": dimension,
                     "ordering_unchanged": all(s == ref for s in sequences.values()),
                     "order_primary": " > ".join(ref), "changes": "; ".join(moved) or "none"})
    return pd.DataFrame(rows)


def robustness(subsets):
    """Friedman, model comparison and ranking tests on each subset."""
    friedman, model, ranks = [], [], []
    for label, d in subsets.items():
        friedman.append(inferential.friedman_by_dimension(d.likert)
                        .assign(subset=label, n_participants=d.N))
        model.append(inferential.model_comparison(d.likert)
                     .assign(subset=label, n_participants=d.N))
        rank_omnibus, _ = inferential.ranking_friedman(d.ranks)
        if len(rank_omnibus):
            ranks.append(rank_omnibus.assign(subset=label, n_participants=d.N))
    return (pd.concat(friedman, ignore_index=True), pd.concat(model, ignore_index=True),
            pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame())


def correlation_leave_one_out(likert, demo):
    """Each background correlation recomputed with every participant left out.

    The AI-familiarity result rests on few raters, one of them both the most
    AI-familiar and the lowest rater. Leaving out every participant in turn
    avoids testing only that one. Holm is redone within each subset.
    """
    frames = []
    for dropped in [None] + demo["pid"].tolist():
        sub_likert = likert if dropped is None else likert[likert["pid"] != dropped]
        sub_demo = demo if dropped is None else demo[demo["pid"] != dropped]
        correlations = inferential.background_correlations(sub_likert, sub_demo)
        if len(correlations):
            frames.append(correlations.assign(dropped=dropped or "(none)"))
    loo = pd.concat(frames, ignore_index=True)

    full = loo[loo["dropped"] == "(none)"].set_index(["dimension", "variable"])["rho"]
    rows = []
    for (dimension, variable), g in loo[loo["dropped"] != "(none)"].groupby(
            ["dimension", "variable"]):
        weakest = g.loc[g["rho"].abs().idxmin()]
        rows.append({"dimension": dimension, "variable": variable,
                     "rho_full": round(full.loc[(dimension, variable)], 3),
                     "rho_min": round(g["rho"].min(), 3), "rho_max": round(g["rho"].max(), 3),
                     "rho_weakest": round(weakest["rho"], 3), "dropping": weakest["dropped"],
                     "n_still_significant": int(g["significant"].sum()),
                     "n_subsets": len(g)})
    return loo.round(3), pd.DataFrame(rows)


def build(data):
    """Write version_checks.txt and return the sensitivity tables."""
    without_p10 = subset(data, drop=NO_SPREAD_PARTICIPANTS)
    v1 = v1_only(data)

    places = placement(data.likert)
    focused = places[places["dimension"].str.contains("Actionability")
                     & places["modality"].isin(CHANGED_MODALITIES)]
    orderings = pd.concat([ordering(v1.likert, "N=9 v1 only"),
                           ordering(without_p10.likert, "N=10 minus P10"),
                           ordering(data.likert, "N=11")], ignore_index=True)
    stability = ordering_stability(orderings)
    ties = tied_raters(data.likert)
    friedman, model, ranks = robustness(
        {"N=11": data, "N=10 minus P10": without_p10, "N=9 v1 only": v1})
    loo, loo_summary = correlation_leave_one_out(data.likert, data.demo)

    tables = {
        "version_placement": places,
        "version_placement_focused": focused,
        "tied_raters": ties,
        "ordering_by_subset": orderings,
        "ordering_stability": stability,
        "robustness_friedman": friedman.round(3),
        "robustness_model": model.round(3),
        "robustness_rankings": ranks.round(3),
    }
    # The thesis sample (N=9), with the same tests as the main run.
    for name, frame in inferential.all_tests(v1).items():
        tables[f"{name}_v1only"] = frame.round(3)
    tables["correlation_loo"] = loo
    tables["correlation_loo_summary"] = loo_summary

    outside = places[[c for c in places.columns if c.endswith("_position")]].apply(
        lambda column: (column != "inside v1 range").sum())
    lines = ["Primary analysis is N=11. The nine-participant run is a sensitivity",
             "check, not the headline: P10 and P11 were recruited after the thesis",
             "was submitted, which is what makes them separable at all.",
             "",
             "Claims here are robustness statements about the ordering of",
             "explanation types. No test compares the subsets: n=2 cannot support",
             "a comparison.",
             "",
             stability.to_string(index=False),
             "",
             "Fully tied raters (identical means across all four types):",
             ties.to_string(index=False),
             "",
             "A tied rater leaves Friedman chi2 and p unchanged and is dropped",
             "from pairwise Wilcoxon, but Kendall's W = chi2 / (n(k-1)) falls by",
             "the deflation factor above. Report that as an artefact of the",
             "formula, not as a weaker effect.",
             "",
             f"v1 participants: {V1_PARTICIPANTS}",
             f"v2 participants: {V2_PARTICIPANTS}",
             f"modalities touched by the v2 changes: {CHANGED_MODALITIES}",
             "",
             "cells outside the v1 range, per v2 participant:",
             outside.to_string()]
    QUANTITATIVE_DIR.mkdir(parents=True, exist_ok=True)
    (QUANTITATIVE_DIR / "version_checks.txt").write_text("\n".join(lines), encoding="utf-8")
    return tables
