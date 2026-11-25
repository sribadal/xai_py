"""Exploratory patterns in the coding, beyond the counts in themes.py.

  - participant x code matrix: is a code widespread, or mostly one person?
  - co-occurrence: codes from different main categories on the same unit;
  - profile per explanation type over the four main categories;
  - change during the session: readability codes by vignette position and
    by first or second exposure to a type;
  - valence by evaluation target;
  - participants whose category profile differs most from the others.

All use the first author's reported units (codeable, raised by the
participant). These analyses were chosen after coding and are descriptive.
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xai_decision_making.analysis.common import (
    QUALITATIVE_DIR, PARTICIPANT_ORDER, MODALITY_ORDER, save_figure)
from xai_decision_making.analysis.qualitative.coding import (
    ALL_CODES, CODE_NAMES, FRAME_ALLOWED, FRAME_CHOOSE_ONE, MAIN_CATEGORIES,
    MAIN_CATEGORY_LABELS, author_units, codes_in_row, split_explanation_types)

FIGURES = QUALITATIVE_DIR / "figures"
CATEGORY_COLOURS = {"A": "#1B4332", "B": "#40916C", "C": "#95D5B2", "D": "#E9C46A"}
QUOTES_PER_PAIR = 3
LABEL_SIZE = 8


def _reported(units):
    return author_units(units).reset_index(drop=True)


def _categories(row):
    return sorted({code[0] for code in codes_in_row(row)})


# --- Participant x code ------------------------------------------------------

def participant_code_matrix(units):
    """Units per participant and code, for all 43 codes."""
    matrix = pd.DataFrame(0, index=PARTICIPANT_ORDER, columns=ALL_CODES, dtype=int)
    for _, row in _reported(units).iterrows():
        for code in codes_in_row(row):
            matrix.at[row["participant"], code] += 1
    return matrix.rename_axis("participant").reset_index()


def code_concentration(units):
    """Per code: units, participants, and the share of units from the one
    participant who raised it most."""
    matrix = participant_code_matrix(units).set_index("participant")
    rows = []
    for code in ALL_CODES:
        counts = matrix[code]
        total = int(counts.sum())
        if not total:
            continue
        rows.append({"code": code, "code_name": CODE_NAMES[code], "units": total,
                     "participants": int((counts > 0).sum()),
                     "top_participant": counts.idxmax(),
                     "top_participant_units": int(counts.max()),
                     "top_participant_share": round(counts.max() / total, 2)})
    return pd.DataFrame(rows).sort_values(["participants", "units"], ascending=False)


# --- Co-occurrence -----------------------------------------------------------

def _frame_label(pair):
    if pair in FRAME_CHOOSE_ONE:
        return "frame: choose one (coding error)"
    if pair in FRAME_ALLOWED:
        return "frame: allowed with condition"
    return "not in frame"


def _pairs(row):
    codes = sorted(codes_in_row(row), key=ALL_CODES.index)
    return [pair for pair in itertools.combinations(codes, 2) if pair[0][0] != pair[1][0]]


def co_occurrence(units):
    """Code pairs from different main categories on the same unit."""
    found = {}
    for _, row in _reported(units).iterrows():
        for pair in _pairs(row):
            entry = found.setdefault(pair, {"units": 0, "participants": set()})
            entry["units"] += 1
            entry["participants"].add(row["participant"])
    rows = [{"code_a": a, "name_a": CODE_NAMES[a], "code_b": b, "name_b": CODE_NAMES[b],
             "categories": f"{a[0]} + {b[0]}", "units": e["units"],
             "participants": len(e["participants"]),
             "participant_list": ", ".join(sorted(e["participants"],
                                                  key=PARTICIPANT_ORDER.index)),
             "frame": _frame_label((a, b))}
            for (a, b), e in found.items()]
    columns = ["code_a", "name_a", "code_b", "name_b", "categories", "units",
               "participants", "participant_list", "frame"]
    return (pd.DataFrame(rows, columns=columns)
            .sort_values(["participants", "units"], ascending=False).reset_index(drop=True))


def co_occurrence_by_category(units):
    """How often each pair of main categories is coded on the same unit."""
    found = {}
    for _, row in _reported(units).iterrows():
        for first, second in itertools.combinations(_categories(row), 2):
            entry = found.setdefault((first, second), {"units": 0, "participants": set()})
            entry["units"] += 1
            entry["participants"].add(row["participant"])
    coded = _reported(units)
    multi = sum(1 for _, row in coded.iterrows() if len(_categories(row)) > 1)
    rows = [{"categories": f"{a} + {b}",
             "names": f"{MAIN_CATEGORY_LABELS[a]} + {MAIN_CATEGORY_LABELS[b]}",
             "units": e["units"], "participants": len(e["participants"]),
             "share_of_multi_category_units": round(e["units"] / multi, 2) if multi else np.nan}
            for (a, b), e in sorted(found.items())]
    return pd.DataFrame(rows)


def co_occurrence_quotes(units, min_participants=2):
    """Up to three units per pair not anticipated by the frame, from different
    participants where possible, with their text. Private: contains speech."""
    pairs = co_occurrence(units)
    pairs = pairs[(pairs["frame"] == "not in frame") & (pairs["participants"] >= min_participants)]
    reported = _reported(units)
    rows = []
    for _, pair in pairs.iterrows():
        key = (pair["code_a"], pair["code_b"])
        hits = reported[reported.apply(lambda row: key in _pairs(row), axis=1)]
        chosen = hits.drop_duplicates("participant").head(QUOTES_PER_PAIR)
        for _, row in chosen.iterrows():
            rows.append({"code_a": key[0], "code_b": key[1],
                         "participant": row["participant"], "unit_id": row.get("unit_id"),
                         "explanation_type": row.get("explanation_type"),
                         "text": row.get("text"), "notes": row.get("notes")})
    return pd.DataFrame(rows)


def figure_co_occurrence(units, min_participants=2):
    """Code pairs raised by at least two participants."""
    pairs = co_occurrence(units)
    pairs = pairs[(pairs["participants"] >= min_participants)
                  & (pairs["frame"] != "frame: choose one (coding error)")]
    if pairs.empty:
        return
    pairs = pairs.iloc[::-1]
    labels = [f"{r.code_a} {r.name_a}  +  {r.code_b} {r.name_b}" for r in pairs.itertuples()]
    colours = ["#2D6A4F" if f == "not in frame" else "#B7E4C7" for f in pairs["frame"]]
    fig, ax = plt.subplots(figsize=(7.0, 0.3 * len(pairs) + 1.0))
    ax.barh(range(len(pairs)), pairs["participants"], color=colours, edgecolor="white")
    for y, (p, u) in enumerate(zip(pairs["participants"], pairs["units"])):
        ax.text(p + 0.1, y, f"{u} units", va="center", fontsize=7, color="#444444")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=LABEL_SIZE)
    ax.set_xlabel(f"Participants (of {len(PARTICIPANT_ORDER)})", fontsize=LABEL_SIZE)
    ax.set_xlim(0, len(PARTICIPANT_ORDER) + 1.5)
    ax.tick_params(axis="x", labelsize=LABEL_SIZE)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#2D6A4F", "#B7E4C7")]
    ax.legend(handles, ["not in the frame", "allowed by the frame, with a condition"],
              frameon=False, fontsize=LABEL_SIZE, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_co_occurrence")


# --- Profile per explanation type --------------------------------------------

def type_profile(units):
    """For each type, the share of its codes in each main category.

    pooled share: all of the type's codes together. Participant share: each
    participant's share for that type, averaged, so one talkative participant
    counts once.
    """
    reported = _reported(units)
    long = []
    for _, row in reported.iterrows():
        for explanation_type in split_explanation_types(row.get("explanation_type")):
            for code in codes_in_row(row):
                long.append({"type": explanation_type, "participant": row["participant"],
                             "category": code[0], "unit_id": row.get("unit_id")})
    long = pd.DataFrame(long, columns=["type", "participant", "category", "unit_id"])
    rows = []
    for explanation_type in MODALITY_ORDER:
        sub = long[long["type"] == explanation_type]
        row = {"type": explanation_type, "units": sub["unit_id"].nunique(),
               "codes": len(sub), "participants": sub["participant"].nunique()}
        per_participant = (sub.groupby(["participant", "category"]).size()
                           .unstack(fill_value=0).reindex(columns=MAIN_CATEGORIES, fill_value=0))
        per_participant = per_participant.div(per_participant.sum(axis=1), axis=0)
        for category in MAIN_CATEGORIES:
            in_category = sub[sub["category"] == category]
            row[f"{category}_codes"] = len(in_category)
            row[f"{category}_share"] = round(len(in_category) / len(sub), 2) if len(sub) else np.nan
            row[f"{category}_participant_share"] = (round(per_participant[category].mean(), 2)
                                                    if len(per_participant) else np.nan)
            row[f"{category}_participants"] = in_category["participant"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def figure_type_profile(units):
    profile = type_profile(units)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    left = np.zeros(len(profile))
    for category in MAIN_CATEGORIES:
        shares = profile[f"{category}_share"].fillna(0).values
        ax.barh(range(len(profile)), shares, left=left, color=CATEGORY_COLOURS[category],
                edgecolor="white", label=f"{category} {MAIN_CATEGORY_LABELS[category]}")
        for y, (start, share) in enumerate(zip(left, shares)):
            if share >= 0.08:
                ax.text(start + share / 2, y, f"{share:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if category in "AB" else "#333333")
        left += shares
    ax.set_yticks(range(len(profile)))
    ax.set_yticklabels([f"{t} (n={u})" for t, u in zip(profile["type"], profile["units"])],
                       fontsize=LABEL_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(labelsize=LABEL_SIZE, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=LABEL_SIZE, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_type_profile")


# --- Change during the session -----------------------------------------------

def _category_shares(frame):
    """Share of coded units carrying a code in each category."""
    out = {"units": len(frame), "participants": frame["participant"].nunique()}
    for category in MAIN_CATEGORIES:
        has = frame.apply(lambda row: category in _categories(row), axis=1) if len(frame) else []
        out[f"{category}_share"] = round(float(np.mean(has)), 2) if len(frame) else np.nan
    return out


def by_position(units):
    """Share of units with a code in each category, by vignette position (1-8).

    Position mixes type and model, and the model order was fixed, so read this
    as description only.
    """
    reported = _reported(units)
    reported = reported[reported["vignette_order"].notna()]
    rows = [{"vignette_order": int(order), **_category_shares(group)}
            for order, group in reported.groupby("vignette_order")]
    return pd.DataFrame(rows)


def by_exposure(units):
    """The same by first or second exposure to a type, per type.

    The model position was fixed, so the second exposure to a type is always
    the other model: exposure and model cannot be separated.
    """
    reported = _reported(units)
    rows = []
    for _, row in reported.iterrows():
        for explanation_type in split_explanation_types(row.get("explanation_type")):
            rows.append({**row.to_dict(), "type": explanation_type})
    expanded = pd.DataFrame(rows)
    if expanded.empty or "exposure" not in expanded:
        return pd.DataFrame()
    expanded = expanded[expanded["exposure"].notna()]
    out = [{"type": t, "exposure": e, **_category_shares(group)}
           for (t, e), group in expanded.groupby(["type", "exposure"])]
    frame = pd.DataFrame(out)
    frame["type"] = pd.Categorical(frame["type"], MODALITY_ORDER, ordered=True)
    return frame.sort_values(["type", "exposure"]).reset_index(drop=True)


def readability_by_half(units):
    """Per participant: share of units with a readability (A) code in the
    first four vignettes and in the last four."""
    reported = _reported(units)
    reported = reported[reported["vignette_order"].notna()]
    rows = []
    for participant in PARTICIPANT_ORDER:
        mine = reported[reported["participant"] == participant]
        if mine.empty:
            continue
        halves = {}
        for label, part in (("first_half", mine[mine["vignette_order"] <= 4]),
                            ("second_half", mine[mine["vignette_order"] > 4])):
            has_a = part.apply(lambda row: "A" in _categories(row), axis=1) if len(part) else []
            halves[f"{label}_units"] = len(part)
            halves[f"{label}_A_share"] = round(float(np.mean(has_a)), 2) if len(part) else np.nan
        change = halves["second_half_A_share"] - halves["first_half_A_share"]
        rows.append({"participant": participant, **halves,
                     "change": round(change, 2) if pd.notna(change) else np.nan,
                     "direction": ("fewer later" if change < 0 else "more later" if change > 0
                                   else "same") if pd.notna(change) else ""})
    return pd.DataFrame(rows)


def figure_by_position(units):
    table = by_position(units)
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for category in MAIN_CATEGORIES:
        ax.plot(table["vignette_order"], table[f"{category}_share"], "-o",
                color=CATEGORY_COLOURS[category], markersize=5, linewidth=1.8,
                label=f"{category} {MAIN_CATEGORY_LABELS[category]}")
    ax.set_xticks(table["vignette_order"])
    ax.set_xlabel("Vignette position", fontsize=LABEL_SIZE)
    ax.set_ylabel("Share of units with a code\nin the category", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(labelsize=LABEL_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=LABEL_SIZE, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_by_position")


# --- Valence by target -------------------------------------------------------

def valence_by_target(units):
    reported = _reported(units).dropna(subset=["valence", "eval_target"])
    table = (reported.groupby(["eval_target", "valence"]).size().unstack(fill_value=0)
             .reindex(columns=["positive", "neutral", "negative"], fill_value=0))
    table["units"] = table.sum(axis=1)
    table["negative_share"] = (table["negative"] / table["units"]).round(2)
    return table.rename_axis(columns=None).reset_index()


def valence_by_target_per_participant(units):
    reported = _reported(units).dropna(subset=["valence", "eval_target"])
    table = (reported.groupby(["participant", "eval_target", "valence"]).size()
             .unstack(fill_value=0)
             .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
             .rename_axis(columns=None).reset_index())
    table["participant"] = pd.Categorical(table["participant"], PARTICIPANT_ORDER, ordered=True)
    return table.sort_values(["participant", "eval_target"]).reset_index(drop=True)


# --- Participants who differ -------------------------------------------------

def deviant_cases(units):
    """Each participant's category profile against everyone else's.

    difference = participant's share minus the others' pooled share, per
    category. Sorted by the largest absolute difference: the top rows are the
    participants worth reading closely. This only flags; the reason has to
    come from the transcripts.
    """
    reported = _reported(units)
    counts = pd.DataFrame(0, index=PARTICIPANT_ORDER, columns=MAIN_CATEGORIES, dtype=int)
    positive = pd.Series(0, index=PARTICIPANT_ORDER)
    for _, row in reported.iterrows():
        for code in codes_in_row(row):
            counts.at[row["participant"], code[0]] += 1
        if row.get("valence") == "positive":
            positive[row["participant"]] += 1
    units_per = reported.groupby("participant").size().reindex(PARTICIPANT_ORDER, fill_value=0)
    rows = []
    for participant in PARTICIPANT_ORDER:
        mine, others = counts.loc[participant], counts.drop(index=participant).sum()
        if mine.sum() == 0:
            continue
        diff = mine / mine.sum() - others / others.sum()
        others_units = units_per.drop(index=participant).sum()
        row = {"participant": participant,
               "version": reported.loc[reported["participant"] == participant, "version"].iloc[0],
               "units": int(units_per[participant])}
        for category in MAIN_CATEGORIES:
            row[f"{category}_share"] = round(mine[category] / mine.sum(), 2)
            row[f"{category}_difference"] = round(diff[category], 2)
        row["positive_share"] = round(positive[participant] / units_per[participant], 2)
        row["positive_difference"] = round(
            positive[participant] / units_per[participant]
            - positive.drop(index=participant).sum() / others_units, 2)
        largest = diff.abs().idxmax()
        row["largest_difference"] = f"{largest} {diff[largest]:+.2f}"
        row["_size"] = abs(diff[largest])
        rows.append(row)
    return (pd.DataFrame(rows).sort_values("_size", ascending=False)
            .drop(columns="_size").reset_index(drop=True))


# --- Output ------------------------------------------------------------------

def tables(units):
    """The public pattern tables, and the private quotes."""
    public = {
        "participant_code_matrix": participant_code_matrix(units),
        "code_concentration": code_concentration(units),
        "co_occurrence": co_occurrence(units),
        "co_occurrence_by_category": co_occurrence_by_category(units),
        "type_profile": type_profile(units),
        "by_position": by_position(units),
        "by_exposure": by_exposure(units),
        "readability_by_half": readability_by_half(units),
        "valence_by_target": valence_by_target(units),
        "valence_target_by_participant": valence_by_target_per_participant(units),
        "deviant_cases": deviant_cases(units),
    }
    private = {"co_occurrence_quotes": co_occurrence_quotes(units)}
    return public, private


def write_figures(units):
    figure_co_occurrence(units)
    figure_type_profile(units)
    figure_by_position(units)
