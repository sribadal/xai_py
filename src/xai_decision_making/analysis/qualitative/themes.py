"""Counts, tables and figures from the first author's coding.

Counts are numbers of participants, not units, unless a table says
otherwise: a participant who raises a theme five times counts once.
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xai_decision_making.analysis.common import (
    QUALITATIVE_DIR, PARTICIPANT_ORDER, MODALITY_ORDER, INTERFACE_VERSION, PAL,
    save_figure)
from xai_decision_making.analysis.qualitative.agreement import (
    table_agreement, table_agreement_by_session)
from xai_decision_making.analysis.qualitative.coding import (
    ALL_CODES, AUTHOR_CODER, CODE_NAMES, INTERVIEWER_SUPPLIED, RESIDUAL_SUFFIX, SCOPE_CODEABLE,
    author_units, codes_in_row, codes_in_row_excluding_residuals, split_explanation_types)

FIGURES = QUALITATIVE_DIR / "figures"

# Themes discussed in the paper, in the order of the reported-themes figure.
REPORTED_THEMES = ["B2", "B3", "B4", "D2", "C1", "C5", "A2", "A3", "A4", "A1", "D4",
                   "B7", "B8", "A6", "C12", "C13"]
# A code "discriminates" between types if its participant counts differ by at
# least this much between the most and least frequent type.
DISCRIMINATING_MIN_SPREAD = 3
GRID_MIN_PARTICIPANTS = 4

EVAL_TARGET_LABELS = {"explanation": "The explanation", "prediction": "The prediction",
                      "input_data": "The input data", "retrieved_cases": "The retrieved cases",
                      "interface": "The interface"}
SOURCE_ORDER = ["spontaneous", "prompted", INTERVIEWER_SUPPLIED]
SOURCE_LABELS = {"spontaneous": "Spontaneous", "prompted": "Prompted",
                 INTERVIEWER_SUPPLIED: "Interviewer supplied"}
VALENCE_ORDER = ["positive", "neutral", "negative"]
TARGET_ORDER = ["explanation", "interface", "input_data", "prediction", "retrieved_cases"]
TARGET_COLOURS = {"explanation": "#1B4332", "interface": "#40916C", "input_data": "#74C69D",
                  "prediction": "#B7E4C7", "retrieved_cases": "#E9C46A"}
UNFILLED_COLOUR = "#E4E9E6"
FIGURE_WIDTH = 7.0
LABEL_SIZE = 8
COUNT_SIZE = 7


# --- Counts ------------------------------------------------------------------

def _presence(units, include_residuals=True):
    """code -> set of participants who raised it."""
    codes = codes_in_row if include_residuals else codes_in_row_excluding_residuals
    presence = {}
    for _, row in author_units(units).iterrows():
        for code in codes(row):
            presence.setdefault(code, set()).add(row["participant"])
    return presence


def participants_per_code(units):
    presence = _presence(units)
    return pd.Series({code: len(presence.get(code, ())) for code in ALL_CODES},
                     name="participants")


def participants_per_code_and_type(units):
    seen = set()
    for _, row in author_units(units).iterrows():
        for code in codes_in_row(row):
            for explanation_type in split_explanation_types(row.get("explanation_type")):
                seen.add((code, explanation_type, row["participant"]))
    matrix = pd.DataFrame(0, index=ALL_CODES, columns=MODALITY_ORDER, dtype=int)
    for (code, explanation_type), total in Counter((c, t) for c, t, _ in seen).items():
        if explanation_type in matrix.columns:
            matrix.at[code, explanation_type] = total
    return matrix.loc[matrix.sum(axis=1) > 0]


def participants_per_code_and_model(units):
    seen = set()
    for _, row in author_units(units).iterrows():
        if pd.isna(row.get("model")):
            continue
        for code in codes_in_row(row):
            seen.add((code, str(row["model"]).upper(), row["participant"]))
    models = sorted({model for _, model, _ in seen})
    matrix = pd.DataFrame(0, index=ALL_CODES, columns=models, dtype=int)
    for (code, model), total in Counter((c, m) for c, m, _ in seen).items():
        matrix.at[code, model] = total
    return matrix.loc[matrix.sum(axis=1) > 0]


def participants_per_target_and_type(units):
    seen = set()
    for _, row in author_units(units).iterrows():
        if pd.isna(row.get("eval_target")):
            continue
        for explanation_type in split_explanation_types(row.get("explanation_type")):
            seen.add((row["eval_target"], explanation_type, row["participant"]))
    targets = [t for t in EVAL_TARGET_LABELS if any(target == t for target, _, _ in seen)]
    matrix = pd.DataFrame(0, index=targets, columns=MODALITY_ORDER, dtype=int)
    for (target, explanation_type), total in Counter((t, e) for t, e, _ in seen).items():
        if explanation_type in matrix.columns and target in matrix.index:
            matrix.at[target, explanation_type] = total
    return matrix


def units_per_source_and_type(units):
    """Units (not participants) by who raised them, interviewer-supplied included."""
    counts = Counter()
    for _, row in author_units(units, exclude_interviewer_supplied=False).iterrows():
        if pd.isna(row.get("source")):
            continue
        for explanation_type in split_explanation_types(row.get("explanation_type")):
            counts[(str(row["source"]), explanation_type)] += 1
    matrix = pd.DataFrame(0, index=SOURCE_ORDER, columns=MODALITY_ORDER, dtype=int)
    for (source, explanation_type), total in counts.items():
        if source in matrix.index and explanation_type in matrix.columns:
            matrix.at[source, explanation_type] = total
    return matrix


def code_spread(matrix):
    """Most minus fewest participants across explanation types, per code."""
    return (matrix.max(axis=1) - matrix.min(axis=1)).sort_values(ascending=False)


def discriminating_codes(matrix):
    spread = code_spread(matrix)
    return [code for code in spread[spread >= DISCRIMINATING_MIN_SPREAD].index
            if not code.endswith(RESIDUAL_SUFFIX)]


def _grid(presence, codes):
    grid = pd.DataFrame(0, index=codes, columns=PARTICIPANT_ORDER, dtype=int)
    for code in codes:
        for participant in presence.get(code, ()):
            if participant in grid.columns:
                grid.at[code, participant] = 1
    return grid


def participant_theme_grid(units):
    """Themes by participant: kept if they discriminate between types or are
    raised by at least GRID_MIN_PARTICIPANTS. A caption must state both."""
    presence = _presence(units, include_residuals=False)
    keep = set(discriminating_codes(participants_per_code_and_type(units)))
    keep |= {code for code, people in presence.items() if len(people) >= GRID_MIN_PARTICIPANTS}
    return _grid(presence, sorted(keep, key=lambda code: (-len(presence.get(code, ())), code)))


def reported_theme_grid(units):
    presence = _presence(units, include_residuals=False)
    return _grid(presence, [code for code in REPORTED_THEMES if presence.get(code)])


# --- Tables ------------------------------------------------------------------

def _with_names(frame):
    frame = frame.copy()
    frame.insert(0, "code_name", [CODE_NAMES[code] for code in frame.index])
    return frame


def table_corpus(units):
    """Units per participant. interviewer_supplied counts codeable units only, so
    codeable minus interviewer_supplied equals reported."""
    rows = []
    for participant, sheet in units[units["coder"] == AUTHOR_CODER].groupby("participant"):
        scope = sheet["scope"].value_counts()
        codeable = sheet[sheet["scope"] == SCOPE_CODEABLE]
        rows.append({
            "participant": participant,
            "version": INTERFACE_VERSION.get(participant),
            "units": len(sheet),
            "codeable": int(scope.get(SCOPE_CODEABLE, 0)),
            "out_of_scope": int(scope.get("out_of_scope", 0)),
            "insufficient_context": int(scope.get("insufficient_context", 0)),
            "interviewer_supplied": int((codeable["source"] == INTERVIEWER_SUPPLIED).sum()),
            "interviewer_supplied_all": int((sheet["source"] == INTERVIEWER_SUPPLIED).sum()),
            "reported": len(author_units(sheet)),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("participant", key=lambda column: column.str[1:].astype(int))


def table_code_participants(units):
    counts = participants_per_code(units)
    frame = pd.DataFrame({"code_name": [CODE_NAMES[code] for code in counts.index],
                          "participants": counts.values}, index=counts.index)
    return frame[frame["participants"] > 0].sort_values("participants", ascending=False)


def table_code_by_type(units):
    matrix = participants_per_code_and_type(units)
    frame = _with_names(matrix)
    frame["spread"] = code_spread(matrix)
    return frame.sort_values("spread", ascending=False)


def table_discriminating_codes(units):
    matrix = participants_per_code_and_type(units)
    selected = discriminating_codes(matrix)
    frame = _with_names(matrix.loc[selected])
    frame["spread"] = code_spread(matrix).loc[selected]
    return frame


def table_code_by_model(units):
    matrix = participants_per_code_and_model(units)
    frame = _with_names(matrix)
    if matrix.shape[1] == 2:
        first, second = matrix.columns
        frame["difference"] = matrix[first] - matrix[second]
    return frame.sort_values("code_name")


def table_eval_target_by_type(units):
    matrix = participants_per_target_and_type(units)
    frame = matrix.copy()
    frame.index = [EVAL_TARGET_LABELS.get(target, target) for target in frame.index]
    frame["total"] = matrix.sum(axis=1).values
    return frame


def table_source_composition(units):
    matrix = units_per_source_and_type(units)
    frame = matrix.copy()
    frame.index = [SOURCE_LABELS[value] for value in matrix.index]
    frame.loc["Spontaneous share"] = (
        matrix.loc["spontaneous"] / matrix.sum(axis=0).replace(0, np.nan)).round(2)
    return frame


def table_participant_themes(units):
    grid = participant_theme_grid(units)
    frame = grid.copy()
    frame.insert(0, "theme", [CODE_NAMES[code] for code in grid.index])
    frame["participants"] = grid.sum(axis=1)
    return frame.sort_values("participants", ascending=False)


def table_valence_by_code(units):
    """Valence per code, in units. Residual (99) codes are left out."""
    counts = Counter()
    for _, row in author_units(units).iterrows():
        if pd.isna(row.get("valence")):
            continue
        for code in codes_in_row_excluding_residuals(row):
            counts[(code, str(row["valence"]))] += 1
    codes = sorted({code for code, _ in counts}, key=ALL_CODES.index)
    frame = pd.DataFrame(0, index=codes, columns=VALENCE_ORDER, dtype=int)
    for (code, valence), total in counts.items():
        if valence in frame.columns:
            frame.at[code, valence] = total
    frame.insert(0, "code_name", [CODE_NAMES[code] for code in codes])
    totals = frame[VALENCE_ORDER].sum(axis=1).replace(0, np.nan)
    frame["dominant"] = frame[VALENCE_ORDER].idxmax(axis=1)
    frame["dominant_share"] = (frame[VALENCE_ORDER].max(axis=1) / totals).round(2)
    directional = frame[["positive", "negative"]].sum(axis=1).replace(0, np.nan)
    frame["directional_units"] = frame[["positive", "negative"]].sum(axis=1)
    frame["dominant_direction"] = frame[["positive", "negative"]].idxmax(axis=1)
    frame["dominant_share_directional"] = (
        frame[["positive", "negative"]].max(axis=1) / directional).round(2)
    return frame.sort_values("dominant_share_directional", ascending=False)


def table_design_suggestions(units):
    reported = units[units["coder"] == AUTHOR_CODER]
    if "design_suggestion" not in reported:
        return pd.DataFrame()
    rows = [{"suggestion": str(row["design_suggestion"]).strip(),
             "participant": row["participant"]}
            for _, row in reported[reported["design_suggestion"].notna()].iterrows()]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    grouped = (frame.groupby(frame["suggestion"].str.lower())
               .agg(suggestion=("suggestion", "first"),
                    participants=("participant", "nunique"),
                    participant_list=("participant", lambda v: ", ".join(sorted(set(v)))),
                    units=("participant", "size"))
               .reset_index(drop=True))
    return grouped.sort_values("participants", ascending=False)


def table_quotable_units(units, codes=REPORTED_THEMES):
    """Every unit carrying a reported theme, with its text. Contains verbatim
    speech, so it stays private."""
    rows = []
    for _, row in units[units["coder"] == AUTHOR_CODER].iterrows():
        for code in [c for c in codes_in_row(row) if c in set(codes)]:
            rows.append({
                "code": code, "code_name": CODE_NAMES[code],
                "participant": row["participant"], "unit_id": row.get("unit_id"),
                "explanation_type": ", ".join(split_explanation_types(
                    row.get("explanation_type"), include_general=True)),
                "model": row.get("model"), "exposure": row.get("exposure"),
                "language": row.get("language"), "source": row.get("source"),
                "eval_target": row.get("eval_target"),
                "quotable": (row.get("scope") == SCOPE_CODEABLE
                             and row.get("source") != INTERVIEWER_SUPPLIED),
                "needs_translation": str(row.get("language")).lower() == "dutch",
                "text": row.get("text"),
            })
    return pd.DataFrame(rows)


def tables(units):
    """The public tables, and separately the two that contain transcript text."""
    by_code = lambda frame: frame.rename_axis("code").reset_index()
    public = {
        "corpus": table_corpus(units).reset_index(drop=True),
        "agreement": table_agreement(units),
        "agreement_by_session": table_agreement_by_session(units),
        "code_participants": by_code(table_code_participants(units)),
        "code_by_type": by_code(table_code_by_type(units)),
        "discriminating_codes": by_code(table_discriminating_codes(units)),
        "code_by_model": by_code(table_code_by_model(units)),
        "participant_themes": by_code(table_participant_themes(units)),
        "eval_target_by_type": table_eval_target_by_type(units)
            .rename_axis("eval_target").reset_index(),
        "source_composition": table_source_composition(units)
            .rename_axis("source").reset_index(),
        "valence_by_code": by_code(table_valence_by_code(units)),
        "design_suggestions": table_design_suggestions(units).reset_index(drop=True),
    }
    private = {
        "units_full": units[units["coder"] == AUTHOR_CODER].reset_index(drop=True),
        "quotable_units": table_quotable_units(units),
    }
    return public, private


# --- Figures -----------------------------------------------------------------

def _draw_grid(grid, name, sort_by_coverage=True):
    """Filled square where a participant raised the theme; totals on the right."""
    totals = grid.sum(axis=1)
    if sort_by_coverage:
        grid = grid.loc[totals.sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 0.18 * len(grid.index) + 0.75))
    for y, code in enumerate(grid.index):
        for x, participant in enumerate(grid.columns):
            ax.add_patch(plt.Rectangle(
                (x - 0.36, y - 0.36), 0.72, 0.72, edgecolor="none",
                facecolor=PAL["Case-based"] if grid.at[code, participant] else UNFILLED_COLOUR))
        ax.text(len(grid.columns) + 0.1, y, str(int(totals[code])), va="center",
                ha="center", fontsize=COUNT_SIZE, color="#666666")
    ax.text(len(grid.columns) + 0.1, -0.95, "Any\ntype", va="center", ha="center",
            fontsize=COUNT_SIZE, color="#666666", linespacing=0.95)
    ax.set_xlim(-0.6, len(grid.columns) - 0.4 + 1.2)
    ax.set_ylim(-1.3, len(grid.index) - 0.4)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([CODE_NAMES[code] for code in grid.index], fontsize=LABEL_SIZE)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, fontsize=LABEL_SIZE)
    ax.xaxis.set_ticks_position("top")
    ax.invert_yaxis()
    ax.tick_params(length=0)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, FIGURES, name)


def figure_eval_target_by_type(units):
    matrix = participants_per_target_and_type(units)
    if matrix.empty:
        return
    n = len(PARTICIPANT_ORDER)
    positions = np.arange(len(matrix.index))
    height = 0.8 / len(MODALITY_ORDER)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 0.62 * len(matrix.index) + 1.2))
    for i, explanation_type in enumerate(MODALITY_ORDER):
        ax.barh(positions + i * height, matrix[explanation_type], height=height,
                color=PAL[explanation_type], edgecolor="white", linewidth=0.6,
                label=explanation_type)
    ax.set_yticks(positions + 1.5 * height)
    ax.set_yticklabels([EVAL_TARGET_LABELS.get(t, t) for t in matrix.index], fontsize=LABEL_SIZE)
    ax.set_xlim(0, n)
    ax.set_xticks(range(0, n + 1, 2))
    ax.tick_params(axis="x", labelsize=LABEL_SIZE)
    ax.set_xlabel(f"Participants (of {n})", fontsize=LABEL_SIZE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=LABEL_SIZE, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_eval_target_by_type")


def figure_target_valence(units):
    """Coded units per type, negative to the left and positive to the right,
    split by what the remark was about."""
    counts = Counter()
    for _, row in author_units(units).iterrows():
        if pd.isna(row.get("eval_target")) or pd.isna(row.get("valence")):
            continue
        for explanation_type in split_explanation_types(row.get("explanation_type")):
            counts[(explanation_type, str(row["eval_target"]), str(row["valence"]))] += 1
    if not counts:
        return
    frame = pd.DataFrame([{"explanation_type": t, "eval_target": target, "valence": v,
                           "units": n} for (t, target, v), n in counts.items()])
    targets = [t for t in TARGET_ORDER if t in set(frame["eval_target"])]
    positions = np.arange(len(MODALITY_ORDER))
    extent = {}
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.4))
    for direction, sign in (("negative", -1.0), ("positive", 1.0)):
        offset = np.zeros(len(MODALITY_ORDER))
        for target in targets:
            widths = np.array([float(frame[(frame["explanation_type"] == t)
                                           & (frame["eval_target"] == target)
                                           & (frame["valence"] == direction)]["units"].sum())
                               for t in MODALITY_ORDER])
            ax.barh(positions, sign * widths, left=offset, height=0.62,
                    color=TARGET_COLOURS[target], edgecolor="white", linewidth=0.7,
                    label=EVAL_TARGET_LABELS[target] if direction == "positive" else None)
            offset = offset + sign * widths
        extent[direction] = offset

    totals = frame.groupby("explanation_type")["units"].sum()
    negative_limit = float(abs(extent["negative"]).max())
    positive_limit = float(extent["positive"].max())
    for i, explanation_type in enumerate(MODALITY_ORDER):
        ax.text(extent["positive"][i] + positive_limit * 0.06, i,
                f"n={int(totals.get(explanation_type, 0))}", va="center", ha="left",
                fontsize=COUNT_SIZE, color="#444444")
    ax.axvline(0, color="#666666", linewidth=0.8, zorder=4)
    ax.set_yticks(positions)
    ax.set_yticklabels(MODALITY_ORDER, fontsize=LABEL_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(-negative_limit * 1.06, positive_limit * 1.45)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: str(int(abs(value)))))
    ax.tick_params(axis="x", labelsize=LABEL_SIZE)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Coded units", fontsize=LABEL_SIZE, labelpad=14)
    below = ax.get_xaxis_transform()
    ax.text(-negative_limit * 0.5, -0.12, "negative", transform=below, ha="center",
            va="top", fontsize=LABEL_SIZE, color="#444444")
    ax.text(positive_limit * 0.5, -0.12, "positive", transform=below, ha="center",
            va="top", fontsize=LABEL_SIZE, color="#444444")
    ax.legend(frameon=False, fontsize=LABEL_SIZE, ncol=len(targets), loc="lower center",
              bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save_figure(fig, FIGURES, "fig_target_valence")


def write_figures(units):
    reported = reported_theme_grid(units)
    if not reported.empty:
        _draw_grid(reported, "fig_reported_themes", sort_by_coverage=False)
    figure_target_valence(units)
    figure_eval_target_by_type(units)
    grid = participant_theme_grid(units)
    if not grid.empty:
        _draw_grid(grid, "fig_participant_themes")
