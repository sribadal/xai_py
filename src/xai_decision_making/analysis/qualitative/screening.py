"""Which cross-tabulations of the coding are worth reporting in the main text.

Every row of each crossing (for example a code against explanation type) is
checked against fixed thresholds: enough participants, one column holding a
clear majority, and that column over-represented relative to the overall
column totals, also when any one participant is left out. Rows that pass go
to the main text, the rest to the supplement.

Not part of the default run: run_all.main(with_screening=True).
"""
import numpy as np
import pandas as pd

from xai_decision_making.analysis.common import (
    PARTICIPANT_ORDER, MODALITY_ORDER)
from xai_decision_making.analysis.qualitative.coding import (
    CODE_NAMES, author_units, codes_in_row, split_explanation_types)
from xai_decision_making.analysis.qualitative.themes import EVAL_TARGET_LABELS

MIN_PARTICIPANTS_TO_REPORT = 5
MIN_PARTICIPANTS_IN_PEAK_CELL = 4
MIN_PEAK_SHARE = 0.50
MIN_CONCENTRATION_RATIO = 1.50
MIN_CONCENTRATION_RATIO_LEAVE_ONE_OUT = 1.25
MIN_PARTICIPANTS_FOR_EXCLUSIVITY = 5
MAX_COLUMNS_FOR_EXCLUSIVITY_CLAIM = 2


def _field(name):
    def values(row):
        value = row.get(name)
        if pd.isna(value):
            return []
        return [str(value).upper()] if name == "model" else [str(value)]
    return values


def _types(row):
    return split_explanation_types(row.get("explanation_type"))


CROSSINGS = {
    "code by explanation type": (codes_in_row, _types),
    "code by evaluation target": (codes_in_row, _field("eval_target")),
    "code by model": (codes_in_row, _field("model")),
    "evaluation target by explanation type": (_field("eval_target"), _types),
    "valence by evaluation target": (_field("valence"), _field("eval_target")),
    "valence by explanation type": (_field("valence"), _types),
    "valence by model": (_field("valence"), _field("model")),
    "source by explanation type": (_field("source"), _types),
}
ROW_LABELS = {**CODE_NAMES, **EVAL_TARGET_LABELS}


def participants_per_cell(units, row_field, column_field, exclude=()):
    reported = author_units(units)
    if exclude:
        reported = reported[~reported["participant"].isin(exclude)]
    cells = {(r, c, row["participant"]) for _, row in reported.iterrows()
             for r in row_field(row) for c in column_field(row)}
    if not cells:
        return pd.DataFrame()
    column_keys = {c for _, c, _ in cells}
    columns = [c for c in MODALITY_ORDER if c in column_keys]
    columns += sorted(c for c in column_keys if c not in columns)
    matrix = pd.DataFrame(0, index=sorted({r for r, _, _ in cells}), columns=columns, dtype=int)
    for r, c, _ in cells:
        matrix.at[r, c] += 1
    return matrix


def participants_per_row(units, row_field):
    pairs = {(key, row["participant"]) for _, row in author_units(units).iterrows()
             for key in row_field(row)}
    counts = pd.Series(0, index=sorted({key for key, _ in pairs}), dtype=int)
    for key, _ in pairs:
        counts[key] += 1
    return counts


def concentration(matrix):
    """For each row: the peak column, its share of the row, and how much more
    often the row falls there than the column total would predict."""
    column_share = matrix.sum(axis=0) / max(matrix.values.sum(), 1)
    row_share = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
    lift = row_share.div(column_share.replace(0, np.nan), axis=1)
    peak = matrix.idxmax(axis=1)
    peak_lift = pd.Series([lift.at[k, peak[k]] for k in matrix.index], index=matrix.index).fillna(0)
    peak_share = pd.Series([row_share.at[k, peak[k]] for k in matrix.index],
                           index=matrix.index).fillna(0)
    return peak_lift, peak, peak_share


def leave_one_out_concentration(units, row_field, column_field):
    worst = {}
    for dropped in PARTICIPANT_ORDER:
        matrix = participants_per_cell(units, row_field, column_field, exclude=[dropped])
        if matrix.empty:
            continue
        for key, value in concentration(matrix)[0].items():
            worst[key] = min(worst.get(key, value), value)
    return pd.Series(worst, dtype=float)


def screen_crossing(units, name, row_field, column_field):
    matrix = participants_per_cell(units, row_field, column_field)
    if matrix.empty:
        return pd.DataFrame()
    totals = participants_per_row(units, row_field)
    lift, peak_column, peak_share = concentration(matrix)
    worst_lift = leave_one_out_concentration(units, row_field, column_field)

    rows = []
    for key in matrix.index:
        present = matrix.loc[key]
        nonzero = present[present > 0]
        total = int(totals.get(key, present.max()))
        exclusive = (nonzero.index[0] if len(nonzero) == 1
                     and matrix.shape[1] > MAX_COLUMNS_FOR_EXCLUSIVITY_CLAIM else "")
        peak_count = int(present[peak_column[key]])
        substantial = (peak_count >= MIN_PARTICIPANTS_IN_PEAK_CELL
                       and peak_share[key] >= MIN_PEAK_SHARE)
        concentrated = lift[key] >= MIN_CONCENTRATION_RATIO
        robust = worst_lift.get(key, 0) >= MIN_CONCENTRATION_RATIO_LEAVE_ONE_OUT

        if total < MIN_PARTICIPANTS_TO_REPORT:
            verdict, reason = "supplementary", "too few participants"
        elif exclusive and total >= MIN_PARTICIPANTS_FOR_EXCLUSIVITY:
            verdict, reason = "main text", f"raised only for {exclusive}"
        elif not substantial:
            verdict, reason = "supplementary", "no column holds a clear majority"
        elif concentrated and robust:
            verdict, reason = "main text", f"concentrated on {peak_column[key]}"
        elif concentrated:
            verdict, reason = "supplementary", "concentration not robust to one participant"
        else:
            verdict, reason = "supplementary", "matches the overall column pattern"

        rows.append({
            "crossing": name, "row": key, "row_label": ROW_LABELS.get(key, key),
            "participants": total, "peak_column": peak_column[key],
            "peak_participants": peak_count, "peak_share": round(float(peak_share[key]), 2),
            "concentration": round(float(lift[key]), 2),
            "concentration_leave_one_out": round(float(worst_lift.get(key, 0)), 2),
            "exclusive_to": exclusive,
            "distribution": " / ".join(f"{c}:{present[c]}" for c in matrix.columns),
            "verdict": verdict, "reason": reason,
        })
    return pd.DataFrame(rows)


def build(units):
    """Return the full inventory and a summary per crossing."""
    frames = [screen_crossing(units, name, row_field, column_field)
              for name, (row_field, column_field) in CROSSINGS.items()]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    inventory = pd.concat(frames, ignore_index=True).sort_values(
        ["verdict", "concentration_leave_one_out", "participants"],
        ascending=[True, False, False])

    summary = []
    for name, group in inventory.groupby("crossing", sort=False):
        promoted = group[group["verdict"] == "main text"]
        summary.append({
            "crossing": name, "rows_screened": len(group),
            "rows_for_main_text": len(promoted),
            "evenly_spread": int(group["reason"].isin(
                ["matches the overall column pattern", "no column holds a clear majority"]).sum()),
            "strongest_row": promoted["row_label"].iloc[0] if len(promoted) else "",
            "strongest_concentration": promoted["concentration"].iloc[0] if len(promoted) else 0.0,
        })
    summary = pd.DataFrame(summary).sort_values("rows_for_main_text", ascending=False)
    print(summary.to_string(index=False))
    print()
    print(inventory[inventory["verdict"] == "main text"][
        ["crossing", "row_label", "participants", "distribution", "reason"]].to_string(index=False))
    return inventory, summary
