"""Agreement between coders (Krippendorff's alpha).

For each main category, two alphas:
  - assigned: did both coders give the unit a code in this category at all;
  - subcategory: among units both coded in the category, the same code?

Comparisons with the model leave out P2, the session used to develop its prompt.
"""
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from xai_decision_making.analysis.common import PARTICIPANT_ORDER
from xai_decision_making.analysis.qualitative.coding import (
    AUTHOR_CODER, SECOND_CODER, MODEL_CODER, ENGLISH_SESSIONS, DUTCH_SESSIONS,
    PROMPT_DEVELOPMENT_SESSION, CODE_COLUMNS, MAIN_CATEGORY_LABELS, aligned_values)


def krippendorff_alpha(observations):
    """Krippendorff's alpha, nominal metric.

    observations: one list of values per unit, one value per coder (None or
    NaN for a missing value).
    """
    observed, totals, grand_total = defaultdict(float), defaultdict(float), 0.0
    for values in observations:
        present = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if len(present) < 2:
            continue
        for first, second in itertools.permutations(present, 2):
            observed[(first, second)] += 1.0 / (len(present) - 1)
        for value in present:
            totals[value] += 1.0
            grand_total += 1.0
    if grand_total < 2:
        return float("nan")
    disagreement = sum(count for (a, b), count in observed.items() if a != b)
    expected = sum(count_a * count_b / (grand_total - 1)
                   for a, count_a in totals.items() for b, count_b in totals.items() if a != b)
    return float("nan") if expected == 0 else 1.0 - disagreement / expected


def coder_agreement(units, coder_a, coder_b, participants=None, label=None):
    selected = units
    if MODEL_CODER in (coder_a, coder_b):
        selected = selected[selected["participant"] != PROMPT_DEVELOPMENT_SESSION]
    if participants:
        selected = selected[selected["participant"].isin(participants)]
    selected = selected[selected["coder"].isin([coder_a, coder_b])]
    rows = []
    for category, column in CODE_COLUMNS.items():
        if column not in selected:
            continue
        assigned, subcategory = [], []
        for _, a, b in aligned_values(selected, coder_a, coder_b, column):
            assigned.append([int(pd.notna(a)), int(pd.notna(b))])
            if pd.notna(a) and pd.notna(b):
                subcategory.append([a, b])
        if not assigned:
            continue
        rows.append({
            "pair": f"{coder_a} vs {coder_b}",
            "subset": label or ("all" if participants is None else ", ".join(participants)),
            "main_category": category,
            "main_category_name": MAIN_CATEGORY_LABELS[category],
            "units_compared": len(assigned),
            "alpha_assigned": round(krippendorff_alpha(assigned), 3),
            "units_both_assigned": len(subcategory),
            "alpha_subcategory": round(krippendorff_alpha(subcategory), 3),
        })
    return pd.DataFrame(rows)


def table_agreement(units):
    """All three pairs on the English sessions, and author vs model on the Dutch ones."""
    return pd.concat([
        coder_agreement(units, AUTHOR_CODER, SECOND_CODER, ENGLISH_SESSIONS, "English sessions"),
        coder_agreement(units, AUTHOR_CODER, MODEL_CODER, ENGLISH_SESSIONS, "English sessions"),
        coder_agreement(units, SECOND_CODER, MODEL_CODER, ENGLISH_SESSIONS, "English sessions"),
        coder_agreement(units, AUTHOR_CODER, MODEL_CODER, DUTCH_SESSIONS, "Dutch sessions"),
    ], ignore_index=True)


def table_agreement_by_session(units):
    frames = []
    for participant in PARTICIPANT_ORDER:
        if participant == PROMPT_DEVELOPMENT_SESSION:
            continue
        pairs = [(AUTHOR_CODER, MODEL_CODER)]
        if participant in ENGLISH_SESSIONS:
            pairs.append((AUTHOR_CODER, SECOND_CODER))
        for coder_a, coder_b in pairs:
            frame = coder_agreement(units, coder_a, coder_b, [participant], label=participant)
            if len(frame):
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
