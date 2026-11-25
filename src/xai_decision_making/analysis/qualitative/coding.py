"""The coding frame, reading the coding sheets, and checks on them.

Each session has one sheet per coder: the first author (all eleven), the
second coder (P1 and P4, the English sessions) and the model (all eleven;
P2 was used to develop the prompt). A sheet has one row per unit with up to
one code in each of the four main categories, A to D.
"""
import glob
import itertools
import os
import re

import numpy as np
import pandas as pd

from xai_decision_making.analysis.common import (
    QUALITATIVE_DIR, SHEET_DIR, PARTICIPANT_ORDER, INTERFACE_VERSION)

# --- Coders and sessions -----------------------------------------------------

AUTHOR_CODER = "shivam"
SECOND_CODER = "shahab"
MODEL_CODER = "model"
CODERS = [AUTHOR_CODER, SECOND_CODER, MODEL_CODER]
ENGLISH_SESSIONS = ["P1", "P4"]
DUTCH_SESSIONS = [p for p in PARTICIPANT_ORDER if p not in ENGLISH_SESSIONS]
SHEETS_EXPECTED = {AUTHOR_CODER: PARTICIPANT_ORDER, SECOND_CODER: ENGLISH_SESSIONS,
                   MODEL_CODER: PARTICIPANT_ORDER}
PROMPT_DEVELOPMENT_SESSION = "P2"

# --- Coding frame ------------------------------------------------------------

MAIN_CATEGORIES = ["A", "B", "C", "D"]
MAIN_CATEGORY_LABELS = {"A": "Readability", "B": "Fit with clinical reasoning",
                        "C": "Credibility and checking", "D": "Usefulness for a next step"}
CODE_COLUMNS = {category: f"code_{category}" for category in MAIN_CATEGORIES}
CODES_PER_CATEGORY = {
    "A": [f"A{n}" for n in range(1, 10)] + ["A99"],
    "B": [f"B{n}" for n in range(1, 9)] + ["B99"],
    "C": [f"C{n}" for n in range(1, 15)] + ["C99"],
    "D": [f"D{n}" for n in range(1, 9)] + ["D99"],
}
ALL_CODES = [code for category in MAIN_CATEGORIES for code in CODES_PER_CATEGORY[category]]
RESIDUAL_SUFFIX = "99"  # A99, B99, ...: "other remark" in each category

CODE_NAMES = {
    "A1": "Reads at once",
    "A2": "Volume overwhelms",
    "A3": "Value hidden behind interaction",
    "A4": "Label too terse",
    "A5": "Label too verbose",
    "A6": "Needs outside help",
    "A7": "Wants a key or heading",
    "A8": "Numeric format hard to grasp",
    "A9": "Encoding changes between panels",
    "A99": "Other remark about readability",
    "B1": "Matches how they reason",
    "B2": "Rejects a variable shown",
    "B3": "Misses a variable",
    "B4": "Variable cannot be relied on as shown",
    "B5": "Too many variables",
    "B6": "Builds a reason for an unexpected variable",
    "B7": "Change is not achievable",
    "B8": "Gain too small to matter",
    "B99": "Other remark about clinical fit",
    "C1": "Plausible on sight",
    "C2": "Accepts without checking",
    "C3": "Not believable overall",
    "C4": "Explanation changes nothing",
    "C5": "Comparison patients not comparable",
    "C6": "Panel disagrees with itself",
    "C7": "Panel disagrees with clinical expectation",
    "C8": "Two explanation types disagree",
    "C9": "Something on screen cannot be right",
    "C10": "Missing data blocks judgement",
    "C11": "Cannot verify without the other variables",
    "C12": "Threshold or label undefined",
    "C13": "Statistic not understood",
    "C14": "Declines to judge",
    "C99": "Other remark about credibility",
    "D1": "Leads to a concrete step",
    "D2": "Nothing follows",
    "D3": "Helps them place the patient",
    "D4": "Suitable for patient conversation",
    "D5": "Unsuitable for patient conversation",
    "D6": "Direction without size",
    "D7": "Would work with treatment variables",
    "D8": "Wants more comparison material",
    "D99": "Other remark about usefulness",
}

# --- Other unit fields -------------------------------------------------------

SCOPE_CODEABLE = "codeable"
SCOPE_VALUES = {SCOPE_CODEABLE, "out_of_scope", "insufficient_context", "not_coded"}
EVAL_TARGETS = {"prediction", "explanation", "input_data", "retrieved_cases", "interface"}
VALENCES = {"positive", "negative", "neutral"}
INTERVIEWER_SUPPLIED = "interviewer_supplied"
SOURCES = {"spontaneous", "prompted", INTERVIEWER_SUPPLIED}
GENERAL_TYPE = "General"

# Filled in by the first author and given to the model as input, so they
# should be identical across a session's sheets.
AUTHOR_FILLED_COLUMNS = ["explanation_type", "model", "horizon", "vignette_order",
                         "exposure", "data_source", "language"]

# Spellings found in the sheets, mapped to the values above.
CODE_NAME_ALIASES = {"A99": ["Other readability remark"], "B99": ["Other clinical fit remark"],
                     "C99": ["Other credibility remark"], "D99": ["Other usefulness remark"]}
VALUE_ALIASES = {
    "out of scope": "out_of_scope", "outofscope": "out_of_scope",
    "insufficient context": "insufficient_context", "not coded": "not_coded",
    "input data": "input_data", "retrieved cases": "retrieved_cases",
    "interviewer supplied": INTERVIEWER_SUPPLIED,
    "prediction or model": "prediction", "model": "prediction",
    "pos": "positive", "neg": "negative", "neu": "neutral",
}
DATA_SOURCE_ALIASES = {
    "transcript": "transcript", "transcript_unchecked": "transcript",
    "verbatim transcript": "transcript", "whisper_with_audio": "whisper_with_audio",
    "notes and recording": "notes_and_recording",
    "interviewer notes and screen recording": "notes_and_recording",
}
LANGUAGE_ALIASES = {"en": "english", "english": "english", "nl": "dutch", "dutch": "dutch"}
TYPE_ALIASES = {
    "attribution": "Attribution", "counterfactual": "Counterfactual",
    "case based": "Case-based", "case-based": "Case-based",
    "case_based": "Case-based", "casebased": "Case-based",
    "rule based": "Rule-based", "rule-based": "Rule-based",
    "rule_based": "Rule-based", "rulebased": "Rule-based",
    "general": GENERAL_TYPE,
}


def _squash(text):
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()


NAME_TO_CODE = {_squash(name): code for code, name in CODE_NAMES.items()}
for _code, _aliases in CODE_NAME_ALIASES.items():
    for _alias in _aliases:
        NAME_TO_CODE[_squash(_alias)] = _code


# --- Reading sheets ----------------------------------------------------------

def find_sheet(participant, coder, sheet_dir=None):
    """The one .xlsx under sheet_dir whose name contains both participant and coder."""
    sheet_dir = str(sheet_dir or SHEET_DIR)
    matches = set()
    for pattern in ("*.xlsx", "*.xlsm"):
        for path in glob.glob(os.path.join(sheet_dir, "**", pattern), recursive=True):
            name = os.path.basename(path).lower()
            if name.startswith("~$"):  # Excel lock file
                continue
            if (re.search(rf"(?<![a-z0-9]){participant.lower()}(?![0-9])", name)
                    and re.search(rf"(?<![a-z0-9]){re.escape(coder.lower())}(?![a-z0-9])", name)):
                matches.add(path)
    if len(matches) > 1:
        raise ValueError(f"{participant}/{coder}: {len(matches)} candidates: {sorted(matches)}")
    return matches.pop() if matches else None


def read_sheet(path, participant):
    """The tab named after the participant, else the first tab with a unit_id column."""
    workbook = pd.ExcelFile(path)
    tab = next((name for name in workbook.sheet_names
                if name.strip().lower() == participant.lower()), None)
    if tab is None:
        tab = next((name for name in workbook.sheet_names
                    if "unit_id" in [str(c).strip() for c in workbook.parse(name, nrows=0).columns]),
                   None)
    if tab is None:
        raise ValueError(f"{os.path.basename(path)}: no tab named {participant!r} and no tab "
                         f"with a unit_id column; tabs are {workbook.sheet_names}")
    frame = workbook.parse(tab, dtype=object)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def normalise_token(value, aliases=None):
    if pd.isna(value) or not str(value).strip():
        return np.nan
    key = str(value).strip().lower()
    if aliases and key in aliases:
        return aliases[key]
    return key.replace(" ", "_")


def normalise_code(value):
    """One code per cell; a code name is accepted in place of its number."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        return np.nan
    if re.search(r"[;,/]", text):
        raise ValueError(f"multiple codes in one cell: {value!r}. Codes within a main "
                         "category are mutually exclusive; split the unit or pick one.")
    if text in ALL_CODES:
        return text
    return NAME_TO_CODE.get(_squash(value), text)


def normalise_units(frame):
    frame = frame.copy()
    for column in ["scope", "eval_target", "valence", "source"]:
        if column in frame:
            frame[column] = frame[column].map(lambda v: normalise_token(v, VALUE_ALIASES))
    if "data_source" in frame:
        frame["data_source"] = frame["data_source"].map(
            lambda v: normalise_token(v, DATA_SOURCE_ALIASES))
    if "language" in frame:
        frame["language"] = frame["language"].map(lambda v: normalise_token(v, LANGUAGE_ALIASES))
    for column in CODE_COLUMNS.values():
        if column in frame:
            frame[column] = frame[column].map(normalise_code)
    for column in ["explanation_type", "model", "exposure", "horizon", "unit_id"]:
        if column in frame:
            frame[column] = frame[column].map(lambda v: np.nan if pd.isna(v) else str(v).strip())
    if "vignette_order" in frame:
        frame["vignette_order"] = pd.to_numeric(frame["vignette_order"], errors="coerce")
    return frame


def load_units(sheet_dir=None, verbose=True):
    """All sheets as one frame with a coder column, plus a list of load problems."""
    frames, problems = [], []
    for coder in CODERS:
        for participant in SHEETS_EXPECTED[coder]:
            try:
                path = find_sheet(participant, coder, sheet_dir)
            except ValueError as error:
                problems.append(str(error))
                continue
            if path is None:
                problems.append(f"MISSING {participant}/{coder}: no file found")
                continue
            try:
                frame = normalise_units(read_sheet(path, participant))
            except ValueError as error:
                raise ValueError(f"{participant}/{coder}: {error}") from error
            frame["coder"] = coder
            frame["source_file"] = os.path.basename(path)
            if "participant" not in frame or frame["participant"].isna().all():
                frame["participant"] = participant
            frame["version"] = INTERFACE_VERSION.get(participant)
            frames.append(frame)
            if verbose:
                print(f"  {participant:>4}/{coder:<7} {len(frame):>4} rows")
    if not frames:
        raise SystemExit(f"No sheets found under {sheet_dir or SHEET_DIR}")
    return pd.concat(frames, ignore_index=True, sort=False), problems


# --- Helpers used by the tables ----------------------------------------------

def author_units(units, codeable_only=True, exclude_interviewer_supplied=True):
    """The first author's units, as reported in the paper.

    By default only codeable units that the participant raised themselves.
    """
    selected = units[units["coder"] == AUTHOR_CODER]
    if codeable_only and "scope" in selected:
        selected = selected[selected["scope"] == SCOPE_CODEABLE]
    if exclude_interviewer_supplied and "source" in selected:
        selected = selected[selected["source"] != INTERVIEWER_SUPPLIED]
    return selected


def canonical_type(value):
    return TYPE_ALIASES.get(str(value).strip().lower(), str(value).strip())


def split_explanation_types(value, include_general=False):
    """A unit can be about more than one type ("Attribution; Case-based")."""
    if pd.isna(value):
        return []
    parts = [canonical_type(part) for part in re.split(r"[;/,]", str(value)) if part.strip()]
    return parts if include_general else [part for part in parts if part != GENERAL_TYPE]


def codes_in_row(row):
    return [row[column] for column in CODE_COLUMNS.values()
            if pd.notna(row.get(column)) and row.get(column) in ALL_CODES]


def codes_in_row_excluding_residuals(row):
    return [code for code in codes_in_row(row) if not code.endswith(RESIDUAL_SUFFIX)]


def aligned_values(units, coder_a, coder_b, column):
    """(key, value_a, value_b) for every unit both coders have."""
    values_a = units[units["coder"] == coder_a].set_index(["participant", "unit_id"])[column]
    values_b = units[units["coder"] == coder_b].set_index(["participant", "unit_id"])[column]
    for coder, values in ((coder_a, values_a), (coder_b, values_b)):
        duplicated = sorted(set(values.index[values.index.duplicated()]))
        if duplicated:
            raise ValueError(f"{coder}: duplicate (participant, unit_id) keys {duplicated[:5]}"
                             f"{' ...' if len(duplicated) > 5 else ''}; fix the coding sheet "
                             "before computing agreement")
    shared = values_a.index.intersection(values_b.index)
    return [(key, values_a.loc[key], values_b.loc[key]) for key in shared]


# --- Checks ------------------------------------------------------------------

def audit_units(units):
    """Values outside the frame, duplicate ids, and codes that contradict scope."""
    findings = []
    for column, allowed in [("scope", SCOPE_VALUES), ("eval_target", EVAL_TARGETS),
                            ("valence", VALENCES), ("source", SOURCES)]:
        if column in units:
            unexpected = set(units[column].dropna().unique()) - allowed
            if unexpected:
                findings.append(f"{column}: unrecognised values {sorted(unexpected)}")
    for (participant, coder), sheet in units.groupby(["participant", "coder"]):
        label = f"{participant}/{coder}"
        ids = sheet["unit_id"].dropna()
        duplicated = sorted(set(ids[ids.duplicated()]))
        if duplicated:
            findings.append(f"{label}: duplicate unit_id {duplicated}")
        if sheet["unit_id"].isna().any():
            findings.append(f"{label}: {int(sheet['unit_id'].isna().sum())} rows with no unit_id")
        for category, column in CODE_COLUMNS.items():
            if column not in sheet:
                findings.append(f"{label}: missing column {column}")
                continue
            unresolved = sheet[sheet[column].notna()
                               & ~sheet[column].isin(CODES_PER_CATEGORY[category])]
            for _, row in unresolved.iterrows():
                findings.append(f"{label}: {row['unit_id']} unresolved {category} "
                                f"value {row[column]!r}")
        if "scope" not in sheet:
            findings.append(f"{label}: missing scope column")
            continue
        has_code = sheet.apply(lambda row: bool(codes_in_row(row)), axis=1)
        for _, row in sheet[(sheet["scope"] != SCOPE_CODEABLE)
                            & sheet["scope"].notna() & has_code].iterrows():
            findings.append(f"{label}: {row['unit_id']} scope={row['scope']} but carries a code")
        for _, row in sheet[(sheet["scope"] == SCOPE_CODEABLE) & ~has_code].iterrows():
            findings.append(f"{label}: {row['unit_id']} codeable but has no code")
        blank_scope = int(sheet["scope"].isna().sum())
        if blank_scope:
            findings.append(f"{label}: {blank_scope} rows with blank scope")
    return findings


def audit_corpus_arithmetic(units):
    """The paper's 'codeable minus interviewer-supplied = reported' needs every
    interviewer-supplied unit to be codeable."""
    reported = units[units["coder"] == AUTHOR_CODER]
    if "scope" not in reported or "source" not in reported:
        return []
    outside = reported[(reported["source"] == INTERVIEWER_SUPPLIED)
                       & (reported["scope"] != SCOPE_CODEABLE)]
    return [f"{participant}: {len(sheet)} interviewer-supplied units outside the codeable "
            "set; codeable minus interviewer_supplied will not equal reported"
            for participant, sheet in outside.groupby("participant")]


def audit_coder_alignment(units):
    """Coders should have the same units and the same author-filled columns."""
    findings = []
    for participant, sheet in units.groupby("participant"):
        ids_by_coder = {coder: set(g["unit_id"].dropna()) for coder, g in sheet.groupby("coder")}
        if len(ids_by_coder) < 2:
            continue
        reference = ids_by_coder.get(AUTHOR_CODER) or next(iter(ids_by_coder.values()))
        for coder, ids in ids_by_coder.items():
            if ids != reference:
                findings.append(f"{participant}: row-set mismatch against {coder} "
                                f"({len(reference - ids)} missing, {len(ids - reference)} extra)")
    for column in AUTHOR_FILLED_COLUMNS:
        if column not in units:
            continue
        for coder_a, coder_b in itertools.combinations(sorted(units["coder"].unique()), 2):
            differences = sum(
                1 for _, a, b in aligned_values(units, coder_a, coder_b, column)
                if not (pd.isna(a) and pd.isna(b)) and a != b)
            if differences:
                findings.append(
                    f"DATA INTEGRITY {column}: {differences} units carry different values "
                    f"in the {coder_a} and {coder_b} sheets. This column is supplied, not "
                    "coded, so this is a sheet mismatch rather than coder disagreement.")
    return findings


def write_audit(units, problems):
    lines = ["LOAD PROBLEMS"] + (problems or ["  none"])
    lines += ["", "UNIT AUDIT"] + (audit_units(units) or ["  none"])
    lines += ["", "CORPUS ARITHMETIC"] + (audit_corpus_arithmetic(units) or ["  none"])
    lines += ["", "CODER ALIGNMENT"] + (audit_coder_alignment(units) or ["  none"])
    QUALITATIVE_DIR.mkdir(parents=True, exist_ok=True)
    (QUALITATIVE_DIR / "audit.txt").write_text("\n".join(map(str, lines)), encoding="utf-8")
