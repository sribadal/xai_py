"""Think-aloud analysis: coding sheets, agreement and themes (Study 2)."""
from xai_decision_making.analysis.common import QUALITATIVE_DIR, write_workbook
from xai_decision_making.analysis.qualitative import coding, screening, themes

WORKBOOK = QUALITATIVE_DIR / "qualitative_tables.xlsx"
PRIVATE_WORKBOOK = QUALITATIVE_DIR / "private" / "units_with_text.xlsx"

DESCRIPTIONS = {
    "corpus": "Units per participant: codeable, out of scope, interviewer-supplied, reported.",
    "agreement": "Krippendorff's alpha per coder pair and main category: whether a code was "
                 "given at all, and which code among units both coders coded.",
    "agreement_by_session": "The same, per session.",
    "code_participants": "Participants raising each code.",
    "code_by_type": "Participants raising each code, per explanation type, and the spread "
                    "between the most and least frequent type.",
    "discriminating_codes": "Codes whose count differs by 3 or more participants between types.",
    "code_by_model": "Participants raising each code, per model.",
    "participant_themes": "Themes by participant: the grid behind fig_participant_themes.",
    "eval_target_by_type": "Participants commenting on each evaluation target, per type.",
    "source_composition": "Units by source (spontaneous, prompted, interviewer-supplied), "
                          "per type.",
    "valence_by_code": "Positive, neutral and negative units per code.",
    "design_suggestions": "Design suggestions and how many participants made each.",
    "screening_inventory": "Every cross-tabulation row checked for the main text, with the "
                           "verdict and reason.",
    "screening_summary": "Rows screened and promoted, per cross-tabulation.",
    "units_full": "Every unit as coded by the first author, with transcript text.",
    "quotable_units": "Units carrying a reported theme, with transcript text.",
}


def build(sheet_dir=None, with_screening=False):
    units, problems = coding.load_units(sheet_dir)
    coding.write_audit(units, problems)
    public, private = themes.tables(units)
    if with_screening:
        public["screening_inventory"], public["screening_summary"] = screening.build(units)
    write_workbook(WORKBOOK, public, DESCRIPTIONS)
    write_workbook(PRIVATE_WORKBOOK, private, DESCRIPTIONS)
    themes.write_figures(units)
    return units
