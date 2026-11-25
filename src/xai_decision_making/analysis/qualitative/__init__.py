"""Think-aloud analysis: coding sheets, agreement and themes (Study 2)."""
from xai_decision_making.analysis.common import QUALITATIVE_DIR, write_workbook
from xai_decision_making.analysis.qualitative import coding, patterns, screening, themes

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
    "participant_code_matrix": "Units per participant and code, for all 43 codes.",
    "code_concentration": "Per code: units, participants, and the share of its units from the "
                          "one participant who raised it most.",
    "co_occurrence": "Code pairs from different main categories on the same unit, and whether "
                     "the coding frame anticipated the pair.",
    "co_occurrence_by_category": "How often each pair of main categories is coded on the "
                                 "same unit.",
    "type_profile": "Per explanation type, the share of its codes in each main category, "
                    "pooled and averaged over participants.",
    "by_position": "Share of units with a code in each main category, by vignette position "
                   "(descriptive; position mixes type and model).",
    "by_exposure": "The same by first or second exposure to a type (the second exposure is "
                   "always the other model).",
    "readability_by_half": "Per participant, the share of units with a readability code in "
                           "the first and last four vignettes.",
    "valence_by_target": "Positive, neutral and negative units per evaluation target.",
    "valence_target_by_participant": "The same per participant.",
    "deviant_cases": "Each participant's category profile against the others, largest "
                     "difference first: who to read closely.",
    "co_occurrence_quotes": "Up to three units with text per co-occurring pair not in the "
                            "frame, from different participants.",
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
    pattern_tables, pattern_private = patterns.tables(units)
    public.update(pattern_tables)
    private.update(pattern_private)
    if with_screening:
        public["screening_inventory"], public["screening_summary"] = screening.build(units)
    write_workbook(WORKBOOK, public, DESCRIPTIONS)
    write_workbook(PRIVATE_WORKBOOK, private, DESCRIPTIONS)
    themes.write_figures(units)
    patterns.write_figures(units)
    return units
