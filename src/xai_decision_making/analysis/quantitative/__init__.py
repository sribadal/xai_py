"""Survey analysis: ratings and rankings (Study 2)."""
from xai_decision_making.analysis.common import QUANTITATIVE_DIR, write_workbook
from xai_decision_making.analysis.quantitative import descriptive, inferential, sensitivity

WORKBOOK = QUANTITATIVE_DIR / "quantitative_tables.xlsx"

_TESTS_V1 = "The same test on the thesis sample, P1-P9 (interface v1)."
DESCRIPTIONS = {
    "participants": "Participant background: role, experience, AI familiarity, specialty, "
                    "interface version.",
    "descriptives": "Mean, median and SD per type and dimension, over all ratings and over "
                    "participant means (each participant's two models averaged).",
    "rankings": "Borda score and mean rank per type, for both ranking questions.",
    "positions": "Position of each type (1 = best) on each rating dimension and ranking.",
    "participant_profiles": "Each participant's mean and SD, and their highest- and "
                            "lowest-rated type.",
    "friedman": "Friedman test across the four types, per dimension.",
    "wilcoxon": "Pairwise Wilcoxon with Holm correction, for dimensions where Friedman "
                "is significant.",
    "model_comparison": "Wilcoxon, RSF against ConDOR, per dimension.",
    "rank_friedman": "Friedman test on each ranking question.",
    "rank_wilcoxon": "Pairwise Wilcoxon with Holm correction, for rankings where Friedman "
                     "is significant.",
    "background_correlations": "Spearman correlations of experience and AI familiarity with "
                               "each dimension, Holm-corrected across all six.",
    "purpose_shift": "Change in each type's rank between overall preference and patient "
                     "communication (Wilcoxon; rule-based primary, others Holm-corrected).",
    "version_placement": "Where P10 and P11 (interface v2) fall within the v1 ratings, per "
                         "type and dimension.",
    "version_placement_focused": "The same, for actionability on the two types changed in v2.",
    "tied_raters": "Participants with the same mean for all four types, and the resulting "
                   "deflation of Kendall's W.",
    "ordering_by_subset": "Order of the four types per dimension in N=11, N=10 (without P10) "
                          "and N=9 (v1 only).",
    "ordering_stability": "Whether that order is the same in all three subsets.",
    "robustness_friedman": "Friedman per dimension in N=11, N=10 and N=9.",
    "robustness_model": "RSF against ConDOR in N=11, N=10 and N=9.",
    "robustness_rankings": "Friedman on the rankings in N=11, N=10 and N=9.",
    "background_correlations_v1only": _TESTS_V1,
    "friedman_v1only": _TESTS_V1,
    "wilcoxon_v1only": _TESTS_V1,
    "model_comparison_v1only": _TESTS_V1,
    "rank_friedman_v1only": _TESTS_V1,
    "rank_wilcoxon_v1only": _TESTS_V1,
    "correlation_loo": "Background correlations recomputed with each participant left out.",
    "correlation_loo_summary": "Range of each correlation across the leave-one-out runs, and "
                               "how often it stays significant.",
}

_ORDER = ["participants", "descriptives", "rankings", "positions", "participant_profiles",
          "friedman", "wilcoxon", "model_comparison", "rank_friedman", "rank_wilcoxon",
          "background_correlations", "purpose_shift"]


def build(data):
    tables = descriptive.build(data)
    tests = inferential.all_tests(data)
    tables.update({name: frame.round(3) for name, frame in tests.items()})
    tables["purpose_shift"] = inferential.purpose_shift(data.ranks).round(3)
    tables = {name: tables[name] for name in _ORDER}
    tables.update(sensitivity.build(data))
    write_workbook(WORKBOOK, tables, DESCRIPTIONS)
    descriptive.write_report(data, tests)
