"""Run the full results pipeline for Study 2 & 3."""

from xai_decision_making.analysis import common, participants, survey_analysis, outlier_and_groups, report
from xai_decision_making.analysis import statistical_analysis as stats


def main():
    data = common.load()

    participants.build()
    survey_analysis.build()

    friedman, wilcoxon, model, rank_f, rank_w, bg = stats.run(
        data.likert, common.Q_ORDER, data.likert, data.ranks, common.RANK_IDS, data.demo)

    bg.round(3).to_csv(common.OUT / "table_background_correlations.csv", index=False)

    friedman.round(3).to_csv(common.OUT / "table_friedman.csv", index=False)
    wilcoxon.round(3).to_csv(common.OUT / "table_wilcoxon.csv", index=False)
    model.round(3).to_csv(common.OUT / "table_model_comparison.csv", index=False)
    rank_f.round(3).to_csv(common.OUT / "table_rank_friedman.csv", index=False)
    rank_w.round(3).to_csv(common.OUT / "table_rank_wilcoxon.csv", index=False)

    outlier_and_groups.build()
    report.build()

    print(f"Done. N={data.N}. Figures, tables, and report.txt in {common.OUT}")


if __name__ == "__main__":
    main()