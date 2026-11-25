"""Run the analysis: `uv run xai analyze [quantitative|qualitative] [--screening]`.

Reads the survey exports from data/ and the coding sheets from
data/coding_sheets/. Each part writes a workbook of tables, its figures and a
text summary to output/quantitative/ or output/qualitative/.
"""
import argparse

from xai_decision_making.analysis import common, qualitative, quantitative


def run_quantitative():
    data = common.load()
    quantitative.build(data)
    print(f"Quantitative: N={data.N}, output in {common.QUANTITATIVE_DIR}")


def run_qualitative(with_screening=False, required=True):
    if not any(common.SHEET_DIR.rglob("*.xls[xm]")):
        message = f"no coding sheets in {common.SHEET_DIR}"
        if required:
            raise SystemExit(f"Qualitative: {message}")
        print(f"Qualitative: skipped, {message}")
        return
    qualitative.build(with_screening=with_screening)
    print(f"Qualitative: output in {common.QUALITATIVE_DIR}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="xai analyze", description="Analysis of the clinician study (Study 2).")
    parser.add_argument("part", nargs="?", default="all",
                        choices=["all", "quantitative", "qualitative"],
                        help="which part to run (default: both)")
    parser.add_argument("--screening", action="store_true",
                        help="also screen the qualitative cross-tabulations")
    args = parser.parse_args(argv)

    if args.part in ("all", "quantitative"):
        run_quantitative()
    if args.part in ("all", "qualitative"):
        run_qualitative(args.screening, required=args.part == "qualitative")


if __name__ == "__main__":
    main()