# Analysis

Analysis of the clinician study: the survey ratings and rankings
(quantitative) and the coded think-aloud sessions (qualitative).

```
uv run xai analyze                           # both parts
uv run xai analyze quantitative              # survey only
uv run xai analyze qualitative               # coding sheets only
uv run xai analyze qualitative --screening   # plus the screening tables
```

Without coding sheets, `xai analyze` runs the survey part and skips the
qualitative part; `xai analyze qualitative` stops with an error.

## Layout

```
analysis/
  run_all.py              runs everything below
  common.py               paths, study constants, survey loader
  quantitative/
    inferential.py        Friedman, Wilcoxon, model comparison, rankings,
                          background correlations, purpose shift
    descriptive.py        descriptive tables, all survey figures, report.txt
    sensitivity.py        v1/v2 placement, N=9 and N=10 reruns, tied raters,
                          leave-one-out correlations
  qualitative/
    coding.py             coding frame, reading the sheets, sheet checks
    agreement.py          Krippendorff's alpha between coders
    themes.py             theme counts, tables and figures
    screening.py          which cross-tabulations go in the main text
                          (not in the default run)
```

## Input

- `data/survey_responses.csv` and `data/survey_responses_followup.csv`:
  the de-identified survey exports (P1–P9 and P10–P11). `common.PARTICIPANTS`
  maps each session to its participant number.
- `data/coding_sheets/`: one `.xlsx` per session and coder, with the
  participant and coder in the file name (`coding_sheet_P3_shivam.xlsx`).
  The sheets contain transcript text and are not public; without them the
  qualitative part is skipped.

## Output

```
output/
  quantitative/
    quantitative_tables.xlsx   contents tab, then one tab per table
                               (tabs ending in _v1only are the N=9 reruns)
    figures/                   fig_*.pdf, one file per figure
    report.txt                 summary of the main results
    version_checks.txt         summary of the sensitivity checks
  qualitative/
    qualitative_tables.xlsx    contents tab, then one tab per table
    figures/                   fig_*.pdf
    audit.txt                  problems found in the coding sheets
    private/
      units_with_text.xlsx     the two tables with transcript text; not public
```

Every table is described on the contents tab of its workbook. The contents
come from `DESCRIPTIONS` in `quantitative/__init__.py` and
`qualitative/__init__.py`.