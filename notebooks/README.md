# Notebooks

Two notebooks that walk through the analysis of the clinician study, table by
table and figure by figure, with a short explanation of each. They are saved
with their output, so you can read them here on GitHub without installing or
running anything.

| Notebook | Contents |
|---|---|
| [`quantitative.ipynb`](quantitative.ipynb) | Survey ratings and rankings: participants, descriptives, statistical tests, rankings, background correlations and sensitivity checks |
| [`qualitative.ipynb`](qualitative.ipynb) | Coded think-aloud sessions: corpus, agreement between coders, themes by participant and by explanation type, design suggestions |

They show the same results as `uv run xai analyze`, which writes the tables
to a workbook instead (see the [analysis README](../src/xai_decision_making/analysis/README.md)).

The qualitative notebook shows no transcript text. Straight after loading the
coding sheets it keeps only the columns the analysis needs (codes, scope,
explanation type, model and so on), so the text never reaches a table.

## Running them yourself

The notebooks read the same input as `uv run xai analyze`:

- the survey exports in `src/xai_decision_making/analysis/data/`;
- for the qualitative notebook, the coding sheets in
  `src/xai_decision_making/analysis/data/coding_sheets/`.

From the repository root:

```bash
uv sync
uv run --with jupyter jupyter lab
```

Open a notebook in the browser tab that appears and choose *Run → Run All
Cells*.

In VS Code you can open the `.ipynb` file directly and pick the project's
`.venv` as the kernel.

To rerun both and save the new output in the files, without opening them:

```bash
uv run --with jupyter jupyter nbconvert --to notebook --execute --inplace \
    notebooks/quantitative.ipynb notebooks/qualitative.ipynb
```