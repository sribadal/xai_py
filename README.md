# XAI Decision Making

This project is an explainability framework for clinical survival and quality-of-life prediction models. It was built for shared decision-making research in oncology. The framework generates patient-level explanations across four explanation modalities (attribution, counterfactual, case-based, and rule-based) and exports structured JSON that is consumed by a separate survey application.

The project was developed as part of a thesis investigating which explanation modalities patients and clinicians find most useful for understanding AI-driven predictions in pancreatic cancer care (PACAP cohort).

---

## Scope and Contributions

This repository contains two layers of code with different origins.

**Original contribution: `explainability/`**

The entire XAI framework is the core deliverable of this thesis. This includes the adapter pattern that wraps the prediction models, all 20+ explainability methods across four modalities, the sanity-check and Co-12 audit pipelines, the per-patient JSON export system, and the confidence metric enrichment. Everything under `src/xai_decision_making/explainability/` was written from scratch for this project. The `scripts/` folder (gitignored) contains local-only utilities such as the Co-12 audit runner and the confidence post-processor.

**Forked and restructured: `preprocessing/`, `models/`, `experiments/`**

These modules originate from a separate research repository that handled data preprocessing, model training (RSF and ConDOR), and evaluation. This code was forked, reorganised into the current `src/xai_decision_making/` package layout, and adapted where needed (import paths, configuration, data loading) to integrate with the explainability layer. The model architectures, training pipelines, and preprocessing logic were not modified. The `analysis/` folder is also forked but gitignored.

In short, someone else built and trained the models. This project introduces explainability to those models.

---

## Overview

The explainability layer wraps the two pre-trained prediction models behind a unified adapter interface and runs a library of XAI methods against each patient in a held-out test set. Each method produces a structured `Explanation` object that is written to per-patient JSON files, enriched with confidence metrics, and validated through an automated audit pipeline.

**Supported models:**

- **RSF** (Random Survival Forest) was trained on the PACAP clinical cohort. It predicts survival probabilities at 3, 6, and 12 months (90, 180, 365 days).
- **ConDOR** is a neural quality-of-life prediction model based on ordinal regression with a temporal treatment block. It predicts QoL trajectories at 1, 3, 6, 9, and 12 months.

**Explanation modalities and methods:**

| Modality | RSF | ConDOR |
|---|---|---|
| Attribution | SurvLIME, Permutation Importance, Minimal Depth, TreeSHAP | SurvLIME, Permutation Importance, DeepLift, Integrated Gradients, DeepLift SHAP |
| Counterfactual | DiCE, C-CHVAE, FACE, LORE, Wachter, REVISE, CounterNet | DiCE, C-CHVAE, FACE, LORE, Wachter, REVISE, CounterNet, Treatment Scenario CF |
| Case-based | KNN, Embedding Retrieval, Prototype | KNN, Embedding Retrieval, Prototype |
| Rule-based | LORE Rules, RuleFit, Anchors, Skope-Rules | LORE Rules, RuleFit, Anchors, Skope-Rules |

---

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. It requires Python 3.11 or higher.

```bash
# Clone the repository
git clone https://github.com/<your-username>/xai-decision-making.git
cd xai-decision-making

# Install with uv
uv sync
```

For GPU support (required for ConDOR and gradient-based attribution methods), add the PyTorch CUDA index to `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }
```

Then reinstall torch:

```bash
uv sync --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
```

Verify:

```bash
uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

---

## Dependencies

The core dependencies include numpy, pandas, scikit-learn, scikit-survival, torch, matplotlib, and seaborn. The explainability layer adds the following libraries:

| Library | Used by |
|---|---|
| `lime` | SurvLIME attribution |
| `shap` / `survshap` | TreeSHAP attribution |
| `captum` | DeepLift, Integrated Gradients, DeepLift SHAP |
| `sage-importance` | SAGE importance (experimental) |
| `dice-ml` | DiCE counterfactuals |
| `alibi` | Anchors rule-based, C-CHVAE counterfactuals |
| `imodels` | RuleFit rule-based |
| `skope-rules` | Skope-Rules rule-based |
| `masurvival` | Missingness-Avoiding survival model utilities |
| `pingouin` | Statistical analysis |

See `pyproject.toml` for the full dependency list with version pins.

---

## Usage

The project provides a single CLI entry point with four subcommands. Default artifacts are injected automatically if not specified.

```bash
uv run xai <command> [args]
```

### Export

Generates per-patient explanation JSON for a given model. Loads the trained artifact, iterates over patients in the test set, runs the configured explainers, and writes one JSON file per patient.

```bash
uv run xai export --model-type rsf --n-patients 100 --time-horizons 90 180 365

uv run xai export --model-type condor --n-patients 509 \
    --test-data data/black_envelop_test_set_pacap.csv \
    --time-horizons 1 3 6 9 12

uv run xai export --model-type rsf --only lime permutation knn rulefit anchors
```

### Enrich

Post-processes exported JSON files with confidence metrics and clinical-unit inversions.

```bash
uv run xai enrich --model-type rsf
```

### Audit

Runs the Co-12 property audit against exported explanations. Scores each method on 11 Nauta properties and writes a CSV ranking and per-method JSON detail files.

```bash
uv run xai audit --refresh

uv run xai audit --only-method lime --only-modality attribution --refresh
```

### Analyze

Runs the full results pipeline for the survey study (Study 2). Produces descriptive tables, statistical tests, and all figures.

```bash
uv run xai analyze
```

---

## Architecture

### Adapter Pattern

All models are wrapped behind the `SurvivalModelAdapter` abstract class defined in `base.py`. This gives every explainer a unified interface for prediction, feature introspection, and immutability constraints, regardless of the underlying model.

- `WrappedSurvivalAdapter` wraps scikit-survival RSF models. It loads `.pkl` artifacts from `artifacts/survival_models/`.
- `WrappedConDORAdapter` wraps the ConDOR neural model. It handles the static-to-z-space transformation, treatment block construction, and inverse scaling (z-space to clinical units) for counterfactual outputs.

### Explainer Registry

Explainers are registered through the `ExplainerRegistry` and grouped by modality. The `build_registry()` function in `run_export.py` returns a dict that maps modality names to method dicts. Some methods are model-specific. For example, RSF gets TreeSHAP and Minimal Depth, while ConDOR gets DeepLift and Integrated Gradients.

### Explanation Object

Every explainer returns an `Explanation` dataclass with four fields:

- `content` is the primary explanation payload, such as feature importances, counterfactual deltas, retrieved cases, or firing rules.
- `metadata` contains auxiliary data like DataFrames of retrieved patients, counterfactual instances, rule text, and confidence metrics.
- `modality` is the `ExplainerModality` enum value.
- `target` is the `ModelTarget` (survival or QoL).

### Export JSON Schema

Each patient file (`exports/<model_type>/patient_<id>.json`) contains:

```json
{
  "patient_id": "...",
  "features": { "cci": 2, "ca199": 150.0, "..." : "..." },
  "features_raw": { "..." : "..." },
  "prediction": { "..." : "..." },
  "time_horizons": { "90": "3_months", "180": "6_months", "365": "12_months" },
  "explanations": {
    "attribution": {
      "lime": { "content": { }, "metadata": { }, "confidence_metrics": { } },
      "permutation": { }
    },
    "counterfactual": {
      "dice": { "content": { }, "raw_query": { }, "raw_cf": { } }
    },
    "case_based": {
      "knn": { "content": { }, "metadata": { } }
    },
    "rule_based": {
      "rulefit": { "content": { }, "metadata": { } }
    }
  }
}
```

A `manifest.json` in each model directory records the export configuration, including the cohort size, time horizons, methods used, artifact path, and timestamp.

---

## Data

The project uses clinical data from the PACAP (Pancreatic Cancer Project) registry linked to the Netherlands Cancer Registry (NCR). Data files are not included in the repository.

**Expected data layout:**

| File | Description |
|---|---|
| `data/dev_set.csv` | Training split (clinical, used by the ConDOR adapter) |
| `data/dev_set_pacap.csv` | Training split (PACAP-linked, used by the ConDOR adapter) |
| `data/dev_set_processed.csv` | Training split (processed features, used by RSF) |
| `data/test_set_processed.csv` | Test split (processed features, used by RSF) |
| `data/black_envelop_test_set_pacap.csv` | Held-out black-envelope test set |

Clinical features include CCI, CA19-9, bilirubin, albumin, LDH, CRP, age category, performance status, tumour differentiation grade, clinical T/M staging, metastasis locations, and others. The RSF model uses 30 features in total.

---

## Related Repository

This project is the backend companion to the XAI Survey Web application:

- [xai_web](https://github.com/sribadal/xai_web) contains the Next.js frontend for presenting patient explanations and collecting survey responses.

---

## Acknowledgements

This project was built as part of a master's thesis on explainable AI for clinical decision support in oncology. It uses the PACAP clinical dataset under a data use agreement.

The preprocessing, model training, and experiment code was forked and restructured from a private research repository. That repository developed the RSF and ConDOR prediction models for the PACAP cohort. The explainability framework in this project was built on top of that foundation.