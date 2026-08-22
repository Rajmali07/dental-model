# Dental Model — Caries & Plaque Detection

![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Python](https://img.shields.io/badge/python-3.11-blue)
![uv](https://img.shields.io/badge/env-uv-purple)

Detects and classifies dental conditions (healthy / plaque / early-stage caries / advanced caries)
from intraoral photographs, using a two-stage detector → classifier pipeline. Deployed as a
Hugging Face Space with weights hosted on the HF Hub.

*(Badges above are placeholders — replace with your actual GitHub Actions badge URL and chosen license once CI and `LICENSE` exist; a static badge claiming "passing" before there's a real workflow is misleading.)*

> **Not a diagnostic device.** This project is a research/portfolio artifact trained on public
> datasets. It is not validated for clinical use and must not be used to make treatment decisions.

## Overview

| | |
|---|---|
| **Task** | Detection (localize regions) + classification (stage/condition) |
| **Data** | 3 Mendeley datasets + 1 Roboflow dataset (see [Data sources](#data-sources)) |
| **Detector** | YOLOv8 (Ultralytics) |
| **Classifier** | `timm` CNN backbone (EfficientNet-B0 / ConvNeXt-Tiny) |
| **Explainability** | Grad-CAM overlays on classifier crops |
| **Deployment** | Gradio app on Hugging Face Spaces, weights on HF Hub |
| **Env/deps** | [`uv`](https://docs.astral.sh/uv/) — no `venv`/`pip` |

## Repository structure

```text
dental-model/
├── AGENTS.md                    # instructions for AI coding agents (Claude Code, etc.) working in this repo
├── pyproject.toml              # uv-managed deps
├── uv.lock                     # reproducible, pinned environment
├── CHANGELOG.md                # version history, tied to model release tags
├── .github/workflows/          # CI (lint + test) and Space deploy
├── configs/
│   ├── detector.yaml
│   ├── classifier.yaml
│   └── data_paths.yaml
├── data/
│   ├── raw/                    # original downloads (gitignored)
│   ├── interim/                # extracted archives, unified layout
│   └── processed/              # final train/valid/test per task
├── src/dental_model/
│   ├── data/                   # extract.py, unify_labels.py, datasets.py
│   ├── detector/               # train.py, infer.py
│   ├── classifier/             # model.py, train.py, infer.py
│   ├── pipeline.py             # detector → crop → classifier, single entrypoint
│   └── utils/                  # metrics.py, viz.py, explainability.py
├── notebooks/                  # 01_eda → 06_evaluation_explainability
├── tests/                      # test_data.py, test_pipeline.py
├── scripts/                    # prepare_data.sh, train_detector.sh, train_classifier.sh
├── app/                        # Hugging Face Space (Gradio app + model card)
├── report/                     # synopsis.md + figures for write-up/portfolio
└── models/                     # exported weights (gitignored; stored on HF Hub, not git)
```

> **If your local repo doesn't match this yet, fix these three drift issues before going further:**
> 1. Move `ci.yml` and `deploy-space.yml` into `.github/workflows/` — GitHub Actions only
>    discovers workflows at that exact path, not the repo root.
> 2. Your dataset folder is currently capitalized `Data/` (carried over from the original
>    dataset README), but every script/config here assumes lowercase `data/`. Git and most
>    CI runners are case-sensitive — rename `Data/` → `data/` and use lowercase consistently
>    everywhere; don't try to support both spellings.
> 3. Delete or relocate any scratch files (e.g. a stray `python.py`) out of the repo root —
>    move exploratory one-off code into `notebooks/` or a gitignored scratch folder instead.

## For AI coding agents

If you're using Claude Code, Cursor, or a similar agent on this repo, read
[`AGENTS.md`](./AGENTS.md) first — it lives at the **repo root** (same level as this README)
and documents conventions the agent should follow here (where configs live, how to run
tests, the `uv`-only rule, dataset status table, etc.) so it doesn't have to re-derive them
from scratch each session.

## Getting started

### 1. Install `uv`

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Set up the environment

```bash
git clone <this-repo-url>
cd dental-model
uv python pin 3.11
uv sync
```

`uv sync` reads `pyproject.toml` + `uv.lock` and builds an exact, reproducible environment —
no `activate` step needed; every command below is run with `uv run`.

### 3. Configure paths and secrets

```bash
cp .env.example .env
# edit .env: set HF_TOKEN if you plan to push models/Space, set DATA_ROOT if not using ./data
```

`.env` is for **local development only**. Once deployed, the Space does not read this file —
set `HF_TOKEN`/`WANDB_API_KEY` under the Space's **Settings → Repository secrets** instead.

### 4. Data versioning

`data/` is gitignored, so the git repo alone can't reproduce a training run. Either track data
with `dvc`, or at minimum regenerate `data/processed/checksums.json` (SHA256 per file) after
`prepare_data.sh` runs, so any model release can be traced back to an exact dataset snapshot.

## Data sources

- [Mendeley Dataset 9jnf2jvghy v2](https://data.mendeley.com/datasets/9jnf2jvghy/2) — Gingivitis intraoral image captioning
- [Mendeley Dataset g8yhdvgjy2 v3](https://data.mendeley.com/datasets/g8yhdvgjy2/3) — Multi-modal dental plaque diagnosis
- [Mendeley Dataset 3253gj88rr v1](https://data.mendeley.com/datasets/3253gj88rr/1) — Caries-Spectra enamel caries dataset
- [Roboflow: Dental Dataset — Healthy and Caries Plaque](https://universe.roboflow.com/dental-data2/dental-dataset-more-healthy-and-caries-plaque) — pre-split detection dataset

Check each source's license before redistributing derived data or weights.

## Usage

### Prepare data

```bash
uv run scripts/prepare_data.sh
```

Extracts archives into `data/interim/`, unifies labels across the three classification
sources and the Roboflow schema, and writes `data/processed/`.

### Train

```bash
uv run scripts/train_detector.sh      # YOLOv8 on the Roboflow split
uv run scripts/train_classifier.sh    # timm backbone on unified caries/plaque labels
```

Hyperparameters live in `configs/detector.yaml` and `configs/classifier.yaml` — edit those
rather than the training scripts.

### Run the full pipeline locally

```bash
uv run python -m dental_model.pipeline --image path/to/photo.jpg
```

Returns detected regions with class and confidence, plus a Grad-CAM overlay per region.

### Run tests / lint

```bash
uv run pytest
uv run ruff check .
```

### Launch the app locally

```bash
uv run python app/app.py
```

## Deployment

Weights are pushed to a Hugging Face Hub model repo (not committed to git); the Gradio
app in `app/` is deployed as a Hugging Face Space and pulls a pinned model revision at
startup. See `app/model_card.md` for intended use, limitations, and data provenance —
it needs YAML frontmatter (`license`, `tags`, `datasets`, `model-index` with eval metrics)
for the Hub to render it correctly. Choose and record the Space's hardware tier here once
decided (CPU is likely fine for the classifier; the detector may need a small GPU tier).

CI (`.github/workflows/ci.yml`) runs lint + tests on every push. On merge to `main`,
`.github/workflows/deploy-space.yml` syncs `app/` to the Space so it never drifts from
the code repo. Model releases are tagged (`v0.1.0-detector`, etc.) and logged in
`CHANGELOG.md`, with `pyproject.toml`'s version bumped to match.

## Development roadmap

1. **Data unification** — extract archives, map all sources to one label schema.
2. **Detector** — YOLOv8 on the Roboflow split; evaluate mAP50-95.
3. **Classifier** — fine-tune a `timm` backbone on Caries-Spectra + Plaque Diagno; track per-class F1 given class imbalance on advanced caries.
4. **Pipeline** — wire detector → crop → classifier into one callable entrypoint.
5. **Documentation** — this README, `report/synopsis.md`, model card.
6. **Packaging/deployment** — CI, versioned model releases, HF Hub + Space.

Full task-by-task detail lives in [`dental-model-implementation-plan.md`](./dental-model-implementation-plan.md).

## Limitations

- Trained on public datasets of varying size/quality; not clinically validated.
- Class imbalance (few advanced-caries examples) limits reliability on that class.
- Sensitive to image quality/lighting; not tested on intraoral camera hardware variation beyond what's present in the source datasets.

## License

Add a `LICENSE` file before publishing — confirm compatibility with each source dataset's
terms first (see [Data sources](#data-sources)).
