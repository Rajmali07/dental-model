# Dental Model — Implementation & Deployment Plan

## 1. What the data actually gives you

Your dataset tree (rename `Data/` → `data/` — lowercase, to match every config/script in
this plan; git and CI are case-sensitive) mixes **four different task types**, not one.
That's the first thing to design around, because it decides whether you build one
multi-task model or several single-task models behind one interface.

| Source | Task type | Labels |
|---|---|---|
| Gingivitis Intraoral Image Dataset | Image captioning | Free-text captions per image |
| Dental Plaque Diagno (multi-modal) | Classification / multi-modal | Plaque presence + auxiliary modality data |
| Caries-Spectra (Enamel Caries) | Classification (3-class) | `AdvanceEnamel_Caries`, `EarlyStageEnamel_Caries`, `NoEnamel_Caries` |
| Roboflow "more healthy and caries + plaque" | Object detection | Bounding boxes: healthy / caries / plaque, pre-split train/valid/test |

**Recommendation:** ship two models, not one Frankenstein model:
- **Detector** (Roboflow data, YOLO-style) → localizes caries/plaque/healthy regions on an intraoral photo.
- **Classifier** (Caries-Spectra + Plaque Diagno) → severity/stage classification, can run on crops from the detector.
- Treat the **captioning dataset** as a stretch goal / v2 (image-to-text head), not part of v1, since it needs a completely different training loop (encoder-decoder) and evaluation metric (BLEU/CIDEr) than the other three.

This staging keeps v1 shippable in weeks instead of months.

**Which datasets are actually required (decided now, not after EDA):**

| Dataset | Needed for v1? | Rationale |
|---|---|---|
| Roboflow (healthy/caries/plaque) | Required | Only pre-labeled detection dataset — the detector stage depends on it entirely. |
| Caries-Spectra (3-class enamel caries) | Required | Only clean multi-class classification labels available — basis for the classifier. |
| Plaque Diagno (multi-modal) | Optional | Roboflow already covers a "plaque" class; only worth including if EDA shows its plaque labels or auxiliary modality add real value over Roboflow's. |
| Gingivitis captioning | Deferred to v2 | Different task family (captioning), out of scope for the detector+classifier v1. |

EDA can still overturn this — e.g. if Caries-Spectra turns out too small/noisy to train on — but the default architecture doesn't wait on EDA to be decided.

## 2. Environment setup with `uv` (no venv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't installed yet

uv init dental-model
cd dental-model
uv python pin 3.11

uv add torch torchvision ultralytics timm albumentations \
       datasets huggingface_hub gradio scikit-learn pandas \
       numpy opencv-python-headless pillow pyyaml

uv add --dev ruff pytest ipykernel pre-commit
```

`uv` gives you `uv.lock` (reproducible), `uv run <script>` (no manual activation),
and `uv sync` for CI/HF Space builds — all of which replace the venv+pip workflow.

## 3. Repo structure (recreated, professional, HF-ready)

```text
dental-model/
├── pyproject.toml              # uv-managed deps, replaces requirements.txt
├── uv.lock
├── README.md                   # project overview (not the dataset README)
├── .pre-commit-config.yaml
├── .gitignore                  # ignore raw archives (.rar/.7z/.zip), checkpoints, data/
├── configs/
│   ├── detector.yaml           # ultralytics/YOLO data + hyperparams
│   ├── classifier.yaml
│   └── data_paths.yaml         # points at raw dataset locations, kept out of git
├── data/
│   ├── raw/                    # original downloads (gitignored)
│   ├── interim/                # extracted .rar/.7z, unified folder layout
│   └── processed/              # final train/valid/test per task, COCO/YOLO/ImageFolder format
├── src/
│   └── dental_model/
│       ├── __init__.py
│       ├── data/
│       │   ├── extract.py      # unrar/7z/unzip → interim/
│       │   ├── unify_labels.py # map 3 caries classes + roboflow classes to one schema
│       │   └── datasets.py     # PyTorch Dataset / HF datasets.Dataset wrappers
│       ├── detector/
│       │   ├── train.py
│       │   └── infer.py
│       ├── classifier/
│       │   ├── model.py        # timm backbone + head
│       │   ├── train.py
│       │   └── infer.py
│       ├── pipeline.py         # detector -> crop -> classifier, single entrypoint
│       └── utils/
│           ├── metrics.py      # mAP, F1, confusion matrix
│           └── viz.py
├── notebooks/
│   ├── 01_eda.ipynb             # exploratory analysis per dataset
│   ├── 02_preprocessing.ipynb   # extraction + label unification (prototype before scripting into src/)
│   ├── 03_baseline_model.ipynb  # simple classifier baseline before the full detector+classifier pipeline
│   ├── 04_detector_finetune.ipynb  # YOLO fine-tuning runs / experiment log
│   ├── 05_classifier_finetune.ipynb # timm backbone fine-tuning runs / experiment log
│   └── 06_evaluation_explainability.ipynb # metrics + Grad-CAM review
├── tests/
│   ├── test_data.py
│   └── test_pipeline.py
├── scripts/
│   ├── prepare_data.sh
│   ├── train_detector.sh
│   └── train_classifier.sh
├── app/                        # Hugging Face Space
│   ├── app.py                  # Gradio interface
│   ├── requirements.txt        # generated via `uv export` for Spaces build
│   └── model_card.md
├── report/                     # project synopsis, write-up, figures for submission/portfolio use
│   ├── synopsis.md
│   └── figures/
└── models/                     # exported weights for HF Hub upload (gitignored, use HF for storage)
```

**Optional, only if you also want a standalone web app (not just the HF Space):** add a `backend/` (Flask `app.py` + `requirements.txt`) and `frontend/` (`templates/` + `static/css`/`static/js`) pair, same split as your cyberbullying-detection reference. Skip this unless you specifically need a self-hosted UI outside Hugging Face — it duplicates what `app/` (Gradio) already does for deployment.

`src/dental_model/utils/` also gets an `explainability.py` alongside `metrics.py` and `viz.py` — Grad-CAM or similar saliency overlays on the classifier's crops, so a reviewer can see *why* a region was flagged, mirroring the role your `explainability.py` (SHAP + trigger-word highlighting) plays in the cyberbullying project.

## 4. Step-by-step implementation plan

**Phase 0 — Data unification (1–2 days)**
1. `src/dental_model/data/extract.py`: decompress the `.rar`/`.7z`/`.zip` archives into `data/interim/` with consistent naming.
2. `unify_labels.py`: map the three caries-stage folders and the Roboflow classes into one label schema (e.g. `healthy`, `plaque`, `caries_early`, `caries_advanced`). Write a single `data/processed/labels.csv`.
3. `01_eda.ipynb` — run EDA on **all four sources**, not one: a per-source section for each of the three Mendeley datasets and the Roboflow dataset, followed by a combined/unified section once labels are mapped. Per-source sections surface source-specific issues (resolution, lighting, label quality) that a single merged view would hide; the combined section is what actually informs training decisions. Lean heavily on visualization here, not just summary stats. At minimum:
   - **Class balance**: bar chart of image counts per unified class, per source dataset (stacked bar) — this is where the advanced-caries imbalance becomes visible early.
   - **Image grids**: `matplotlib`/`torchvision.utils.make_grid` sampled thumbnails per class, so imbalance and label-quality issues are seen, not just counted.
   - **Image size/aspect-ratio distribution**: histogram/scatter across the four sources, since Mendeley and Roboflow images won't share resolution or aspect ratio — this drives your resize/augmentation choices.
   - **Duplicate/near-duplicate detection**: perceptual-hash (`imagehash`) similarity heatmap across sources, since the same tooth/photo can plausibly appear in more than one Mendeley dataset.
   - **Color/intensity distribution**: per-channel histograms, before/after any normalization — useful because intraoral photos vary a lot in lighting/white balance across sources.
   - **Bounding box visualization** (Roboflow data only): overlay boxes on sample images to sanity-check the exported annotations before training the detector.
   - **Before/after preprocessing comparisons**: side-by-side original vs. cleaned/resized/augmented image, so every transform in `unify_labels.py`/the preprocessing pipeline is visually verified, not just assumed correct.
   - Save the key plots into `report/figures/` as you go — they get reused directly in `report/synopsis.md` rather than being redone later.

**Phase 1 — Detector (Roboflow data)**
1. Since it's already train/valid/test split in Roboflow format, export as YOLO format directly.
2. `uv run yolo train data=configs/detector.yaml model=yolov8n.pt epochs=100 imgsz=640` (swap in `ultralytics` CLI or your own training loop in `detector/train.py`).
3. Evaluate mAP50-95 on the held-out `test/` split.

**Phase 2 — Classifier (Caries-Spectra + Plaque Diagno)**
1. `timm` backbone (e.g. `efficientnet_b0` or `convnext_tiny`) fine-tuned on the unified 3–4 class label set.
2. Stratified split (the raw data isn't pre-split like Roboflow's), 70/15/15.
3. Track accuracy, per-class F1, confusion matrix — dental data is usually class-imbalanced (few "advanced caries" images), so weight the loss or oversample.

**Phase 3 — Pipeline glue**
`pipeline.py`: run detector → crop detected regions → run classifier on crops → return structured JSON (`{bbox, class, confidence}` list). This is what the Gradio app calls.

**Phase 4 — Packaging & HF deployment**
1. Push weights to a HF Hub model repo (`huggingface_hub.upload_folder`), not to git — keeps the code repo light.
2. Build `app/app.py` with Gradio: upload image → run `pipeline.py` → draw boxes + labels.
3. `app/model_card.md`: intended use, training data provenance (link back to the three Mendeley DOIs + Roboflow project), limitations (small/imbalanced medical dataset, not a diagnostic device), license.
4. `uv export --format requirements-txt > app/requirements.txt` so the Space builds without uv itself needing to be present at HF's build step (Spaces currently expect pip-style requirements; keep `uv` as your local/dev tool and export for the Space).
5. Create the Space (`Gradio` SDK), push `app/` as the Space repo, reference the HF Hub model repo inside `app.py` via `hf_hub_download` or `from_pretrained`.

**Phase 5 — Documentation: README + report**
1. Rewrite `README.md` at the repo root (this replaces the current dataset-listing README) to include: project summary, the repo structure above, `uv` setup commands, how to run each phase (`scripts/prepare_data.sh`, `train_detector.sh`, `train_classifier.sh`), and a link to the deployed HF Space.
2. Fold a condensed version of this implementation plan into the README as a "Development Roadmap" section, so the plan lives with the code, not just as a standalone file.
3. `report/synopsis.md`: longer-form write-up for submission/portfolio — problem statement, dataset provenance (the three Mendeley DOIs + Roboflow project link), methodology, results tables, Grad-CAM figures from `report/figures/`, limitations.

## 6. Production-readiness checklist

Beyond "it trains and demos on Gradio," these are the gaps between a notebook project and something you'd point a reviewer or employer at:

**Data versioning**
- `data/` is gitignored, which means the repo alone can't reproduce a training run. Track data with `dvc` (or at minimum, commit a `data/processed/checksums.json` of SHA256 hashes) so any given model version can be traced back to an exact dataset snapshot.

**Reproducibility**
- Fix and log random seeds (Python, NumPy, torch, and CUDA determinism flags) in every training script — without this, "we got 0.87 mAP" isn't reproducible and isn't defensible in a report.

**Robustness at inference time**
- `pipeline.py` must validate inputs before running the model: reject non-image files, cap image size/resolution, handle corrupt/truncated uploads gracefully — this matters most in `app.py`, since it's the one place an untrusted user directly controls the input.
- Wrap inference in try/except with a clear user-facing error message (not a raw stack trace) and structured logging (`logging` module, not `print`) so failures are diagnosable from Space logs.

**Hugging Face specifics**
- Secrets (HF write token, W&B API key) go in the **Space's Settings → Repository secrets**, not `.env` — `.env` is for local dev only and must stay gitignored; it does not carry over to a deployed Space.
- `app/model_card.md` needs the YAML frontmatter HF expects (`license`, `tags`, `datasets`, and `model-index` with your eval metrics) — without it, the Hub won't render license/metrics on the model page.
- Pick and document the Space's hardware tier (CPU basic is usually fine for the classifier; the detector may need a small GPU tier for acceptable latency) — this is a cost/latency decision, not a code decision, and should be recorded in the README.

**CI/CD**
- `.github/workflows/ci.yml`: on every push/PR, `uv sync`, `uv run ruff check .`, `uv run pytest`. Fail the build on lint/test failure.
- Separate `.github/workflows/deploy-space.yml` that pushes `app/` to the HF Space repo on merge to `main` (via `huggingface_hub` API or the HF git remote), so the Space is never manually out of sync with the code repo.

**Config & secrets**
- All hyperparameters live in `configs/*.yaml`, never hardcoded in scripts — makes runs reproducible and diffable.
- `.env.example` checked in, real `.env` gitignored. Any HF token used for pushing models/Space is read from environment, never committed.

**Versioning & experiment tracking**
- Tag model releases (`v0.1.0-detector`, `v0.1.0-classifier`) both in git and as revisions on the HF Hub model repo, so the Space can pin to a known-good revision instead of always pulling `main`.
- Log runs with `mlflow` or the free `wandb` tier — even lightweight tracking (loss curves, mAP/F1 per run) makes the results in `report/synopsis.md` defensible instead of anecdotal.
- Keep a `CHANGELOG.md` at the repo root and bump `pyproject.toml`'s `version` field on each tagged release — without this, "which code produced which weights" becomes guesswork after a few iterations.

**Testing**
- `tests/test_data.py`: label schema is respected after `unify_labels.py` runs (no unmapped classes, no empty label sets).
- `tests/test_pipeline.py`: the detector→crop→classifier path runs end-to-end on a tiny fixture image and returns well-formed JSON, so a broken pipeline is caught in CI, not after deployment.

**Packaging & reproducibility**
- `pyproject.toml` pins exact versions via `uv.lock` — anyone can `uv sync` and get byte-identical environments.
- Optional `Dockerfile` (multi-stage: `uv sync --no-dev` in build stage, slim runtime image) if you want the Space or an API to run outside HF's managed build too.

**Documentation & compliance**
- `app/model_card.md` includes intended use, out-of-scope use ("not a diagnostic device, not a substitute for a dentist"), training data provenance and licenses for the three Mendeley datasets + Roboflow project, and known failure modes (class imbalance on advanced caries, image quality sensitivity).
- `LICENSE` file at repo root — check each source dataset's license before choosing your repo's license, since Mendeley/Roboflow datasets can carry their own redistribution terms that constrain what you can ship.
- `AGENTS.md` at the **repo root** (same level as `README.md`) — a short agent-facing doc (not for humans) covering the `uv`-only rule, where configs/secrets live, the two-model architecture, and what not to do (no root-level `requirements.txt`, no committing `data/`, no moving workflow files out of `.github/workflows/`). This keeps any AI coding agent you use on the repo consistent across sessions instead of re-deriving conventions each time.

**Repo hygiene**
- Workflow files (`ci.yml`, `deploy-space.yml`) must live under `.github/workflows/`, never the repo root — GitHub Actions won't discover them anywhere else.
- No scratch files (e.g. an untracked `python.py`) committed at the repo root — move exploratory code into `notebooks/` or a gitignored scratch directory.
- Exactly one casing for the data folder (`data/`, lowercase) used everywhere — mixed `Data/`/`data/` references are a common source of "works on my machine" bugs on case-sensitive filesystems (Linux CI runners, most deployment targets).

## 7. Immediate next actions
- Confirm whether you want detector+classifier as two models (recommended) or you specifically need one multi-task network — that changes `configs/` and `src/dental_model/` layout.
- Decide the unified label taxonomy in Phase 0 before writing any training code — this is the one decision that's expensive to change later.

Let me know which phase you want to start scaffolding first and I'll generate the actual code files (extract.py, unify_labels.py, configs, or the Gradio app) rather than just the plan.
