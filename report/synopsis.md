# 🦷 Dental Model: Technical Synopsis & Architecture Report

**Project Title:** Dental Disease Detection & Severity Classification System  
**Package:** `dental_model` (`src/dental_model/`)  
**Environment:** [`uv`](https://docs.astral.sh/uv/) (Python 3.11)  
**Target Hardware:** NVIDIA GeForce RTX 4070 (12.88 GB VRAM), CUDA 12.1  
**Deployment Target:** Hugging Face Spaces (Gradio) & Hugging Face Hub  

---

## 1. Executive Summary & Problem Context

Dental caries (tooth decay) and dental plaque accumulation are among the most prevalent chronic oral diseases globally. Early detection and accurate staging are critical for preventive dentistry and minimally invasive interventions. However, manual visual inspection of intraoral photographs is susceptible to intra-observer variability, differences in photographic illumination, and subtle initial enamel demineralization signs.

This project implements an **end-to-end, two-stage modular computer vision system** designed to automate the screening, localization, and severity classification of dental conditions from standard intraoral clinical photographs.

### Core Architectural Philosophy: Two-Stage Modular Pipeline

The system decouples object detection from fine-grained lesion classification:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Input Intraoral Image                  │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │          Stage 1: YOLOv8 Object Detector               │
                  │   - Localizes: healthy teeth, plaque, caries lesions   │
                  │   - Predicts bounding boxes + class labels + conf      │
                  └───────────────┬──────────────────────┬─────────────────┘
                                  │                      │
                   [Full Detections Overlay]    [Caries / Plaque BBox Crops]
                                  │                      │
                                  │                      ▼
                                  │  ┌─────────────────────────────────────┐
                                  │  │    Stage 2: timm Lesion Classifier   │
                                  │  │  - Severity: early vs. advanced     │
                                  │  │  - Class-weighted / Focal Loss      │
                                  │  └───────────────────┬─────────────────┘
                                  │                      │
                                  ▼                      ▼
                  ┌────────────────────────────────────────────────────────┐
                  │              Unified Pipeline Interface                │
                  │  - Formats structured JSON: [{bbox, label, severity}]  │
                  │  - Overlays Grad-CAM visual explainability heatmaps    │
                  │  - Serves interactive Gradio web application (HF)      │
                  └────────────────────────────────────────────────────────┘
```

---

## 2. Complete Repository Blueprint

```text
dental-model/
├── .github/workflows/                 # Automated CI/CD (lint, pytest, HF deployment)
├── configs/                           # Centralized YAML configurations
│   ├── data_paths.yaml                # Dataset registry and raw paths
│   ├── detector.yaml                  # YOLOv8 baseline config (v0)
│   ├── detector_v1.yaml               # YOLOv8 tuned config (v1)
│   └── classifier.yaml                # timm classifier hyperparameters & backbone config
├── data/                              # Dataset Directory (gitignored)
│   ├── raw/                           # Original dataset archives
│   ├── interim/                       # Extracted standardized archives
│   └── processed/                     # Final unified datasets
│       ├── checksums.json             # SHA-256 integrity hashes (12,379 files tracked)
│       ├── classifier/labels.csv      # Stratified 70/15/15 classification split (2,000 images)
│       └── detector/data.yaml         # YOLO format (4,881 train / 692 val / 615 test)
├── models/                            # Model Checkpoints & Training Outputs (gitignored)
│   ├── detector_runs/                 # YOLOv8 runs (v0 Nano, v1 Small)
│   └── classifier_runs/               # timm EfficientNet-B0 runs (v0)
├── notebooks/                         # Pre-executed Jupyter research & exploration notebooks
│   ├── 01_eda.ipynb                   # Phase 0: Multi-source EDA & figures
│   ├── 02_preprocessing.ipynb         # Phase 0: Label unification & splitting prototype
│   ├── 04_detector_finetune.ipynb     # Phase 1: YOLOv8 training, metrics & visual bbox inference
│   └── 05_classifier_finetune.ipynb   # Phase 2: timm classifier fine-tuning & evaluation
├── report/                            # Technical reports & figures
│   ├── figures/                       # EDA figures, diagnostic Grad-CAM, and pipeline outputs
│   └── synopsis.md                    # Detailed architectural & results synopsis
├── scripts/                           # Multi-platform execution scripts (.ps1 and .sh)
├── src/dental_model/                  # Modular Python Package
│   ├── data/ (extract.py, unify_labels.py)
│   ├── detector/ (train.py, infer.py)
│   ├── classifier/ (model.py, train.py, infer.py)
│   ├── pipeline.py                    # Unified detector -> crop -> classifier pipeline
│   └── utils/ (viz.py)                # Grad-CAM and detection overlay utilities
├── tests/                             # Automated Test Suite (pytest, 35 passing tests)
├── app/                               # Hugging Face Space Gradio Deployment
├── AGENTS.md                          # Coding conventions & constraints
├── pyproject.toml                     # uv project configuration & CUDA wheel index
└── report.md                          # Root-level codebase analysis & progress report
```

---

## 3. Data Synthesis & EDA Insights

Four raw datasets were audited in Phase 0:
1. **Roboflow Dental Dataset (Required - Detection):** 6 classes remapped to 3 hard-tissue classes (`healthy: 0`, `plaque: 1`, `caries: 2`). Soft tissue (`gingivitis`, `gum_swelling`) deferred to v2. Total 6,188 annotated images across train (4,881), val (692), and test (615).
2. **Caries-Spectra Enamel Caries (Required - Classification):** 2,000 high-resolution images mapped to `healthy` (400), `caries_early` (800), and `caries_advanced` (800). Stratified 70/15/15 train/val/test splits created with fixed seed 42.
3. **Dental Plaque Diagno (Optional):** Evaluated during EDA.
4. **Gingivitis Intraoral Image Dataset (Deferred v2):** Captioning task.

All 12,379 processed files are tracked with SHA-256 integrity hashes in `data/processed/checksums.json`.

---

## 4. Benchmark Results & Verification

### 4.1 Stage 1 Detector (YOLOv8s Runs v1 & v2)
- **Run v1 (3-Class Hard Tissue):** Overall Precision: 0.5352, Recall: 0.5475, mAP@50: 0.4692, Caries mAP@50: 0.384.
- **Run v2 (5-Class Multi-Disease):** Expanded to soft-tissue periodontal conditions (`gingivitis` mAP@50: **0.541**, `gum_swelling` mAP@50: 0.166, `healthy` mAP@50: 0.782, `caries` mAP@50: 0.324, `plaque` mAP@50: 0.191). Overall mAP@50: **0.4010**.

### 4.2 Stage 2 Classifier (EfficientNet-B0 Run v0)
- **Test Accuracy:** 95.67%
- **Macro F1 Score:** 0.964
- **Grad-CAM Sanity Check:** Verified that attention is centered on tooth enamel and occlusal fissure lines ($1.7\times - 4.35\times$ center-to-border activation ratio), with zero shortcut learning on photographic borders.

### 4.3 Phase 3 Unified Pipeline (`DentalPipeline`)
- Implemented in `src/dental_model/pipeline.py`, combining `DentalDetector` $\rightarrow$ bounding box cropping $\rightarrow$ `DentalClassifierInferer` severity classification $\rightarrow$ `generate_gradcam` visual explainability heatmaps $\rightarrow$ structured JSON output.

### 4.4 Verification & Quality Assurance
- Automated test suite (`pytest`) executing **36 unit tests** across `tests/test_app.py` (1), `tests/test_data.py` (4), `tests/test_detector.py` (11), `tests/test_classifier.py` (10), and `tests/test_pipeline.py` (10) all passing (100%).
- Code linting and formatting configured via `ruff` with 0 errors across `src/`, `app/`, `notebooks/`, and `tests/`.
- Gradio web app (`app/app.py`) equipped with doctor-centric clinical presets, top-level diagnostic triage card, reconciled clinical terminology, and clinical cyan/slate blue UI.
