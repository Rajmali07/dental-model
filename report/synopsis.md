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
                                  └───────────────────┬─────────────────┘
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
│   └── detector_runs/                 # YOLOv8 runs (v0 Nano, v1 Small)
├── notebooks/                         # Pre-executed Jupyter research & exploration notebooks
│   ├── 01_eda.ipynb                   # Phase 0: Multi-source EDA & figures
│   ├── 02_preprocessing.ipynb         # Phase 0: Label unification & splitting prototype
│   └── 04_detector_finetune.ipynb     # Phase 1: YOLOv8 training, metrics & visual bbox inference
├── report/                            # Technical reports & figures
│   ├── figures/                       # 7 High-resolution EDA figures
│   └── synopsis.md                    # Detailed architectural & results synopsis
├── scripts/                           # Multi-platform execution scripts (.ps1 and .sh)
├── src/dental_model/                  # Modular Python Package
│   ├── data/ (extract.py, unify_labels.py)
│   ├── detector/ (train.py, infer.py)
│   ├── classifier/ (model.py, train.py, infer.py)
│   ├── pipeline.py                    # Unified detector -> crop -> classifier pipeline
│   └── utils/ (viz.py, metrics.py)
├── tests/                             # Automated Test Suite (pytest, 15 passing tests)
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

## 4. Detector Benchmark Results (Phase 1)

Trained on NVIDIA RTX 4070 (CUDA 12.1):

| Metric | Run v0 (YOLOv8 Nano, 50 ep) | Run v1 (YOLOv8 Small, 100 ep) | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Precision** | 0.4560 | **0.5352** | **+7.9%** |
| **Overall Recall** | 0.4888 | **0.5475** | **+5.9%** |
| **mAP@50 (Overall)** | 0.4267 (Peak: 0.430) | **0.4692 (Peak: 0.476)** | **+4.6%** |
| **mAP@50-95 (Overall)** | 0.2372 (Peak: 0.239) | **0.2619 (Peak: 0.267)** | **+2.8%** |
| `healthy` mAP@50 | 0.790 | **0.812** | +2.2% |
| `caries` mAP@50 | 0.315 | **0.384** | +6.9% |
| `plaque` mAP@50 | 0.185 | **0.212** | +2.7% |

---

- Automated test suite (`pytest`) executing **25 unit tests** across `tests/test_data.py` (4), `tests/test_detector.py` (11), and `tests/test_classifier.py` (10) all passing (100% in ~2.9s).
- Code linting and formatting configured via `ruff` with 0 errors across `src/` and `tests/`.
- Git release tag `v0.1.0-detector` created locally.
- Phase 2 Classifier module (`src/dental_model/classifier/model.py`, `train.py`) verified and ready for training.
