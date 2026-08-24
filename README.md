# 🦷 Dental Disease Detection & Severity Classification

An end-to-end computer vision system for automated detection and severity classification of dental conditions (**healthy teeth**, **dental plaque**, and **caries lesions**) from intraoral dental imagery.

---

## 📌 Project Overview & Architecture

The system uses a **two-stage modular architecture** (per [`AGENTS.md`](./AGENTS.md) and [`dental-model-implementation-plan.md`](./dental-model-implementation-plan.md)):
1. **Stage 1: Object Detector (YOLOv8)** — Scans intraoral photographs, locates tooth and lesion boundaries, and detects conditions (`healthy`, `plaque`, `caries`).
2. **Stage 2: Lesion Classifier (`timm` Backbone)** — Crops detected lesion regions and classifies severity (early vs. advanced caries) with class-imbalance weighting.
3. **Pipeline & Web Demo** — Wires the two models together and serves an interactive web app on **Hugging Face Spaces** (Gradio).

| Aspect | Implementation |
| :--- | :--- |
| **Stage 1 Detector** | YOLOv8 nano fine-tuned on Roboflow Dental Dataset (`healthy: 0`, `plaque: 1`, `caries: 2`) |
| **Stage 2 Classifier** | `timm` ConvNeXt / EfficientNet fine-tuned on Caries-Spectra & Plaque Diagno |
| **Hardware / Acceleration** | NVIDIA GeForce RTX 4070 (12.88 GB VRAM), CUDA 12.1 (`torch==2.5.1+cu121`) |
| **Environment & Tooling** | [`uv`](https://docs.astral.sh/uv/) (strictly no manual `venv` or `pip`) |
| **Deployment** | Gradio App on Hugging Face Spaces + Model Weights on Hugging Face Hub |

---

## 🗂️ Current Repository Structure

```text
dental-model/
├── .github/workflows/                 # CI/CD Workflows
│   ├── ci.yml                         # Automated lint (ruff) + unit tests (pytest)
│   └── deploy-space.yml               # Hugging Face Space auto-deploy on main
├── configs/                           # Centralized YAML Configurations
│   ├── data_paths.yaml                # Dataset registry (raw, interim, processed paths)
│   ├── detector.yaml                  # YOLOv8 hyperparameters, image size, batch size
│   └── classifier.yaml                # timm classifier hyperparameters & backbone config
├── data/                              # Dataset Directory (gitignored)
│   ├── raw/                           # Original dataset archives & downloads
│   ├── interim/                       # Extracted standardized archives
│   └── processed/                     # Final unified datasets
│       ├── checksums.json             # SHA-256 integrity hashes (12,379 files tracked)
│       ├── classifier/labels.csv      # Stratified train/val/test splits (2,000 images)
│       └── detector/                  # YOLO format (4,881 train / 692 val / 615 test)
│           └── data.yaml              # Ultralytics dataset configuration
├── models/                            # Model Checkpoints & Training Outputs (gitignored)
│   └── detector_runs/v0/              # YOLOv8 fine-tuned artifacts
│       ├── weights/best.pt            # Best detector checkpoint (6.25 MB)
│       ├── weights/last.pt            # Final epoch checkpoint
│       ├── results.csv                # Epoch-by-epoch losses and mAP logs
│       ├── results.png                # Loss & validation curves plot
│       └── confusion_matrix.png       # Per-class validation confusion matrix
├── notebooks/                         # Interactive Exploration & Visual Evaluation
│   ├── 01_eda.ipynb                   # ✅ Phase 0: Exploratory data analysis (7 saved figures)
│   ├── 02_preprocessing.ipynb         # ✅ Phase 0: Label unification & splitting prototype
│   ├── 03_baseline_model.ipynb        # ⏳ Phase 2: Simple baseline classifier (Next)
│   ├── 04_detector_finetune.ipynb     # ✅ Phase 1: YOLOv8 training, metrics & visual bbox inference
│   ├── 05_classifier_finetune.ipynb   # ⏳ Phase 2: timm backbone fine-tuning & focal loss
│   └── 06_evaluation_explainability.ipynb # ⏳ Phase 3: Grad-CAM heatmaps & error analysis
├── report/                            # Project Documentation & Artifacts
│   ├── figures/                       # High-resolution figures generated from EDA
│   │   ├── 01_source_inventory.png
│   │   ├── 02_class_balance.png
│   │   ├── 03_sample_grids.png
│   │   ├── 04_image_sizes.png
│   │   ├── 05_duplicates.png
│   │   ├── 06_color_distribution.png
│   │   └── 07_before_after_preprocessing.png
│   └── synopsis.md                    # ⏳ Phase 5: Technical write-up & portfolio report
├── scripts/                           # Multi-Platform Execution Scripts
│   ├── prepare_data.sh / .ps1         # Extract raw datasets & generate processed splits
│   ├── train_detector.sh / .ps1       # Run YOLOv8 detector training
│   └── train_classifier.sh / .ps1     # ⏳ Phase 2: Run classifier training
├── src/dental_model/                  # Modular, Installable Python Package
│   ├── __init__.py
│   ├── data/                          # Data extraction, unification, and verification
│   │   ├── __init__.py
│   │   ├── extract.py
│   │   └── unify_labels.py
│   ├── detector/                      # Stage 1: Detection subsystem
│   │   ├── __init__.py
│   │   ├── train.py                   # YOLOv8 trainer with seed fixing and CUDA support
│   │   └── infer.py                   # DentalDetector class with validation & batch inference
│   ├── classifier/                    # ⏳ Stage 2: Classification subsystem (Phase 2)
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── infer.py
│   ├── pipeline.py                    # ⏳ Stage 3: End-to-end detector -> crop -> classifier (Phase 3)
│   └── utils/                         # ⏳ Stage 3: Metrics, Grad-CAM visualization
├── tests/                             # Automated Test Suite (pytest)
│   ├── test_data.py                   # Data config, remapping, sha256 checksum tests (4/4 passed)
│   └── test_detector.py               # Detector config, seed fixing, device & inference tests (11/11 passed)
├── app/                               # Hugging Face Space Gradio Deployment
│   ├── app.py                         # ⏳ Phase 4: Gradio UI demo
│   ├── requirements.txt               # ⏳ Phase 4: Exported dependencies for HF Space
│   └── model_card.md                  # Intended use, clinical disclaimers, data provenance
├── AGENTS.md                          # Coding conventions & constraints for AI agents
├── dental-model-implementation-plan.md # Architectural roadmap & engineering specifications
├── pyproject.toml                     # uv project configuration & CUDA wheel index
└── README.md                          # Project documentation
```

---

## 🚦 Roadmap & Implementation Progress

| Phase | Description | Key Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Phase 0** | **Data Unification & EDA** | • Data extraction pipeline (`extract.py`)<br>• Label taxonomy mapper (`unify_labels.py`)<br>• SHA-256 checksums (`checksums.json`)<br>• Full EDA notebook (`01_eda.ipynb`) + 7 figures | ✅ **Completed** |
| **Phase 1** | **YOLOv8 Detector** | • Detector module (`src/dental_model/detector/`)<br>• CUDA 12.1 training on RTX 4070<br>• Best weights saved (`models/detector_runs/v0/weights/best.pt`)<br>• Pre-executed interactive notebook (`04_detector_finetune.ipynb`)<br>• 15/15 unit tests passing (`pytest`) | ✅ **Completed** |
| **Phase 2** | **Classifier Model** | • `timm` backbone (`src/dental_model/classifier/`)<br>• Stratified dataset loader with class weighting / focal loss<br>• Training script (`train_classifier.ps1` / `.sh`)<br>• Notebooks `03_baseline_model.ipynb` & `05_classifier_finetune.ipynb` | ⏳ **Next Step** |
| **Phase 3** | **Pipeline Glue & Explainability** | • End-to-end `pipeline.py` (detector → crop → classifier)<br>• Grad-CAM lesion explainability overlay (`viz.py`)<br>• Evaluation notebook `06_evaluation_explainability.ipynb` | ⏳ **Upcoming** |
| **Phase 4** | **Packaging & Deployment** | • Gradio web app (`app/app.py`)<br>• Upload weights to Hugging Face Hub (`Bihari04/dental-model`)<br>• Deploy Space (`Bihari04/dental-model-demo`)<br>• Model card with medical disclaimers | ⏳ **Upcoming** |
| **Phase 5** | **Documentation & Report** | • `report/synopsis.md` submission write-up<br>• Benchmark comparison tables & metrics | ⏳ **Upcoming** |

---

## ⚡ Quickstart & Reproduction Guide

### 1. Environment Setup

This project strictly uses **`uv`**. Do not create virtual environments manually with `venv` or `pip`.

```powershell
# Windows (PowerShell)
uv sync --all-extras --dev
```

### 2. Verify GPU & Environment

```powershell
uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

### 3. Run Data Pipeline

```powershell
.\scripts\prepare_data.ps1
```

### 4. Train the YOLOv8 Detector

```powershell
.\scripts\train_detector.ps1 -Config configs\detector.yaml
```

### 5. Run Tests & Linter

```powershell
uv run pytest
uv run ruff check .
```

---

## 📊 Detector Benchmark Results (Phase 1)

Trained on 4,881 images (640x640) for 50 epochs on NVIDIA RTX 4070 (CUDA 12.1):

| Class | Precision | Recall | **mAP50** | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Overall** | **0.450** | **0.498** | **0.430** | **0.239** |
| `healthy` | 0.657 | 0.802 | **0.790** | 0.477 |
| `caries` | 0.430 | 0.364 | **0.315** | 0.163 |
| `plaque` | 0.261 | 0.328 | **0.185** | 0.077 |

---

## ⚠️ Clinical Disclaimer
This project is developed for **academic and research demonstration purposes only**. It is **not** a certified medical diagnostic device and is not intended to replace professional dental diagnosis, examination, or treatment by a licensed dentist.
