# 🦷 Dental Model: Comprehensive Codebase Analysis & Progress Report

**Project Title:** Dental Disease Detection & Severity Classification System  
**Repository Root:** `c:\Users\user\Downloads\dental-model`  
**Package Name:** `dental_model` (`src/dental_model/`)  
**Package Manager:** [`uv`](https://docs.astral.sh/uv/) (Python 3.11, strictly no manual `venv`/`pip`)  
**Target Hardware / Runtime:** NVIDIA GeForce RTX 4070 (12.88 GB VRAM), CUDA 12.1 (`torch==2.5.1+cu121`)  
**Deployment Target:** Hugging Face Spaces (Gradio) & Hugging Face Hub  

---

## 1. Executive Summary & Problem Context

Dental caries (tooth decay) and dental plaque accumulation are among the most prevalent chronic oral diseases globally. Early detection and accurate staging are critical for preventive dentistry and minimally invasive interventions. However, manual visual inspection of intraoral photographs is susceptible to intra-observer variability, differences in photographic illumination, and subtle initial enamel demineralization signs.

This project implements an **end-to-end, two-stage modular computer vision system** designed to automate the screening, localization, and severity classification of dental conditions from standard intraoral clinical photographs.

### Core Architectural Philosophy: Two-Stage Modular vs. Monolithic Network

Rather than forcing heterogeneous clinical datasets with fundamentally different label semantics into a single monolithic network, the system separates detection from fine-grained staging:

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

#### Why this architecture was chosen:
1. **Label Semantic Separation:** Detection datasets annotate spatial bounding boxes on full intraoral scenes (`healthy`, `plaque`, `caries`), whereas specialized caries datasets annotate whole tooth/crop severity stages (`NoEnamel_Caries`, `EarlyStageEnamel_Caries`, `AdvanceEnamel_Caries`).
2. **Independent Optimization:** The detector (YOLOv8) optimizes spatial IoU and localization recall, while the classifier (`timm` EfficientNet/ConvNeXt) optimizes fine-grained texture demineralization and resolves extreme class imbalance.
3. **Clinical Explainability:** Allows isolating individual suspicious lesions and generating focused Grad-CAM saliency maps on the specific lesion crop rather than diffuse activations over the entire mouth.

---

## 2. Complete Repository Blueprint & Architecture

The repository is structured as a modular, installable Python package following production ML engineering standards, strict configuration decoupling, and reproducible environment pinning with `uv`.

```text
dental-model/
├── .github/
│   └── workflows/
│       ├── ci.yml                     # GitHub Actions CI: ruff linting + pytest test execution
│       └── deploy-space.yml           # Automated CD pipeline pushing app/ to Hugging Face Spaces
├── .env.example                       # Template for local API tokens (HF_TOKEN, WANDB_API_KEY)
├── .gitignore                         # Strict exclusion of data/, checkpoints, archives, caches
├── AGENTS.md                          # Mandatory agent guidelines (uv usage, two-model rule, CI constraints)
├── CHANGELOG.md                       # Semantic versioning changelog following Keep a Changelog
├── README.md                          # Comprehensive project landing page, quickstart, and benchmarks
├── SETUP.md                           # Environment setup and developer onboarding guide
├── dental-model-implementation-plan.md # Architectural roadmap, dataset specifications, and checklists
├── pyproject.toml                     # uv project configuration, dependencies, dev tools, and PyTorch CUDA index
├── uv.lock                            # Deterministic lockfile for reproducible, byte-identical builds
├── configs/                           # Centralized YAML configuration files (no hardcoded parameters)
│   ├── data_paths.yaml                # Dataset registry: URLs, raw paths, interim/processed paths, status
│   ├── detector.yaml                  # YOLOv8 baseline config (Nano, 50 epochs, batch 16, lr0 0.01)
│   ├── detector_v1.yaml               # YOLOv8 tuned config (Small, 100 epochs, batch 16, lr0 0.01)
│   └── classifier.yaml                # timm classifier hyperparameters, backbone architecture, splits
├── data/                              # Dataset Directory (gitignored, managed via prepare_data)
│   ├── raw/                           # Original dataset archives (.zip / directories)
│   ├── interim/                       # Extracted standardized intermediate datasets
│   └── processed/                     # Final unified datasets ready for model consumption
│       ├── checksums.json             # SHA-256 verification hashes for all 12,379 processed files
│       ├── labels.csv                 # Root unified labels manifest
│       ├── classifier/
│       │   └── labels.csv             # Stratified 70/15/15 classification split (2,000 images)
│       └── detector/
│           ├── data.yaml              # Ultralytics dataset configuration (classes: healthy, plaque, caries)
│           ├── train/ (images, labels)# 4,881 training images and YOLO annotations
│           ├── val/ (images, labels)  # 692 validation images and YOLO annotations
│           └── test/ (images, labels) # 615 test images and YOLO annotations
├── models/                            # Training runs, weight checkpoints, and validation logs (gitignored)
│   └── detector_runs/
│       ├── v0/                        # Baseline YOLOv8n run artifacts (best.pt, results.csv, confusion matrix)
│       └── v1/                        # Tuned YOLOv8s run artifacts (best.pt, results.csv, PR curves)
├── notebooks/                         # Pre-executed Jupyter research & exploration notebooks
│   ├── 01_eda.ipynb                   # Phase 0: Multi-source EDA, distributions, duplicate detection, 7 figures
│   ├── 02_preprocessing.ipynb         # Phase 0: Label unification, filtering, and dataset split prototyping
│   └── 04_detector_finetune.ipynb     # Phase 1: YOLOv8 detector fine-tuning, metric evaluation, inference demo
├── report/                            # Technical reports, documentation, and visual figures
│   ├── figures/                       # High-resolution figures exported during EDA
│   │   ├── 01_source_inventory.png
│   │   ├── 02_class_balance.png
│   │   ├── 03_sample_grids.png
│   │   ├── 04_image_sizes.png
│   │   ├── 05_duplicates.png
│   │   ├── 06_color_distribution.png
│   │   └── 07_before_after_preprocessing.png
│   └── synopsis.md                    # Technical project synopsis for portfolio/academic review
├── scripts/                           # Multi-platform automation and execution scripts
│   ├── prepare_data.ps1 / .sh         # End-to-end data extraction and label unification runner
│   └── train_detector.ps1 / .sh       # Automated environment sync, GPU check, and YOLOv8 training runner
├── src/dental_model/                  # Core installable Python package
│   ├── __init__.py                    # Package initialization
│   ├── data/                          # Phase 0 data extraction, mapping, and integrity verification
│   │   ├── __init__.py
│   │   ├── extract.py                 # Multi-format archive extraction (.zip/.rar/.7z) to interim/
│   │   └── unify_labels.py            # Taxonomy remapping, stratified splitting, SHA-256 generation
│   ├── detector/                      # Phase 1 detection subsystem
│   │   ├── __init__.py
│   │   ├── train.py                   # YOLOv8 trainer with deterministic seed fixing & artifact sync
│   │   └── infer.py                   # DentalDetector class: image validation, single & batch inference
│   ├── classifier/                    # Phase 2 classification subsystem (in development)
│   │   └── (model.py, train.py, infer.py)
│   ├── pipeline.py                    # Phase 3 detector -> crop -> classifier end-to-end pipeline (planned)
│   └── utils/                         # Phase 3 metrics and Grad-CAM explainability utilities (planned)
├── tests/                             # Automated test suite (pytest)
│   ├── test_data.py                   # Unit tests for data configs, YOLO remapping, SHA-256 hashing
│   └── test_detector.py               # Unit tests for detector configs, seeds, device fallback, validation
└── app/                               # Hugging Face Space Gradio application
    ├── model_card.md                  # Model card with clinical disclaimers, data provenance, and tags
    └── (app.py, requirements.txt)     # Gradio UI & exported pip requirements (Phase 4)
```

---

## 3. Phase 0: Dataset Engineering & Multi-Source Synthesis

### 3.1 Raw Dataset Analysis & Scoping Decisions

The project analyzes four distinct source datasets, cataloged centrally in `configs/data_paths.yaml`:

| Dataset Source | Modality / Task | Raw Classes / Format | Scope / Status | Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **Roboflow Dental Dataset** | Object Detection | 6 classes: `caries`, `cavity`, `gingivitis`, `gum_swelling`, `healthy`, `plaque` | **Required** (Phase 1) | Only source with pre-annotated bounding boxes; feeds Stage 1 Detector. |
| **Caries-Spectra (Enamel Caries)** | 3-Class Image Classification | `AdvanceEnamel_Caries`, `EarlyStageEnamel_Caries`, `NoEnamel_Caries` | **Required** (Phase 2) | Clean clinical classification labels for caries progression; feeds Stage 2 Classifier. |
| **Dental Plaque Diagno** | Multi-modal Classification | Plaque presence + auxiliary modalities | **Optional** | Plaque is already represented in Roboflow; auxiliary modalities evaluated during EDA. |
| **Gingivitis Intraoral Images** | Image Captioning | Free-text clinical captions per intraoral image | **Deferred (v2)** | Captioning requires an encoder-decoder architecture (LLaVA/BLIP), out of scope for v1. |

### 3.2 Label Taxonomy & Remapping Strategy

Heterogeneous annotations from the raw sources are unified into a standard taxonomy via `src/dental_model/data/unify_labels.py`:

#### 1. Roboflow Detection Remapping:
- **Hard vs. Soft Tissue Filtering:** Periodontal soft-tissue conditions (`gingivitis` [class 2] and `gum_swelling` [class 3]) are filtered out to keep Stage 1 strictly focused on hard-tissue tooth conditions.
- **Synonym Consolidation:** `caries` (0) and `cavity` (1) are merged into target class `caries` (index 2).
- **Target Schema:**
  - `0`: `healthy` (from raw 4)
  - `1`: `plaque` (from raw 5)
  - `2`: `caries` (from raw 0 & 1)
- **Config Documentation Confirmation:** The remapping decision rationale is explicitly documented as formal comments inside [`configs/data_paths.yaml`](file:///c:/Users/user/Downloads/dental-model/configs/data_paths.yaml) (in both the top header documentation block and directly preceding the `roboflow_detection:` YAML key).

#### 2. Caries-Spectra Classification Remapping:
- `NoEnamel_Caries` $\rightarrow$ `healthy`
- `EarlyStageEnamel_Caries` $\rightarrow$ `caries_early`
- `AdvanceEnamel_Caries` $\rightarrow$ `caries_advanced`
- **Splits:** Stratified 70% Train (1,400 images), 15% Validation (300 images), 15% Test (300 images) with fixed random seed `42`.

```text
Classifier Dataset Split Breakdown (2,000 Total Images):
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│ Split            │ healthy      │ caries_early │ caries_adv   │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Train (70%)      │ 280 images   │ 560 images   │ 560 images   │
│ Val (15%)        │ 60 images    │ 120 images   │ 120 images   │
│ Test (15%)       │ 60 images    │ 120 images   │ 120 images   │
│ Total (100%)     │ 400 images   │ 800 images   │ 800 images   │
└──────────────────┴──────────────┴──────────────┴──────────────┘
```

#### 3. Data Integrity & Provenance Tracking
`src/dental_model/data/unify_labels.py` computes SHA-256 hashes for every generated image and annotation file, saving them to `data/processed/checksums.json` (tracking **12,379 files**). This guarantees that training runs are fully auditable and traceable back to exact dataset snapshots.

---

## 4. Exploratory Data Analysis (EDA) Insights

The exploratory analysis conducted in `notebooks/01_eda.ipynb` generated seven high-resolution diagnostic figures saved in `report/figures/`:

1. **`01_source_inventory.png` (Source Inventory & Breakdown):** Details the distribution of images across the four repositories, demonstrating the dominance of Roboflow in volume (6,188 detection frames) and Caries-Spectra in high-resolution tooth photos (2,000 images).
2. **`02_class_balance.png` (Class Distributions):** Visualizes the class counts across tasks. Highlights the class imbalance in classification (`healthy` constitutes 20% vs 40% `caries_early` and 40% `caries_advanced`), informing the necessity of class-weighted cross-entropy loss in Stage 2.
3. **`03_sample_grids.png` (Visual Inspection Grids):** A visual grid displaying representative samples for each class, validating clean separation between sound enamel, white spot early demineralization, and cavitated advanced caries.
4. **`04_image_sizes.png` (Image Resolution & Aspect Ratio Distribution):** Reveals significant resolution variance across sources (from $640 \times 640$ standard detection crops to $1920 \times 1080$ clinical macro captures). Validates standardizing detector inputs to $640 \times 640$ and classifier inputs to $224 \times 224$.
5. **`05_duplicates.png` (Perceptual Hash Duplicate Detection):** Uses difference hashing (`imagehash.dhash`) to check for cross-dataset leakage and duplicate patient photos across splits.
6. **`06_color_distribution.png` (Color & Illumination Histograms):** Plots per-channel (RGB) intensity histograms across different photographic devices, demonstrating significant white-balance shift and motivating color jitter and normalization augmentations.
7. **`07_before_after_preprocessing.png` (Transformation Verification):** Validates the data preprocessing pipeline, verifying that bounding boxes align accurately after normalization, resizing, and augmentation.

---

## 5. Phase 1: YOLOv8 Object Detector Implementation & Results

### 5.1 Detector Engineering & Implementation

The detection subsystem is fully implemented in `src/dental_model/detector/`:

- **`train.py` (Deterministic Training Engine):**
  - Fixes global random seeds across Python `random`, `numpy`, `torch`, and CUDA determinism (`torch.backends.cudnn.deterministic = True`).
  - Automatically resolves compute device (`cuda` if available on RTX 4070, graceful fallback to `cpu`).
  - Reads configuration strictly from YAML (`configs/detector.yaml` or `configs/detector_v1.yaml`).
  - Automatically synchronizes Ultralytics output directories to versioned project paths (`models/detector_runs/v0/`, `v1/`).
- **`infer.py` (Production Inference Wrapper - `DentalDetector`):**
  - Validates image files before execution (validates file existence, supported extensions, and guards against memory exhaustion with a `MAX_IMAGE_DIM = 4096px` cap).
  - Implements `predict(image_path)` and `predict_batch(image_paths)`.
  - Formats output as clean, serializable JSON dictionaries: `[{"bbox": [x1, y1, x2, y2], "class_id": int, "class_name": str, "confidence": float}]`.
  - Provides a complete CLI interface for standalone testing.
- **Git Release Tagging:**
  - Git tag `v0.1.0-detector` is created locally with annotated tag message: `"YOLOv8s detector, mAP50-95=0.262 (v1), see report.md for full v0 vs v1 comparison"`.
- **Notebook & Script Synchronization:**
  - `notebooks/04_detector_finetune.ipynb` is synchronized to load, evaluate, and visualize the superior Run v1 (`models/detector_runs/v1/`) artifacts (`results.csv`, `results.png`, `confusion_matrix.png`, `weights/best.pt`).
  - `scripts/train_detector.ps1` (PowerShell) and `scripts/train_detector.sh` (Bash) handle automated environment synchronization (`uv sync`), GPU verification, and model training.

### 5.2 Training Runs & Benchmark Comparison

Two complete training iterations were conducted and logged under `models/detector_runs/`:

```text
Detector Benchmark Comparison on NVIDIA RTX 4070 (CUDA 12.1):
┌───────────────────────┬─────────────────────────┬─────────────────────────┐
│ Metric / Parameter    │ Run v0 (YOLOv8 Nano)    │ Run v1 (YOLOv8 Small)   │
├───────────────────────┼─────────────────────────┼─────────────────────────┤
│ Architecture          │ yolov8n.pt (3.2M params)│ yolov8s.pt (11.2M params│
│ Input Resolution      │ 640 x 640               │ 640 x 640               │
│ Epochs Trained        │ 50 epochs               │ 100 epochs              │
│ Batch Size            │ 16                      │ 16                      │
│ Training Duration     │ ~18.1 minutes           │ ~57.9 minutes           │
├───────────────────────┼─────────────────────────┼─────────────────────────┤
│ Overall Precision (B) │ 0.4560                  │ 0.5352 (+7.9%)          │
│ Overall Recall (B)    │ 0.4888                  │ 0.5475 (+5.9%)          │
│ **mAP@50 (B)**        │ **0.4267** (Peak 0.430) │ **0.4692** (Peak 0.476) │
│ **mAP@50-95 (B)**     │ **0.2372** (Peak 0.239) │ **0.2619** (Peak 0.267) │
├───────────────────────┼─────────────────────────┼─────────────────────────┤
│ `healthy` mAP@50      │ 0.790                   │ 0.812 (+2.2%)           │
│ `caries` mAP@50       │ 0.315                   │ 0.384 (+6.9%)           │
│ `plaque` mAP@50       │ 0.185                   │ 0.212 (+2.7%)           │
└───────────────────────┴─────────────────────────┴─────────────────────────┘
```

#### Key Detector Takeaways & Known Limitations:
1. **Model Capacity Scaling:** Upgrading from YOLOv8n (Run v0) to YOLOv8s (Run v1) yielded a **+4.6% boost in overall mAP50** and a **+6.9% boost in caries detection mAP50**.
2. **Precision vs. Recall:** YOLOv8s demonstrated markedly fewer false positive detections on plaque and tooth fissures, improving overall precision from 0.456 to 0.535.
3. **Plaque Detection Limitation:** Plaque detection remains modest (mAP50=0.212), which is explicitly acknowledged as a known clinical data limitation due to diffuse visual plaque boundaries and variable clinical staining.

---

## 6. Verification, Testing & Code Quality

The codebase enforces strict unit testing and linting standards before any milestone is marked complete.

### 6.1 Automated Test Suite (`pytest`)

The test suite in `tests/` contains **25 unit tests** across data, detector, and classifier modules:

```text
tests/
├── test_data.py (4 tests)
│   ├── test_load_data_config          # Verifies YAML parsing of dataset paths
│   ├── test_remap_yolo_label_file     # Verifies YOLO bbox class filtering and remapping logic
│   ├── test_compute_sha256            # Verifies SHA-256 hash generation
│   └── test_generate_checksums        # Verifies checksums.json generation
├── test_detector.py (11 tests)
│   ├── TestLoadConfig (3 tests)       # YAML parsing, missing file, missing sections
│   ├── TestSetSeeds (1 test)          # Reproducibility seed verification
│   ├── TestResolveDevice (2 tests)    # CUDA/CPU device resolution and fallback
│   ├── TestDentalDetectorValidateImage (4 tests) # Image validity, dimension cap (4096px)
│   └── TestDentalDetectorInit (1 test)# Checkpoint missing exception handling
└── test_classifier.py (10 tests)
    ├── TestClassifierConfig (3 tests) # Classifier YAML schema and config parser
    ├── TestDentalClassifierModel (2 tests) # Forward pass tensor shape check (B, 3) & from_config
    ├── TestClassWeightsComputation (2 tests) # Inverse-frequency class weighting math check
    └── TestTransformsAndDevice (3 tests) # Albumentations transforms, device fallback, seeds
```

**Test Execution Output:**
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\user\Downloads\dental-model
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.2
collected 25 items

tests\test_classifier.py ..........                                      [ 40%]
tests\test_data.py ....                                                  [ 56%]
tests\test_detector.py ...........                                       [100%]

============================= 25 passed in 2.87s ==============================
```

---

## 7. Current Project Progress Tracker (What We Have Done Till Now)

The following table summarizes the implementation status across all planned phases:

| Phase | Component / Task | Deliverable / File Path | Status |
| :---: | :--- | :--- | :---: |
| **Phase 0** | **Environment & Tooling** | `pyproject.toml`, `uv.lock`, `.env.example`, `SETUP.md` | ✅ **Completed** |
| **Phase 0** | **Data Registry & Configs** | `configs/data_paths.yaml`, `.gitignore`, `AGENTS.md` | ✅ **Completed** |
| **Phase 0** | **Data Extraction Engine** | `src/dental_model/data/extract.py` | ✅ **Completed** |
| **Phase 0** | **Taxonomy & Label Mapper** | `src/dental_model/data/unify_labels.py` | ✅ **Completed** |
| **Phase 0** | **Data Integrity Checksums** | `data/processed/checksums.json` (12,379 files) | ✅ **Completed** |
| **Phase 0** | **Exploratory Data Analysis** | `notebooks/01_eda.ipynb` + 7 figures in `report/figures/` | ✅ **Completed** |
| **Phase 0** | **Data Preprocessing Proto** | `notebooks/02_preprocessing.ipynb` | ✅ **Completed** |
| **Phase 1** | **Detector Configuration** | `configs/detector.yaml`, `configs/detector_v1.yaml` | ✅ **Completed** |
| **Phase 1** | **YOLOv8 Trainer** | `src/dental_model/detector/train.py` | ✅ **Completed** |
| **Phase 1** | **Detector Inference Wrapper**| `src/dental_model/detector/infer.py` (`DentalDetector`) | ✅ **Completed** |
| **Phase 1** | **Detector Training Runs** | `models/detector_runs/v0/` (Nano) & `v1/` (Small) | ✅ **Completed** |
| **Phase 1** | **Git Release Tagging** | `v0.1.0-detector` created locally | ✅ **Completed** |
| **Phase 1** | **Interactive Detector Demo**| `notebooks/04_detector_finetune.ipynb` (synchronized to v1) | ✅ **Completed** |
| **Phase 1** | **Detector Test Suite** | `tests/test_data.py`, `tests/test_detector.py` (15 tests) | ✅ **Completed** |
| **Phase 1** | **Multi-Platform Scripts** | `scripts/prepare_data.*`, `scripts/train_detector.*` | ✅ **Completed** |
| **Phase 2** | **Classifier Architecture** | `src/dental_model/classifier/model.py` (`DentalClassifier`) | ✅ **Completed (Code Ready)** |
| **Phase 2** | **Classifier Data & Trainer**| `src/dental_model/classifier/train.py` (Class weighting, Albumentations) | ✅ **Completed (Code Ready)** |
| **Phase 2** | **Classifier Test Suite** | `tests/test_classifier.py` (10 unit tests) | ✅ **Completed** |
| **Phase 2** | **Classifier Training Run** | Execute `uv run python -m dental_model.classifier.train` | ⏳ **Awaiting User Review** |
| **Phase 3** | **Pipeline Integration** | `src/dental_model/pipeline.py` (detector $\rightarrow$ crop $\rightarrow$ classifier) | ⏳ **Upcoming** |
| **Phase 3** | **Explainability Heatmaps** | `src/dental_model/utils/viz.py` (Grad-CAM overlays) | ⏳ **Upcoming** |
| **Phase 4** | **Gradio Web Demo** | `app/app.py` + `app/requirements.txt` | ⏳ **Upcoming** |
| **Phase 4** | **HF Hub Deployment** | Hub model release + Space sync via `.github/workflows/` | ⏳ **Upcoming** |
| **Phase 5** | **Portfolio Documentation** | `report/synopsis.md`, `README.md` updates | ⏳ **Upcoming** |

---

## 8. Next Technical Steps & Implementation Recommendations

To advance the project to full deployment, the immediate technical priorities are:

1. **Implement Stage 2 Classifier (`src/dental_model/classifier/`):**
   - Build `model.py` utilizing `timm` architectures (e.g., `efficientnet_b0` or `convnext_tiny`).
   - Implement `train.py` consuming `data/processed/classifier/labels.csv` with class-weighted cross-entropy or focal loss to handle the 20/40/40 class distribution.
   - Implement `infer.py` for evaluating cropped lesion patches.
2. **Build End-to-End Pipeline (`src/dental_model/pipeline.py`):**
   - Connect `DentalDetector` and `DentalClassifier` into a unified inference function:
     $\text{Input Image} \rightarrow \text{YOLO Bounding Boxes} \rightarrow \text{Bounding Box Crops} \rightarrow \text{Classifier Severity} \rightarrow \text{Structured Predictions}$.
3. **Implement Grad-CAM Explainability (`src/dental_model/utils/viz.py`):**
   - Provide visual saliency heatmaps for cropped lesions so clinicians can inspect which textural features triggered an `early` vs. `advanced` caries classification.
4. **Deploy Interactive Gradio Interface (`app/app.py`):**
   - Build an intuitive web application allowing users to upload intraoral photos, adjust confidence thresholds, view bounding box overlays, and inspect fine-grained severity ratings with Grad-CAM overlays.
   - Deploy to Hugging Face Spaces with automated GitHub Actions synchronization.

---

## 9. Clinical & Ethical Disclaimers

> [!WARNING]
> **Academic & Research Demonstration Only:** This software and its associated model checkpoints are developed solely for academic research, benchmarking, and portfolio demonstration.
>
> 1. **Not a Medical Device:** The system has not been cleared, certified, or approved by the FDA, CE, or any international medical device regulatory body.
> 2. **No Substitute for Clinical Diagnosis:** The models should never be used as a standalone diagnostic tool or as a substitute for professional dental examination, radiography, and diagnosis by a qualified, licensed dentist.
> 3. **Dataset Provenance & Redistribution:** Source datasets originate from published academic repositories (Mendeley Data DOIs `9jnf2jvghy`, `g8yhdvgjy2`, `3253gj88rr` and Roboflow Universe). Derived model checkpoints must comply with all respective source dataset licenses.
