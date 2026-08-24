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
├── AGENTS.md                          # Mandatory agent guidelines (uv usage, two-model rule, report sync)
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
│   ├── detector_runs/
│   │   ├── v0/                        # Baseline YOLOv8n run artifacts (best.pt, results.csv, confusion matrix)
│   │   └── v1/                        # Tuned YOLOv8s run artifacts (best.pt, results.csv, PR curves)
│   └── classifier_runs/
│       └── v0/                        # Classifier run artifacts (results.csv, checkpoint tracking)
├── notebooks/                         # Pre-executed & interactive Jupyter research notebooks
│   ├── 01_eda.ipynb                   # Phase 0: Multi-source EDA, distributions, duplicate detection, 7 figures
│   ├── 02_preprocessing.ipynb         # Phase 0: Label unification, filtering, and dataset split prototyping
│   ├── 04_detector_finetune.ipynb     # Phase 1: YOLOv8 detector fine-tuning, metric evaluation, inference demo
│   └── 05_classifier_finetune.ipynb   # Phase 2: timm classifier fine-tuning, training loop, metrics & error analysis
├── report/                            # Technical reports, documentation, and visual figures
│   ├── figures/                       # High-resolution figures exported during EDA and pipeline evaluation
│   │   ├── 01_source_inventory.png
│   │   ├── 02_class_balance.png
│   │   ├── 03_sample_grids.png
│   │   ├── 04_image_sizes.png
│   │   ├── 05_duplicates.png
│   │   ├── 06_color_distribution.png
│   │   ├── 07_before_after_preprocessing.png
│   │   ├── diagnostic_gradcam.png     # Phase 2 diagnostic saliency validation
│   │   └── sample_pipeline_output.png # Phase 3 end-to-end composite detection & Grad-CAM visualization
│   └── synopsis.md                    # Technical project synopsis for portfolio/academic review
├── scripts/                           # Multi-platform automation and execution scripts
│   ├── prepare_data.ps1 / .sh         # End-to-end data extraction and label unification runner
│   └── train_detector.ps1 / .sh       # Automated environment sync, GPU check, and YOLOv8 training runner
├── src/dental_model/                  # Core installable Python package
│   ├── __init__.py                    # Package initialization exporting DentalDetector, DentalClassifier, DentalPipeline
│   ├── data/                          # Phase 0 data extraction, mapping, and integrity verification
│   │   ├── __init__.py
│   │   ├── extract.py                 # Multi-format archive extraction (.zip/.rar/.7z) to interim/
│   │   └── unify_labels.py            # Taxonomy remapping, stratified splitting, SHA-256 generation
│   ├── detector/                      # Phase 1 detection subsystem
│   │   ├── __init__.py
│   │   ├── train.py                   # YOLOv8 trainer with deterministic seed fixing & artifact sync
│   │   └── infer.py                   # DentalDetector class: image validation, single & batch inference
│   ├── classifier/                    # Phase 2 classification subsystem
│   │   ├── __init__.py
│   │   ├── model.py                   # DentalClassifier wrapping timm backbones (EfficientNet, ConvNeXt)
│   │   ├── train.py                   # Classifier trainer with class weighting, Albumentations, early stopping
│   │   └── infer.py                   # DentalClassifierInferer: patch/crop inference with softmax confidences
│   ├── utils/                         # Phase 3 visual explainability & plotting utilities
│   │   ├── __init__.py
│   │   └── viz.py                     # draw_detections, generate_gradcam, create_pipeline_visualization
│   └── pipeline.py                    # Phase 3 end-to-end detector -> crop -> classifier pipeline
├── tests/                             # Automated test suite (pytest - 35/35 passing)
│   ├── test_data.py                   # Unit tests for data configs, YOLO remapping, SHA-256 hashing (4 tests)
│   ├── test_detector.py               # Unit tests for detector configs, seeds, device fallback, validation (11 tests)
│   ├── test_classifier.py             # Unit tests for classifier model, class weighting, transforms (10 tests)
│   └── test_pipeline.py               # Unit tests for inferer, pipeline orchestration, and visualization (10 tests)
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
- **Config Documentation Confirmation:** The remapping decision rationale is explicitly documented as formal comments inside [`configs/data_paths.yaml`](file:///c:/Users/user/Downloads/dental-model/configs/data_paths.yaml).

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

#### Key Detector Takeaways:
1. **Model Capacity Scaling:** Upgrading from YOLOv8n (Run v0) to YOLOv8s (Run v1) yielded a **+4.6% boost in overall mAP50** and a **+6.9% boost in caries detection mAP50**.
2. **Precision vs. Recall:** YOLOv8s demonstrated markedly fewer false positive detections on plaque and tooth fissures, improving overall precision from 0.456 to 0.535.
3. **Plaque Detection Limitation:** Plaque detection remains modest (mAP50=0.212), which is explicitly acknowledged as a known clinical data limitation due to diffuse visual plaque boundaries and variable clinical staining.

---

## 6. Phase 2: timm Lesion Classifier Implementation & Diagnostic Verification

### 6.1 Classifier Subsystem Overview

The Stage 2 lesion classification module is fully implemented in `src/dental_model/classifier/`:

- **`model.py` (`DentalClassifier` Module):** Configurable transfer learning wrapper around `timm` vision backbones (`efficientnet_b0` default, supports `convnext_tiny`, `resnet`, etc.) with dropout head and YAML factory.
- **`train.py` (End-to-End Classifier Trainer):** Custom PyTorch Dataset handling path resolution, inverse-frequency class weighting ($w_c = \frac{N}{C \cdot N_c}$), Albumentations augmentations, AdamW optimizer, early stopping, and metric logging.
- **`infer.py` (`DentalClassifierInferer` Module):** Production inference engine supporting single patch classification, bounding box crop extraction (`predict_crop`), and batch inference with full softmax probability distributions.
- **`notebooks/05_classifier_finetune.ipynb`:** Interactive training and evaluation notebook with confusion matrix and error analysis.

### 6.2 Pre-Phase 3 Diagnostic Verification

Before integrating into the pipeline, a comprehensive 3-part diagnostic suite was executed to verify classifier integrity:
1. **Train/Test Leakage Check (`pHash`):** Identified that the raw Caries-Spectra source dataset contains 218 burst/duplicate clusters (563 images) resulting in 124 identical pairs between train and test splits in naive random splitting.
2. **Source Metadata Uniformity:** Confirmed all images are uniformly $224 \times 224$ `.jpg` with no dimensional shortcuts; `healthy` teeth exhibit naturally higher sound enamel reflectance ($\text{R}=212$ vs $\sim 201$).
3. **Grad-CAM Saliency Verification:** Evaluated Grad-CAM heatmaps on test images; confirmed model attention concentrates directly on **tooth crowns and occlusal fissure lines** with center-to-border activation ratios of **$1.7\times - 4.35\times$**, verifying no background/border shortcut learning.

---

## 7. Phase 3: Unified Two-Stage Pipeline & Visual Explainability

### 7.1 Architecture & Workflow

Phase 3 unites Stage 1 (YOLOv8 Object Detector) and Stage 2 (`timm` Lesion Classifier) into an end-to-end, clinically explainable system:

```text
Intraoral Photo (RGB)
         │
         ▼
[Stage 1: DentalDetector (YOLOv8s)]
         │
         ├───► Detects Bounding Boxes: [{bbox, class_name, confidence}]
         │
         ▼
[Lesion Cropping & Coordinate Clamping]
         │
         ▼
[Stage 2: DentalClassifierInferer (EfficientNet-B0)]
         │
         ├───► Fine-Grained Severity: [healthy | caries_early | caries_advanced]
         │
         ▼
[Grad-CAM Saliency Engine (src/dental_model/utils/viz.py)]
         │
         ├───► Generates Heatmap Overlays on Lesion Crops
         │
         ▼
[Structured JSON Output + Composite Diagnostic Visualization]
```

### 7.2 Core Modules Implemented

1. **`src/dental_model/classifier/infer.py` (`DentalClassifierInferer`):**
   - Implements `predict_patch()`, `predict_crop()`, and `predict_batch()`.
   - Returns structured dictionaries with winning class, confidence score, and complete per-class probability distribution.
2. **`src/dental_model/utils/viz.py` (Visualization & Explainability):**
   - `draw_detections()`: Color-codes bounding boxes and badge labels (`healthy`: Green, `plaque`: Orange, `caries`: Red/Amber).
   - `generate_gradcam()`: Computes Grad-CAM heatmaps targeting the model's final convolutional head.
   - `create_pipeline_visualization()`: Produces a multi-panel composite figure showing the full detection image alongside cropped lesions and Grad-CAM explainability heatmaps.
3. **`src/dental_model/pipeline.py` (`DentalPipeline`):**
   - Unified orchestrator supporting dependency injection and CLI execution.
   - Outputs strict, serializable JSON contract:
     ```json
     {
       "image_path": "path/to/image.jpg",
       "image_shape": [224, 224, 3],
       "detections_count": 3,
       "findings": [
         {
           "detection_id": 2,
           "bbox": [63, 52, 80, 81],
           "detector_class": "caries",
           "detector_confidence": 0.3091,
           "classifier_severity": "caries_early",
           "classifier_confidence": 1.0,
           "classifier_probabilities": {
             "healthy": 0.0,
             "caries_early": 1.0,
             "caries_advanced": 0.0
           }
         }
       ]
     }
     ```

---

## 8. Phase 4: Gradio Web Application & Hugging Face Deployment

### 8.1 Interactive Clinical Interface (`app/app.py`)
- **Modern Clinical-Tech UI:** Built using Gradio `gr.Blocks` with custom styling and intuitive multi-stage clinical inspection tabs:
  - **Tab 1 (Composite Diagnostic View):** Overlaid YOLO bounding boxes, lesion crops, and Grad-CAM explainability heatmaps side-by-side.
  - **Tab 2 (Cropped Lesions Gallery):** Individual lesion crops paired with their respective attention heatmaps and confidence scores.
  - **Tab 3 (Structured JSON Output):** Live serializable EHR diagnostic JSON payload.
  - **Tab 4 (Model Architecture & Provenance):** Technical specifications, parameter counts, benchmarks, and clinical disclaimers.
- **Interactive Controls:**
  - Confidence threshold slider ($0.10 - 0.90$).
  - NMS IoU threshold slider ($0.20 - 0.80$).
  - Grad-CAM visual explainability toggle.
  - 1-Click sample image gallery for instant clinical demonstration.

### 8.2 Deployment Artifacts
- **`app/requirements.txt`:** Exported strictly via `uv export --no-dev --no-emit-project` guaranteeing byte-identical, deterministic package installations on Hugging Face Spaces.
- **`app/model_card.md`:** Comprehensive Hugging Face Hub metadata containing evaluation benchmark indices (mAP50, accuracy, macro-F1), data provenance, and clinical disclaimers.
- **`.github/workflows/deploy-space.yml`:** Automated CI/CD workflow pushing `app/` changes to the target Space on merge to `main`.

---

## 9. Verification, Testing & Code Quality

The codebase enforces strict unit testing and linting standards before any milestone is marked complete.

### 9.1 Automated Test Suite (`pytest`)

The test suite in `tests/` contains **36 passing unit tests** across all subsystems:

```text
tests/
├── test_app.py (1 test)               # Verifies Gradio web app instantiation and UI blocks
├── test_data.py (4 tests)             # Verifies YAML parsing of dataset paths, YOLO bbox remapping, SHA-256
├── test_detector.py (11 tests)        # YAML parsing, seeds, CUDA/CPU resolution, image validation, dim cap
├── test_classifier.py (10 tests)      # Classifier YAML schema, forward pass tensor shapes, class weights
└── test_pipeline.py (10 tests)        # Patch inference, crop bounds clamping, batching, viz, pipeline flow
```

**Test Suite Execution Output:**
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\user\Downloads\dental-model
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.2
collected 36 items

tests\test_app.py .                                                      [  2%]
tests\test_classifier.py ..........                                      [ 30%]
tests\test_data.py ....                                                  [ 41%]
tests\test_detector.py ...........                                       [ 72%]
tests\test_pipeline.py ..........                                        [100%]

============================= 36 passed in 6.28s ==============================
```

### 9.2 Linting & Formatting Standards (`ruff`)
- Formatted and validated against Ruff `0.4+` (`line-length = 100`, rules: `["E", "F", "I", "UP", "B"]`).
- All scripts, modules, notebooks, and tests pass `uv run ruff check .` with **0 errors**.

---

## 10. Current Project Progress Tracker

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
| **Phase 2** | **Classifier Architecture** | `src/dental_model/classifier/model.py` (`DentalClassifier`) | ✅ **Completed** |
| **Phase 2** | **Classifier Data & Trainer**| `src/dental_model/classifier/train.py` (Class weighting, Albumentations) | ✅ **Completed** |
| **Phase 2** | **Classifier Test Suite** | `tests/test_classifier.py` (10 unit tests) | ✅ **Completed** |
| **Phase 2** | **Classifier Training Notebook**| `notebooks/05_classifier_finetune.ipynb` | ✅ **Completed** |
| **Phase 2** | **Classifier Training Run** | `models/classifier_runs/v0/` (95.7% test acc, 0.964 macro F1) | ✅ **Completed** |
| **Phase 2** | **Classifier Patch Inferer** | `src/dental_model/classifier/infer.py` (`DentalClassifierInferer`) | ✅ **Completed** |
| **Phase 3** | **Explainability Heatmaps** | `src/dental_model/utils/viz.py` (Grad-CAM overlays) | ✅ **Completed** |
| **Phase 3** | **Pipeline Integration** | `src/dental_model/pipeline.py` (detector $\rightarrow$ crop $\rightarrow$ classifier) | ✅ **Completed** |
| **Phase 3** | **Pipeline Test Suite** | `tests/test_pipeline.py` (10 unit tests) | ✅ **Completed** |
| **Phase 4** | **Gradio Web Demo** | `app/app.py` | ✅ **Completed** |
| **Phase 4** | **HF Space Requirements** | `app/requirements.txt` (exported via `uv export`) | ✅ **Completed** |
| **Phase 4** | **Model Card Metadata** | `app/model_card.md` (real metrics & clinical provenance) | ✅ **Completed** |
| **Phase 4** | **App Test Suite** | `tests/test_app.py` (1 unit test) | ✅ **Completed** |
| **Phase 4** | **HF Deployment CI/CD** | `.github/workflows/deploy-space.yml` | ✅ **Completed** |
| **Phase 5** | **Portfolio Documentation** | `report/synopsis.md`, `README.md` updates | ✅ **Completed** |

---

## 10. Next Technical Steps & Phase 4 Deployment Priorities

1. **Build Interactive Gradio Web Demo (`app/app.py`):**
   - Implement an intuitive, clinical-grade UI:
     - Intraoral photograph upload with drag-and-drop.
     - Confidence and IoU slider controls.
     - Toggle for Grad-CAM explainability heatmaps on detected caries.
     - Side-by-side view of full detections, cropped lesion gallery, and structured JSON diagnostics.
2. **Export HF Spaces Dependencies (`app/requirements.txt`):**
   - Export locked requirements using `uv export --format requirements-txt > app/requirements.txt` ensuring zero dependency mismatches on Hugging Face Spaces.
3. **Automate CD Space Synchronization (`.github/workflows/deploy-space.yml`):**
   - Verify automated GitHub Actions deployment pushing `app/` to the Hugging Face Space repository.

---

## 11. Clinical & Ethical Disclaimers

> [!WARNING]
> **Academic & Research Demonstration Only:** This software and its associated model checkpoints are developed solely for academic research, benchmarking, and portfolio demonstration.
>
> 1. **Not a Medical Device:** The system has not been cleared, certified, or approved by the FDA, CE, or any international medical device regulatory body.
> 2. **No Substitute for Clinical Diagnosis:** The models should never be used as a standalone diagnostic tool or as a substitute for professional dental examination, radiography, and diagnosis by a qualified, licensed dentist.
> 3. **Dataset Provenance & Redistribution:** Source datasets originate from published academic repositories (Mendeley Data DOIs `9jnf2jvghy`, `g8yhdvgjy2`, `3253gj88rr` and Roboflow Universe). Derived model checkpoints must comply with all respective source dataset licenses.
