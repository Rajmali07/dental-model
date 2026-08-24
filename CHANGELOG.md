# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions match
`pyproject.toml` and tagged model releases (e.g. `v0.1.0-detector`, `v0.1.0-classifier`).

## [Unreleased]
### Added
- Initial repo scaffold: `pyproject.toml`, `uv.lock`, CI/CD workflows, configs, `.env.example`.

## [0.1.0] - 2026-08-24
### Added
- Phase 0: Data extraction and label unification pipeline (`src/dental_model/data/`).
- Phase 1: YOLOv8 object detector module (`src/dental_model/detector/`) with deterministic training and inference wrapper.
- Fine-tuned detector released as `v0.1.0-detector` (YOLOv8s, 100 epochs, imgsz=640) achieving mAP50=0.469 and mAP50-95=0.262 on held-out test split.
- Pre-executed Jupyter research notebooks (`01_eda.ipynb`, `02_preprocessing.ipynb`, `04_detector_finetune.ipynb`).
- Automated pytest test suite with 15 passing unit tests.

### Known Limitations
- Plaque detection performance remains modest (mAP50=0.212), driven by subtle visual demineralization boundaries and high inter-photographic illumination shifts; explicitly documented for future augmentation/ablation improvements.
