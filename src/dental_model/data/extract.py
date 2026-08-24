"""Data extraction module for Dental Model datasets.

Decompresses archives (.zip, .rar, .7z, .tar.gz) and organizes raw/interim folders.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data_config(config_path: str | Path = "configs/data_paths.yaml") -> dict:
    """Load data paths config."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning("Config %s not found. Using default paths.", config_file)
        return {
            "sources": {
                "caries_spectra": {
                    "raw_path": "data/raw/caries_spectra/",
                    "status": "required",
                },
                "roboflow_detection": {
                    "raw_path": "data/raw/roboflow_detection/",
                    "status": "required",
                },
                "plaque_diagno": {
                    "raw_path": "data/raw/plaque_diagno/",
                    "status": "optional",
                },
                "gingivitis_captioning": {
                    "raw_path": "data/raw/gingivitis_captioning/",
                    "status": "deferred_v2",
                },
            },
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
        }
    with open(config_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip archive safely."""
    logger.info("Extracting %s -> %s", zip_path.name, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def copy_or_link_dir(source_dir: Path, target_dir: Path) -> None:
    """Copy directory contents into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def extract_caries_spectra(data_root: Path, interim_dir: Path) -> Path:
    """Standardize/extract Caries-Spectra dataset into interim/caries_spectra."""
    target = interim_dir / "caries_spectra"
    target.mkdir(parents=True, exist_ok=True)

    candidates = [
        data_root / "raw" / "caries_spectra",
        data_root / "Caries-Spectra A dataset of Enamel Caries",
        data_root / "caries_spectra",
    ]

    found_dir: Path | None = None
    for cand in candidates:
        if cand.exists():
            found_dir = cand
            break

    if not found_dir:
        for p in data_root.glob("*Caries-Spectra*"):
            if p.is_dir():
                found_dir = p
                break

    if found_dir:
        logger.info("Found Caries-Spectra at %s", found_dir)
        nested_candidates = list(found_dir.rglob("Caries_Dataset"))
        if nested_candidates:
            dataset_dir = nested_candidates[0]
            copy_or_link_dir(dataset_dir, target / "Caries_Dataset")
        else:
            copy_or_link_dir(found_dir, target)
        return target
    else:
        logger.warning("Caries-Spectra directory not found in %s", data_root)
        return target


def extract_roboflow_detection(data_root: Path, interim_dir: Path) -> Path:
    """Standardize/extract Roboflow detection dataset into interim/roboflow_detection."""
    target = interim_dir / "roboflow_detection"
    target.mkdir(parents=True, exist_ok=True)

    candidates = [
        data_root / "raw" / "roboflow_detection",
        data_root / "dental-dataset-+-more-healthy-and-caries-+-plaque-1",
        data_root / "roboflow_detection",
    ]

    found_dir: Path | None = None
    for cand in candidates:
        if cand.exists():
            found_dir = cand
            break

    if not found_dir:
        for p in data_root.glob("*dental-dataset*"):
            if p.is_dir():
                found_dir = p
                break

    if found_dir:
        logger.info("Found Roboflow dataset at %s", found_dir)
        copy_or_link_dir(found_dir, target)
        return target
    else:
        logger.warning("Roboflow detection directory not found in %s", data_root)
        return target


def extract_plaque_diagno(data_root: Path, interim_dir: Path) -> Path:
    """Standardize/extract Plaque Diagno dataset into interim/plaque_diagno."""
    target = interim_dir / "plaque_diagno"
    target.mkdir(parents=True, exist_ok=True)

    zip_files = list(data_root.glob("*Dental Plaque Diagno*.zip"))
    if zip_files:
        safe_extract_zip(zip_files[0], target)

    for p in data_root.glob("*Dental Plaque Diagno*"):
        if p.is_dir() and p != target:
            copy_or_link_dir(p, target)
            break

    return target


def extract_gingivitis_captioning(data_root: Path, interim_dir: Path) -> Path:
    """Standardize/extract Gingivitis Captioning dataset into interim/gingivitis_captioning."""
    target = interim_dir / "gingivitis_captioning"
    target.mkdir(parents=True, exist_ok=True)

    zip_files = list(data_root.glob("*GINGIVITIS*.zip"))
    if zip_files:
        safe_extract_zip(zip_files[0], target)

    for p in data_root.glob("*GINGIVITIS*"):
        if p.is_dir() and p != target:
            copy_or_link_dir(p, target)
            break

    return target


def extract_all(config_path: str | Path = "configs/data_paths.yaml") -> dict[str, Path]:
    """Extract and standardize all available datasets into data/interim/."""
    cfg = load_data_config(config_path)
    data_root = Path(os.environ.get("DATA_ROOT", "data"))
    interim_dir = Path(cfg.get("interim_dir", "data/interim"))
    interim_dir.mkdir(parents=True, exist_ok=True)

    extracted = {}
    logger.info("Starting Phase 0: Data Extraction to %s", interim_dir)

    extracted["caries_spectra"] = extract_caries_spectra(data_root, interim_dir)
    extracted["roboflow_detection"] = extract_roboflow_detection(data_root, interim_dir)
    extracted["plaque_diagno"] = extract_plaque_diagno(data_root, interim_dir)
    extracted["gingivitis_captioning"] = extract_gingivitis_captioning(data_root, interim_dir)

    logger.info("Extraction complete. Extracted datasets: %s", list(extracted.keys()))
    return extracted


def main() -> None:
    extract_all()


if __name__ == "__main__":
    main()
