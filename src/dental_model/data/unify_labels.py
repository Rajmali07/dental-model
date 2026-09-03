"""Data label unification module for Dental Model datasets.

Maps heterogeneous source labels into a single unified taxonomy and produces
clean datasets for detector and classifier training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard taxonomy mapping for Caries-Spectra
CARIES_SPECTRA_MAP = {
    "AdvanceEnamel_Caries": "caries_advanced",
    "EarlyStageEnamel_Caries": "caries_early",
    "NoEnamel_Caries": "healthy",
}

# Standard taxonomy mapping for Roboflow Detection classes (v1 - 3 classes)
# Roboflow: 0: caries, 1: cavity, 2: gingivitis, 3: gum_swelling, 4: healthy, 5: plaque
# Target 3-class schema: 0: healthy, 1: plaque, 2: caries
ROBOFLOW_CLASS_MAP = {
    0: (2, "caries"),       # caries -> caries
    1: (2, "caries"),       # cavity -> caries
    4: (0, "healthy"),      # healthy -> healthy
    5: (1, "plaque"),       # plaque -> plaque
}

# Version 2 taxonomy mapping: includes soft-tissue periodontal conditions (5 classes)
# Target 5-class schema: 0: healthy, 1: plaque, 2: caries, 3: gingivitis, 4: gum_swelling
ROBOFLOW_CLASS_MAP_V2 = {
    0: (2, "caries"),
    1: (2, "caries"),
    2: (3, "gingivitis"),
    3: (4, "gum_swelling"),
    4: (0, "healthy"),
    5: (1, "plaque"),
}

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_checksums(processed_dir: Path, output_file: Path) -> dict[str, str]:
    """Generate SHA256 checksums for files in processed directory."""
    logger.info("Generating dataset checksums for %s -> %s", processed_dir, output_file)
    checksums = {}
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for p in sorted(processed_dir.rglob("*")):
        if p.is_file() and p != output_file:
            rel_path = p.relative_to(processed_dir.parent).as_posix()
            checksums[rel_path] = compute_sha256(p)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    logger.info("Saved %d file checksums to %s", len(checksums), output_file)
    return checksums


def unify_caries_spectra(
    interim_dir: Path,
    processed_dir: Path,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Unify Caries-Spectra classification labels and create stratified splits."""
    caries_dir = interim_dir / "caries_spectra"
    records = []

    for raw_cat, unified_label in CARIES_SPECTRA_MAP.items():
        cat_dirs = list(caries_dir.rglob(raw_cat))
        for d in cat_dirs:
            for img in d.iterdir():
                if img.is_file() and img.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    records.append({
                        "image_path": str(img.resolve()),
                        "relative_path": img.relative_to(interim_dir.parent).as_posix(),
                        "label": unified_label,
                        "source": "caries_spectra",
                        "raw_label": raw_cat,
                    })

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No classification images found in %s", caries_dir)
        return df

    logger.info(
        "Caries-Spectra loaded: %d images. Class counts:\n%s",
        len(df),
        df["label"].value_counts(),
    )

    # Stratified Train/Val/Test Split
    train_val_df, test_df = train_test_split(
        df, test_size=test_split, random_state=seed, stratify=df["label"]
    )
    adjusted_val_size = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    classifier_processed_dir = processed_dir / "classifier"
    classifier_processed_dir.mkdir(parents=True, exist_ok=True)
    classifier_csv = classifier_processed_dir / "labels.csv"
    combined.to_csv(classifier_csv, index=False)

    root_csv = processed_dir / "labels.csv"
    combined.to_csv(root_csv, index=False)
    logger.info(
        "Saved unified classification labels to %s (%d rows)",
        classifier_csv,
        len(combined),
    )

    return combined


def remap_yolo_label_file(
    src_label_file: Path,
    dst_label_file: Path,
    class_map: dict[int, tuple[int, str]],
) -> int:
    """Remap YOLO format bounding boxes to target classes and write to destination."""
    dst_label_file.parent.mkdir(parents=True, exist_ok=True)
    remapped_lines = []

    if src_label_file.exists():
        with open(src_label_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    if cls_id in class_map:
                        new_cls_id, _ = class_map[cls_id]
                        remapped_lines.append(f"{new_cls_id} {' '.join(parts[1:])}\n")
                except (ValueError, IndexError):
                    continue

    with open(dst_label_file, "w", encoding="utf-8") as f:
        f.writelines(remapped_lines)

    return len(remapped_lines)


def unify_roboflow_detection(interim_dir: Path, processed_dir: Path) -> Path:
    """Remap Roboflow detection dataset to unified 3-class YOLO structure."""
    roboflow_dir = interim_dir / "roboflow_detection"
    detector_processed_dir = processed_dir / "detector"
    detector_processed_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    split_map = {"train": "train", "valid": "val", "test": "test"}

    for split in splits:
        src_img_dir = roboflow_dir / split / "images"
        src_lbl_dir = roboflow_dir / split / "labels"
        dst_split = split_map[split]
        dst_img_dir = detector_processed_dir / dst_split / "images"
        dst_lbl_dir = detector_processed_dir / dst_split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            logger.warning("Roboflow split %s not found at %s", split, src_img_dir)
            continue

        for img_path in src_img_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                dst_img = dst_img_dir / img_path.name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

                lbl_name = img_path.stem + ".txt"
                src_lbl = src_lbl_dir / lbl_name
                dst_lbl = dst_lbl_dir / lbl_name
                remap_yolo_label_file(src_lbl, dst_lbl, ROBOFLOW_CLASS_MAP)

    data_yaml_path = detector_processed_dir / "data.yaml"
    detector_data = {
        "path": detector_processed_dir.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 3,
        "names": ["healthy", "plaque", "caries"],
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(detector_data, f, sort_keys=False)

    logger.info("Saved unified detector dataset config to %s", data_yaml_path)
    return data_yaml_path


def unify_roboflow_detection_v2(interim_dir: Path, processed_dir: Path) -> Path:
    """Remap Roboflow detection dataset to unified 5-class YOLO structure for v2."""
    roboflow_dir = interim_dir / "roboflow_detection"
    detector_v2_dir = processed_dir / "detector_v2"
    detector_v2_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    split_map = {"train": "train", "valid": "val", "test": "test"}

    for split in splits:
        src_img_dir = roboflow_dir / split / "images"
        src_lbl_dir = roboflow_dir / split / "labels"
        dst_split = split_map[split]
        dst_img_dir = detector_v2_dir / dst_split / "images"
        dst_lbl_dir = detector_v2_dir / dst_split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            logger.warning("Roboflow split %s not found at %s", split, src_img_dir)
            continue

        for img_path in src_img_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                dst_img = dst_img_dir / img_path.name
                if not dst_img.exists():
                    try:
                        os.link(img_path, dst_img)
                    except OSError:
                        shutil.copy2(img_path, dst_img)

                lbl_name = img_path.stem + ".txt"
                src_lbl = src_lbl_dir / lbl_name
                dst_lbl = dst_lbl_dir / lbl_name
                remap_yolo_label_file(src_lbl, dst_lbl, ROBOFLOW_CLASS_MAP_V2)

    data_yaml_path = detector_v2_dir / "data.yaml"
    detector_data = {
        "path": detector_v2_dir.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 5,
        "names": ["healthy", "plaque", "caries", "gingivitis", "gum_swelling"],
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(detector_data, f, sort_keys=False)

    logger.info("Saved unified v2 detector dataset config to %s", data_yaml_path)
    return data_yaml_path


def unify_all(
    config_path: str | Path = "configs/data_paths.yaml",
) -> tuple[pd.DataFrame, Path, Path]:
    """Run full label unification pipeline for classifier and detector."""
    config_file = Path(config_path)
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")

    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            interim_dir = Path(cfg.get("interim_dir", "data/interim"))
            processed_dir = Path(cfg.get("processed_dir", "data/processed"))

    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 0: Unify Labels...")
    classifier_df = unify_caries_spectra(interim_dir, processed_dir)
    detector_yaml = unify_roboflow_detection(interim_dir, processed_dir)

    checksums_file = processed_dir / "checksums.json"
    generate_checksums(processed_dir, checksums_file)

    logger.info("Phase 0 label unification complete!")
    return classifier_df, detector_yaml, checksums_file


def main() -> None:
    unify_all()


if __name__ == "__main__":
    main()
