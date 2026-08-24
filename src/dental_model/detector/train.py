"""
src/dental_model/detector/train.py
Train YOLOv8 on data/processed/detector using settings from configs/detector.yaml.

Usage (from repo root):
    uv run python -m dental_model.detector.train
    uv run python -m dental_model.detector.train --config configs/detector_v1.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

REPO_ROOT = Path(__file__).resolve().parents[3]  # dental-model/


def _set_seeds(seed: int) -> None:
    """Fix random seeds for Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Seeds fixed to %d", seed)


def _resolve_device(requested: str) -> str:
    """Return 'cuda' if available when requested, otherwise fall back to 'cpu'."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available – falling back to CPU.")
        return "cpu"
    return requested


def load_config(config_path: str | Path) -> dict:
    """Load and validate the detector YAML config."""
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    required_keys = {"data", "model", "train", "output"}
    missing = required_keys - set(cfg.keys())
    if missing:
        raise KeyError(f"Config missing required sections: {missing}")
    return cfg


def train(config_path: str | Path = "configs/detector.yaml") -> Path:
    """
    Run YOLOv8 fine-tuning and return the path to the best weights file.

    Parameters
    ----------
    config_path: path to detector.yaml (relative paths resolved from repo root).

    Returns
    -------
    Path to best.pt produced by this run.
    """
    cfg = load_config(config_path)

    seed: int = cfg["train"].get("seed", 42)
    _set_seeds(seed)

    device: str = _resolve_device(str(cfg["train"].get("device", "cpu")))
    logger.info("Training device: %s", device)

    data_yaml = REPO_ROOT / cfg["data"]["yaml_path"]
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. "
            "Run scripts/prepare_data.ps1 first."
        )

    model_arch: str = cfg["model"]["architecture"]
    pretrained: bool = cfg["model"].get("pretrained", True)
    logger.info("Loading model: %s (pretrained=%s)", model_arch, pretrained)
    model = YOLO(model_arch)

    project_dir = (REPO_ROOT / cfg["output"]["project_dir"]).resolve()
    run_name: str = cfg["output"]["run_name"]
    project_dir.mkdir(parents=True, exist_ok=True)

    tracker: str = cfg.get("logging", {}).get("tracker", "none")
    if tracker == "wandb":
        try:
            import wandb  # noqa: F401
            os.environ.setdefault("WANDB_PROJECT", "dental-model-detector")
        except ImportError:
            logger.warning("wandb requested but not installed – disabling tracking.")
            tracker = "none"
    elif tracker == "mlflow":
        try:
            import mlflow  # noqa: F401
        except ImportError:
            logger.warning("mlflow requested but not installed – disabling tracking.")
            tracker = "none"

    train_kwargs: dict = {
        "data": str(data_yaml.resolve()),
        "epochs": cfg["train"].get("epochs", 100),
        "imgsz": cfg["data"].get("image_size", 640),
        "batch": cfg["train"].get("batch_size", 16),
        "patience": cfg["train"].get("patience", 20),
        "device": device,
        "seed": seed,
        "optimizer": cfg["train"].get("optimizer", "auto"),
        "lr0": cfg["train"].get("lr0", 0.01),
        "workers": cfg["train"].get("workers", 4),
        "project": str(project_dir),
        "name": run_name,
        "pretrained": pretrained,
        "exist_ok": True,
        "verbose": True,
    }
    if tracker != "none":
        train_kwargs["tracker"] = tracker

    logger.info("Starting YOLO training with kwargs: %s", train_kwargs)
    results = model.train(**train_kwargs)

    # Resolve target directory
    target_run_dir = project_dir / run_name
    target_run_dir.mkdir(parents=True, exist_ok=True)

    # Check if Ultralytics outputted to runs/detect/<project>/<name> or save_dir
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    if save_dir and save_dir.exists() and save_dir.resolve() != target_run_dir.resolve():
        logger.info("Syncing training artifacts from %s to %s", save_dir, target_run_dir)
        for item in save_dir.iterdir():
            dest = target_run_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    best_weights = target_run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = target_run_dir / "weights" / "last.pt"

    if best_weights.exists():
        logger.info("Training complete. Best weights: %s", best_weights)
    else:
        logger.warning("Could not locate best.pt/last.pt in %s", target_run_dir)

    return best_weights


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 dental detector")
    parser.add_argument(
        "--config",
        default="configs/detector.yaml",
        help="Path to detector config YAML (default: configs/detector.yaml)",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    _cli()
