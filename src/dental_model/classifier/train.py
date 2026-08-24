"""
src/dental_model/classifier/train.py
Train a timm backbone classifier on data/processed/classifier/labels.csv
using hyperparameters from configs/classifier.yaml.

Usage (from repo root):
    uv run python -m dental_model.classifier.train
    uv run python -m dental_model.classifier.train --config configs/classifier.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from dental_model.classifier.model import DentalClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _set_seeds(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Seeds fixed to %d", seed)


def _resolve_device(requested: str) -> torch.device:
    """Return torch.device('cuda') if available, otherwise fall back to CPU."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available – falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_config(config_path: str | Path = "configs/classifier.yaml") -> dict[str, Any]:
    """Load and validate the classifier YAML config."""
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    required_sections = {"data", "model", "train", "output"}
    missing = required_sections - set(cfg.keys())
    if missing:
        raise KeyError(f"Config missing required sections: {missing}")
    return cfg


def compute_class_weights(
    labels: list[int] | np.ndarray,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for balanced cross-entropy loss.
    weight[c] = total_samples / (num_classes * count[c])
    """
    labels_arr = np.asarray(labels, dtype=int)
    counts = np.bincount(labels_arr, minlength=num_classes)
    total = len(labels_arr)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


def get_transforms(
    imgsz: int = 224,
    is_train: bool = True,
    aug_cfg: dict[str, Any] | None = None,
) -> A.Compose:
    """Build Albumentations transform pipeline."""
    if aug_cfg is None:
        aug_cfg = {}

    if is_train:
        transforms_list: list[Any] = []
        if aug_cfg.get("random_crop", True):
            transforms_list.append(A.RandomResizedCrop(imgsz, imgsz, scale=(0.8, 1.0), p=1.0))
        else:
            transforms_list.append(A.Resize(imgsz, imgsz))

        if aug_cfg.get("horizontal_flip", True):
            transforms_list.append(A.HorizontalFlip(p=0.5))

        rot_deg = aug_cfg.get("rotation_degrees", 15)
        if rot_deg > 0:
            transforms_list.append(A.Rotate(limit=rot_deg, p=0.5))

        jitter = aug_cfg.get("color_jitter", 0.1)
        if jitter > 0:
            transforms_list.append(
                A.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=jitter,
                    hue=0.05,
                    p=0.5,
                )
            )

        transforms_list.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        return A.Compose(transforms_list)
    else:
        return A.Compose([
            A.Resize(imgsz, imgsz),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class DentalClassificationDataset(Dataset):
    """PyTorch Dataset for dental lesion images."""

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform: A.Compose | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        raw_p = Path(row["image_path"])
        if raw_p.exists():
            return raw_p
        if "relative_path" in row:
            cand = REPO_ROOT / "data" / str(row["relative_path"])
            if cand.exists():
                return cand
        cand2 = REPO_ROOT / raw_p
        if cand2.exists():
            return cand2
        raise FileNotFoundError(f"Image not found for row: {row.to_dict()}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row)

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img)
            tensor_img = augmented["image"]
        else:
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label_name = str(row["label"])
        label_idx = self.class_to_idx[label_name]

        return tensor_img, label_idx


def build_dataloaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], torch.Tensor]:
    """
    Load CSV, build Dataset objects using existing train/val/test splits,
    compute class weights, and return DataLoaders.
    """
    csv_path = REPO_ROOT / cfg["data"]["processed_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Labels CSV not found at {csv_path}. Run scripts/prepare_data.ps1 first."
        )

    df = pd.read_csv(csv_path)
    classes = cfg["data"]["classes"]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    logger.info(
        "Dataset splits loaded: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(df),
    )

    train_labels = [class_to_idx[lbl] for lbl in train_df["label"]]
    class_weights = compute_class_weights(train_labels, num_classes=len(classes))
    logger.info("Computed train-split class weights: %s", class_weights.tolist())

    imgsz = cfg["data"].get("image_size", 224)
    aug_cfg = cfg.get("augmentation", {})
    train_transform = get_transforms(imgsz=imgsz, is_train=True, aug_cfg=aug_cfg)
    eval_transform = get_transforms(imgsz=imgsz, is_train=False)

    train_dataset = DentalClassificationDataset(train_df, class_to_idx, transform=train_transform)
    val_dataset = DentalClassificationDataset(val_df, class_to_idx, transform=eval_transform)
    test_dataset = DentalClassificationDataset(test_df, class_to_idx, transform=eval_transform)

    batch_size = cfg["train"].get("batch_size", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    # Clamp workers on Windows if needed
    if os.name == "nt" and num_workers > 4:
        num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_to_idx, class_weights


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model on a DataLoader, returning (loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def train(config_path: str | Path = "configs/classifier.yaml") -> Path:
    """
    Run classifier training according to YAML config and return path to best checkpoint.
    """
    cfg = load_config(config_path)

    seed = cfg["train"].get("seed", 42)
    _set_seeds(seed)

    device = _resolve_device(cfg["train"].get("device", "cuda"))
    logger.info("Training on device: %s", device)

    train_loader, val_loader, test_loader, class_to_idx, class_weights = build_dataloaders(cfg)

    model = DentalClassifier.from_config(cfg).to(device)

    use_weighting = cfg["train"].get("class_weighting", True)
    if use_weighting:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info(
            "Using class-weighted CrossEntropyLoss: %s",
            class_weights.tolist(),
        )
    else:
        criterion = nn.CrossEntropyLoss()

    lr = float(cfg["train"].get("lr", 3e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = cfg["train"].get("epochs", 30)
    patience = cfg["train"].get("early_stopping_patience", 7)

    project_dir = REPO_ROOT / cfg["output"]["project_dir"]
    run_name = cfg["output"]["run_name"]
    output_dir = project_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    results_csv_path = output_dir / "results.csv"

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, Any]] = []

    logger.info("Starting classifier training for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds: list[int] = []
        train_targets: list[int] = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %02d/%02d - Train Loss: %.4f, Train Acc: %.4f | "
            "Val Loss: %.4f, Val Acc: %.4f, Val F1 (macro): %.4f",
            epoch,
            epochs,
            avg_train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "class_to_idx": class_to_idx,
                    "config": cfg,
                },
                best_checkpoint_path,
            )
            logger.info("Saved new best model checkpoint to %s", best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", epoch)
                break

    # Save last checkpoint and history CSV
    torch.save(model.state_dict(), last_checkpoint_path)
    pd.DataFrame(history).to_csv(results_csv_path, index=False)
    logger.info("Saved training history to %s", results_csv_path)

    # Evaluate best model on test split
    if best_checkpoint_path.exists():
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
        logger.info(
            "Final Test Split Evaluation (Best Checkpoint): "
            "Test Loss: %.4f, Test Acc: %.4f, Test F1 (macro): %.4f",
            test_loss,
            test_acc,
            test_f1,
        )

    return best_checkpoint_path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train timm dental lesion classifier")
    parser.add_argument(
        "--config",
        default="configs/classifier.yaml",
        help="Path to classifier config YAML (default: configs/classifier.yaml)",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    _cli()
