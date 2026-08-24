"""
tests/test_classifier.py
Unit tests for the Stage 2 lesion classifier subsystem (model & data loading).

Run with: uv run pytest tests/test_classifier.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dental_model.classifier.model import DentalClassifier
from dental_model.classifier.train import (
    _resolve_device,
    _set_seeds,
    compute_class_weights,
    get_transforms,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_minimal_classifier_config(tmp_path: Path) -> dict:
    """Return a minimal valid classifier config dict."""
    return {
        "data": {
            "processed_csv": "data/processed/classifier/labels.csv",
            "classes": ["healthy", "caries_early", "caries_advanced"],
            "image_size": 224,
            "val_split": 0.15,
            "test_split": 0.15,
            "stratify": True,
        },
        "model": {
            "backbone": "efficientnet_b0",
            "pretrained": False,
            "dropout": 0.2,
        },
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "class_weighting": True,
            "seed": 42,
            "device": "cpu",
            "early_stopping_patience": 3,
            "num_workers": 0,
        },
        "augmentation": {
            "horizontal_flip": True,
            "rotation_degrees": 15,
            "color_jitter": 0.1,
            "random_crop": False,
        },
        "output": {
            "project_dir": str(tmp_path / "classifier_runs"),
            "run_name": "test_run",
        },
        "logging": {"tracker": "none"},
    }


class TestClassifierConfig:
    def test_load_valid_config(self, tmp_path: Path) -> None:
        """load_config parses valid YAML file without errors."""
        cfg_dict = _make_minimal_classifier_config(tmp_path)
        cfg_file = tmp_path / "classifier.yaml"
        cfg_file.write_text(yaml.dump(cfg_dict), encoding="utf-8")

        loaded = load_config(cfg_file)
        assert loaded["model"]["backbone"] == "efficientnet_b0"
        assert len(loaded["data"]["classes"]) == 3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_missing_section_raises(self, tmp_path: Path) -> None:
        incomplete = {"data": {}, "model": {}}
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(yaml.dump(incomplete), encoding="utf-8")

        with pytest.raises(KeyError, match="Config missing required sections"):
            load_config(cfg_file)


class TestDentalClassifierModel:
    def test_forward_pass_shape(self) -> None:
        """Forward pass on dummy tensor produces expected (B, num_classes) logits."""
        model = DentalClassifier(
            backbone="efficientnet_b0",
            num_classes=3,
            pretrained=False,
            dropout=0.2,
        )
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 3)

    def test_from_config(self, tmp_path: Path) -> None:
        """from_config initializes DentalClassifier with correct parameters."""
        cfg_dict = _make_minimal_classifier_config(tmp_path)
        model = DentalClassifier.from_config(cfg_dict)

        assert model.num_classes == 3
        assert model.backbone_name == "efficientnet_b0"
        assert model.drop_rate == 0.2


class TestClassWeightsComputation:
    def test_balanced_distribution(self) -> None:
        """Equally distributed classes receive equal weights of 1.0."""
        labels = [0, 0, 1, 1, 2, 2]
        weights = compute_class_weights(labels, num_classes=3)
        assert torch.allclose(weights, torch.tensor([1.0, 1.0, 1.0]))

    def test_imbalanced_distribution(self) -> None:
        """Imbalanced classes receive inverse-frequency weights."""
        # 280 healthy, 560 caries_early, 560 caries_advanced (Total = 1400)
        labels = [0] * 280 + [1] * 560 + [2] * 560
        weights = compute_class_weights(labels, num_classes=3)

        expected_w0 = 1400.0 / (3.0 * 280.0)  # ~1.6667
        expected_w1 = 1400.0 / (3.0 * 560.0)  # ~0.8333
        expected_w2 = 1400.0 / (3.0 * 560.0)  # ~0.8333

        assert pytest.approx(weights[0].item(), rel=1e-3) == expected_w0
        assert pytest.approx(weights[1].item(), rel=1e-3) == expected_w1
        assert pytest.approx(weights[2].item(), rel=1e-3) == expected_w2


class TestTransformsAndDevice:
    def test_transforms_output_shape(self) -> None:
        """Albumentations pipeline produces correctly shaped normalized tensor."""
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        transform = get_transforms(imgsz=224, is_train=True)
        transformed = transform(image=dummy_img)["image"]

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)

    def test_device_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        dev = _resolve_device("cuda")
        assert dev == torch.device("cpu")

    def test_seed_setting(self) -> None:
        _set_seeds(42)
