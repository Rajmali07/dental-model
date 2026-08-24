"""
tests/test_detector.py
Unit tests for src/dental_model/detector/train.py and infer.py.

Run with: uv run pytest tests/test_detector.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_config(tmp_path: Path, data_yaml_path: str | None = None) -> dict:
    """Return a minimal valid detector config dict."""
    if data_yaml_path is None:
        data_yaml_path = "data/processed/detector/data.yaml"
    return {
        "data": {"yaml_path": data_yaml_path, "image_size": 640},
        "model": {"architecture": "yolov8n.pt", "pretrained": True},
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "patience": 5,
            "device": "cpu",
            "seed": 42,
            "optimizer": "auto",
            "lr0": 0.01,
            "workers": 0,
        },
        "output": {"project_dir": str(tmp_path / "runs"), "run_name": "test_run"},
        "logging": {"tracker": "none"},
    }


# ---------------------------------------------------------------------------
# train.py tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_valid_config(self, tmp_path: Path) -> None:
        """load_config should parse a correct YAML without raising."""
        from dental_model.detector.train import load_config

        cfg_dict = _make_minimal_config(tmp_path)
        cfg_file = tmp_path / "detector.yaml"
        cfg_file.write_text(yaml.dump(cfg_dict))

        cfg = load_config(cfg_file)
        assert cfg["train"]["seed"] == 42
        assert cfg["model"]["architecture"] == "yolov8n.pt"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from dental_model.detector.train import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_missing_required_section_raises(self, tmp_path: Path) -> None:
        from dental_model.detector.train import load_config

        incomplete = {"data": {}, "model": {}}  # missing "train" and "output"
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(yaml.dump(incomplete))

        with pytest.raises(KeyError):
            load_config(cfg_file)


class TestSetSeeds:
    def test_set_seeds_does_not_raise(self) -> None:
        from dental_model.detector.train import _set_seeds

        _set_seeds(0)
        _set_seeds(42)
        _set_seeds(99)


class TestResolveDevice:
    def test_cpu_always_available(self) -> None:
        from dental_model.detector.train import _resolve_device

        assert _resolve_device("cpu") == "cpu"

    def test_cuda_fallback_to_cpu_when_unavailable(self, monkeypatch) -> None:
        import torch

        from dental_model.detector.train import _resolve_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        result = _resolve_device("cuda")
        assert result == "cpu"


# ---------------------------------------------------------------------------
# infer.py tests
# ---------------------------------------------------------------------------

class TestDentalDetectorValidateImage:
    def test_valid_jpg(self, tmp_path: Path) -> None:
        from PIL import Image

        from dental_model.detector.infer import DentalDetector

        img_path = tmp_path / "test.jpg"
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)
        result = DentalDetector._validate_image(img_path)
        assert result == img_path

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        from dental_model.detector.infer import DentalDetector

        with pytest.raises(FileNotFoundError):
            DentalDetector._validate_image(tmp_path / "ghost.jpg")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        from dental_model.detector.infer import DentalDetector

        p = tmp_path / "file.xyz"
        p.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported image format"):
            DentalDetector._validate_image(p)

    def test_oversized_image_raises(self, tmp_path: Path, monkeypatch) -> None:
        from PIL import Image

        import dental_model.detector.infer as infer_mod
        from dental_model.detector.infer import DentalDetector

        # Temporarily lower the limit
        monkeypatch.setattr(infer_mod, "MAX_IMAGE_DIM", 50)

        img_path = tmp_path / "big.jpg"
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)
        with pytest.raises(ValueError, match="too large"):
            DentalDetector._validate_image(img_path)


class TestDentalDetectorInit:
    def test_missing_weights_raises(self, tmp_path: Path) -> None:
        from dental_model.detector.infer import DentalDetector

        with pytest.raises(FileNotFoundError):
            DentalDetector(weights_path=tmp_path / "nosuchfile.pt")
