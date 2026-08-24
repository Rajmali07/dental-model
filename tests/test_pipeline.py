"""
tests/test_pipeline.py
Unit tests for the Phase 3 inference, visual explainability, and unified pipeline subsystems.

Run with: uv run pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from dental_model.classifier.infer import DentalClassifierInferer
from dental_model.classifier.model import DentalClassifier
from dental_model.pipeline import DentalPipeline
from dental_model.utils.viz import (
    create_pipeline_visualization,
    draw_detections,
    generate_gradcam,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dummy_rgb_image() -> np.ndarray:
    """Return a synthetic RGB image (300 x 400 x 3)."""
    return np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)


@pytest.fixture
def mock_classifier_checkpoint(tmp_path: Path) -> Path:
    """Create a valid dummy checkpoint for DentalClassifierInferer."""
    model = DentalClassifier(backbone="efficientnet_b0", num_classes=3, pretrained=False)
    ckpt_path = tmp_path / "mock_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model": {"backbone": "efficientnet_b0", "pretrained": False, "dropout": 0.2},
                "data": {
                    "classes": ["healthy", "caries_early", "caries_advanced"],
                    "image_size": 224,
                },
            },
            "class_to_idx": {"healthy": 0, "caries_early": 1, "caries_advanced": 2},
        },
        ckpt_path,
    )
    return ckpt_path


class TestClassifierInferer:
    def test_inferer_predict_patch(
        self, mock_classifier_checkpoint: Path, dummy_rgb_image: np.ndarray
    ) -> None:
        """predict_patch returns structured dictionary with expected keys and types."""
        inferer = DentalClassifierInferer(
            weights_path=mock_classifier_checkpoint,
            device="cpu",
        )
        res = inferer.predict_patch(dummy_rgb_image)

        assert "class_id" in res
        assert "class_name" in res
        assert "confidence" in res
        assert "probabilities" in res
        assert res["class_name"] in ["healthy", "caries_early", "caries_advanced"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["probabilities"]) == 3
        assert pytest.approx(sum(res["probabilities"].values()), rel=1e-2) == 1.0

    def test_inferer_predict_crop_bounds(
        self, mock_classifier_checkpoint: Path, dummy_rgb_image: np.ndarray
    ) -> None:
        """predict_crop clamps bounding box coordinates and classifies patch."""
        inferer = DentalClassifierInferer(
            weights_path=mock_classifier_checkpoint,
            device="cpu",
        )
        # Bounding box exceeding image bounds [x1, y1, x2, y2]
        res = inferer.predict_crop(dummy_rgb_image, bbox=[-20, -10, 1000, 1000])
        assert "class_name" in res

    def test_inferer_invalid_crop_raises(
        self, mock_classifier_checkpoint: Path, dummy_rgb_image: np.ndarray
    ) -> None:
        """Degenerate bounding box raises ValueError."""
        inferer = DentalClassifierInferer(
            weights_path=mock_classifier_checkpoint,
            device="cpu",
        )
        with pytest.raises(ValueError, match="Invalid crop dimensions"):
            inferer.predict_crop(dummy_rgb_image, bbox=[100, 100, 50, 50])

    def test_inferer_predict_batch(
        self, mock_classifier_checkpoint: Path, dummy_rgb_image: np.ndarray
    ) -> None:
        """predict_batch handles list of image inputs."""
        inferer = DentalClassifierInferer(
            weights_path=mock_classifier_checkpoint,
            device="cpu",
        )
        patches = [dummy_rgb_image, Image.fromarray(dummy_rgb_image)]
        results = inferer.predict_batch(patches)
        assert len(results) == 2
        assert all("class_name" in r for r in results)

    def test_inferer_missing_weights_raises(self, tmp_path: Path) -> None:
        """Missing weights file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DentalClassifierInferer(weights_path=tmp_path / "nonexistent.pt")


class TestVisualization:
    def test_draw_detections(self, dummy_rgb_image: np.ndarray) -> None:
        """draw_detections draws bounding boxes and returns RGB array."""
        detections = [
            {
                "bbox": [20.0, 30.0, 120.0, 150.0],
                "class_name": "caries",
                "confidence": 0.85,
            },
            {
                "bbox": [150.0, 50.0, 250.0, 200.0],
                "classifier_severity": "caries_early",
                "classifier_confidence": 0.92,
            },
        ]
        annotated = draw_detections(dummy_rgb_image, detections)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == dummy_rgb_image.shape
        assert annotated.dtype == np.uint8

    def test_generate_gradcam(
        self, mock_classifier_checkpoint: Path, dummy_rgb_image: np.ndarray
    ) -> None:
        """generate_gradcam creates saliency overlay on crop."""
        inferer = DentalClassifierInferer(
            weights_path=mock_classifier_checkpoint,
            device="cpu",
        )
        crop = dummy_rgb_image[:200, :200]
        overlay = generate_gradcam(
            classifier_model=inferer.model,
            crop_image=crop,
            device=torch.device("cpu"),
            imgsz=224,
        )
        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == (224, 224, 3)

    def test_create_pipeline_visualization(
        self, dummy_rgb_image: np.ndarray, tmp_path: Path
    ) -> None:
        """create_pipeline_visualization builds multi-panel diagnostic figure."""
        pipeline_output = {
            "image_path": "test.jpg",
            "image_shape": [300, 400, 3],
            "detections_count": 1,
            "findings": [
                {
                    "detection_id": 0,
                    "bbox": [50, 50, 200, 200],
                    "detector_class": "caries",
                    "detector_confidence": 0.89,
                    "classifier_severity": "caries_advanced",
                    "classifier_confidence": 0.95,
                    "crop_image": dummy_rgb_image[50:200, 50:200],
                    "gradcam_overlay": np.zeros((224, 224, 3), dtype=np.uint8),
                }
            ],
        }
        save_file = tmp_path / "viz_test.png"
        rendered = create_pipeline_visualization(
            image=dummy_rgb_image,
            pipeline_result=pipeline_output,
            save_path=save_file,
        )
        assert isinstance(rendered, np.ndarray)
        assert save_file.exists()


class TestDentalPipeline:
    def test_pipeline_predict_flow(
        self,
        dummy_rgb_image: np.ndarray,
    ) -> None:
        """Pipeline coordinates detector -> crop -> classifier end-to-end."""
        # Configure Mock Detector
        mock_detector = MagicMock()
        mock_detector.predict.return_value = [
            {
                "bbox": [10.0, 10.0, 100.0, 100.0],
                "class_id": 2,
                "class_name": "caries",
                "confidence": 0.88,
            },
            {
                "bbox": [120.0, 20.0, 220.0, 120.0],
                "class_id": 1,
                "class_name": "plaque",
                "confidence": 0.75,
            },
        ]
        mock_detector.device = "cpu"

        # Configure Mock Classifier
        mock_classifier = MagicMock()
        mock_classifier.predict_patch.return_value = {
            "class_id": 2,
            "class_name": "caries_advanced",
            "confidence": 0.94,
            "probabilities": {
                "healthy": 0.01,
                "caries_early": 0.05,
                "caries_advanced": 0.94,
            },
        }
        mock_classifier.device = torch.device("cpu")
        mock_classifier.imgsz = 224

        pipeline = DentalPipeline(
            detector=mock_detector,
            classifier=mock_classifier,
        )

        res = pipeline.predict(
            dummy_rgb_image,
            generate_saliency=False,
            return_crops=True,
        )

        assert res["detections_count"] == 2
        assert len(res["findings"]) == 2

        # Caries finding has severity staging
        caries_finding = res["findings"][0]
        assert caries_finding["detector_class"] == "caries"
        assert caries_finding["classifier_severity"] == "caries_advanced"
        assert caries_finding["classifier_confidence"] == 0.94
        assert "crop_image" in caries_finding

        # Plaque finding preserves detector class
        plaque_finding = res["findings"][1]
        assert plaque_finding["detector_class"] == "plaque"
        assert plaque_finding["classifier_severity"] == "plaque"

    def test_pipeline_empty_detections(
        self,
        dummy_rgb_image: np.ndarray,
    ) -> None:
        """Pipeline handles zero detection case gracefully."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = []
        mock_detector.device = "cpu"

        mock_classifier = MagicMock()
        mock_classifier.device = torch.device("cpu")

        pipeline = DentalPipeline(
            detector=mock_detector,
            classifier=mock_classifier,
        )

        res = pipeline.predict(dummy_rgb_image)
        assert res["detections_count"] == 0
        assert res["findings"] == []
