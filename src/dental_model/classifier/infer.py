"""
src/dental_model/classifier/infer.py
Run fine-grained dental lesion classification on image crops.

Usage (from repo root):
    uv run python -m dental_model.classifier.infer \
        --weights models/classifier_runs/v0/best.pt \
        --image path/to/patch.jpg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image

from dental_model.classifier.model import DentalClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_IMAGE_DIM = 4096
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class DentalClassifierInferer:
    """
    Inference engine for fine-grained dental lesion severity classification.

    Attributes
    ----------
    weights_path : Path
        Path to the trained PyTorch checkpoint (.pt).
    device : torch.device
        Device used for inference ('cuda' or 'cpu').
    classes : list[str]
        Ordered list of class names.
    model : DentalClassifier
        Loaded and evaluated PyTorch model.
    transform : A.Compose
        Albumentations evaluation transform pipeline.
    """

    def __init__(
        self,
        weights_path: str | Path = "models/classifier_runs/v0/best.pt",
        config_path: str | Path | None = None,
        device: str | None = None,
        imgsz: int = 224,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_absolute():
            self.weights_path = REPO_ROOT / self.weights_path

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {self.weights_path}"
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            "Loading DentalClassifier from %s on device '%s'",
            self.weights_path,
            self.device,
        )

        ckpt = torch.load(self.weights_path, map_location=self.device)

        # Resolve config from checkpoint or external YAML
        if "config" in ckpt and isinstance(ckpt["config"], dict):
            cfg = ckpt["config"]
        elif config_path is not None:
            cfg_p = Path(config_path)
            if not cfg_p.is_absolute():
                cfg_p = REPO_ROOT / cfg_p
            with cfg_p.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg_p = REPO_ROOT / "configs/classifier.yaml"
            if cfg_p.exists():
                with cfg_p.open(encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
            else:
                cfg = {
                    "model": {"backbone": "efficientnet_b0", "pretrained": False, "dropout": 0.2},
                    "data": {"classes": ["healthy", "caries_early", "caries_advanced"]},
                }

        self.cfg = cfg
        self.classes: list[str] = cfg.get("data", {}).get(
            "classes", ["healthy", "caries_early", "caries_advanced"]
        )

        if "class_to_idx" in ckpt and isinstance(ckpt["class_to_idx"], dict):
            self.class_to_idx = ckpt["class_to_idx"]
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        else:
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

        self.model = DentalClassifier.from_config(cfg).to(self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.imgsz = imgsz or cfg.get("data", {}).get("image_size", 224)
        self.transform = A.Compose([
            A.Resize(self.imgsz, self.imgsz),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def predict_patch(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
    ) -> dict[str, Any]:
        """
        Run classification on a single image patch.

        Parameters
        ----------
        image_input : str, Path, np.ndarray, or PIL.Image
            Input image patch.

        Returns
        -------
        dict[str, Any]
            Prediction dictionary containing:
            - class_id: int
            - class_name: str
            - confidence: float
            - probabilities: dict[str, float]
        """
        rgb_img = self._to_rgb_array(image_input)

        augmented = self.transform(image=rgb_img)
        tensor_img = augmented["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = self.idx_to_class.get(pred_idx, str(pred_idx))
        confidence = float(probs[pred_idx])

        probabilities = {
            self.idx_to_class.get(i, str(i)): round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return {
            "class_id": pred_idx,
            "class_name": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }

    def predict_crop(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        bbox: list[float] | tuple[float, float, float, float],
    ) -> dict[str, Any]:
        """
        Crop bounding box from full image and classify lesion severity.

        Parameters
        ----------
        image_input : str, Path, np.ndarray, or PIL.Image
            Full-size image.
        bbox : list or tuple of 4 floats [x1, y1, x2, y2]
            Bounding box coordinates.

        Returns
        -------
        dict[str, Any]
            Classification result for the cropped patch.
        """
        rgb_img = self._to_rgb_array(image_input)
        h, w, _ = rgb_img.shape

        x1 = max(0, int(round(bbox[0])))
        y1 = max(0, int(round(bbox[1])))
        x2 = min(w, int(round(bbox[2])))
        y2 = min(h, int(round(bbox[3])))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop dimensions after clamping: [{x1}, {y1}, {x2}, {y2}]")

        crop = rgb_img[y1:y2, x1:x2]
        return self.predict_patch(crop)

    def predict_batch(
        self,
        patches: list[str | Path | np.ndarray | Image.Image],
    ) -> list[dict[str, Any]]:
        """
        Run classification on a list of image patches.

        Returns
        -------
        list[dict[str, Any]]
            List of prediction dictionaries.
        """
        results: list[dict[str, Any]] = []
        for patch in patches:
            try:
                results.append(self.predict_patch(patch))
            except Exception as exc:
                logger.warning("Patch classification failed: %s", exc)
                results.append({
                    "class_id": -1,
                    "class_name": "unknown",
                    "confidence": 0.0,
                    "probabilities": {},
                })
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _to_rgb_array(
        cls,
        image_input: str | Path | np.ndarray | Image.Image,
    ) -> np.ndarray:
        """Convert various image input types to uint8 RGB NumPy array."""
        if isinstance(image_input, (str, Path)):
            path = cls._validate_image_path(image_input)
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Failed to read image at {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                return image_input.astype(np.uint8)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            raise ValueError(f"Unsupported numpy array shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def _validate_image_path(image_path: str | Path) -> Path:
        """Validate existence, format, and dimension of image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in VALID_SUFFIXES:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. Supported: {sorted(VALID_SUFFIXES)}"
            )
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception as exc:
            raise ValueError(f"Cannot open image {path}: {exc}") from exc

        if max(w, h) > MAX_IMAGE_DIM:
            raise ValueError(
                f"Image {path} too large ({w}x{h}). Max allowed: {MAX_IMAGE_DIM}px."
            )
        return path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run DentalClassifier inferer on an image patch")
    parser.add_argument(
        "--weights",
        default="models/classifier_runs/v0/best.pt",
        help="Path to best.pt classifier checkpoint",
    )
    parser.add_argument("--image", required=True, help="Path to input image patch")
    parser.add_argument("--device", default=None, help="Device override (cuda/cpu)")
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image size")
    args = parser.parse_args()

    inferer = DentalClassifierInferer(
        weights_path=args.weights,
        device=args.device,
        imgsz=args.imgsz,
    )
    result = inferer.predict_patch(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
