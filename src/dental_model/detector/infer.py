"""
src/dental_model/detector/infer.py
Run YOLOv8 inference with a trained detector checkpoint.

Usage (from repo root):
    uv run python -m dental_model.detector.infer \
        --weights models/detector_runs/v0/weights/best.pt \
        --image path/to/image.jpg
    uv run python -m dental_model.detector.infer \
        --weights models/detector_runs/v0/weights/best.pt \
        --image path/to/image.jpg --conf 0.3 --iou 0.45
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Maximum image dimension accepted at inference (guard against huge inputs)
MAX_IMAGE_DIM = 4096


class DentalDetector:
    """
    Lightweight wrapper around a trained YOLOv8 checkpoint for dental detection.

    Attributes
    ----------
    weights_path : Path
        Absolute path to the .pt checkpoint file.
    conf_threshold : float
        Minimum confidence to keep a detection (0–1).
    iou_threshold : float
        IoU threshold for NMS (0–1).
    imgsz : int
        Inference image size (square).
    device : str
        Torch device string ("cuda" or "cpu").
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        if not self.weights_path.is_absolute():
            self.weights_path = REPO_ROOT / self.weights_path
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "Loading detector from %s on device '%s'", self.weights_path, self.device
        )
        self.model = YOLO(str(self.weights_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image_path: str | Path,
    ) -> list[dict[str, Any]]:
        """
        Run detection on a single image file.

        Parameters
        ----------
        image_path : path to the image (jpg, png, etc.)

        Returns
        -------
        List of detection dicts, each with keys:
            bbox        : [x1, y1, x2, y2]  (pixel coords, float)
            class_id    : int
            class_name  : str
            confidence  : float (0–1)
        Sorted by confidence descending.

        Raises
        ------
        FileNotFoundError  if the image does not exist.
        ValueError         if the file is not a recognised image format
                           or exceeds MAX_IMAGE_DIM.
        RuntimeError       if inference fails for any other reason.
        """
        path = self._validate_image(image_path)

        try:
            results = self.model.predict(
                source=str(path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Inference failed on {path}: {exc}") from exc

        detections: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                cls_name = (
                    result.names[cls_id]
                    if result.names and cls_id in result.names
                    else str(cls_id)
                )
                detections.append(
                    {
                        "bbox": [round(v, 2) for v in xyxy],
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                    }
                )

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def predict_batch(
        self,
        image_paths: list[str | Path],
    ) -> list[list[dict[str, Any]]]:
        """
        Run detection on a list of images.

        Returns a list (one entry per image) of detection lists.
        Failed images produce an empty list and a logged warning.
        """
        outputs: list[list[dict[str, Any]]] = []
        for img in image_paths:
            try:
                outputs.append(self.predict(img))
            except Exception as exc:
                logger.warning("Skipping %s – %s", img, exc)
                outputs.append([])
        return outputs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image_path: str | Path) -> Path:
        """Return resolved Path after basic sanity checks."""
        from PIL import Image

        VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in VALID_SUFFIXES:
            raise ValueError(
                f"Unsupported image format '{path.suffix}'. "
                f"Supported: {sorted(VALID_SUFFIXES)}"
            )
        # Check dimensions without loading full array
        try:
            with Image.open(path) as img:
                w, h = img.size
        except Exception as exc:
            raise ValueError(f"Cannot open image {path}: {exc}") from exc

        if max(w, h) > MAX_IMAGE_DIM:
            raise ValueError(
                f"Image {path} is too large ({w}x{h}). "
                f"Maximum allowed dimension: {MAX_IMAGE_DIM}px."
            )
        return path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 dental detector on an image")
    parser.add_argument("--weights", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default=None, help="Device override: cuda or cpu")
    args = parser.parse_args()

    detector = DentalDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )
    detections = detector.predict(args.image)
    print(json.dumps(detections, indent=2))


if __name__ == "__main__":
    _cli()
