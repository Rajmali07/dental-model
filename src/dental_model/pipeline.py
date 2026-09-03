"""
src/dental_model/pipeline.py
Unified two-stage dental computer vision pipeline:
1. Stage 1: YOLOv8 detector localizes tooth regions (healthy, plaque, caries).
2. Stage 2: timm classifier assesses fine-grained lesion severity (healthy, early, advanced).
3. Explainability: Grad-CAM visual heatmaps on cropped lesion patches.

Usage (from repo root):
    uv run python -m dental_model.pipeline \
        --image path/to/image.jpg \
        --save-viz report/figures/sample_pipeline_output.png \
        --gradcam
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from dental_model.classifier.infer import DentalClassifierInferer
from dental_model.detector.infer import DentalDetector
from dental_model.utils.viz import create_pipeline_visualization, draw_detections, generate_gradcam

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class DentalPipeline:
    """
    End-to-end dental disease detection and severity classification pipeline.

    Attributes
    ----------
    detector : DentalDetector
        Stage 1 YOLOv8 object detector instance.
    classifier : DentalClassifierInferer
        Stage 2 timm lesion classifier instance.
    """

    def __init__(
        self,
        detector: DentalDetector | None = None,
        classifier: DentalClassifierInferer | None = None,
        detector_weights: str | Path = "models/detector_runs/v2/weights/best.pt",
        classifier_weights: str | Path = "models/classifier_runs/v0/best.pt",
        detector_conf: float = 0.25,
        detector_iou: float = 0.45,
        detector_imgsz: int = 640,
        classifier_imgsz: int = 224,
        device: str | None = None,
    ) -> None:
        if detector is not None:
            self.detector = detector
        else:
            det_path = Path(detector_weights)
            if not det_path.is_absolute():
                det_path = REPO_ROOT / det_path

            # Fallback to v2, v1, or v0 detector if specified weights not found
            if not det_path.exists():
                for candidate in ["v2", "v1", "v0"]:
                    cand_path = REPO_ROOT / f"models/detector_runs/{candidate}/weights/best.pt"
                    if cand_path.exists():
                        logger.warning(
                            "Detector %s not found. Falling back to %s", det_path, cand_path
                        )
                        det_path = cand_path
                        break

            self.detector = DentalDetector(
                weights_path=det_path,
                conf_threshold=detector_conf,
                iou_threshold=detector_iou,
                imgsz=detector_imgsz,
                device=device,
            )

        if classifier is not None:
            self.classifier = classifier
        else:
            cls_path = Path(classifier_weights)
            if not cls_path.is_absolute():
                cls_path = REPO_ROOT / cls_path

            self.classifier = DentalClassifierInferer(
                weights_path=cls_path,
                device=device,
                imgsz=classifier_imgsz,
            )

        logger.info(
            "DentalPipeline successfully initialized on device '%s'",
            getattr(self.detector, "device", "cpu"),
        )

    def predict(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        generate_saliency: bool = False,
        return_crops: bool = True,
        classify_all_regions: bool = False,
    ) -> dict[str, Any]:
        """
        Run end-to-end pipeline on an intraoral photo.

        Parameters
        ----------
        image_input : str, Path, np.ndarray, or PIL.Image
            Input intraoral photo.
        conf_threshold : float, optional
            Detection confidence threshold override.
        iou_threshold : float, optional
            IoU NMS threshold override.
        generate_saliency : bool
            Whether to compute Grad-CAM heatmap overlays for classified crops.
        return_crops : bool
            Whether to attach cropped NumPy RGB arrays in findings dicts.
        classify_all_regions : bool
            If True, classifies healthy/plaque crops in addition to caries crops.

        Returns
        -------
        dict[str, Any]
            Structured pipeline output containing image metadata and findings list.
        """
        rgb_img = self._load_rgb(image_input)
        h, w, c = rgb_img.shape

        # Temporarily adjust thresholds if provided
        if conf_threshold is not None:
            self.detector.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.detector.iou_threshold = iou_threshold

        # Stage 1: Detection
        if isinstance(image_input, (str, Path)):
            detections = self.detector.predict(image_input)
        else:
            # Temporary file conversion for YOLO API if array passed
            temp_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            temp_path = REPO_ROOT / ".cache_pipeline_input.jpg"
            cv2.imwrite(str(temp_path), temp_bgr)
            try:
                detections = self.detector.predict(temp_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        findings: list[dict[str, Any]] = []

        # Stage 2: Crop Extraction & Lesion Classification
        for idx, det in enumerate(detections):
            bbox = det["bbox"]
            x1 = max(0, min(w - 1, int(round(bbox[0]))))
            y1 = max(0, min(h - 1, int(round(bbox[1]))))
            x2 = max(0, min(w - 1, int(round(bbox[2]))))
            y2 = max(0, min(h - 1, int(round(bbox[3]))))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = rgb_img[y1:y2, x1:x2]
            det_class = det["class_name"]

            finding: dict[str, Any] = {
                "detection_id": idx,
                "bbox": [x1, y1, x2, y2],
                "detector_class": det_class,
                "detector_confidence": round(float(det["confidence"]), 4),
            }

            # Classify caries (or all hard tissue regions if configured)
            should_classify = (det_class == "caries") or classify_all_regions

            if should_classify and crop.size > 0:
                cls_res = self.classifier.predict_patch(crop)
                finding["classifier_severity"] = cls_res["class_name"]
                finding["classifier_confidence"] = cls_res["confidence"]
                finding["classifier_probabilities"] = cls_res["probabilities"]

                if generate_saliency:
                    try:
                        gradcam_overlay = generate_gradcam(
                            classifier_model=self.classifier.model,
                            crop_image=crop,
                            target_class=cls_res["class_id"],
                            device=self.classifier.device,
                            imgsz=self.classifier.imgsz,
                        )
                        finding["gradcam_overlay"] = gradcam_overlay
                    except Exception as exc:
                        logger.warning("Grad-CAM generation failed for finding #%d: %s", idx, exc)
                        finding["gradcam_overlay"] = None
            else:
                # Stage 1 label preserved for plaque/healthy
                finding["classifier_severity"] = det_class
                finding["classifier_confidence"] = round(float(det["confidence"]), 4)
                finding["classifier_probabilities"] = {
                    det_class: round(float(det["confidence"]), 4)
                }

            if return_crops:
                finding["crop_image"] = crop

            findings.append(finding)

        output: dict[str, Any] = {
            "image_path": str(image_input) if isinstance(image_input, (str, Path)) else None,
            "image_shape": [h, w, c],
            "detections_count": len(findings),
            "findings": findings,
        }

        return output

    def visualize(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        pipeline_output: dict[str, Any],
        save_path: str | Path | None = None,
        composite: bool = True,
    ) -> np.ndarray:
        """
        Render visualization of detections and explainability overlays.

        Parameters
        ----------
        image_input : str, Path, np.ndarray, or PIL.Image
            Input intraoral photo.
        pipeline_output : dict[str, Any]
            Output from predict().
        save_path : str or Path, optional
            Path to save the generated image.
        composite : bool
            If True, generates multi-panel figure. If False, draws simple bounding boxes.

        Returns
        -------
        np.ndarray
            RGB annotated image array.
        """
        rgb_img = self._load_rgb(image_input)

        if composite:
            return create_pipeline_visualization(
                image=rgb_img,
                pipeline_result=pipeline_output,
                save_path=save_path,
            )
        else:
            annotated = draw_detections(rgb_img, pipeline_output.get("findings", []))
            if save_path is not None:
                p = Path(save_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(p), bgr)
            return annotated

    @staticmethod
    def _load_rgb(image_input: str | Path | np.ndarray | Image.Image) -> np.ndarray:
        """Load any image input as RGB uint8 NumPy array."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image not found at {p}")
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise ValueError(f"Failed to read image at {p}")
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
            raise ValueError(f"Unsupported array shape: {image_input.shape}")
        raise TypeError(f"Unsupported input type: {type(image_input)}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run DentalPipeline end-to-end inference")
    parser.add_argument("--image", required=True, help="Path to input intraoral image")
    parser.add_argument(
        "--detector-weights",
        default="models/detector_runs/v1/weights/best.pt",
        help="Path to YOLOv8 detector weights",
    )
    parser.add_argument(
        "--classifier-weights",
        default="models/classifier_runs/v0/best.pt",
        help="Path to classifier weights",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="Detector IoU threshold")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmaps")
    parser.add_argument("--save-viz", default=None, help="Path to save composite visualization")
    args = parser.parse_args()

    pipeline = DentalPipeline(
        detector_weights=args.detector_weights,
        classifier_weights=args.classifier_weights,
        detector_conf=args.conf,
        detector_iou=args.iou,
    )

    results = pipeline.predict(
        image_input=args.image,
        generate_saliency=args.gradcam,
        return_crops=True,
    )

    # Print JSON without raw numpy arrays
    printable_findings = []
    for f in results.get("findings", []):
        f_copy = {k: v for k, v in f.items() if k not in ("crop_image", "gradcam_overlay")}
        printable_findings.append(f_copy)

    summary = {
        "image_path": results["image_path"],
        "image_shape": results["image_shape"],
        "detections_count": results["detections_count"],
        "findings": printable_findings,
    }
    print(json.dumps(summary, indent=2))

    if args.save_viz:
        pipeline.visualize(
            image_input=args.image,
            pipeline_output=results,
            save_path=args.save_viz,
            composite=True,
        )


if __name__ == "__main__":
    _cli()
