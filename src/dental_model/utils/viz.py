"""
src/dental_model/utils/viz.py
Visualization utilities for detection bounding boxes and Grad-CAM explainability heatmaps.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = logging.getLogger(__name__)

# Standard color palette (RGB)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "healthy": (34, 139, 34),          # Forest Green
    "plaque": (255, 140, 0),           # Dark Orange
    "caries": (220, 20, 60),           # Crimson
    "caries_early": (255, 191, 0),     # Amber
    "caries_advanced": (178, 34, 34),  # Firebrick Red
}
DEFAULT_COLOR = (70, 130, 180)         # Steel Blue


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw color-coded bounding boxes and label badges on an RGB image.

    Parameters
    ----------
    image : np.ndarray
        Input image in RGB format (H, W, 3).
    detections : list[dict[str, Any]]
        List of detections or pipeline findings with 'bbox' and class labels.
    line_thickness : int
        Bounding box outline thickness.
    font_scale : float
        Text label font scale.

    Returns
    -------
    np.ndarray
        Annotated image in RGB format.
    """
    annotated = image.copy()
    h, w, _ = annotated.shape

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1 = max(0, min(w - 1, int(round(bbox[0]))))
        y1 = max(0, min(h - 1, int(round(bbox[1]))))
        x2 = max(0, min(w - 1, int(round(bbox[2]))))
        y2 = max(0, min(h - 1, int(round(bbox[3]))))

        cls_name = det.get("classifier_severity") or det.get("detector_class") or det.get(
            "class_name", "lesion"
        )
        conf = det.get("classifier_confidence") or det.get("detector_confidence") or det.get(
            "confidence", 1.0
        )

        color_rgb = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, line_thickness)

        label_text = f"{cls_name} {conf * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        badge_y1 = max(0, y1 - text_h - 6)
        badge_y2 = y1
        badge_x2 = min(w, x1 + text_w + 6)

        cv2.rectangle(
            annotated,
            (x1, badge_y1),
            (badge_x2, badge_y2),
            color_rgb,
            -1,
        )

        cv2.putText(
            annotated,
            label_text,
            (x1 + 3, badge_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def generate_gradcam(
    classifier_model: nn.Module,
    crop_image: np.ndarray,
    target_layer: Any = None,
    target_class: int | None = None,
    device: torch.device | None = None,
    imgsz: int = 224,
) -> np.ndarray:
    """
    Generate a Grad-CAM saliency overlay on a cropped lesion patch.

    Parameters
    ----------
    classifier_model : nn.Module
        Trained classifier instance.
    crop_image : np.ndarray
        Cropped image patch in RGB format.
    target_layer : Any, optional
        Target convolutional layer. Defaults to model's last conv head if available.
    target_class : int, optional
        Target class index for gradient computation.
    device : torch.device, optional
        Compute device.

    Returns
    -------
    np.ndarray
        Heatmap overlay on the crop in RGB format (imgsz, imgsz, 3).
    """
    if device is None:
        device = next(classifier_model.parameters()).device

    classifier_model.eval()

    if target_layer is None:
        if hasattr(classifier_model, "model") and hasattr(classifier_model.model, "conv_head"):
            target_layer = classifier_model.model.conv_head
        elif hasattr(classifier_model, "conv_head"):
            target_layer = classifier_model.conv_head
        else:
            # Fallback: find last Conv2d layer in the model
            conv_layers = [m for m in classifier_model.modules() if isinstance(m, nn.Conv2d)]
            if conv_layers:
                target_layer = conv_layers[-1]
            else:
                raise ValueError("Could not automatically locate a Conv2d layer for Grad-CAM.")

    eval_transform = A.Compose([
        A.Resize(imgsz, imgsz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    resized_rgb = cv2.resize(crop_image, (imgsz, imgsz))
    rgb_float = np.float32(resized_rgb) / 255.0

    augmented = eval_transform(image=crop_image)
    tensor_img = augmented["image"].unsqueeze(0).to(device)

    cam = GradCAM(model=classifier_model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    grayscale_cam = cam(input_tensor=tensor_img, targets=targets)[0, :]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return overlay


def create_pipeline_visualization(
    image: np.ndarray,
    pipeline_result: dict[str, Any],
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Create a composite diagnostic figure displaying full detections, lesion crops,
    and Grad-CAM explainability heatmaps.

    Parameters
    ----------
    image : np.ndarray
        Original intraoral image (RGB).
    pipeline_result : dict[str, Any]
        Output dictionary from DentalPipeline.predict().
    save_path : str | Path, optional
        File path to save the generated visualization.

    Returns
    -------
    np.ndarray
        Rendered RGB visualization array.
    """
    findings = pipeline_result.get("findings", [])
    num_findings = len(findings)

    annotated_full = draw_detections(image, findings)

    if num_findings == 0:
        # Single image visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(annotated_full)
        ax.set_title("Detection Output (No lesions detected)", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
    else:
        # Multi-panel visualization: Left = Full image, Right = Crop + Grad-CAM rows
        fig = plt.figure(figsize=(14, max(6, num_findings * 3)))
        gs = fig.add_gridspec(num_findings, 3, width_ratios=[1.5, 1.0, 1.0])

        ax_full = fig.add_subplot(gs[:, 0])
        ax_full.imshow(annotated_full)
        ax_full.set_title(
            f"Stage 1: Detections ({num_findings} regions)",
            fontsize=12,
            fontweight="bold",
        )
        ax_full.axis("off")

        for idx, finding in enumerate(findings):
            crop = finding.get("crop_image")
            gradcam = finding.get("gradcam_overlay")
            cls_name = finding.get("classifier_severity", finding.get("detector_class", "unknown"))
            conf = finding.get("classifier_confidence", finding.get("detector_confidence", 1.0))

            ax_crop = fig.add_subplot(gs[idx, 1])
            if crop is not None:
                ax_crop.imshow(crop)
            else:
                ax_crop.text(0.5, 0.5, "Crop unavailable", ha="center", va="center")
            ax_crop.set_title(
                f"Crop #{idx + 1}: {cls_name} ({conf * 100:.1f}%)",
                fontsize=10,
                fontweight="bold",
            )
            ax_crop.axis("off")

            ax_cam = fig.add_subplot(gs[idx, 2])
            if gradcam is not None:
                ax_cam.imshow(gradcam)
                ax_cam.set_title("Grad-CAM Explainability", fontsize=10, fontweight="bold")
            else:
                ax_cam.text(0.5, 0.5, "Grad-CAM N/A", ha="center", va="center")
                ax_cam.set_title("Grad-CAM", fontsize=10)
            ax_cam.axis("off")

        plt.suptitle(
            "Dental Computer Vision Pipeline — Multi-Stage Diagnostic",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=150, bbox_inches="tight")
        logger.info("Saved pipeline visualization to %s", p)

    # Convert canvas to RGB array
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb_output = rgba[:, :, :3]
    plt.close(fig)

    return rgb_output
