"""
app/app.py
Interactive Gradio Web Application for Dental Disease Detection and Severity Staging.
Two-Stage Computer Vision Pipeline: YOLOv8s Detector + timm EfficientNet-B0 Classifier + Grad-CAM.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import gradio as gr
import numpy as np

# Ensure project src is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dental_model.pipeline import DentalPipeline  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

# Global pipeline instance cache
_PIPELINE: DentalPipeline | None = None


def get_pipeline() -> DentalPipeline:
    """Lazily load and cache the DentalPipeline instance."""
    global _PIPELINE
    if _PIPELINE is None:
        det_weights = REPO_ROOT / "models/detector_runs/v1/weights/best.pt"
        if not det_weights.exists():
            det_weights = REPO_ROOT / "models/detector_runs/v0/weights/best.pt"

        cls_weights = REPO_ROOT / "models/classifier_runs/v0/best.pt"

        logger.info("Initializing DentalPipeline with weights:")
        logger.info("  • Detector  : %s", det_weights)
        logger.info("  • Classifier: %s", cls_weights)

        _PIPELINE = DentalPipeline(
            detector_weights=det_weights,
            classifier_weights=cls_weights,
            detector_conf=0.25,
            detector_iou=0.45,
        )
    return _PIPELINE


def process_intraoral_image(
    image: np.ndarray | None,
    conf_threshold: float,
    iou_threshold: float,
    enable_gradcam: bool,
    classify_all: bool,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, str]], str, str]:
    """
    Execute end-to-end pipeline on uploaded intraoral image.

    Returns
    -------
    tuple:
        - Composite visualization image (RGB)
        - Gallery of cropped lesion patches with labels
        - Formatted Markdown summary table
        - Structured JSON findings string
    """
    if image is None:
        return None, [], "### ⚠️ Please upload an intraoral photograph to analyze.", "{}"

    pipeline = get_pipeline()

    try:
        results = pipeline.predict(
            image_input=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            generate_saliency=enable_gradcam,
            return_crops=True,
            classify_all_regions=classify_all,
        )

        findings = results.get("findings", [])
        num_findings = len(findings)

        vis_rgb = pipeline.visualize(
            image_input=image,
            pipeline_output=results,
            composite=True,
        )

        gallery_items: list[tuple[np.ndarray, str]] = []
        for idx, f in enumerate(findings):
            crop = f.get("crop_image")
            saliency = f.get("gradcam_overlay")
            sev = f.get("classifier_severity", f.get("detector_class", "unknown"))
            conf = f.get("classifier_confidence", f.get("detector_confidence", 1.0)) * 100

            if crop is not None:
                gallery_items.append((crop, f"Crop #{idx + 1}: {sev} ({conf:.1f}%)"))
            if saliency is not None:
                gallery_items.append((saliency, f"Grad-CAM #{idx + 1}: {sev} Attention"))

        if num_findings == 0:
            md_summary = (
                "### ✅ Analysis Result: No Suspicious Lesions Detected\n\n"
                "*The detector did not locate any hard-tissue lesions above the threshold.*"
            )
        else:
            md_summary = (
                f"### 🔍 Analysis Result: {num_findings} Region(s) Localized\n\n"
                "| # | Localized Condition | Staged Severity | Confidence | Top Probabilities |\n"
                "| :-: | :--- | :--- | :-: | :--- |\n"
            )
            for idx, f in enumerate(findings):
                det_cls = f.get("detector_class", "unknown")
                sev = f.get("classifier_severity", det_cls)
                conf = f.get("classifier_confidence", f.get("detector_confidence", 1.0)) * 100
                probs = f.get("classifier_probabilities", {})
                prob_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in probs.items() if v > 0.01])

                badge = "🔴" if "advanced" in sev else "🟡" if "early" in sev else "🟢"
                md_summary += (
                    f"| **{idx + 1}** | `{det_cls}` | {badge} **`{sev}`** | "
                    f"**{conf:.1f}%** | *{prob_str}* |\n"
                )

        clean_findings = []
        for f in findings:
            clean_f = {k: v for k, v in f.items() if k not in ("crop_image", "gradcam_overlay")}
            clean_findings.append(clean_f)

        json_out = json.dumps(
            {
                "image_shape": results.get("image_shape"),
                "detections_count": num_findings,
                "findings": clean_findings,
            },
            indent=2,
        )

        return vis_rgb, gallery_items, md_summary, json_out

    except Exception as exc:
        logger.exception("Error during pipeline execution: %s", exc)
        return None, [], f"### ❌ Error during analysis: {exc}", "{}"


# Find available test examples for Gradio demo
EXAMPLES = []
test_img_dir = REPO_ROOT / "data/processed/detector/test/images"
if test_img_dir.exists():
    sample_files = list(test_img_dir.glob("*.jpg"))[:5]
    for sf in sample_files:
        EXAMPLES.append([str(sf), 0.25, 0.45, True, False])


def build_app() -> gr.Blocks:
    """Construct the Gradio UI Blocks interface."""
    with gr.Blocks(title="🦷 Dental Model — Clinical AI Screening Demo") as demo:
        with gr.Column():
            gr.Markdown(
                """
                # 🦷 Dental Disease Detection & Severity Classification System
                **Two-Stage Deep Learning Pipeline:** YOLOv8s Spatial Detector +
                timm EfficientNet-B0 Lesion Classifier + Grad-CAM Visual Explainability.
                """
            )

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 📸 1. Input Clinical Photograph")
                input_image = gr.Image(
                    label="Upload Intraoral Photo",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=320,
                )

                gr.Markdown("### ⚙️ 2. Detection & Staging Parameters")
                conf_slider = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.25,
                    step=0.05,
                    label="Detector Confidence Threshold",
                    info="Minimum confidence score for YOLOv8 bounding box proposals.",
                )
                iou_slider = gr.Slider(
                    minimum=0.20,
                    maximum=0.80,
                    value=0.45,
                    step=0.05,
                    label="NMS IoU Threshold",
                    info="Intersection-over-Union threshold for Non-Maximum Suppression.",
                )

                with gr.Row():
                    gradcam_toggle = gr.Checkbox(
                        value=True,
                        label="Generate Grad-CAM Saliency",
                        info="Compute visual explainability heatmaps on lesion crops.",
                    )
                    classify_all_toggle = gr.Checkbox(
                        value=False,
                        label="Classify All Regions",
                        info="Run Stage 2 classifier on healthy/plaque crops as well.",
                    )

                analyze_btn = gr.Button(
                    "🔬 Analyze Intraoral Image",
                    variant="primary",
                    size="lg",
                )

                if EXAMPLES:
                    gr.Markdown("### 💡 Click an Example Image:")
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[
                            input_image,
                            conf_slider,
                            iou_slider,
                            gradcam_toggle,
                            classify_all_toggle,
                        ],
                    )

            with gr.Column(scale=6):
                gr.Markdown("### 📊 3. Multi-Stage Clinical Findings")

                with gr.Tabs():
                    with gr.TabItem("🖼️ Composite Diagnostic View"):
                        output_composite = gr.Image(
                            label="Annotated Detections, Lesion Crops & Grad-CAM Heatmaps",
                            type="numpy",
                            interactive=False,
                        )
                        output_summary = gr.Markdown(
                            value="*Upload an intraoral image and click 'Analyze'.*"
                        )

                    with gr.TabItem("🔎 Cropped Lesions Gallery"):
                        output_gallery = gr.Gallery(
                            label="Extracted Regions & Explainability Saliency",
                            columns=2,
                            rows=2,
                            height=400,
                            object_fit="contain",
                        )

                    with gr.TabItem("📋 Structured JSON Output"):
                        output_json = gr.Code(
                            label="Serializable EHR JSON Contract",
                            language="json",
                            lines=12,
                        )

                    with gr.TabItem("ℹ️ Model Architecture & Provenance"):
                        gr.Markdown(
                            """
                            ### System Architecture Overview
                            - **Stage 1 (Object Detector):** YOLOv8s (`11.2M params`) trained on
                              hard-tissue Roboflow annotations (`healthy`, `plaque`, `caries`).
                              (mAP50: **0.4692**, Caries mAP50: **0.384**).
                            - **Stage 2 (Lesion Classifier):** `timm` EfficientNet-B0 trained on
                              Caries-Spectra (Test Accuracy: **95.67%**, Macro F1: **0.964**).
                            - **Stage 3 (Explainability):** Grad-CAM saliency on final conv head
                              verifying localized enamel demineralization cues.

                            ### Clinical Disclaimers
                            > **Academic & Research Demonstration Only:** This AI system is
                            > developed for academic research. It is **not a medical device**.
                            """
                        )

        analyze_btn.click(
            fn=process_intraoral_image,
            inputs=[
                input_image,
                conf_slider,
                iou_slider,
                gradcam_toggle,
                classify_all_toggle,
            ],
            outputs=[
                output_composite,
                output_gallery,
                output_summary,
                output_json,
            ],
        )

    return demo


if __name__ == "__main__":
    demo_app = build_app()
    demo_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
