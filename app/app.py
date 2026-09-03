"""
app/app.py
Interactive Gradio Web Application for Dental Disease Detection and Severity Staging.
Two-Stage Computer Vision Pipeline: YOLOv8s Detector + timm EfficientNet-B0 Classifier + Grad-CAM.
Designed for Clinician Decision-Support with Clinical Presets & Reconciled Diagnostic Triage.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import gradio as gr
import numpy as np

# Ensure project src or bundled app package is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dental_model.pipeline import DentalPipeline  # noqa: E402

try:
    import spaces
except ImportError:
    spaces = None


def maybe_gpu(fn):
    """Wrap function with @spaces.GPU for Hugging Face ZeroGPU compatibility."""
    if spaces is not None and hasattr(spaces, "GPU"):
        return spaces.GPU(fn)
    return fn


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
        det_weights = REPO_ROOT / "models/detector_runs/v2/weights/best.pt"
        if not det_weights.exists():
            det_weights = REPO_ROOT / "models/detector_runs/v1/weights/best.pt"
        if not det_weights.exists():
            det_weights = REPO_ROOT / "models/detector_runs/v0/weights/best.pt"
        if not det_weights.exists():
            from huggingface_hub import hf_hub_download

            logger.info("Downloading detector v2 weights from HF Hub (Bihari04/dental-model)...")
            det_weights = Path(
                hf_hub_download(repo_id="Bihari04/dental-model", filename="detector_v2_best.pt")
            )

        cls_weights = REPO_ROOT / "models/classifier_runs/v0/best.pt"
        if not cls_weights.exists():
            from huggingface_hub import hf_hub_download

            logger.info("Downloading classifier weights from HF Hub (Bihari04/dental-model)...")
            cls_weights = Path(
                hf_hub_download(repo_id="Bihari04/dental-model", filename="classifier_best.pt")
            )

        logger.info("Initializing DentalPipeline with weights:")
        logger.info("  • Detector  : %s", det_weights)
        logger.info("  • Classifier: %s", cls_weights)

        _PIPELINE = DentalPipeline(
            detector_weights=det_weights,
            classifier_weights=cls_weights,
            detector_conf=0.35,
            detector_iou=0.45,
        )
    return _PIPELINE


# Clinical terminology dictionary for clinicians
CLINICAL_TAXONOMY = {
    "caries_advanced": {
        "label": "Advanced Dental Caries",
        "badge": "🔴",
        "category": "High Risk / Active Cavitation",
        "severity": "Stage 2 (Dentin Cavitation)",
        "recommendation": "Restorative excavation & filling; x-ray recommended.",
        "rank": 3,
    },
    "caries_early": {
        "label": "Incipient / Early Caries",
        "badge": "🟡",
        "category": "Moderate Risk / Demineralization",
        "severity": "Stage 1 (Enamel Micro-Lesion)",
        "recommendation": "Non-invasive: topical fluoride varnish & 3-mo recall.",
        "rank": 2,
    },
    "plaque": {
        "label": "Dental Plaque Biofilm",
        "badge": "🟣",
        "category": "Surface Biofilm Accumulation",
        "severity": "Hygiene Deficit",
        "recommendation": "Professional prophylaxis scaling and hygiene instructions.",
        "rank": 1,
    },
    "gingivitis": {
        "label": "Gingivitis (Gum Inflammation)",
        "badge": "🟠",
        "category": "Periodontal Soft-Tissue Inflammation",
        "severity": "Stage 1 (Marginal Erythema)",
        "recommendation": "Professional scaling, chlorhexidine rinse & flossing instructions.",
        "rank": 2,
    },
    "gum_swelling": {
        "label": "Periodontal Gum Swelling",
        "badge": "🔴",
        "category": "Active Periodontal Edema",
        "severity": "Stage 2 (Edema / Infection Risk)",
        "recommendation": "Periodontal probing, bone evaluation & anti-inflammatory triage.",
        "rank": 3,
    },
    "healthy": {
        "label": "Sound Enamel",
        "badge": "🟢",
        "category": "No Active Pathology",
        "severity": "Sound Enamel",
        "recommendation": "Routine preventive recall exam in 6 months.",
        "rank": 0,
    },
    "ruled_out": {
        "label": "Sound Surface (Ruled Out)",
        "badge": "🟢",
        "category": "Benign Surface / False-Alarm Resolved",
        "severity": "Sound Enamel",
        "recommendation": "Classifier verified sound enamel. Benign anatomical fissure.",
        "rank": 0,
    },
}

# Clinical preset parameter profiles
PRESETS = {
    "Routine Screening (Recommended)": {
        "conf": 0.35,
        "iou": 0.45,
        "guidance": (
            "🎯 **Routine Screening (Balanced):** Optimal balance of sensitivity and "
            "low false alarms for general checkups (Conf: **0.35**, IoU: **0.45**)."
        ),
    },
    "Early Prevention (High Sensitivity)": {
        "conf": 0.20,
        "iou": 0.40,
        "guidance": (
            "🔍 **Early Prevention (Sensitive):** Lowers bar to catch subtle, "
            "incipient white-spot demineralization (Conf: **0.20**, IoU: **0.40**)."
        ),
    },
    "Definitive Diagnosis (High Specificity)": {
        "conf": 0.50,
        "iou": 0.50,
        "guidance": (
            "🛡️ **Definitive Diagnosis (Strict):** Strictly highlights unambiguous, "
            "cavitated caries lesions (Conf: **0.50**, IoU: **0.50**)."
        ),
    },
}


@maybe_gpu
def process_intraoral_image(
    image: np.ndarray | None,
    conf_threshold: float,
    iou_threshold: float,
    enable_gradcam: bool,
    classify_all: bool,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, str]], str, str]:
    """
    Execute end-to-end pipeline on uploaded intraoral image with clinical translation.

    Returns
    -------
    tuple:
        - Composite visualization image (RGB)
        - Gallery of cropped lesion patches with labels
        - Formatted Markdown clinical summary & table
        - Structured JSON findings string
    """
    if image is None:
        warning_msg = (
            "<div class='clinical-alert warning'>"
            "⚠️ <strong>Please upload an intraoral photo to begin analysis.</strong>"
            "</div>"
        )
        return (None, [], warning_msg, "{}")

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

        # Triage counters
        count_advanced = 0
        count_early = 0
        count_plaque = 0
        count_sound = 0
        count_gingivitis = 0
        count_swelling = 0

        clinical_findings = []

        for idx, f in enumerate(findings):
            crop = f.get("crop_image")
            saliency = f.get("gradcam_overlay")
            det_cls = f.get("detector_class", "unknown")
            raw_sev = f.get("classifier_severity", det_cls)
            cls_conf = f.get("classifier_confidence", f.get("detector_confidence", 1.0)) * 100

            # Reconcile detector vs classifier
            if det_cls == "caries":
                if raw_sev == "healthy":
                    taxon_key = "ruled_out"
                    count_sound += 1
                elif raw_sev == "caries_early":
                    taxon_key = "caries_early"
                    count_early += 1
                else:
                    taxon_key = "caries_advanced"
                    count_advanced += 1
            elif det_cls == "plaque":
                taxon_key = "plaque"
                count_plaque += 1
            elif det_cls == "gingivitis":
                taxon_key = "gingivitis"
                count_gingivitis += 1
            elif det_cls == "gum_swelling":
                taxon_key = "gum_swelling"
                count_swelling += 1
            else:
                taxon_key = "healthy"
                count_sound += 1

            meta = CLINICAL_TAXONOMY[taxon_key]

            display_title = f"Site #{idx + 1}: {meta['label']} ({cls_conf:.1f}%)"
            if crop is not None:
                gallery_items.append((crop, display_title))
            if saliency is not None:
                gallery_items.append((saliency, f"Grad-CAM #{idx + 1}: {meta['label']} Saliency"))

            clinical_findings.append({
                "index": idx + 1,
                "bbox": f.get("bbox"),
                "detector_class": det_cls,
                "meta": meta,
                "confidence": cls_conf,
                "probabilities": f.get("classifier_probabilities", {}),
            })

        # Overall Case Triage Assessment
        active_lesions = count_advanced + count_early + count_swelling + count_gingivitis

        if num_findings == 0:
            md_summary = (
                "<div class='triage-card success'>"
                "<div class='triage-header'>"
                "<span class='triage-badge badge-success'>🟢 Sound Dentition</span>"
                "<h3 style='margin: 0.5rem 0 0.2rem 0;'>No Suspicious Lesions Detected</h3>"
                "</div>"
                "<p>No active caries or periodontal pathology identified above threshold.</p>"
                "<div class='triage-footer'>"
                "<strong>Action:</strong> Routine 6-month preventive recall examination."
                "</div>"
                "</div>"
            )
        else:
            if count_advanced > 0 or count_swelling > 0:
                triage_class = "urgent"
                triage_badge = "🔴 RESTORATIVE / PERIODONTAL ATTENTION"
                badge_class = "badge-urgent"
                if count_advanced > 0 and count_swelling > 0:
                    primary_dx = "Cavitated Dental Caries & Periodontal Gum Swelling"
                    clinical_action = (
                        "Urgent dental & periodontal exam; radiographs and restorative triage."
                    )
                elif count_advanced > 0:
                    primary_dx = "Active Cavitated Dental Caries (Dentin Involvement)"
                    clinical_action = (
                        "Clinical exam, bitewing/periapical x-rays, and restorative treatment."
                    )
                else:
                    primary_dx = "Periodontal Gum Swelling (Edema / Infection Risk)"
                    clinical_action = (
                        "Periodontal probing, bone evaluation, and anti-inflammatory therapy."
                    )
            elif count_early > 0:
                triage_class = "warning"
                triage_badge = "🟡 PREVENTIVE INTERVENTION NEEDED"
                badge_class = "badge-warning"
                primary_dx = "Incipient Enamel Demineralization / Early Caries"
                clinical_action = (
                    "Topical fluoride varnish, remineralizing regimen, and 3-month follow-up."
                )
            elif count_gingivitis > 0:
                triage_class = "warning"
                triage_badge = "🟠 PERIODONTAL CARE INDICATED"
                badge_class = "badge-warning"
                primary_dx = "Gingivitis (Marginal Gum Inflammation)"
                clinical_action = (
                    "Scaling, root planning, chlorhexidine rinse, and oral hygiene instruction."
                )
            elif count_plaque > 0:
                triage_class = "info"
                triage_badge = "🟣 ORAL PROPHYLAXIS INDICATED"
                badge_class = "badge-info"
                primary_dx = "Dental Plaque Biofilm Accumulation"
                clinical_action = "Professional dental scaling and oral hygiene instruction."
            else:
                triage_class = "success"
                triage_badge = "🟢 SOUND TOOTH SURFACES"
                badge_class = "badge-success"
                primary_dx = "Sound Enamel (Candidate Ruled Out as Benign)"
                clinical_action = "Maintain standard oral hygiene and regular checkups."

            # Construct Clinical Diagnostic Card
            urgent_count = count_advanced + count_swelling
            early_count = count_early + count_gingivitis
            md_summary = f"""
<div class="triage-card {triage_class}">
    <div class="triage-header">
        <span class="triage-badge {badge_class}">{triage_badge}</span>
        <h2 style="margin: 0.4rem 0 0.2rem 0; font-size: 1.3rem;">{primary_dx}</h2>
    </div>
    <div class="triage-stats-grid">
        <div class="stat-box">
            <span class="stat-number">{num_findings}</span>
            <span class="stat-label">Sites Inspected</span>
        </div>
        <div class="stat-box">
            <span class="stat-number" style="color: #ef4444;">{urgent_count}</span>
            <span class="stat-label">Urgent (Cavities/Swelling)</span>
        </div>
        <div class="stat-box">
            <span class="stat-number" style="color: #f59e0b;">{early_count}</span>
            <span class="stat-label">Early / Gingivitis</span>
        </div>
        <div class="stat-box">
            <span class="stat-number" style="color: #10b981;">{count_sound}</span>
            <span class="stat-label">Sound / Ruled-Out</span>
        </div>
    </div>
    <div class="triage-footer">
        <strong>📋 Doctor Recommendation:</strong> {clinical_action}
    </div>
</div>

### 🦷 Detailed Diagnostic Breakdown by Region

| # | Localized Site | Clinical Verdict | Stage | Confidence | Recommendation |
| :-: | :--- | :--- | :--- | :-: | :--- |
"""
            for item in clinical_findings:
                m = item["meta"]
                idx = item["index"]
                conf = item["confidence"]
                det_label = item["detector_class"].capitalize()
                action_text = m["recommendation"]
                badge = m["badge"]
                label_text = m["label"]
                stage_text = m["severity"]

                md_summary += (
                    f"| **{idx}** | `{det_label}` | {badge} **{label_text}** | "
                    f"*{stage_text}* | **{conf:.1f}%** | {action_text} |\n"
                )

            # Footnote explaining reconciliation
            if count_sound > 0 and active_lesions > 0:
                md_summary += (
                    "\n> 💡 **Doctor Note on Reconciled Verdicts:** Where Stage 1 detector "
                    "flagged a candidate but Stage 2 confirmed sound enamel with high certainty, "
                    "the verdict is recorded as **Sound Surface (Ruled Out)**.\n"
                )

        clean_findings = []
        for f in findings:
            clean_f = {k: v for k, v in f.items() if k not in ("crop_image", "gradcam_overlay")}
            clean_findings.append(clean_f)

        json_out = json.dumps(
            {
                "image_shape": results.get("image_shape"),
                "detections_count": num_findings,
                "active_caries_count": active_lesions,
                "findings": clean_findings,
            },
            indent=2,
        )

        return vis_rgb, gallery_items, md_summary, json_out

    except Exception as exc:
        logger.exception("Error during pipeline execution: %s", exc)
        err_msg = (
            f"<div class='clinical-alert error'>"
            f"❌ <strong>Error during analysis:</strong> {exc}</div>"
        )
        return (None, [], err_msg, "{}")


# Available test examples
EXAMPLES = []
test_img_dir = REPO_ROOT / "data/processed/detector/test/images"
if test_img_dir.exists():
    sample_files = list(test_img_dir.glob("*.jpg"))[:5]
    for sf in sample_files:
        EXAMPLES.append([str(sf), 0.35, 0.45, True, False])

# Custom CSS for modern medical aesthetic (Light & Dark Mode Compatible)
CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

.header-container {
    background: linear-gradient(135deg, #0f172a 0%, #0369a1 50%, #0284c7 100%);
    color: white !important;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(2, 132, 199, 0.2);
}

.header-container h1 {
    color: white !important;
    font-weight: 700;
    margin-bottom: 0.3rem;
    font-size: 1.9rem;
}

.header-subtitle {
    color: #bae6fd !important;
    font-size: 1rem;
    margin: 0;
}

/* Preset Guidance Box - High Contrast in Light & Dark Mode */
.preset-guidance-box {
    background: #f0f9ff !important;
    border: 1px solid #bae6fd !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    margin-top: 0.5rem !important;
    font-size: 0.88rem !important;
}

.preset-guidance-box,
.preset-guidance-box p,
.preset-guidance-box strong,
.preset-guidance-box span {
    color: #0369a1 !important;
}

.dark .preset-guidance-box {
    background: rgba(12, 74, 110, 0.35) !important;
    border: 1px solid #0284c7 !important;
}

.dark .preset-guidance-box,
.dark .preset-guidance-box p,
.dark .preset-guidance-box strong,
.dark .preset-guidance-box span {
    color: #7dd3fc !important;
}

/* Triage Cards - Light Mode */
.triage-card {
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.triage-card.urgent {
    background: #fff5f5 !important;
    border: 1px solid #fecaca !important;
    border-left: 6px solid #ef4444 !important;
}

.triage-card.urgent h2,
.triage-card.urgent p,
.triage-card.urgent strong,
.triage-card.urgent .triage-footer {
    color: #7f1d1d !important;
}

.triage-card.warning {
    background: #fffbeb !important;
    border: 1px solid #fde68a !important;
    border-left: 6px solid #f59e0b !important;
}

.triage-card.warning h2,
.triage-card.warning p,
.triage-card.warning strong,
.triage-card.warning .triage-footer {
    color: #78350f !important;
}

.triage-card.info {
    background: #f5f3ff !important;
    border: 1px solid #ddd6fe !important;
    border-left: 6px solid #8b5cf6 !important;
}

.triage-card.info h2,
.triage-card.info p,
.triage-card.info strong,
.triage-card.info .triage-footer {
    color: #4c1d95 !important;
}

.triage-card.success {
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    border-left: 6px solid #10b981 !important;
}

.triage-card.success h2,
.triage-card.success p,
.triage-card.success strong,
.triage-card.success .triage-footer {
    color: #14532d !important;
}

/* Triage Cards - Dark Mode */
.dark .triage-card.urgent {
    background: rgba(153, 27, 27, 0.25) !important;
    border: 1px solid #991b1b !important;
    border-left: 6px solid #ef4444 !important;
}

.dark .triage-card.urgent h2,
.dark .triage-card.urgent p,
.dark .triage-card.urgent strong,
.dark .triage-card.urgent .triage-footer {
    color: #fecaca !important;
}

.dark .triage-card.warning {
    background: rgba(146, 64, 14, 0.25) !important;
    border: 1px solid #92400e !important;
    border-left: 6px solid #f59e0b !important;
}

.dark .triage-card.warning h2,
.dark .triage-card.warning p,
.dark .triage-card.warning strong,
.dark .triage-card.warning .triage-footer {
    color: #fde68a !important;
}

.dark .triage-card.info {
    background: rgba(91, 33, 182, 0.25) !important;
    border: 1px solid #5b21b6 !important;
    border-left: 6px solid #8b5cf6 !important;
}

.dark .triage-card.info h2,
.dark .triage-card.info p,
.dark .triage-card.info strong,
.dark .triage-card.info .triage-footer {
    color: #ddd6fe !important;
}

.dark .triage-card.success {
    background: rgba(22, 101, 52, 0.25) !important;
    border: 1px solid #166534 !important;
    border-left: 6px solid #10b981 !important;
}

.dark .triage-card.success h2,
.dark .triage-card.success p,
.dark .triage-card.success strong,
.dark .triage-card.success .triage-footer {
    color: #bbf7d0 !important;
}

.triage-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-urgent { background: #fee2e2 !important; color: #b91c1c !important; }
.badge-warning { background: #fef3c7 !important; color: #b45309 !important; }
.badge-info { background: #ede9fe !important; color: #6d28d9 !important; }
.badge-success { background: #dcfce7 !important; color: #15803d !important; }

.dark .badge-urgent { background: #7f1d1d !important; color: #fecaca !important; }
.dark .badge-warning { background: #78350f !important; color: #fef3c7 !important; }
.dark .badge-info { background: #4c1d95 !important; color: #ede9fe !important; }
.dark .badge-success { background: #14532d !important; color: #dcfce7 !important; }

.triage-stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}

.stat-box {
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    text-align: center !important;
}

.stat-number {
    display: block;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a !important;
}

.stat-label {
    display: block;
    font-size: 0.78rem;
    color: #64748b !important;
    text-transform: uppercase;
    font-weight: 600;
}

.dark .stat-box {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid #334155 !important;
}

.dark .stat-box .stat-number {
    color: #f8fafc !important;
}

.dark .stat-box .stat-label {
    color: #94a3b8 !important;
}

.triage-footer {
    border-top: 1px solid rgba(0, 0, 0, 0.08);
    padding-top: 0.75rem;
    margin-top: 0.75rem;
    font-size: 0.95rem;
}

.dark .triage-footer {
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

button.primary-btn {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
}
"""


CLINICAL_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)


def build_app() -> gr.Blocks:
    """Construct the Gradio UI Blocks interface with doctor-centric clinical design."""
    with gr.Blocks(title="🦷 Dental Model — Clinical AI Screening Demo") as demo:
        with gr.Column(elem_classes=["header-container"]):
            gr.HTML(
                """
                <div>
                    <h1>🦷 DentalAI Clinical Screening & Lesion Staging</h1>
                    <p class="header-subtitle">
                        Two-Stage Decision-Support: YOLOv8s Spatial Detector +
                        timm EfficientNet-B0 Staging Classifier + Grad-CAM Explainability
                    </p>
                </div>
                """
            )

        with gr.Row():
            # Left Panel: Clinical Inputs & Triage Controls
            with gr.Column(scale=4):
                gr.Markdown("### 📸 1. Input Intraoral Photograph")
                input_image = gr.Image(
                    label="Intraoral Photo (Clinical Capture)",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=290,
                )

                gr.Markdown("### 🩺 2. Clinical Operating Mode")
                preset_radio = gr.Radio(
                    choices=[
                        "Routine Screening (Recommended)",
                        "Early Prevention (High Sensitivity)",
                        "Definitive Diagnosis (High Specificity)",
                        "Custom Parameters",
                    ],
                    value="Routine Screening (Recommended)",
                    label="Doctor Presets",
                    info="Auto-configures detection thresholds for your workflow.",
                )

                preset_guidance = gr.Markdown(
                    value=PRESETS["Routine Screening (Recommended)"]["guidance"],
                    elem_classes=["preset-guidance-box"],
                )

                with gr.Accordion("⚙️ Fine-Tune Diagnostic Thresholds", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        value=0.35,
                        step=0.05,
                        label="Detector Sensitivity (Confidence Threshold)",
                        info="Higher = obvious lesions only. Lower = catch faint white-spots.",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.20,
                        maximum=0.80,
                        value=0.45,
                        step=0.05,
                        label="Overlap Suppression (IoU Threshold)",
                        info="Merges overlapping bounding boxes on the same tooth.",
                    )
                    with gr.Row():
                        gradcam_toggle = gr.Checkbox(
                            value=True,
                            label="Grad-CAM Saliency",
                            info="Compute attention heatmap overlays on lesions.",
                        )
                        classify_all_toggle = gr.Checkbox(
                            value=False,
                            label="Classify All Regions",
                            info="Run micro-staging on healthy/plaque crops.",
                        )

                analyze_btn = gr.Button(
                    "🔬 Execute Clinical Analysis",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"],
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

            # Right Panel: Clinical Findings & Visualizations
            with gr.Column(scale=6):
                gr.Markdown("### 📊 3. Clinical Diagnostic Findings")

                with gr.Tabs():
                    with gr.TabItem("📋 Diagnostic Impression & Table"):
                        output_summary = gr.Markdown(
                            value=(
                                "*Upload an intraoral photograph on the left "
                                "and click 'Execute Clinical Analysis'.*"
                            )
                        )

                    with gr.TabItem("🖼️ Multi-Panel Composite"):
                        output_composite = gr.Image(
                            label="Annotated Detections, Lesion Crops & Grad-CAM Heatmaps",
                            type="numpy",
                            interactive=False,
                        )

                    with gr.TabItem("🔎 Individual Lesion Crops"):
                        output_gallery = gr.Gallery(
                            label="High-Resolution Lesion Patches & Attention Saliency",
                            columns=2,
                            rows=2,
                            height=400,
                            object_fit="contain",
                        )

                    with gr.TabItem("📋 Structured EHR Contract (JSON)"):
                        output_json = gr.Code(
                            label="Serializable Clinical Findings (HL7 / FHIR Ready)",
                            language="json",
                            lines=12,
                        )

                    with gr.TabItem("ℹ️ Clinical AI Provenance"):
                        gr.Markdown(
                            """
                            ### System Clinical Provenance
                            - **Stage 1 (Lesion Detector):** YOLOv8s (`11.2M params`) trained on
                              intraoral photographs (`healthy`, `plaque`, `caries`).
                            - **Stage 2 (Lesion Stager):** `timm` EfficientNet-B0 trained on
                              Caries-Spectra dataset (Macro F1: **0.964**, Accuracy: **95.7%**).
                            - **Stage 3 (Explainability):** Grad-CAM heatmaps highlight exact
                              enamel/dentin demineralization gradients verifying model focus.

                            ### Diagnostic Disclaimer
                            > **Clinical Research Demonstration Only:** This system is an
                            > investigational decision-support prototype. It is **not a medical
                            > device**. All findings must be corroborated by a licensed dentist
                            > with physical examination and radiographic evaluation.
                            """
                        )

        # Reactive Preset Handler
        def on_preset_change(preset_name: str) -> tuple[float, float, str]:
            if preset_name in PRESETS:
                p = PRESETS[preset_name]
                return p["conf"], p["iou"], p["guidance"]
            return (
                0.35,
                0.45,
                "⚙️ **Custom Parameters:** Manual adjustment of detection and IoU thresholds.",
            )

        preset_radio.change(
            fn=on_preset_change,
            inputs=[preset_radio],
            outputs=[conf_slider, iou_slider, preset_guidance],
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
        theme=CLINICAL_THEME,
        css=CUSTOM_CSS,
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
