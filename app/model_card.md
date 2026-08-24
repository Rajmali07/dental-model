---
license: cc-by-4.0
tags:
  - image-classification
  - object-detection
  - dental
  - medical-imaging
  - gradio
datasets:
  - Mendeley-3253gj88rr
  - Roboflow-dental-dataset
model-index:
  - name: dental-model-detector-yolov8s
    results:
      - task:
          type: object-detection
        metrics:
          - type: mAP50
            value: 0.4692
          - type: mAP50-95
            value: 0.2619
          - type: precision
            value: 0.5352
          - type: recall
            value: 0.5475
  - name: dental-model-classifier-efficientnet-b0
    results:
      - task:
          type: image-classification
        metrics:
          - type: accuracy
            value: 0.9567
          - type: macro-f1
            value: 0.9640
---

# 🦷 Dental Disease Detection & Severity Staging System

## Intended Use
Research and portfolio demonstration of a two-stage modular computer vision pipeline (`detector` $\rightarrow$ `crop` $\rightarrow$ `classifier` $\rightarrow$ `Grad-CAM explainability`) for identifying and staging healthy teeth, dental plaque, and enamel caries lesions from intraoral clinical photographs.

## Clinical & Regulatory Disclaimers
- **Not a Diagnostic Device:** This model has not been evaluated, cleared, or approved by the FDA, CE, or any international medical regulatory agency.
- **Not a Substitute for Professional Examination:** It should never be used as a primary diagnostic tool or as a replacement for clinical evaluation, radiography, and consultation with a licensed dentist.

## Training Data Provenance
1. **Stage 1 Detector (YOLOv8s):** Trained on Roboflow Dental Universe dataset (6,188 annotated images; 4,881 train / 692 val / 615 test) remapped to hard-tissue classes (`healthy: 0`, `plaque: 1`, `caries: 2`).
2. **Stage 2 Classifier (EfficientNet-B0):** Trained on Caries-Spectra dataset ([Mendeley 3253gj88rr v1](https://data.mendeley.com/datasets/3253gj88rr/1)) across 2,000 images (stratified 70/15/15 train/val/test) for lesion severity staging:
   - `healthy` (sound enamel)
   - `caries_early` (initial enamel demineralization)
   - `caries_advanced` (cavitated enamel decay)

## Model Benchmarks & Performance
- **Detector (YOLOv8s, 100 Epochs):**
  - Overall mAP@50: **0.4692** (Peak: 0.476)
  - Overall mAP@50-95: **0.2619** (Peak: 0.267)
  - Caries Localization mAP@50: **0.384**
  - Healthy Tooth mAP@50: **0.812**
- **Classifier (EfficientNet-B0, Class-Weighted AdamW):**
  - Test Accuracy: **95.67%** (held-out 300 test images)
  - Macro F1 Score: **0.9640**
- **Explainability:** Grad-CAM attention heatmaps concentrate on tooth crown structures and occlusal fissures ($1.7\times - 4.35\times$ center-to-border saliency ratio).

## Known Limitations
- Plaque localization exhibits modest mAP@50 (0.212) due to diffuse visual plaque boundaries and inconsistent clinical disclosing agent staining.
- Performance on low-resolution mobile phone captures or off-axis angles may vary compared to direct clinical macro captures.
