---
license: TBD
tags:
  - image-classification
  - object-detection
  - dental
  - medical-imaging
datasets:
  - custom (see Training Data below)
model-index:
  - name: dental-model-detector
    results:
      - task:
          type: object-detection
        metrics:
          - type: mAP50-95
            value: TBD
  - name: dental-model-classifier
    results:
      - task:
          type: image-classification
        metrics:
          - type: f1
            value: TBD
---

# Dental Model — Detector + Classifier

## Intended use
Research/portfolio demonstration of a two-stage pipeline (detector → crop → classifier) for
identifying healthy / plaque / caries regions in intraoral photographs.

## Out-of-scope use
- **Not a diagnostic device.** Not validated for clinical use, not a substitute for examination
  by a licensed dentist.
- Not tested on intraoral camera hardware beyond what appears in the training data.

## Training data
- [Mendeley 9jnf2jvghy v2](https://data.mendeley.com/datasets/9jnf2jvghy/2) — deferred, not used in v1
- [Mendeley g8yhdvgjy2 v3](https://data.mendeley.com/datasets/g8yhdvgjy2/3) — optional, TBD after EDA
- [Mendeley 3253gj88rr v1](https://data.mendeley.com/datasets/3253gj88rr/1) — used (classifier)
- [Roboflow dental-dataset-more-healthy-and-caries-plaque](https://universe.roboflow.com/dental-data2/dental-dataset-more-healthy-and-caries-plaque) — used (detector)

Confirm each source's license before redistributing derived weights.

## Known limitations
- Class imbalance: few "advanced caries" examples in the source data.
- Sensitivity to image quality/lighting; source images vary across datasets.

## Evaluation
Fill in after training: mAP50-95 (detector), per-class F1 + confusion matrix (classifier).
