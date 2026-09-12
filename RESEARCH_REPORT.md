# BharatDrishti Research Report Scaffold

This report intentionally contains no fabricated dataset counts, demographic claims, accuracy values, or clinical claims.

## Problem
BharatDrishti is an AI-assisted retinal image screening prototype. Its output is for screening/research assistance and does not replace ophthalmological diagnosis.

## Dataset documentation
Candidate sources and access notes are recorded in `backend/datasets/dataset_catalog.json`. No images are bundled or downloaded. Image counts, class distributions, demographics, devices, and licensing details remain `not available` until verified from source metadata.

## Split methodology
`backend/datasets/prepare_manifest.py` accepts `image_path,label,patient_id`. When patient IDs are present, all images from one patient are assigned to one partition. Without patient IDs, it records the limitation and performs an image-level split that may retain leakage risk.

## Preprocessing
The improved pipeline performs quality checks, conservative retinal-region cropping, LAB-space CLAHE, resizing, and ImageNet normalization. The baseline evaluator uses basic resize and normalization. The pipeline returns a visualization image for original-versus-processed review.

## Models
- Baseline: supplied ResNet50 checkpoint `backend/bharatdrishti_resnet50_best.pth`.
- Proposed: ResNet50 with CBAM channel/spatial attention, LayerNorm, dropout, and a classification head. This is a proposed project architecture, not a verified novelty claim, and requires a locally trained compatible checkpoint.

## Evaluation
`backend/evaluation/evaluate.py` generates real accuracy, precision, recall/sensitivity, specificity, macro F1, balanced accuracy, ROC-AUC when applicable, confusion matrix, and per-class metrics. `compare_preprocessing.py` and `plot_metrics.py` consume only generated JSON outputs. No results are available until real data and checkpoints are supplied.

## Explainability and workflow
The API performs image quality assessment, preprocessing, model prediction, confidence gating, and Grad-CAM. The frontend displays original/preprocessed images, prediction, confidence, quality details, risk-oriented interpretation, heatmap, and disclaimer. Grad-CAM indicates model attribution and does not prove diagnosis or lesion localization.

## Limitations and ethics
Dataset coverage, patient metadata, calibration, external validation, subgroup performance, and clinical utility are not available from this repository. These must be measured before claims about generalization, fairness, or deployment readiness. Handle patient data under applicable consent, privacy, and dataset-license requirements.
