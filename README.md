# BharatDrishti

BharatDrishti is a retinal-image screening prototype. The static frontend remains in `frontend/`; the FastAPI inference service and reproducible research utilities live in `backend/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Serve `frontend/` with a static server. For local development, use `?api=http://127.0.0.1:8000`. For production, configure the API URL through `window.__BHARATDRISHTI_API_URL__` or the query parameter; production code does not require localhost.

## Models and inference

The default production model is the supplied five-class ResNet50 checkpoint. The proposed model is an optional ResNet50 + CBAM attention architecture and is not used unless a compatible trained checkpoint exists. Inference performs quality checking, preprocessing, prediction, confidence gating, and Grad-CAM.

## Dataset preparation

No datasets are downloaded or bundled. Review `backend/datasets/dataset_catalog.json`, then create a CSV with:

```text
image_path,label,patient_id
```

Generate leakage-aware manifests:

```powershell
python -m backend.datasets.prepare_manifest --csv data/labels.csv --output-dir data/manifests
```

Counts, class distributions, demographics, imaging conditions, and licensing details remain `not available` until verified from the actual source metadata.

## Training and evaluation

Training is optional and is not run automatically. Keep the final test set isolated:

```text
data/images/train/<class>/*.jpg
data/images/validation/<class>/*.jpg
data/images/test/<class>/*.jpg
```

```powershell
python -m backend.training.train --data-root data/images --output backend/bharatdrishti_resnet50_cbam_best.pth --variant proposed --epochs 10
python -m backend.evaluation.evaluate --data-dir data/images/test --checkpoint backend/bharatdrishti_resnet50_best.pth --output outputs/baseline_metrics.json --variant baseline
python -m backend.evaluation.evaluate --data-dir data/images/test --checkpoint backend/bharatdrishti_resnet50_cbam_best.pth --output outputs/proposed_metrics.json --variant proposed
python -m backend.evaluation.compare_preprocessing --baseline outputs/baseline_metrics.json --improved outputs/proposed_metrics.json --output outputs/comparison.json
python -m backend.evaluation.plot_metrics --baseline outputs/baseline_metrics.json --proposed outputs/proposed_metrics.json --output outputs/comparison.png
```

These commands generate real results only after valid datasets and checkpoints are supplied. See `RESEARCH_REPORT.md` for the report structure and limitations.

## Deployment

The FastAPI entry point is `backend.app:app`. The repository root contains a `Procfile` with the production command:

```text
web: uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

For Render, Railway, or another Procfile-compatible host, use that command as the start command. Set the backend environment variable `CORS_ORIGINS` to the deployed Vercel URL, for example:

```text
CORS_ORIGINS=https://your-project.vercel.app
```

Multiple frontend origins may be comma-separated. `VERCEL_FRONTEND_URL` is also accepted for a single frontend origin. The model path is resolved relative to `backend/config.py`, so startup does not depend on the platform's working directory. The frontend remains a separate Vercel static deployment.