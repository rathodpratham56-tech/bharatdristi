# BharatDrishti

BharatDrishti is a retinal-image screening prototype. The existing frontend remains a static site in `frontend/`; the FastAPI inference service and research code live in `backend/`.

## Verified project status

- **Baseline model:** the supplied five-class ResNet50 checkpoint, `backend/bharatdrishti_resnet50_best.pth`.
- **Model:** the supplied five-class ResNet50 checkpoint is retained; no retraining is performed by this project update.
- **XAI:** Grad-CAM generated from the actual selected model prediction. It is an attribution visualization, not proof of diagnosis.
- **Dataset documentation placeholders:** source: not available; number of images: not available; classes: not available from dataset metadata; class distribution: not available. No dataset is downloaded or bundled.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the backend from the repository root:

```powershell
uvicorn backend.app:app --reload --port 8000
```

The static frontend can be served from `frontend/` with any static web server. For local API use, open it with `?api=http://127.0.0.1:8000`, or define `window.__BHARATDRISHTI_API_URL__` before the controller script. The production frontend does not hardcode localhost; it defaults to its own origin and supports a deployed API URL at runtime.

Quality thresholds are configurable with `MIN_BRIGHTNESS`, `MAX_BRIGHTNESS`, `MIN_BLUR_SCORE`, and `MIN_RETINAL_AREA`.

## Deployment

Keep Vercel's Root Directory set to `frontend`; it is a static HTML/CSS/JavaScript site with no build command. Deploy the FastAPI backend separately, then configure the frontend runtime API URL with `window.__BHARATDRISHTI_API_URL__` or the `?api=` query parameter. Configure backend `CORS_ORIGINS` to the deployed frontend origin rather than leaving unrestricted CORS in a production deployment.

The system is intended for screening/research assistance and does not replace professional ophthalmological diagnosis.