# Image Quality & Defect Detector

Upload an image, get back a quality score (0-100), a label (ACCEPTABLE /
DEGRADED / DEFECTIVE), and a list of detected issues (blur, over/under
exposure, noise, corruption, potential visual defect) with severity and
confidence -- all from a self-trained CV/ML pipeline, no external AI APIs.

See `PRD_Image_Quality_Defect_Detection.md` for the full spec and
`EVALUATION.md` for real evaluation results from a training run.

## Quick start (Docker, recommended)

```bash
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend API: http://localhost:8000 (docs at http://localhost:8000/docs)

That's it for running the app. `backend/app/ml/models/` ships with a trained
classifier and PCA anomaly detector (see `EVALUATION.md` for real results) --
both load automatically at startup, and neither requires PyTorch. **They
were trained on synthetic placeholder images, not real photos -- retrain
before treating this as a finished submission** (see "Training on real
images" below).

## Project structure

```
backend/
  app/
    ml/               feature extraction, synthetic degradation, classifier + anomaly detection
      pca_anomaly.py       PCA reconstruction-error anomaly detector (runs anywhere, no torch needed)
      autoencoder.py       PyTorch conv autoencoder (optional, higher-capacity upgrade)
      models/         trained model artifacts (classifier.joblib, pca_anomaly.joblib, autoencoder.pt)
    db/               SQLAlchemy models + session
    schemas.py        Pydantic request/response models
    main.py           FastAPI app and routes
  train/
    train_classifier.py    trains the RandomForest issue-type/severity classifier
    train_pca_anomaly.py   trains the PCA anomaly detector (default, no torch needed)
    train_autoencoder.py   trains the optional PyTorch conv autoencoder
  tests/
    test_api.py       basic API tests (pytest + FastAPI TestClient)
  requirements.txt
  Dockerfile
frontend/
  src/                React app (upload, results, history views)
  Dockerfile
  nginx.conf          serves the built app, proxies /api to the backend
samples/              example images covering each quality condition
eval_artifacts/       reconstruction-error heatmaps from the PCA anomaly detector
generate_samples.py   regenerates samples/ from the degradation code
docker-compose.yml
EVALUATION.md
```

## Data sourcing

The classifier needs a pool of clean, good-quality images to learn from (it
then generates its own labeled degraded examples -- see "How training works"
below). Options, fastest first:

- **Your own photos** -- 100-300 phone photos, varied lighting/subjects. No
  licensing concerns, most realistic option.
- **Kodak24** -- 24 high-quality reference images, unrestricted use:
  `huggingface.co/datasets/Freed-Wu/kodak` or `r0k.us/graphics/kodak`.
- **COCO val2017 subset** -- diverse, free to use:
  `images.cocodataset.org/zips/val2017.zip` (grab a random few hundred, not
  the whole 5k).
- **Unsplash / Pexels** -- free-license stock photos for extra variety.

## How training works

`train/train_classifier.py`, `train/train_pca_anomaly.py`, and
`train/train_autoencoder.py` all:

1. Load clean images from `--images-dir` (or fall back to generated synthetic
   placeholder images if no directory is given -- useful for a quick sanity
   check, not for a real submission).
2. Programmatically generate labeled degraded versions using
   `app/ml/synth_degrade.py` (blur, over/underexposure, noise, corruption at
   three severities each).
3. Hold out an unseen split for evaluation.
4. Train and save the model to `app/ml/models/`.

```bash
cd backend
pip install -r requirements.txt
python train/train_classifier.py --images-dir /path/to/your/clean/images
python train/train_pca_anomaly.py --images-dir /path/to/your/clean/images
```

`train_pca_anomaly.py` needs only scikit-learn (already in
`requirements.txt`) and is what the API uses for defect detection by
default. If you also want the higher-capacity PyTorch conv autoencoder:

```bash
pip install torch  # if not already installed
python train/train_autoencoder.py --images-dir /path/to/your/clean/images
```

`app/ml/inference.py` defaults to the PCA detector (ROC-AUC 0.895) because 
evaluation showed that the higher-capacity PyTorch autoencoder (ROC-AUC 0.785)
is *too good* at reconstructing corrupted blocks and noise, causing it to miss 
actual defects. See `EVALUATION.md` for the full breakdown of reconstruction 
error asymmetry.

All three scripts print real evaluation output (classification report,
confusion matrix, ROC-AUC) to the console -- copy results into
`EVALUATION.md` after re-running on real photos.

The FastAPI app loads whatever's in `app/ml/models/` at startup, so re-run
training and restart the API (or rebuild the Docker image) to pick up a new
model.

## API

Interactive docs at `/docs` (Swagger UI) once the backend is running.

**`POST /analyze`** -- multipart form upload, field name `file`.

```bash
curl -F "file=@samples/blurry.jpg" http://localhost:8000/analyze
```

```json
{
  "id": "a1b2c3...",
  "quality_score": 87,
  "quality_label": "ACCEPTABLE",
  "issues": [{"type": "blur", "severity": "medium", "confidence": 0.65}],
  "features": {"sharpness": 12.3, "brightness": 0.49, "contrast": 0.11, "noise_estimate": 0.02, "saturation": 0.31, "blockiness": 1.4, "edge_density": 0.08},
  "anomaly_score": null,
  "image_url": "/uploads/a1b2c3....jpg",
  "created_at": "2026-08-28T10:00:00Z"
}
```

**`GET /results/{id}`** -- fetch one stored analysis.
**`GET /results?limit=20&offset=0`** -- paginated history.
**`GET /health`** -- `{"status": "ok", "model_loaded": true, "db": "ok"}`.

Invalid or unreadable uploads return `400`, oversized uploads return `413`;
the API never returns `500` for bad input.

## Database

SQLite by default (`DATABASE_URL` in `.env`, see `backend/.env.example`),
stored in a Docker volume (`backend_data`) so results survive container
restarts. Swap `DATABASE_URL` for a Postgres connection string if you'd
rather run Postgres -- SQLAlchemy handles both without code changes.

Schema (`backend/app/db/models.py`): id, filename, created_at, image_path,
quality_score, quality_label, issues (JSON), features (JSON), anomaly_score,
heatmap_path.

## Sample images

`samples/` contains one example per condition (acceptable, blurry,
overexposed, underexposed, noisy, corrupted), generated by
`generate_samples.py` using the same degradation code as training --
useful for manually testing `/analyze` or the upload UI. Regenerate with:

```bash
python generate_samples.py
```

`eval_artifacts/heatmaps/` contains reconstruction-error heatmaps from the
PCA anomaly detector run on these sample images -- visual proof of the
defect-localization bonus item. See `EVALUATION.md` for the full writeup.

## Testing

```bash
cd backend
pytest
```

Covers `/health`, a valid upload, a non-image upload (expects 400), an empty
file (expects 400), fetching a result by id, a 404 for a missing id, and
pagination on `/results`.

Manual end-to-end checks worth doing before submission:
- Walk through upload -> result -> history in a browser.
- Upload a corrupted/renamed non-image file through the UI and confirm a
  clean error message, not a crash.
- `docker compose down -v && docker compose up --build` from a fully clean
  state to confirm nothing depends on leftover local state.

## Environment variables

See `backend/.env.example` and `frontend/.env.example`. In Docker Compose,
the frontend never needs a backend URL baked in -- nginx proxies `/api/*` to
the `backend` service by Docker network name, so the same built image works
regardless of where it's deployed.

## Deployment

Local Docker Compose is sufficient per the assessment brief. If you deploy
this online, note the URL here. Otherwise: **local-only deployment via
`docker compose up`, as permitted by the brief.**
