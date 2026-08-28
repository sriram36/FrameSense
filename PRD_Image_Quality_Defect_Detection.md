# PRD: AI-Powered Image Quality & Defect Detection

**Owner:** Applicant (internship technical assessment)
**Timebox:** 48 hours
**Status:** Draft for build

---

## 1. Purpose & Background

Build a full-stack application that accepts a user-uploaded image, analyzes it using a self-trained AI/ML pipeline (no external AI or vision APIs), and returns a structured quality assessment: an overall score, a quality label, and a list of detected issues with severity and confidence. Results are persisted and retrievable via a history view.

This document translates the assessment brief into concrete requirements, an architecture, a data model, an API contract, an ML design, and a build plan, so the project can be executed without re-deriving decisions mid-build.

## 2. Goals

- Demonstrate genuine CV/ML understanding: meaningful engineered features plus a trained decision component (not simple thresholding).
- Ship a working, containerized, end-to-end system: upload → analyze → persist → retrieve.
- Produce defensible evaluation results (metrics, confusion matrix, failure cases) on unseen data, from an actual training run, not a notebook that was never executed.
- Keep the system reproducible by a third party from the README alone.

## 3. Non-Goals

- No external AI/vision API calls, no API keys.
- No requirement for cloud deployment (local Docker Compose is sufficient).
- No requirement for a large curated defect dataset — synthetic degradation of clean images is acceptable and expected.
- Visual polish is secondary to functionality on the frontend.

## 4. Users & Use Case

Single-user technical evaluator: uploads one image at a time, reviews the returned assessment, and can browse a history of past analyses. No auth/multi-tenancy required.

## 5. Functional Requirements

### 5.1 Detection capabilities (required)
- Blur / insufficient sharpness
- Underexposure
- Overexposure
- Image noise
- Image corruption / severe degradation
- Potential visual defect (anomaly-style, distinct from the above degradations)

### 5.2 Backend
- `POST /analyze` — accepts an image file, validates it, runs the ML pipeline, persists the result, returns structured JSON.
- `GET /results/{id}` — returns a single stored analysis.
- `GET /results` — returns paginated history (list view).
- `GET /health` — liveness/readiness check (model loaded, DB reachable).
- Invalid/unreadable files return a 4xx with a clear error body, never a 500.
- All analysis results persisted in a relational DB.

### 5.3 Frontend
- Upload view: drag-and-drop or file picker, image preview, loading state during analysis.
- Results view: uploaded image, overall score, quality label, issue list with severity + confidence, key image statistics (sharpness, brightness, contrast, noise).
- History view: list/grid of past analyses, click through to a stored result.
- Explicit loading / success / error states throughout.
- Responsive layout (desktop + mobile width).

### 5.4 ML / AI
- Hybrid approach (see §7): engineered image-quality features → trained classifier for degradation-type issues, plus a lightweight autoencoder for anomaly-style defect detection.
- Explainability: raw feature values reported alongside each issue; autoencoder reconstruction-error heatmap for defect localization.

## 6. Non-Functional Requirements

- **Reproducibility:** `docker compose up` brings up frontend + backend + DB with no manual steps beyond documented env vars.
- **Performance:** single-image analysis returns in well under a few seconds on CPU.
- **Robustness:** corrupted/non-image uploads must not crash the service.
- **Config:** model paths, DB connection, CORS origin, and upload size limit are environment variables, not hardcoded.

## 7. System Architecture

Three containers behind Docker Compose:

- **Frontend** (React + Vite, served by nginx in production build) — calls the backend REST API.
- **Backend** (FastAPI) — owns validation, the ML pipeline, persistence, and the REST contract.
- **Database** (SQLite file on a mounted volume, or Postgres container) — stores analysis records.

Request flow: Browser → Frontend → `POST /analyze` on Backend → feature extraction → classifier + autoencoder inference → score/label computed → row written to DB → JSON returned → Frontend renders result. `GET /results` reads the same table for history.

## 8. Data Sourcing

Before any training can happen you need a small pool of clean, good-quality images plus a plan to generate degraded versions.

**Where to get clean source images (pick one or combine):**
- **Self-collected:** 100–200 phone photos across varied lighting/subjects. Zero licensing risk, fastest to start.
- **Kodak24** — 24 high-quality reference images released for unrestricted use; standard in compression/quality research. `huggingface.co/datasets/Freed-Wu/kodak` or `r0k.us/graphics/kodak`. Good as a clean baseline, too small alone for a full train set.
- **COCO val2017 subset** — diverse, free-to-use images: `images.cocodataset.org/zips/val2017.zip`. Pull a random few hundred rather than the full 5k.
- **Unsplash / Pexels** — free-license stock photos for additional variety.
- **Optional, if you want ready distorted+labeled pairs instead of only synthetic degradation:** TID2013, KADID-10k, or LIVE IQA — established image-quality-assessment datasets with pristine/distorted pairs and human quality scores. Heavier to integrate; only pursue if time allows.

**Recommendation:** self-collected photos or a COCO subset as the clean pool, degraded programmatically per §10. This is the fastest path to a labeled, reproducible dataset within the timebox.

## 9. Data Model

**`analyses` table**

| Field | Type | Notes |
|---|---|---|
| id | UUID / int PK | |
| filename | string | original filename |
| created_at | timestamp | |
| image_path | string | stored path or blob ref |
| quality_score | int (0–100) | |
| quality_label | enum | ACCEPTABLE / DEGRADED / DEFECTIVE |
| issues | JSON | list of `{type, severity, confidence}` |
| features | JSON | raw stats: sharpness, brightness, contrast, noise, saturation |
| anomaly_score | float | autoencoder reconstruction error |
| heatmap_path | string, nullable | saved localization map, if generated |

## 10. API Contract

**`POST /analyze`** (multipart/form-data, field `file`)
```json
{
  "id": "a1b2c3",
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [
    {"type": "noise", "severity": "low", "confidence": 0.71}
  ],
  "features": {
    "sharpness": 145.2,
    "brightness": 0.53,
    "contrast": 0.31,
    "noise_estimate": 0.04
  },
  "created_at": "2026-08-27T10:00:00Z"
}
```
Errors: `400` invalid/unreadable file, `413` file too large, `500` only for genuine unhandled faults (should be rare and logged).

**`GET /results/{id}`** → same shape as above.
**`GET /results?limit=&offset=`** → `{ "items": [...], "total": n }`.
**`GET /health`** → `{ "status": "ok", "model_loaded": true, "db": "ok" }`.

## 11. ML Pipeline Design

**Features (engineered, per image):**
Sharpness (Laplacian variance), brightness (mean/histogram of luminance), contrast (std dev of luminance), noise estimate (high-frequency residual after denoising), saturation/colorfulness.

**Training data generation:**
From the clean pool in §8, programmatically produce labeled examples: Gaussian blur (varying sigma) for blur, gamma/exposure shift for over/underexposure, additive Gaussian/salt-and-pepper noise for noise, JPEG re-compression/block corruption for corruption. Hold out an unseen split of both clean and degraded images for evaluation — this split must not touch training at all.

**Models:**
1. Classifier (RandomForest or GradientBoosting) on engineered features → predicts issue type + severity for blur/exposure/noise/corruption.
2. Convolutional autoencoder (PyTorch), trained only on clean/acceptable images → reconstruction error flags "potential visual defect" as a genuine anomaly-detection formulation, independent of the labeled degradation classes.

**This step must actually be executed, not just scripted.** Run the training, save the resulting model artifacts to disk, and load them at API startup rather than retraining per request.

**Score composition:** start at 100, subtract per-issue penalty weighted by severity × confidence; autoencoder anomaly score above threshold force-sets `quality_label = DEFECTIVE` regardless of the computed score.

**Explainability:** report raw feature values with every response; save the autoencoder's per-pixel reconstruction-error map as an optional heatmap image for defect localization (bonus item).

## 12. Evaluation Plan (must be run, results must be real)

- Classifier: accuracy, precision/recall/F1 per issue type, confusion matrix, all on the held-out synthetic-degradation split.
- Autoencoder: reconstruction-error distribution on clean vs. degraded/corrupted hold-out, ROC-AUC for anomaly detection, chosen threshold justified against that curve.
- Write up 3–5 concrete failure cases (false positive and false negative) using actual misclassified images from your hold-out set, with the feature values that likely caused the miss.
- Save all of this as `EVALUATION.md` — plots, numbers, and the failure-case writeup, not just code that could theoretically produce them.

## 13. Documentation Requirements

- **README.md**: one-command setup (`docker compose up`), how models are trained and loaded at startup, example API requests (curl or httpie), DB setup/schema, and how to reproduce the evaluation — written so a stranger can follow it with no other context.
- **EVALUATION.md**: metrics, confusion matrix, ROC-AUC, and the failure-case discussion from §12.
- **API docs**: FastAPI's auto-generated `/docs` satisfies the "API documentation" requirement — confirm it renders correctly and link to it from the README.
- **.env.example**: every configurable variable (API base URL, upload size limit, model paths, CORS origin, DB connection) with sane defaults.

## 14. Testing & Verification Requirements

Code existing is not the same as code working — verify before considering any phase complete:
- Manually run `/analyze` against a real image, a corrupted/truncated file, and a non-image file (e.g. a `.txt` renamed to `.jpg`); confirm 4xx responses for bad input, never a 500 or crash.
- Manually walk through the frontend end to end in a browser: upload → result rendering → history list → click into a past result.
- Bring the entire stack up via a clean `docker compose up` (no leftover local state) and confirm frontend and backend talk to each other correctly through the configured API URL.
- If time allows: add a few automated backend tests (e.g. pytest hitting `/analyze` and `/health`) as bonus credit toward code quality.

## 15. Sample Images for Submission

Separate from your training data: curate a small `samples/` folder with a handful of images that clearly demonstrate each condition — one acceptable, one blurry, one over-exposed, one under-exposed, one noisy, one corrupted — for the reviewer to test against directly.

## 16. Deployment & Packaging Requirements

- `docker-compose.yml` bringing up frontend, backend, and DB (or DB volume) together.
- `.env.example` documenting every configurable value (see §13).
- `/health` endpoint exposed and used as the compose healthcheck.
- Frontend calls the backend via an environment variable, not a hardcoded localhost path, so it survives containerization.
- If deployed online, record the URL in the README; otherwise state explicitly that it's local-only (acceptable per the brief).

## 17. Submission Checklist

- [ ] Full source (frontend, backend, ML) with clear structure
- [ ] Clean image pool sourced (§8) and synthetic degradation pipeline actually run
- [ ] Classifier + autoencoder actually trained; model artifacts saved to disk
- [ ] `EVALUATION.md`: real metrics, confusion matrix, ROC-AUC, 3–5 failure cases
- [ ] `README.md`: setup, training/loading, API examples, DB setup, deployment
- [ ] `.env.example` covering every configurable variable
- [ ] `samples/` folder covering each quality condition
- [ ] Manual end-to-end browser test completed (upload → result → history)
- [ ] Bad-input handling verified (corrupted file, non-image file → 4xx not 500)
- [ ] Clean `docker compose up` verified from a fresh checkout
- [ ] API docs reachable (FastAPI `/docs`) and linked from README
- [ ] Deployed URL noted, or explicitly stated as local-only

## 18. Milestones (48-hour plan)

| Hours | Work |
|---|---|
| 0–6 | Source clean images (§8), build synthetic degradation pipeline, generate labeled dataset |
| 6–16 | Train classifier + autoencoder, run evaluation, write `EVALUATION.md` with failure cases |
| 16–28 | Backend API: analyze, results, health, validation, persistence |
| 28–38 | Frontend: upload, results, history, loading/error states |
| 38–44 | Docker Compose, `.env.example`, `samples/` folder, manual end-to-end + bad-input testing |
| 44–48 | README finalization, clean `docker compose up` check, buffer |

## 19. Risks & Assumptions

- **Risk:** synthetic degradations may not generalize to real defective images → mitigated by evaluating on a small manually-collected set in addition to synthetic hold-out.
- **Risk:** autoencoder threshold tuning is subjective → mitigated by picking the threshold from the ROC curve and stating the trade-off explicitly.
- **Assumption:** CPU-only training is acceptable given the lightweight model choices.

## 20. Optional / Bonus (time-permitting)

Batch analysis endpoint, quality heatmaps (already partially covered by the autoencoder), confidence calibration, model versioning, automated tests, concurrent-request handling, CI/CD, basic logging/monitoring.

## Appendix: Assessment Scoring Weights (for self-check before submission)

| Area | Weight |
|---|---|
| Computer vision understanding & feature reasoning | 15% |
| AI/ML/DL implementation | 25% |
| Model evaluation & experimental rigor | 15% |
| Backend/API implementation | 15% |
| Frontend functionality & usability | 10% |
| Deployment & reproducibility | 10% |
| Code quality & documentation | 10% |
