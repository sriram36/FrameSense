import os
import uuid

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .db.models import Analysis, get_db, init_db
from .ml.inference import pipeline
from .schemas import AnalysisList, AnalysisResult, HealthStatus

app = FastAPI(title="Image Quality & Defect Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins (localhost and Vercel)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "data/uploads")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024


@app.on_event("startup")
def on_startup():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    init_db()
    app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


def serialize(record: Analysis) -> AnalysisResult:
    result = AnalysisResult.model_validate(record)
    result.image_url = f"/uploads/{os.path.basename(record.image_path)}"
    return result


@app.get("/health", response_model=HealthStatus)
def health(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.execute("SELECT 1")
    except Exception:
        db_ok = False
    return HealthStatus(status="ok" if db_ok else "degraded", model_loaded=pipeline.is_ready, db="ok" if db_ok else "unreachable")


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...), db: Session = Depends(get_db)):
    raw = await file.read()

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file exceeds maximum upload size")

    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="file is not a readable image")

    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="model not loaded -- run train_classifier.py first")

    result = pipeline.analyze(bgr)

    record_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    image_path = os.path.join(UPLOAD_DIR, f"{record_id}{ext}")
    with open(image_path, "wb") as f:
        f.write(raw)

    record = Analysis(
        id=record_id,
        filename=file.filename or "upload",
        image_path=image_path,
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        issues=result["issues"],
        features=result["features"],
        anomaly_score=result["anomaly_score"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return serialize(record)


@app.get("/results/{result_id}", response_model=AnalysisResult)
def get_result(result_id: str, db: Session = Depends(get_db)):
    record = db.get(Analysis, result_id)
    if record is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return serialize(record)


@app.get("/results", response_model=AnalysisList)
def list_results(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    total = db.query(Analysis).count()
    items = (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return AnalysisList(items=[serialize(i) for i in items], total=total)
