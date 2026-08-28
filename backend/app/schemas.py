from datetime import datetime

from pydantic import BaseModel


class Issue(BaseModel):
    type: str
    severity: str
    confidence: float


class AnalysisResult(BaseModel):
    id: str
    filename: str
    quality_score: int
    quality_label: str
    issues: list[Issue]
    features: dict[str, float]
    anomaly_score: float | None = None
    image_url: str | None = None
    heatmap_url: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True  # allows AnalysisResult.model_validate(sqlalchemy_obj)


class AnalysisList(BaseModel):
    items: list[AnalysisResult]
    total: int


class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    db: str
