import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./data/analyses.db")

# check_same_thread only matters for SQLite; harmless to set unconditionally
# only when the URL is actually sqlite.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    image_path = Column(String, nullable=False)
    quality_score = Column(Integer, nullable=False)
    quality_label = Column(String, nullable=False)  # ACCEPTABLE / DEGRADED / DEFECTIVE
    issues = Column(JSON, nullable=False)            # list[{type, severity, confidence}]
    features = Column(JSON, nullable=False)          # raw feature dict
    anomaly_score = Column(Float, nullable=True)
    heatmap_path = Column(String, nullable=True)


def init_db():
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
