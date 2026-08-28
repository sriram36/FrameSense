"""
Basic backend tests. Not executed in the assessment sandbox (fastapi
isn't installed there), but ready to run once dependencies are installed:

    pip install -r requirements.txt
    pytest

Uses a temporary SQLite DB so tests never touch app/data/.
"""

import io
import os

os.environ["DATABASE_URL"] = "sqlite:///./data/test_analyses.db"

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _fake_jpeg_bytes(size=(64, 64), color=(120, 130, 110)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("ok", "degraded")
    assert "model_loaded" in body


def test_analyze_valid_image():
    files = {"file": ("test.jpg", _fake_jpeg_bytes(), "image/jpeg")}
    res = client.post("/analyze", files=files)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["quality_score"] <= 100
    assert body["quality_label"] in ("ACCEPTABLE", "DEGRADED", "DEFECTIVE")
    assert isinstance(body["issues"], list)
    assert "sharpness" in body["features"]


def test_analyze_rejects_non_image():
    files = {"file": ("not_an_image.txt", b"hello world", "text/plain")}
    res = client.post("/analyze", files=files)
    assert res.status_code == 400


def test_analyze_rejects_empty_file():
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    res = client.post("/analyze", files=files)
    assert res.status_code == 400


def test_get_result_not_found():
    res = client.get("/results/does-not-exist")
    assert res.status_code == 404


def test_analyze_then_fetch_result():
    files = {"file": ("test2.jpg", _fake_jpeg_bytes(), "image/jpeg")}
    created = client.post("/analyze", files=files).json()
    res = client.get(f"/results/{created['id']}")
    assert res.status_code == 200
    assert res.json()["id"] == created["id"]


def test_list_results_pagination():
    for i in range(3):
        client.post("/analyze", files={"file": (f"t{i}.jpg", _fake_jpeg_bytes(), "image/jpeg")})
    res = client.get("/results?limit=2&offset=0")
    assert res.status_code == 200
    body = res.json()
    assert len(body["items"]) <= 2
    assert body["total"] >= 3
