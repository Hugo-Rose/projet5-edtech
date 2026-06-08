"""Tests API FastAPI — sans DB ni modèle réel (mocks)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_db, get_model
from src.api.main import app

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_db():
    """Session DB factice."""
    db = MagicMock()
    yield db


def _mock_model_ready():
    """ModelBundle prêt avec un XGBoost factice."""
    import numpy as np
    bundle = MagicMock()
    bundle.ready = True
    bundle.threshold = 0.5
    bundle.version   = "test-v1"
    bundle.preprocessor.transform.return_value = np.zeros((1, 23))
    bundle.model.predict_proba.return_value     = np.array([[0.3, 0.7]])
    return bundle


def _mock_model_not_ready():
    bundle = MagicMock()
    bundle.ready = False
    return bundle


@pytest.fixture
def client_no_db():
    app.dependency_overrides[get_db]    = _mock_db
    app.dependency_overrides[get_model] = _mock_model_not_ready
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_with_model():
    app.dependency_overrides[get_db]    = _mock_db
    app.dependency_overrides[get_model] = _mock_model_ready
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Tests meta ────────────────────────────────────────────────────────────────

def test_root(client_no_db):
    r = client_no_db.get("/")
    assert r.status_code == 200
    assert "EdTech" in r.json()["message"]


def test_health_structure(client_no_db):
    with patch("src.data.db.check_connection", return_value=False):
        r = client_no_db.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status"  in data
    assert "db"      in data
    assert "model"   in data
    assert "version" in data


def test_docs_available(client_no_db):
    r = client_no_db.get("/docs")
    assert r.status_code == 200


# ── Tests /students ───────────────────────────────────────────────────────────

def test_list_students_structure(client_no_db):
    db = MagicMock()
    db.execute.return_value.mappings.return_value.all.return_value = []
    db.execute.return_value.scalar.return_value = 0

    with patch("src.api.crud.get_students", return_value=(0, [])):
        r = client_no_db.get("/students")
    assert r.status_code == 200
    data = r.json()
    assert "total"    in data
    assert "students" in data
    assert isinstance(data["students"], list)


def test_list_students_pagination_params(client_no_db):
    with patch("src.api.crud.get_students", return_value=(0, [])):
        r = client_no_db.get("/students?limit=10&offset=20")
    assert r.status_code == 200
    data = r.json()
    assert data["limit"]  == 10
    assert data["offset"] == 20


def test_get_student_not_found(client_no_db):
    with patch("src.api.crud.get_student_by_id", return_value=None):
        r = client_no_db.get("/students/99999")
    assert r.status_code == 404


def test_get_student_found(client_no_db):
    fake_student = {
        "student_id": 1, "cohort_id": 1,
        "first_name": "Alice", "last_name": "Dupont",
        "email": "alice@test.fr", "age": 22,
        "status": "enrolled", "enrollment_date": "2024-09-02",
        "has_job": False, "scholarship": True,
        "distance_km": 10.0, "entry_grade": 14.5, "dropout_date": None,
        "dropout_prob": None, "risk_label": None,
        "predicted_at": None, "model_version": None, "prediction_id": None,
    }
    with patch("src.api.crud.get_student_by_id", return_value=fake_student), \
         patch("src.api.crud.get_student_weekly_trend", return_value=[]):
        r = client_no_db.get("/students/1")
    assert r.status_code == 200
    assert r.json()["email"] == "alice@test.fr"


def test_predict_student_model_not_ready(client_no_db):
    r = client_no_db.post("/students/1/predict")
    assert r.status_code == 503


# ── Tests /cohorts ────────────────────────────────────────────────────────────

def test_list_cohorts(client_no_db):
    fake_cohorts = [{"cohort_id": 1, "name": "Promo 2024",
                     "program": "DS", "start_date": "2024-09-02",
                     "end_date": "2026-06-30", "capacity": 35}]
    with patch("src.api.crud.get_cohorts", return_value=fake_cohorts):
        r = client_no_db.get("/cohorts")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_cohort_stats_not_found(client_no_db):
    with patch("src.api.crud.get_cohort_stats", return_value=None):
        r = client_no_db.get("/cohorts/999/stats")
    assert r.status_code == 404


def test_cohort_stats_found(client_no_db):
    fake = {
        "cohort_id": 1, "name": "Promo 2024", "program": "DS",
        "start_date": "2024-09-02", "end_date": "2026-06-30", "capacity": 35,
        "total_students": 30, "active_students": 25,
        "dropout_count": 5, "dropout_rate": 0.1667,
        "avg_login_week": 5.2, "avg_score": 68.4,
    }
    with patch("src.api.crud.get_cohort_stats", return_value=fake), \
         patch("src.api.crud.get_cohort_risk_distribution",
               return_value={"low": 10, "medium": 8, "high": 5, "critical": 2}):
        r = client_no_db.get("/cohorts/1/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["dropout_count"] == 5
    assert "risk_distribution" in data


# ── Tests /predictions ────────────────────────────────────────────────────────

def test_list_predictions(client_no_db):
    with patch("src.api.crud.get_predictions", return_value=(0, [])):
        r = client_no_db.get("/predictions")
    assert r.status_code == 200
    assert "predictions" in r.json()


# ── Tests /alerts ─────────────────────────────────────────────────────────────

def test_list_alerts(client_no_db):
    with patch("src.api.crud.get_alerts", return_value=(0, [])):
        r = client_no_db.get("/alerts")
    assert r.status_code == 200
    assert "alerts" in r.json()


def test_resolve_alert_not_found(client_no_db):
    with patch("src.api.crud.resolve_alert", return_value=None):
        r = client_no_db.patch("/alerts/999/resolve",
                               json={"resolved_by": "prof@edtech.fr"})
    assert r.status_code == 404


def test_resolve_alert_ok(client_no_db):
    fake_alert = {
        "alert_id": 1, "student_id": 42,
        "alert_type": "dropout_risk", "severity": "critical",
        "message": "Risque élevé", "triggered_at": "2025-01-15T10:00:00",
        "resolved_at": "2025-01-16T09:00:00",
    }
    with patch("src.api.crud.resolve_alert", return_value=fake_alert):
        r = client_no_db.patch("/alerts/1/resolve",
                               json={"resolved_by": "prof@edtech.fr"})
    assert r.status_code == 200
    assert r.json()["resolved_at"] is not None


# ── Tests latence middleware ───────────────────────────────────────────────────

def test_process_time_header(client_no_db):
    r = client_no_db.get("/")
    assert "X-Process-Time-Ms" in r.headers
