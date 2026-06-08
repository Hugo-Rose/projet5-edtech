"""Schémas Pydantic — request / response de l'API."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


# ── Prédictions ───────────────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    prediction_id: Optional[int] = None
    student_id:    int
    dropout_prob:  float = Field(ge=0.0, le=1.0)
    risk_label:    str
    model_version: Optional[str] = None
    predicted_at:  Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PredictRequest(BaseModel):
    student_id: int


class BatchPredictRequest(BaseModel):
    student_ids: Optional[list[int]] = None
    week:        Optional[str]       = Field(
        default=None, description="YYYY-MM-DD — si absent, dernière semaine disponible"
    )


class BatchPredictResponse(BaseModel):
    week:    str
    count:   int
    results: list[PredictionOut]


# ── Étudiants ─────────────────────────────────────────────────────────────────

class StudentBase(BaseModel):
    student_id:      int
    cohort_id:       Optional[int]
    first_name:      Optional[str]
    last_name:       Optional[str]
    email:           str
    age:             Optional[int]
    status:          str
    enrollment_date: Optional[date]

    model_config = ConfigDict(from_attributes=True)


class StudentDetail(StudentBase):
    has_job:         Optional[bool]
    scholarship:     Optional[bool]
    distance_km:     Optional[float]
    entry_grade:     Optional[float]
    dropout_date:    Optional[date]
    latest_risk:     Optional[PredictionOut] = None
    weekly_trend:    Optional[list[dict]]    = None


class StudentListResponse(BaseModel):
    total:    int
    limit:    int
    offset:   int
    students: list[StudentBase]


# ── Cohortes ──────────────────────────────────────────────────────────────────

class CohortBase(BaseModel):
    cohort_id:  int
    name:       str
    program:    Optional[str]
    start_date: Optional[date]
    end_date:   Optional[date]
    capacity:   Optional[int]

    model_config = ConfigDict(from_attributes=True)


class RiskDistribution(BaseModel):
    low:      int = 0
    medium:   int = 0
    high:     int = 0
    critical: int = 0


class CohortStats(CohortBase):
    total_students:   int
    active_students:  int
    dropout_count:    int
    dropout_rate:     float
    avg_login_week:   Optional[float]
    avg_score:        Optional[float]
    risk_distribution: Optional[RiskDistribution] = None


class WeeklyEngagement(BaseModel):
    week_start:      str
    avg_logins:      float
    avg_time_min:    float
    avg_quiz_pass:   Optional[float]
    dropout_risk_p50: Optional[float]


# ── Alertes ───────────────────────────────────────────────────────────────────

class AlertOut(BaseModel):
    alert_id:    int
    student_id:  int
    alert_type:  Optional[str]
    severity:    Optional[str]
    message:     Optional[str]
    triggered_at: Optional[datetime]
    resolved_at:  Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class ResolveAlertRequest(BaseModel):
    resolved_by: str


class AlertListResponse(BaseModel):
    total:  int
    alerts: list[AlertOut]


# ── Santé ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:   str
    db:       str
    model:    str
    version:  str
