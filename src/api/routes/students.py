"""Routes /students."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import crud
from src.api.dependencies import ModelBundle, get_db, get_model
from src.api.schemas import (
    PredictionOut,
    PredictRequest,
    RecommendationsResponse,
    StudentDetail,
    StudentListResponse,
)
from src.models.constants import FEATURE_COLS
from src.models.evaluate import risk_label as compute_risk_label

router = APIRouter(prefix="/students", tags=["students"])


@router.get("", response_model=StudentListResponse)
def list_students(
    cohort_id:  Optional[int] = Query(None),
    status:     Optional[str] = Query(None),
    risk_label: Optional[str] = Query(None),
    limit:  int = Query(50,  ge=1, le=500),
    offset: int = Query(0,   ge=0),
    db: Session = Depends(get_db),
):
    total, students = crud.get_students(
        db, cohort_id=cohort_id, status=status,
        risk_label=risk_label, limit=limit, offset=offset,
    )
    return {"total": total, "limit": limit, "offset": offset, "students": students}


@router.get("/{student_id}", response_model=StudentDetail)
def get_student(
    student_id: int,
    trend_weeks: int = Query(12, ge=1, le=52),
    db: Session = Depends(get_db),
):
    row = crud.get_student_by_id(db, student_id)
    if not row:
        raise HTTPException(status_code=404, detail="Étudiant introuvable")

    trend = crud.get_student_weekly_trend(db, student_id, trend_weeks)

    latest_risk = None
    if row.get("dropout_prob") is not None:
        latest_risk = {
            "prediction_id": row.get("prediction_id"),
            "student_id":    student_id,
            "dropout_prob":  float(row["dropout_prob"]),
            "risk_label":    row.get("risk_label"),
            "model_version": row.get("model_version"),
            "predicted_at":  row.get("predicted_at"),
        }

    return {**row, "latest_risk": latest_risk, "weekly_trend": trend}


@router.post("/{student_id}/predict", response_model=PredictionOut)
def predict_student(
    student_id: int,
    db:    Session      = Depends(get_db),
    model: ModelBundle  = Depends(get_model),
):
    if not model.ready:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Lancez src.models.train_dropout.",
        )

    # Récupère les dernières features hebdomadaires de l'étudiant
    sql_features = """
        SELECT {cols}
        FROM weekly_features
        WHERE student_id = :sid
        ORDER BY week_start DESC
        LIMIT 1
    """.format(cols=", ".join(FEATURE_COLS))

    from sqlalchemy import text
    row = db.execute(text(sql_features), {"sid": student_id}).mappings().first()
    if not row:
        raise HTTPException(
            status_code=404,
            detail="Aucune feature disponible pour cet étudiant.",
        )

    import pandas as pd
    X = pd.DataFrame([dict(row)])[FEATURE_COLS]
    X_pp = model.preprocessor.transform(X)
    prob = float(model.model.predict_proba(X_pp)[0, 1])
    label = compute_risk_label(prob, model.threshold)

    saved = crud.save_prediction(
        db, student_id=student_id,
        dropout_prob=prob, risk_label=label,
        model_version=model.version,
        features_snapshot=dict(row),
    )
    return saved


@router.get("/{student_id}/recommendations", response_model=RecommendationsResponse)
def get_recommendations(
    student_id: int,
    db: Session = Depends(get_db),
):
    """
    Retourne les 3 recommandations personnalisées basées sur le cluster K-Means
    et la corrélation comportement → amélioration du score.
    """
    # Vérifier que l'étudiant existe
    student = crud.get_student_by_id(db, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Étudiant introuvable")

    # Charger les weekly_features depuis la DB
    sql = text("""
        SELECT wf.student_id, wf.week_start,
               wf.login_count, wf.total_time_min, wf.videos_watched,
               wf.quiz_attempts, wf.quiz_pass_rate, wf.avg_score,
               wf.forum_posts, wf.assignments_on_time, wf.assignments_late
        FROM weekly_features wf
        JOIN students s ON s.student_id = wf.student_id
        WHERE s.cohort_id = (SELECT cohort_id FROM students WHERE student_id = :sid)
        ORDER BY wf.week_start
    """)
    rows = db.execute(sql, {"sid": student_id}).mappings().all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune donnée hebdomadaire disponible.",
        )

    weekly_df = pd.DataFrame([dict(r) for r in rows])

    from src.models.recommender import get_recommendations_for_student
    result = get_recommendations_for_student(student_id, weekly_df)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Impossible de calculer des recommandations pour cet étudiant.",
        )

    return {
        "student_id":      result.student_id,
        "cluster_id":      result.cluster_id,
        "cluster_name":    result.cluster_name,
        "recommendations": [
            {
                "rank":                 r.rank,
                "action_type":          r.action_type,
                "title":                r.title,
                "description":          r.description,
                "current_value":        r.current_value,
                "target_value":         r.target_value,
                "correlation":          r.correlation,
                "unit":                 r.unit,
                "expected_improvement": r.expected_improvement,
            }
            for r in result.recommendations
        ],
        "computed_at": result.computed_at,
    }
