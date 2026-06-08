"""Routes /cohorts."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api import crud
from src.api.dependencies import get_db
from src.api.schemas import CohortBase, CohortStats, WeeklyEngagement

router = APIRouter(prefix="/cohorts", tags=["cohorts"])


@router.get("", response_model=list[CohortBase])
def list_cohorts(db: Session = Depends(get_db)):
    return crud.get_cohorts(db)


@router.get("/{cohort_id}/stats", response_model=CohortStats)
def cohort_stats(
    cohort_id: int,
    db: Session = Depends(get_db),
):
    stats = crud.get_cohort_stats(db, cohort_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Cohorte introuvable")

    risk_dist = crud.get_cohort_risk_distribution(db, cohort_id)
    return {**stats, "risk_distribution": risk_dist}


@router.get("/{cohort_id}/engagement", response_model=list[WeeklyEngagement])
def cohort_engagement(
    cohort_id: int,
    weeks: int = Query(16, ge=1, le=104),
    db: Session = Depends(get_db),
):
    rows = crud.get_cohort_weekly_engagement(db, cohort_id, weeks)
    if not rows:
        raise HTTPException(status_code=404, detail="Aucune donnée d'engagement")
    return [
        {**r, "week_start": str(r["week_start"])} for r in rows
    ]
