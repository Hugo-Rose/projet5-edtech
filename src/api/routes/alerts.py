"""Routes /alerts."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api import crud
from src.api.dependencies import get_db
from src.api.schemas import AlertListResponse, AlertOut, ResolveAlertRequest

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=AlertListResponse)
def list_alerts(
    student_id: Optional[int]  = Query(None),
    severity:   Optional[str]  = Query(None),
    resolved:   Optional[bool] = Query(None, description="true=résolues, false=ouvertes"),
    limit:  int = Query(100, ge=1, le=500),
    offset: int = Query(0,   ge=0),
    db: Session = Depends(get_db),
):
    total, alerts = crud.get_alerts(
        db, student_id=student_id, severity=severity,
        resolved=resolved, limit=limit, offset=offset,
    )
    return {"total": total, "alerts": alerts}


@router.patch("/{alert_id}/resolve", response_model=AlertOut)
def resolve_alert(
    alert_id: int,
    body: ResolveAlertRequest,
    db: Session = Depends(get_db),
):
    updated = crud.resolve_alert(db, alert_id, body.resolved_by)
    if not updated:
        raise HTTPException(status_code=404, detail="Alerte introuvable")
    return updated
