"""Routes /predictions."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api import crud
from src.api.dependencies import ModelBundle, get_db, get_model
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionOut,
)
from src.models.constants import FEATURE_COLS
from src.models.evaluate import risk_label as compute_risk_label

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=dict)
def list_predictions(
    student_id:  Optional[int] = Query(None),
    risk_label:  Optional[str] = Query(None),
    limit:  int = Query(100, ge=1, le=1000),
    offset: int = Query(0,   ge=0),
    db: Session = Depends(get_db),
):
    total, preds = crud.get_predictions(
        db, student_id=student_id,
        risk_label=risk_label,
        limit=limit, offset=offset,
    )
    return {"total": total, "limit": limit, "offset": offset, "predictions": preds}


@router.post("/batch", response_model=BatchPredictResponse)
def batch_predict(
    body:  BatchPredictRequest,
    db:    Session     = Depends(get_db),
    model: ModelBundle = Depends(get_model),
):
    if not model.ready:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Lancez src.models.train_dropout.",
        )

    # Sélectionne la semaine cible
    if body.week:
        week_filter = "AND wf.week_start = :week"
        params: dict = {"week": body.week}
    else:
        week_filter = "AND wf.week_start = (SELECT MAX(week_start) FROM weekly_features)"
        params = {}

    id_filter = ""
    if body.student_ids:
        id_filter = "AND wf.student_id = ANY(:ids)"
        params["ids"] = body.student_ids

    sql = text(f"""
        SELECT wf.student_id, wf.week_start, {', '.join(f'wf.{c}' for c in FEATURE_COLS)}
        FROM weekly_features wf
        WHERE 1=1 {week_filter} {id_filter}
        ORDER BY wf.student_id
    """)

    rows = db.execute(sql, params).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail="Aucune feature pour cette semaine")

    df = pd.DataFrame([dict(r) for r in rows])
    week_str = str(df["week_start"].iloc[0])

    X_pp = model.preprocessor.transform(df[FEATURE_COLS])
    probas = model.model.predict_proba(X_pp)[:, 1]

    results = []
    for i, row in df.iterrows():
        prob  = float(probas[i])
        label = compute_risk_label(prob, model.threshold)
        saved = crud.save_prediction(
            db,
            student_id=int(row["student_id"]),
            dropout_prob=prob, risk_label=label,
            model_version=model.version,
        )
        results.append(saved)

    return {"week": week_str, "count": len(results), "results": results}
