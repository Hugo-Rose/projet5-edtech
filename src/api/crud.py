"""Requêtes SQL — séparation logique métier / routes."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

# ── Étudiants ─────────────────────────────────────────────────────────────────

def get_students(
    db: Session,
    cohort_id:  Optional[int] = None,
    status:     Optional[str] = None,
    risk_label: Optional[str] = None,
    limit:  int = 50,
    offset: int = 0,
) -> tuple[int, list[dict]]:
    filters = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if cohort_id:
        filters.append("s.cohort_id = :cohort_id")
        params["cohort_id"] = cohort_id
    if status:
        filters.append("s.status = :status")
        params["status"] = status
    if risk_label:
        filters.append("latest_pred.risk_label = :risk_label")
        params["risk_label"] = risk_label

    where = " AND ".join(filters)

    sql = text(f"""
        WITH latest_pred AS (
            SELECT DISTINCT ON (student_id)
                student_id, dropout_prob, risk_label, predicted_at, model_version
            FROM predictions
            ORDER BY student_id, predicted_at DESC
        )
        SELECT
            s.student_id, s.cohort_id, s.first_name, s.last_name,
            s.email, s.age, s.status, s.enrollment_date,
            lp.dropout_prob, lp.risk_label, lp.predicted_at, lp.model_version
        FROM students s
        LEFT JOIN latest_pred lp ON lp.student_id = s.student_id
        WHERE {where}
        ORDER BY s.student_id
        LIMIT :limit OFFSET :offset
    """)

    count_sql = text(f"""
        WITH latest_pred AS (
            SELECT DISTINCT ON (student_id)
                student_id, risk_label
            FROM predictions ORDER BY student_id, predicted_at DESC
        )
        SELECT COUNT(*) FROM students s
        LEFT JOIN latest_pred lp ON lp.student_id = s.student_id
        WHERE {where}
    """)

    rows  = db.execute(sql, params).mappings().all()
    total = db.execute(count_sql, {k: v for k, v in params.items()
                                   if k not in ("limit", "offset")}).scalar()
    return int(total or 0), [dict(r) for r in rows]


def get_student_by_id(db: Session, student_id: int) -> Optional[dict]:
    sql = text("""
        SELECT
            s.*,
            lp.dropout_prob, lp.risk_label, lp.predicted_at,
            lp.model_version, lp.prediction_id
        FROM students s
        LEFT JOIN (
            SELECT DISTINCT ON (student_id)
                student_id, dropout_prob, risk_label,
                predicted_at, model_version, prediction_id
            FROM predictions
            ORDER BY student_id, predicted_at DESC
        ) lp ON lp.student_id = s.student_id
        WHERE s.student_id = :sid
    """)
    row = db.execute(sql, {"sid": student_id}).mappings().first()
    return dict(row) if row else None


def get_student_weekly_trend(db: Session, student_id: int,
                             n_weeks: int = 12) -> list[dict]:
    sql = text("""
        SELECT week_start, login_count, total_time_min,
               quiz_pass_rate, avg_score, forum_posts,
               dropout_risk_score
        FROM weekly_features
        WHERE student_id = :sid
        ORDER BY week_start DESC
        LIMIT :n
    """)
    rows = db.execute(sql, {"sid": student_id, "n": n_weeks}).mappings().all()
    return [dict(r) for r in rows]


# ── Cohortes ──────────────────────────────────────────────────────────────────

def get_cohorts(db: Session) -> list[dict]:
    sql = text("""
        SELECT
            c.cohort_id, c.name, c.program, c.start_date, c.end_date, c.capacity,
            COUNT(s.student_id)                                     AS total_students,
            COUNT(s.student_id) FILTER (WHERE s.status != 'dropped_out') AS active_students,
            COUNT(s.student_id) FILTER (WHERE s.status = 'dropped_out')  AS dropout_count,
            ROUND(
                COUNT(s.student_id) FILTER (WHERE s.status = 'dropped_out')::NUMERIC
                / NULLIF(COUNT(s.student_id), 0), 4
            )                                                        AS dropout_rate
        FROM cohorts c
        LEFT JOIN students s ON s.cohort_id = c.cohort_id
        GROUP BY c.cohort_id
        ORDER BY c.start_date DESC
    """)
    return [dict(r) for r in db.execute(sql).mappings().all()]


def get_cohort_stats(db: Session, cohort_id: int) -> Optional[dict]:
    sql = text("""
        SELECT
            c.cohort_id, c.name, c.program, c.start_date, c.end_date, c.capacity,
            COUNT(s.student_id)                                     AS total_students,
            COUNT(s.student_id) FILTER (WHERE s.status != 'dropped_out') AS active_students,
            COUNT(s.student_id) FILTER (WHERE s.status = 'dropped_out')  AS dropout_count,
            ROUND(
                COUNT(s.student_id) FILTER (WHERE s.status = 'dropped_out')::NUMERIC
                / NULLIF(COUNT(s.student_id), 0), 4
            )                                                        AS dropout_rate,
            ROUND(AVG(wf.login_count), 2)                           AS avg_login_week,
            ROUND(AVG(wf.avg_score), 2)                             AS avg_score
        FROM cohorts c
        LEFT JOIN students s  ON s.cohort_id = c.cohort_id
        LEFT JOIN weekly_features wf ON wf.student_id = s.student_id
        WHERE c.cohort_id = :cid
        GROUP BY c.cohort_id
    """)
    row = db.execute(sql, {"cid": cohort_id}).mappings().first()
    return dict(row) if row else None


def get_cohort_risk_distribution(db: Session, cohort_id: int) -> dict:
    sql = text("""
        WITH latest_pred AS (
            SELECT DISTINCT ON (p.student_id)
                p.student_id, p.risk_label
            FROM predictions p
            JOIN students s ON s.student_id = p.student_id
            WHERE s.cohort_id = :cid
            ORDER BY p.student_id, p.predicted_at DESC
        )
        SELECT risk_label, COUNT(*) AS cnt
        FROM latest_pred
        GROUP BY risk_label
    """)
    rows = db.execute(sql, {"cid": cohort_id}).mappings().all()
    dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for r in rows:
        if r["risk_label"] in dist:
            dist[r["risk_label"]] = int(r["cnt"])
    return dist


def get_cohort_weekly_engagement(db: Session, cohort_id: int,
                                 n_weeks: int = 16) -> list[dict]:
    sql = text("""
        SELECT
            wf.week_start,
            ROUND(AVG(wf.login_count), 2)        AS avg_logins,
            ROUND(AVG(wf.total_time_min), 2)     AS avg_time_min,
            ROUND(AVG(wf.quiz_pass_rate), 4)     AS avg_quiz_pass,
            ROUND(PERCENTILE_CONT(0.5)
                  WITHIN GROUP (ORDER BY wf.dropout_risk_score), 4) AS dropout_risk_p50
        FROM weekly_features wf
        JOIN students s ON s.student_id = wf.student_id
        WHERE s.cohort_id = :cid
        GROUP BY wf.week_start
        ORDER BY wf.week_start DESC
        LIMIT :n
    """)
    rows = db.execute(sql, {"cid": cohort_id, "n": n_weeks}).mappings().all()
    return [dict(r) for r in rows]


# ── Prédictions ───────────────────────────────────────────────────────────────

def get_predictions(
    db: Session,
    student_id:  Optional[int] = None,
    risk_label:  Optional[str] = None,
    limit:  int = 100,
    offset: int = 0,
) -> tuple[int, list[dict]]:
    filters = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if student_id:
        filters.append("student_id = :student_id")
        params["student_id"] = student_id
    if risk_label:
        filters.append("risk_label = :risk_label")
        params["risk_label"] = risk_label

    where = " AND ".join(filters)
    sql = text(f"""
        SELECT prediction_id, student_id, model_version,
               dropout_prob, risk_label, predicted_at
        FROM predictions WHERE {where}
        ORDER BY predicted_at DESC
        LIMIT :limit OFFSET :offset
    """)
    count_sql = text(f"SELECT COUNT(*) FROM predictions WHERE {where}")

    rows  = db.execute(sql, params).mappings().all()
    total = db.execute(count_sql, {k: v for k, v in params.items()
                                   if k not in ("limit", "offset")}).scalar()
    return int(total or 0), [dict(r) for r in rows]


def save_prediction(db: Session, student_id: int, dropout_prob: float,
                    risk_label: str, model_version: str,
                    features_snapshot: Optional[dict] = None) -> dict:
    import json as _json
    sql = text("""
        INSERT INTO predictions
            (student_id, model_version, dropout_prob, risk_label,
             predicted_at, features_snapshot)
        VALUES
            (:sid, :mv, :prob, :rl, NOW(),
             :snap::jsonb)
        RETURNING prediction_id, student_id, model_version,
                  dropout_prob, risk_label, predicted_at
    """)
    row = db.execute(sql, {
        "sid":  student_id,
        "mv":   model_version,
        "prob": dropout_prob,
        "rl":   risk_label,
        "snap": _json.dumps(features_snapshot) if features_snapshot else None,
    }).mappings().first()
    db.commit()
    return dict(row)


# ── Alertes ───────────────────────────────────────────────────────────────────

def get_alerts(
    db: Session,
    student_id: Optional[int] = None,
    severity:   Optional[str] = None,
    resolved:   Optional[bool] = None,
    limit:  int = 100,
    offset: int = 0,
) -> tuple[int, list[dict]]:
    filters = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if student_id:
        filters.append("student_id = :student_id")
        params["student_id"] = student_id
    if severity:
        filters.append("severity = :severity")
        params["severity"] = severity
    if resolved is not None:
        if resolved:
            filters.append("resolved_at IS NOT NULL")
        else:
            filters.append("resolved_at IS NULL")

    where = " AND ".join(filters)
    sql = text(f"""
        SELECT alert_id, student_id, alert_type, severity,
               message, triggered_at, resolved_at
        FROM alerts WHERE {where}
        ORDER BY triggered_at DESC
        LIMIT :limit OFFSET :offset
    """)
    count_sql = text(f"SELECT COUNT(*) FROM alerts WHERE {where}")

    rows  = db.execute(sql, params).mappings().all()
    total = db.execute(count_sql, {k: v for k, v in params.items()
                                   if k not in ("limit", "offset")}).scalar()
    return int(total or 0), [dict(r) for r in rows]


def resolve_alert(db: Session, alert_id: int, resolved_by: str) -> Optional[dict]:
    sql = text("""
        UPDATE alerts
           SET resolved_at = NOW(), resolved_by = :by
         WHERE alert_id = :aid
        RETURNING alert_id, student_id, alert_type, severity,
                  message, triggered_at, resolved_at
    """)
    row = db.execute(sql, {"aid": alert_id, "by": resolved_by}).mappings().first()
    db.commit()
    return dict(row) if row else None
