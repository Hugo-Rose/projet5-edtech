"""
Chargement des données pour le dashboard Streamlit.

Essaie PostgreSQL → Parquet → données mockées (mode démo).
Toutes les fonctions publiques retournent des DataFrames prêts à l'emploi.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PARQUET_DIR = Path("data/features")
DEMO_LABEL  = "demo"

CLUSTER_NAMES = {0: "Très engagé", 1: "Engagé", 2: "Passif", 3: "À risque"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _db_available() -> bool:
    try:
        from src.data.db import check_connection
        return check_connection()
    except Exception:
        return False


def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """Score 0-100 composite : logins 30% + temps 30% + quiz 40%."""
    out = pd.Series(index=df.index, dtype=float)
    for col, w, cap in [
        ("login_count",     0.30, 15),
        ("total_time_min",  0.30, 300),
        ("quiz_pass_rate",  0.40, 1.0),
    ]:
        if col in df.columns:
            out = out.add(
                df[col].clip(0, cap).fillna(0) / cap * w * 100,
                fill_value=0,
            )
    return out.clip(0, 100).round(1)


def assign_clusters(weekly_df: pd.DataFrame,
                    n_clusters: int = 4) -> pd.DataFrame:
    """K-Means sur les features agrégées par étudiant."""
    agg = (
        weekly_df
        .groupby("student_id")[["login_count", "total_time_min",
                                 "quiz_pass_rate", "avg_score"]]
        .mean()
        .fillna(0)
        .reset_index()
    )
    X = StandardScaler().fit_transform(agg.drop(columns="student_id"))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    agg["cluster_id"] = km.fit_predict(X)

    # Nommer les clusters : engagement décroissant → "Très engagé" … "À risque"
    centroids = pd.DataFrame(
        km.cluster_centers_,
        columns=["login", "time", "quiz", "score"],
    )
    rank = centroids.mean(axis=1).rank(ascending=False).astype(int) - 1
    agg["cluster_name"] = agg["cluster_id"].map(rank.to_dict()).map(CLUSTER_NAMES)
    return agg[["student_id", "cluster_id", "cluster_name"]]


# ── Chargement depuis DB ──────────────────────────────────────────────────────

def _from_db() -> dict[str, pd.DataFrame]:
    from src.data.db import get_engine
    engine = get_engine()

    students = pd.read_sql("SELECT * FROM students", engine)
    cohorts  = pd.read_sql("SELECT * FROM cohorts",  engine)
    weekly   = pd.read_sql(
        "SELECT * FROM weekly_features ORDER BY week_start", engine
    )
    preds = pd.read_sql("""
        SELECT DISTINCT ON (student_id)
            student_id, dropout_prob, risk_label, predicted_at, model_version
        FROM predictions ORDER BY student_id, predicted_at DESC
    """, engine)
    alerts = pd.read_sql(
        "SELECT * FROM alerts WHERE resolved_at IS NULL ORDER BY triggered_at DESC",
        engine,
    )
    forecasts_path = PARQUET_DIR / "engagement_forecasts.parquet"
    forecasts = (pd.read_parquet(forecasts_path)
                 if forecasts_path.exists() else pd.DataFrame())

    return dict(students=students, cohorts=cohorts, weekly=weekly,
                predictions=preds, alerts=alerts, forecasts=forecasts)


# ── Données mockées (mode démo) ───────────────────────────────────────────────

def _mock_data(n_students: int = 150, n_weeks: int = 12) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(99)

    cohorts = pd.DataFrame([
        {"cohort_id": 1, "name": "Promo 2023 — Data Science",
         "program": "Data Science & IA", "capacity": 35},
        {"cohort_id": 2, "name": "Promo 2024 — Dev Web",
         "program": "Développement Web", "capacity": 40},
        {"cohort_id": 3, "name": "Promo 2024 — Cybersécurité",
         "program": "Cybersécurité",     "capacity": 30},
    ])

    # Étudiants
    profiles = rng.choice(["engaged", "average", "at_risk"],
                          p=[0.35, 0.45, 0.20], size=n_students)
    students = pd.DataFrame({
        "student_id":  range(1, n_students + 1),
        "cohort_id":   rng.integers(1, 4, n_students),
        "first_name":  [f"Étudiant {i}" for i in range(1, n_students + 1)],
        "last_name":   [f"Nom{i}"       for i in range(1, n_students + 1)],
        "email":       [f"s{i}@demo.fr" for i in range(1, n_students + 1)],
        "age":         rng.integers(19, 35, n_students),
        "gender":      rng.choice(["M", "F", "Other"], p=[0.48, 0.48, 0.04],
                                  size=n_students),
        "has_job":     rng.random(n_students) < 0.30,
        "scholarship": rng.random(n_students) < 0.25,
        "distance_km": rng.exponential(20, n_students).round(1),
        "entry_grade": np.clip(rng.normal(12.5, 2.5, n_students), 0, 20).round(2),
        "status":      rng.choice(["enrolled", "graduated", "dropped_out"],
                                  p=[0.65, 0.20, 0.15], size=n_students),
        "_profile":    profiles,
    })

    # Weekly features
    base_date = pd.Timestamp("2025-10-06")
    weeks     = [base_date - pd.Timedelta(weeks=i) for i in range(n_weeks - 1, -1, -1)]
    wf_rows   = []
    for _, s in students.iterrows():
        p = s["_profile"]
        for w in weeks:
            logins = int(rng.integers(*(
                (6,12) if p=="engaged" else (2,6) if p=="average" else (0,3)
            )))
            qpr = float(np.clip(
                rng.normal(*(0.82,0.08) if p=="engaged"
                            else (0.60,0.12) if p=="average"
                            else (0.35,0.15)), 0, 1))
            wf_rows.append({
                "student_id":         int(s["student_id"]),
                "week_start":         w,
                "login_count":        logins,
                "total_time_min":     round(logins * rng.uniform(15, 45), 1),
                "videos_watched":     int(rng.integers(0, logins + 1)),
                "quiz_attempts":      int(rng.integers(0, 3)),
                "quiz_pass_rate":     round(qpr, 4),
                "avg_score":          round(float(rng.normal(
                    70 if p=="engaged" else 58 if p=="average" else 42, 10
                )), 2),
                "forum_posts":        int(rng.integers(0, 4)),
                "assignments_on_time": int(rng.integers(0, 2)),
                "assignments_late":    int(rng.integers(0, 2)),
                "dropout_risk_score":  round(float(
                    rng.beta(1, 9) if p=="engaged"
                    else rng.beta(2, 5) if p=="average"
                    else rng.beta(4, 3)), 4),
                "label_dropout":       1 if (p=="at_risk" and rng.random()<0.4) else 0,
            })
    weekly = pd.DataFrame(wf_rows)

    # Prédictions
    latest_wf = weekly.groupby("student_id").last().reset_index()
    risk_map   = lambda p: (
        "critical" if p > 0.65 else "high" if p > 0.40
        else "medium" if p > 0.20 else "low"
    )
    predictions = pd.DataFrame({
        "student_id":    latest_wf["student_id"],
        "dropout_prob":  latest_wf["dropout_risk_score"],
        "risk_label":    latest_wf["dropout_risk_score"].apply(risk_map),
        "predicted_at":  pd.Timestamp.now(),
        "model_version": "demo-v1",
    })

    # Alertes
    critical_ids = predictions[
        predictions["risk_label"].isin(["high", "critical"])
    ]["student_id"].head(20)
    alerts = pd.DataFrame({
        "alert_id":    range(1, len(critical_ids) + 1),
        "student_id":  critical_ids.values,
        "alert_type":  "dropout_risk",
        "severity":    predictions.set_index("student_id").loc[
                           critical_ids, "risk_label"
                       ].map({"high": "warning", "critical": "critical"}).values,
        "message":     [f"Risque détecté (student {sid})"
                        for sid in critical_ids.values],
        "triggered_at": pd.Timestamp.now(),
    })

    # Forecasts mock (Prophet-like structure)
    future_weeks = [pd.Timestamp.now() + pd.Timedelta(weeks=i+1) for i in range(4)]
    fc_rows = []
    for cid in [1, 2, 3]:
        for w in future_weeks:
            for metric in ["avg_logins", "avg_time_min",
                           "avg_quiz_pass_rate", "dropout_risk_p50"]:
                base = {"avg_logins": 5, "avg_time_min": 120,
                        "avg_quiz_pass_rate": 0.65, "dropout_risk_p50": 0.25}[metric]
                fc_rows.append({
                    "cohort_id":   cid,
                    "cohort_name": f"Promo {cid}",
                    "metric":      metric,
                    "ds":          w,
                    "yhat":        base * rng.uniform(0.85, 1.15),
                    "yhat_lower":  base * 0.75,
                    "yhat_upper":  base * 1.25,
                    "is_future":   True,
                })
    forecasts = pd.DataFrame(fc_rows)

    students = students.drop(columns=["_profile"])
    return dict(students=students, cohorts=cohorts, weekly=weekly,
                predictions=predictions, alerts=alerts, forecasts=forecasts)


# ── Point d'entrée public ─────────────────────────────────────────────────────

def load_all_data() -> tuple[dict[str, pd.DataFrame], bool]:
    """
    Retourne (data_dict, is_demo).
    data_dict contient : students, cohorts, weekly, predictions, alerts, forecasts.
    Ajoute automatiquement engagement_score et cluster dans weekly + students.
    """
    is_demo = False
    if _db_available():
        try:
            data = _from_db()
            logger.info("Données chargées depuis PostgreSQL.")
        except Exception as e:
            logger.warning(f"Erreur DB ({e}) — mode démo activé.")
            data   = _mock_data()
            is_demo = True
    else:
        logger.info("DB inaccessible — mode démo activé.")
        data   = _mock_data()
        is_demo = True

    # Enrichissements communs
    data["weekly"]["engagement_score"] = compute_engagement_score(data["weekly"])
    clusters = assign_clusters(data["weekly"])
    data["students"] = data["students"].merge(clusters, on="student_id", how="left")

    # Merge prédictions → étudiants
    data["students"] = data["students"].merge(
        data["predictions"][["student_id", "dropout_prob", "risk_label"]],
        on="student_id", how="left",
    )
    return data, is_demo
