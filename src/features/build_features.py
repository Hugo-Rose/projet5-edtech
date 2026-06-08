"""
Feature engineering : agrégats hebdomadaires par étudiant.

Lit lms_events + students depuis PostgreSQL,
produit weekly_features et exporte en Parquet.

Usage:
    python -m src.features.build_features [--output data/features] [--weeks-back 0]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.data.db import engine

PARQUET_OUT = Path("data/features")

# ── Requêtes SQL ──────────────────────────────────────────────────────────────

SQL_EVENTS = """
SELECT
    e.student_id,
    e.event_type,
    e.event_ts,
    e.duration_sec,
    e.score,
    DATE_TRUNC('week', e.event_ts)::DATE AS week_start
FROM lms_events e
ORDER BY e.event_ts
"""

SQL_STUDENTS = """
SELECT student_id, cohort_id, status, dropout_date,
       entry_grade, has_job, scholarship, distance_km, age, gender
FROM students
"""


# ── Agrégation hebdomadaire ───────────────────────────────────────────────────

def _agg_week(grp: pd.DataFrame) -> pd.Series:
    logins       = (grp["event_type"] == "login").sum()
    total_time   = grp.loc[grp["event_type"] == "login", "duration_sec"].sum() / 60
    videos       = (grp["event_type"] == "video_view").sum()

    quiz_att     = (grp["event_type"] == "quiz_attempt").sum()
    quiz_pass    = (grp["event_type"] == "quiz_pass").sum()
    pass_rate    = (quiz_pass / quiz_att) if quiz_att > 0 else np.nan

    score_rows   = grp.loc[grp["score"].notna(), "score"]
    avg_score    = score_rows.mean() if len(score_rows) > 0 else np.nan

    forum        = grp["event_type"].isin(["forum_post", "forum_reply"]).sum()
    on_time      = (grp["event_type"] == "assignment_submit").sum()
    late         = (grp["event_type"] == "assignment_late").sum()

    return pd.Series({
        "login_count":          int(logins),
        "total_time_min":       round(float(total_time), 2),
        "videos_watched":       int(videos),
        "quiz_attempts":        int(quiz_att),
        "quiz_pass_rate":       round(float(pass_rate), 4) if not np.isnan(pass_rate) else None,
        "avg_score":            round(float(avg_score),  2) if not np.isnan(avg_score)  else None,
        "forum_posts":          int(forum),
        "assignments_on_time":  int(on_time),
        "assignments_late":     int(late),
    })


def build_weekly_features(events: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Agrégation de {len(events):,} événements...")
    wf = (
        events
        .groupby(["student_id", "week_start"])
        .apply(_agg_week, include_groups=False)
        .reset_index()
    )
    logger.info(f"  → {len(wf):,} snapshots hebdomadaires")
    return wf


# ── Features glissantes (tendances sur 4 semaines) ───────────────────────────

def add_rolling_features(wf: pd.DataFrame) -> pd.DataFrame:
    wf = wf.sort_values(["student_id", "week_start"])
    cols_to_roll = ["login_count", "total_time_min", "quiz_pass_rate", "avg_score"]

    for col in cols_to_roll:
        wf[f"{col}_trend4"] = (
            wf.groupby("student_id")[col]
            .transform(lambda s: s.rolling(4, min_periods=1).mean())
        )
        wf[f"{col}_delta"] = (
            wf.groupby("student_id")[col]
            .transform(lambda s: s.diff())
        )
    return wf


# ── Jointure données statiques étudiants ─────────────────────────────────────

def merge_student_info(wf: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
    meta = students[["student_id", "cohort_id", "status", "dropout_date",
                      "entry_grade", "has_job", "scholarship", "distance_km",
                      "age", "gender"]]
    df = wf.merge(meta, on="student_id", how="left")

    # Label binaire pour ML : décrochage survenu après cette semaine ?
    df["dropout_date"] = pd.to_datetime(df["dropout_date"])
    df["week_start"]   = pd.to_datetime(df["week_start"])
    df["label_dropout"] = (
        (df["status"] == "dropped_out") &
        (df["dropout_date"] >= df["week_start"])
    ).astype(int)

    # Encodage gender
    df["gender_enc"] = df["gender"].map({"M": 0, "F": 1, "Other": 2}).fillna(-1).astype(int)
    df["has_job"]    = df["has_job"].astype(int)
    df["scholarship"] = df["scholarship"].astype(int)

    return df


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save_to_db(wf: pd.DataFrame) -> None:
    cols_db = [
        "student_id", "week_start", "login_count", "total_time_min",
        "videos_watched", "quiz_attempts", "quiz_pass_rate", "avg_score",
        "forum_posts", "assignments_on_time", "assignments_late",
    ]
    subset = wf[cols_db].copy()
    subset["week_start"] = subset["week_start"].dt.date

    with engine.connect() as conn:
        conn.execute(
            __import__("sqlalchemy").text(
                "TRUNCATE TABLE weekly_features"
            )
        )
        conn.commit()

    subset.to_sql(
        "weekly_features", engine,
        if_exists="append", index=False,
        chunksize=50_000, method="multi",
    )
    logger.success(f"weekly_features : {len(subset):,} lignes insérées")


def save_to_parquet(wf: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "weekly_features.parquet"
    wf.to_parquet(path, index=False)
    logger.success(f"Parquet exporté → {path}  ({path.stat().st_size / 1e6:.1f} Mo)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(output_dir: str = "data/features", weeks_back: int = 0) -> pd.DataFrame:
    logger.info("=== Feature Engineering — weekly_features ===")

    logger.info("Lecture lms_events depuis PostgreSQL...")
    events = pd.read_sql(SQL_EVENTS, engine)

    if weeks_back > 0:
        cutoff = events["week_start"].max() - pd.Timedelta(weeks=weeks_back)
        events = events[events["week_start"] >= cutoff]
        logger.info(f"Filtré sur les {weeks_back} dernières semaines")

    logger.info("Lecture students...")
    students = pd.read_sql(SQL_STUDENTS, engine)

    wf = build_weekly_features(events)
    wf = add_rolling_features(wf)
    wf = merge_student_info(wf, students)

    save_to_db(wf)
    save_to_parquet(wf, Path(output_dir))

    # Résumé
    logger.info(f"\n=== Résumé features ===")
    logger.info(f"  Snapshots totaux   : {len(wf):,}")
    logger.info(f"  Étudiants couverts : {wf['student_id'].nunique():,}")
    logger.info(f"  Semaines           : {wf['week_start'].nunique():,}")
    logger.info(f"  Taux label=1 (drop): {wf['label_dropout'].mean()*100:.1f}%")
    logger.info(f"  Colonnes           : {list(wf.columns)}")

    return wf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",     default="data/features")
    parser.add_argument("--weeks-back", type=int, default=0,
                        help="0 = tout l'historique")
    args = parser.parse_args()
    main(output_dir=args.output, weeks_back=args.weeks_back)
