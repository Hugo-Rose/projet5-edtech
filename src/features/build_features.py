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

from src.data.db import get_engine

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
    # Colonnes requises toujours présentes
    required = ["student_id"]
    # Colonnes optionnelles — ajout de valeurs par défaut si absentes
    optional_defaults: dict = {
        "cohort_id":     1,
        "status":        "enrolled",
        "dropout_date":  pd.NaT,
        "entry_grade":   12.0,
        "has_job":       0,
        "scholarship":   0,
        "distance_km":   0.0,
        "age":           23,
        "gender":        "Other",
    }
    for col, default in optional_defaults.items():
        if col not in students.columns:
            students = students.copy()
            students[col] = default

    cols_to_use = required + [c for c in optional_defaults if c in students.columns]
    meta = students[cols_to_use]
    df = wf.merge(meta, on="student_id", how="left")

    # Remplissage des valeurs manquantes après merge
    for col, default in optional_defaults.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(default if not pd.isna(default) else 0)

    # Label binaire pour ML
    df["week_start"] = pd.to_datetime(df["week_start"])
    if "label_dropout" not in df.columns:
        # Si déjà présent dans wf (source UCI/OULAD), on le conserve
        if "dropout_date" in df.columns and df["dropout_date"].notna().any():
            df["dropout_date"] = pd.to_datetime(df["dropout_date"])
            df["label_dropout"] = (
                (df["status"] == "dropped_out") &
                (df["dropout_date"] >= df["week_start"])
            ).astype(int)
        else:
            df["label_dropout"] = (df["status"] == "dropped_out").astype(int)

    # Encodage gender
    df["gender_enc"]  = df["gender"].map({"M": 0, "F": 1, "Other": 2}).fillna(-1).astype(int)
    df["has_job"]     = df["has_job"].astype(int)
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

    with get_engine().connect() as conn:
        conn.execute(
            __import__("sqlalchemy").text(
                "TRUNCATE TABLE weekly_features"
            )
        )
        conn.commit()

    subset.to_sql(
        "weekly_features",
        get_engine(),
        if_exists="append", index=False,
        chunksize=50_000, method="multi",
    )
    logger.success(f"weekly_features : {len(subset):,} lignes insérées")


def save_to_parquet(wf: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "weekly_features.parquet"
    wf.to_parquet(path, index=False)
    logger.success(f"Parquet exporté → {path}  ({path.stat().st_size / 1e6:.1f} Mo)")


# ── Loaders par source ────────────────────────────────────────────────────────

def _load_from_db(weeks_back: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Source par défaut : PostgreSQL lms_events."""
    logger.info("Lecture lms_events depuis PostgreSQL...")
    events = pd.read_sql(SQL_EVENTS, get_engine())
    if weeks_back > 0:
        cutoff = events["week_start"].max() - pd.Timedelta(weeks=weeks_back)
        events = events[events["week_start"] >= cutoff]
        logger.info(f"Filtré sur les {weeks_back} dernières semaines")
    students = pd.read_sql(SQL_STUDENTS, get_engine())
    return events, students


def _load_from_uci(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Source UCI : lit data/raw/uci_dropout.csv (weekly snapshots pré-calculés)
    ou le génère à la volée si absent.
    """
    weekly_path  = raw_dir / "uci_dropout.csv"
    student_path = raw_dir / "uci_students.csv"

    if not weekly_path.exists():
        logger.info("Fichier UCI absent — téléchargement en cours...")
        from src.data.load_uci_dropout import main as load_uci
        load_uci(str(raw_dir))

    logger.info(f"Lecture UCI weekly features : {weekly_path}")
    weekly   = pd.read_csv(weekly_path, parse_dates=["week_start"])
    students = pd.read_csv(student_path) if student_path.exists() else pd.DataFrame()

    # weekly UCI est déjà agrégé — on le remet dans le format events-like
    # pour pouvoir passer par build_weekly_features standard.
    # On simule des événements à partir des colonnes agrégées.
    events = _expand_weekly_to_events(weekly)
    return events, students


def _expand_weekly_to_events(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstitue un DataFrame events-like depuis des weekly features pré-agrégées.
    Permet de réutiliser build_weekly_features sans duplication de code.
    """
    rows = []
    for _, r in weekly.iterrows():
        sid = int(r["student_id"])
        ws  = r["week_start"]
        for _ in range(int(r.get("login_count", 0))):
            rows.append({"student_id": sid, "event_type": "login",
                         "event_ts": ws, "duration_sec": 2700,
                         "score": None, "week_start": ws})
        for _ in range(int(r.get("videos_watched", 0))):
            rows.append({"student_id": sid, "event_type": "video_view",
                         "event_ts": ws, "duration_sec": 900,
                         "score": None, "week_start": ws})
        n_att = int(r.get("quiz_attempts", 0))
        qpr   = float(r.get("quiz_pass_rate") or 0)
        for i in range(n_att):
            passed = i < int(n_att * qpr)
            sc = float(r.get("avg_score") or 60)
            rows.append({"student_id": sid, "event_type": "quiz_attempt",
                         "event_ts": ws, "duration_sec": 600,
                         "score": sc, "week_start": ws})
            rows.append({"student_id": sid,
                         "event_type": "quiz_pass" if passed else "quiz_fail",
                         "event_ts": ws, "duration_sec": None,
                         "score": sc, "week_start": ws})
        for _ in range(int(r.get("assignments_on_time", 0))):
            rows.append({"student_id": sid, "event_type": "assignment_submit",
                         "event_ts": ws, "duration_sec": None,
                         "score": float(r.get("avg_score") or 65), "week_start": ws})
        for _ in range(int(r.get("assignments_late", 0))):
            rows.append({"student_id": sid, "event_type": "assignment_late",
                         "event_ts": ws, "duration_sec": None,
                         "score": float(r.get("avg_score") or 50), "week_start": ws})

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["student_id", "event_type", "event_ts",
                 "duration_sec", "score", "week_start"]
    )


def _load_from_oulad(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Source OULAD : lit data/raw/oulad_weekly.parquet (pré-agrégé).
    Génère à la volée si absent.
    """
    weekly_path  = raw_dir / "oulad_weekly.parquet"
    student_path = raw_dir / "oulad_students.csv"

    if not weekly_path.exists():
        logger.info("Fichier OULAD absent — téléchargement en cours...")
        from src.data.load_oulad import main as load_oulad
        load_oulad(str(raw_dir))

    logger.info(f"Lecture OULAD weekly features : {weekly_path}")
    weekly   = pd.read_parquet(weekly_path)
    students = pd.read_csv(student_path) if student_path.exists() else pd.DataFrame()

    events = _expand_weekly_to_events(weekly)
    return events, students


def _load_from_synthetic(raw_dir: Path,
                         n_students: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback : génère des données synthétiques si aucun dataset réel n'est dispo."""
    logger.warning("Mode fallback : génération de données synthétiques...")
    from src.data.generate_synthetic_data import (
        generate_students, generate_modules, generate_lms_events, COHORTS
    )
    students_df = generate_students(n_students)
    modules_df  = generate_modules()
    events_df   = generate_lms_events(students_df, modules_df)
    return events_df, students_df


# ── Main ──────────────────────────────────────────────────────────────────────

SOURCE_CHOICES = ("db", "uci", "oulad", "synthetic")


def main(
    output_dir: str = "data/features",
    weeks_back: int = 0,
    source:     str = "db",
    raw_dir:    str = "data/raw",
) -> pd.DataFrame:
    logger.info(f"=== Feature Engineering — source={source} ===")

    raw = Path(raw_dir)

    if source == "db":
        events, students = _load_from_db(weeks_back)
    elif source == "uci":
        events, students = _load_from_uci(raw)
    elif source == "oulad":
        events, students = _load_from_oulad(raw)
    elif source == "synthetic":
        events, students = _load_from_synthetic(raw)
    else:
        raise ValueError(f"Source inconnue : {source!r}. Choisir parmi {SOURCE_CHOICES}")

    if events.empty:
        logger.error("Aucun événement chargé — abandon.")
        raise SystemExit(1)

    wf = build_weekly_features(events)
    wf = add_rolling_features(wf)

    if not students.empty:
        wf = merge_student_info(wf, students)

    if source == "db":
        save_to_db(wf)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    parquet_name = f"weekly_features_{source}.parquet" if source != "db" \
                   else "weekly_features.parquet"
    wf.to_parquet(out / parquet_name, index=False)

    logger.info("\n=== Résumé features ===")
    logger.info(f"  Source             : {source}")
    logger.info(f"  Snapshots totaux   : {len(wf):,}")
    logger.info(f"  Etudiants couverts : {wf['student_id'].nunique():,}")
    logger.info(f"  Semaines           : {wf['week_start'].nunique():,}")
    if "label_dropout" in wf.columns:
        logger.info(f"  Taux label=1 (drop): {wf['label_dropout'].mean()*100:.1f}%")

    return wf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",     default="data/features")
    parser.add_argument("--weeks-back", type=int, default=0,
                        help="0 = tout l'historique (source=db uniquement)")
    parser.add_argument("--source",     default="db", choices=SOURCE_CHOICES,
                        help="Source des données : db | uci | oulad | synthetic")
    parser.add_argument("--raw-dir",    default="data/raw")
    args = parser.parse_args()
    main(output_dir=args.output, weeks_back=args.weeks_back,
         source=args.source, raw_dir=args.raw_dir)
