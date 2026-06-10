"""
Loader — UCI Dataset ID=697 « Predict students' Dropout and Academic Success »
https://archive.ics.uci.edu/dataset/697

Colonnes clés → schéma EdTech :
  Gender, Age at enrollment, Admission grade, Scholarship holder,
  Daytime/evening attendance, Curricular units 1/2 sem (grade, approved,
  enrolled, evaluations) → students + weekly_features (2 snapshots/étudiant)

Target : Dropout → label_dropout=1 ; Enrolled/Graduate → 0

Usage:
    pip install ucimlrepo
    python -m src.data.load_uci_dropout [--output data/raw]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

OUTPUT_DIR = Path("data/raw")

# ── Mapping colonnes UCI → schéma interne ────────────────────────────────────

UCI_RENAME = {
    "Gender":                               "gender_raw",
    "Age at enrollment":                    "age",
    "Admission grade":                      "admission_grade",
    "Scholarship holder":                   "scholarship",
    "Daytime/evening attendance":           "daytime",
    "Curricular units 1st sem (grade)":     "sem1_grade",
    "Curricular units 1st sem (approved)":  "sem1_approved",
    "Curricular units 1st sem (enrolled)":  "sem1_enrolled",
    "Curricular units 1st sem (evaluations)": "sem1_evals",
    "Curricular units 2nd sem (grade)":     "sem2_grade",
    "Curricular units 2nd sem (approved)":  "sem2_approved",
    "Curricular units 2nd sem (enrolled)":  "sem2_enrolled",
    "Curricular units 2nd sem (evaluations)": "sem2_evals",
    "Course":                               "course_raw",
    "Target":                               "target_raw",
    "Debtor":                               "has_debt",
    "Tuition fees up to date":              "fees_up_to_date",
    "International":                        "international",
}

# UCI course IDs → cohort_id (simplifié : 4 groupes)
COURSE_TO_COHORT: dict[int, int] = {}  # rempli dynamiquement


# ── Téléchargement ────────────────────────────────────────────────────────────

def fetch_raw() -> pd.DataFrame:
    """Télécharge le dataset UCI 697 via ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo requis : pip install ucimlrepo"
        )

    logger.info("Téléchargement UCI dataset 697...")
    dataset = fetch_ucirepo(id=697)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    logger.success(f"UCI chargé : {len(df):,} étudiants, {df.shape[1]} colonnes")
    return df


# ── Transformation ────────────────────────────────────────────────────────────

def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne (students_df, weekly_features_df).
    weekly_features contient 2 snapshots par étudiant (sem 1 et sem 2).
    """
    df = df.copy()

    # Renommage colonnes disponibles
    rename = {k: v for k, v in UCI_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)

    # ── Étudiants ─────────────────────────────────────────────────────────────
    df["student_id"] = range(1, len(df) + 1)

    df["gender"] = df["gender_raw"].map({1: "M", 0: "F"}).fillna("Other")
    df["scholarship"] = df["scholarship"].astype(bool)
    df["has_job"] = (~df["daytime"].astype(bool))   # soir → emploi probable

    # Normalise note d'admission 0-200 → 0-20
    df["entry_grade"] = (df["admission_grade"] / 10).clip(0, 20).round(2)

    # Cohort : 4 groupes de cours
    unique_courses = sorted(df["course_raw"].unique())
    n = len(unique_courses)
    course_map = {c: (i * 4 // n) + 1 for i, c in enumerate(unique_courses)}
    df["cohort_id"] = df["course_raw"].map(course_map)

    # Statut final
    status_map = {
        "Dropout":  "dropped_out",
        "Graduate": "graduated",
        "Enrolled": "enrolled",
    }
    df["status"]         = df["target_raw"].map(status_map).fillna("enrolled")
    df["label_dropout"]  = (df["target_raw"] == "Dropout").astype(int)

    students = df[[
        "student_id", "cohort_id", "gender", "age", "scholarship",
        "has_job", "entry_grade", "status", "label_dropout",
    ]].copy()
    students["email"] = students["student_id"].apply(
        lambda i: f"uci_{i}@edtech-real.eu"
    )

    # ── Weekly features (2 snapshots par étudiant) ────────────────────────────
    rows = []
    ref_date = pd.Timestamp("2022-09-05")

    for _, s in df.iterrows():
        for sem, week_offset in [(1, 16), (2, 32)]:
            g    = s.get(f"sem{sem}_grade",    0) or 0
            app  = s.get(f"sem{sem}_approved", 0) or 0
            enr  = s.get(f"sem{sem}_enrolled", 1) or 1
            evs  = s.get(f"sem{sem}_evals",    0) or 0

            score       = float(np.clip(g * 5, 0, 100))          # 0-20 → 0-100
            pass_rate   = float(np.clip(app / max(enr, 1), 0, 1))
            login_count = int(min(evs, 15))                       # proxy engagement
            time_min    = login_count * 45.0

            rows.append({
                "student_id":         int(s["student_id"]),
                "week_start":         ref_date + pd.Timedelta(weeks=week_offset),
                "login_count":        login_count,
                "total_time_min":     time_min,
                "videos_watched":     int(login_count * 0.4),
                "quiz_attempts":      int(evs),
                "quiz_pass_rate":     round(pass_rate, 4),
                "avg_score":          round(score, 2),
                "forum_posts":        0,
                "assignments_on_time": int(app),
                "assignments_late":    max(0, int(enr) - int(app)),
                "label_dropout":      int(s["label_dropout"]),
            })

    weekly = pd.DataFrame(rows)
    return students, weekly


# ── Export ────────────────────────────────────────────────────────────────────

def main(output_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_raw = fetch_raw()
    students, weekly = transform(df_raw)

    s_path = out / "uci_students.csv"
    w_path = out / "uci_dropout.csv"
    students.to_csv(s_path, index=False)
    weekly.to_csv(w_path, index=False)

    n_drop = students["label_dropout"].sum()
    logger.success(f"UCI students  -> {s_path}  ({len(students):,} lignes)")
    logger.success(f"UCI weekly    -> {w_path}  ({len(weekly):,} lignes)")
    logger.info(
        f"Taux décrochage : {n_drop/len(students)*100:.1f}%  "
        f"({n_drop}/{len(students)})"
    )
    return students, weekly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()
    main(args.output)
