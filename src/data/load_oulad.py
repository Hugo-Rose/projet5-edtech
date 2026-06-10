"""
Loader — OULAD (Open University Learning Analytics Dataset)
Hugging Face : IxaSuCo/open-university-learning-analytics

Tables utilisées :
  - studentVle   : clics VLE par jour → événements LMS hebdomadaires
  - studentInfo  : démographie + résultats finaux → students
  - studentAssessment + assessments : scores → quiz_pass_rate, avg_score

Schéma cible :
  data/raw/oulad_students.csv
  data/raw/oulad_weekly.parquet   (weekly_features)

Usage:
    pip install datasets huggingface_hub
    python -m src.data.load_oulad [--output data/raw] [--streaming]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

HF_DATASET = "IxaSuCo/open-university-learning-analytics"
OUTPUT_DIR = Path("data/raw")

# Colonnes studentInfo → schéma students
INFO_RENAME = {
    "id_student":          "student_id",
    "gender":              "gender",
    "age_band":            "age_band",
    "highest_education":   "education",
    "imd_band":            "imd_band",
    "num_of_prev_attempts": "prev_attempts",
    "studied_credits":     "studied_credits",
    "disability":          "disability",
    "final_result":        "final_result",
    "code_module":         "module",
    "code_presentation":   "presentation",
}

RESULT_MAP = {
    "Withdrawn":    "dropped_out",
    "Fail":         "dropped_out",
    "Pass":         "graduated",
    "Distinction":  "graduated",
}

AGE_BAND_MAP = {
    "0-35":   25,
    "35-55":  45,
    "55<=":   60,
}


# ── Chargement HuggingFace ────────────────────────────────────────────────────

def _load_hf_table(name: str, streaming: bool = False) -> pd.DataFrame:
    """Charge une table OULAD depuis HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets requis : pip install datasets")

    logger.info(f"Chargement HuggingFace : {HF_DATASET} / {name}...")
    try:
        ds = load_dataset(HF_DATASET, name=name, streaming=streaming, trust_remote_code=True)
        split = list(ds.keys())[0]
        if streaming:
            rows = list(ds[split].take(200_000))
            df = pd.DataFrame(rows)
        else:
            df = ds[split].to_pandas()
        logger.success(f"  {name}: {len(df):,} lignes")
        return df
    except Exception as e:
        logger.warning(f"Impossible de charger {name} depuis HF : {e}")
        return pd.DataFrame()


def load_all_tables(streaming: bool = False) -> dict[str, pd.DataFrame]:
    tables = {}
    for tbl in ["studentVle", "studentInfo", "studentAssessment", "assessments"]:
        df = _load_hf_table(tbl, streaming=streaming)
        if not df.empty:
            tables[tbl] = df
    return tables


# ── Transformation studentInfo → students ────────────────────────────────────

def build_students(info_df: pd.DataFrame) -> pd.DataFrame:
    df = info_df.rename(columns={k: v for k, v in INFO_RENAME.items()
                                 if k in info_df.columns})

    # Déduplique par étudiant (peut être inscrit à plusieurs modules)
    df = df.drop_duplicates(subset=["student_id"]).copy()
    df = df.reset_index(drop=True)

    df["gender"] = df["gender"].map({"M": "M", "F": "F"}).fillna("Other")
    df["age"]    = df["age_band"].map(AGE_BAND_MAP).fillna(28).astype(int)

    df["status"] = df["final_result"].map(RESULT_MAP).fillna("enrolled")
    df["label_dropout"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

    df["scholarship"]  = False
    df["has_job"]      = df.get("disability", "N") == "Y"  # proxy
    df["entry_grade"]  = np.clip(
        df.get("prev_attempts", 0).fillna(0) * (-1) + 14, 8, 20
    ).round(1)

    # cohort_id : basé sur la présentation (année)
    if "presentation" in df.columns:
        presentations = sorted(df["presentation"].dropna().unique())
        pmap = {p: (i % 4) + 1 for i, p in enumerate(presentations)}
        df["cohort_id"] = df["presentation"].map(pmap).fillna(1).astype(int)
    else:
        df["cohort_id"] = 1

    df["email"] = df["student_id"].apply(lambda i: f"oulad_{i}@open.ac.uk")

    return df[[
        "student_id", "cohort_id", "gender", "age", "scholarship",
        "has_job", "entry_grade", "status", "label_dropout", "email",
    ]]


# ── Transformation studentVle → weekly_features ──────────────────────────────

def build_weekly_from_vle(
    vle_df:   pd.DataFrame,
    info_df:  pd.DataFrame,
    assess_df: pd.DataFrame | None = None,
    sa_df:    pd.DataFrame | None  = None,
) -> pd.DataFrame:
    """
    Agrège les clics VLE en weekly_features.

    vle_df colonnes attendues :
        id_student, date (jours depuis début du cours), sum_click,
        code_module, code_presentation
    """
    if vle_df.empty:
        return pd.DataFrame()

    df = vle_df.copy()

    # Normalise les noms de colonnes
    df = df.rename(columns={"id_student": "student_id"})
    if "date" not in df.columns:
        logger.warning("Colonne 'date' absente de studentVle")
        return pd.DataFrame()

    df["date"] = pd.to_numeric(df["date"], errors="coerce").fillna(0)

    # Convertit les jours en semaines relatives au cours
    df["week_num"] = (df["date"] // 7).astype(int)

    # Référence temporelle : départ fictif au 2020-02-03 (présentation typique B)
    ref = pd.Timestamp("2020-02-03")
    df["week_start"] = df["week_num"].apply(
        lambda w: ref + pd.Timedelta(weeks=max(0, w))
    )

    # ── Agrégation hebdomadaire ───────────────────────────────────────────────
    agg = (
        df.groupby(["student_id", "week_start"])
        .agg(
            total_clicks = ("sum_click", "sum"),
            n_sessions   = ("sum_click", "count"),
        )
        .reset_index()
    )

    # Proxy des métriques LMS à partir des clics
    agg["login_count"]    = agg["n_sessions"].clip(0, 15).astype(int)
    agg["total_time_min"] = (agg["total_clicks"] * 0.5).clip(0, 600).round(1)
    agg["videos_watched"] = (agg["total_clicks"] / 20).clip(0, 20).round(0).astype(int)

    # ── Scores depuis studentAssessment ──────────────────────────────────────
    if sa_df is not None and not sa_df.empty and assess_df is not None and not assess_df.empty:
        try:
            sa = sa_df.rename(columns={"id_student": "student_id"}).copy()
            sa["score"] = pd.to_numeric(sa.get("score", 0), errors="coerce").fillna(0)

            if "id_assessment" in sa.columns and "id_assessment" in assess_df.columns:
                sa = sa.merge(
                    assess_df[["id_assessment", "date"]].rename(columns={"date": "assess_date"}),
                    on="id_assessment", how="left",
                )
                sa["assess_date"] = pd.to_numeric(sa["assess_date"], errors="coerce").fillna(0)
                sa["week_start"] = (sa["assess_date"] // 7).apply(
                    lambda w: ref + pd.Timedelta(weeks=max(0, int(w)))
                )

                score_agg = (
                    sa.groupby(["student_id", "week_start"])
                    .agg(
                        avg_score    = ("score", "mean"),
                        quiz_attempts = ("score", "count"),
                        quiz_passed  = ("score", lambda x: (x >= 40).sum()),
                    )
                    .reset_index()
                )
                score_agg["quiz_pass_rate"] = (
                    score_agg["quiz_passed"] / score_agg["quiz_attempts"].clip(1)
                ).clip(0, 1).round(4)

                agg = agg.merge(
                    score_agg[["student_id", "week_start",
                               "avg_score", "quiz_attempts", "quiz_pass_rate"]],
                    on=["student_id", "week_start"], how="left",
                )
        except Exception as e:
            logger.warning(f"Merge assessments échoué : {e}")

    # Valeurs par défaut si scores absents
    if "avg_score"     not in agg.columns: agg["avg_score"]     = np.nan
    if "quiz_attempts" not in agg.columns: agg["quiz_attempts"] = 0
    if "quiz_pass_rate" not in agg.columns: agg["quiz_pass_rate"] = np.nan

    # ── Jointure statut décrochage ────────────────────────────────────────────
    if not info_df.empty and "student_id" in info_df.columns:
        status_cols = ["student_id", "final_result"] if "final_result" in info_df.columns \
                       else ["student_id"]
        agg = agg.merge(
            info_df[status_cols].drop_duplicates("student_id"),
            on="student_id", how="left",
        )
        if "final_result" in agg.columns:
            agg["label_dropout"] = agg["final_result"].isin(
                ["Withdrawn", "Fail"]
            ).astype(int)
        else:
            agg["label_dropout"] = 0
    else:
        agg["label_dropout"] = 0

    # Colonnes manquantes à 0
    for col in ["forum_posts", "assignments_on_time", "assignments_late"]:
        if col not in agg.columns:
            agg[col] = 0

    return agg[[
        "student_id", "week_start", "login_count", "total_time_min",
        "videos_watched", "quiz_attempts", "quiz_pass_rate", "avg_score",
        "forum_posts", "assignments_on_time", "assignments_late", "label_dropout",
    ]]


# ── Pipeline principal ────────────────────────────────────────────────────────

def main(
    output_dir: str  = "data/raw",
    streaming:  bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = load_all_tables(streaming=streaming)

    if not tables:
        logger.error(
            "Aucune table OULAD chargée. Vérifiez votre connexion HuggingFace."
        )
        raise SystemExit(1)

    info_df   = tables.get("studentInfo",      pd.DataFrame())
    vle_df    = tables.get("studentVle",        pd.DataFrame())
    sa_df     = tables.get("studentAssessment", None)
    assess_df = tables.get("assessments",       None)

    students = build_students(info_df) if not info_df.empty else pd.DataFrame()
    weekly   = build_weekly_from_vle(vle_df, info_df, assess_df, sa_df)

    if not students.empty:
        s_path = out / "oulad_students.csv"
        students.to_csv(s_path, index=False)
        logger.success(f"OULAD students -> {s_path}  ({len(students):,} lignes)")

    if not weekly.empty:
        w_path = out / "oulad_weekly.parquet"
        weekly.to_parquet(w_path, index=False)
        sz = w_path.stat().st_size / 1e6
        logger.success(f"OULAD weekly   -> {w_path}  ({len(weekly):,} lignes, {sz:.1f} Mo)")

    return students, weekly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",    default="data/raw")
    parser.add_argument("--streaming", action="store_true",
                        help="Mode streaming HuggingFace (économise la RAM)")
    args = parser.parse_args()
    main(output_dir=args.output, streaming=args.streaming)
