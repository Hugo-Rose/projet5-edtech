"""
Rapport de data drift et target drift avec Evidently AI.

Compare la semaine courante vs une baseline (4 semaines glissantes précédentes).
Exporte le rapport HTML et logue le résumé dans MLflow.

Usage:
    python -m src.monitoring.drift_report [--week 2025-01-06] [--baseline-weeks 4]
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently import ColumnMapping
    EVIDENTLY_OK = True
except ImportError:
    EVIDENTLY_OK = False
    logger.warning("evidently non installé — fallback scipy KS-test")

from src.data.db import get_engine
from src.models.constants import FEATURE_COLS

REPORTS_DIR = Path("data/reports")
FEATURE_COLS_NO_STATIC = [
    c for c in FEATURE_COLS
    if c not in ("entry_grade", "has_job", "scholarship",
                 "distance_km", "age", "gender_enc")
]


# ── Chargement données ────────────────────────────────────────────────────────

def load_features_window(week_start: date, n_weeks: int) -> pd.DataFrame:
    engine = get_engine()
    sql = """
        SELECT wf.*, s.cohort_id
        FROM weekly_features wf
        JOIN students s ON s.student_id = wf.student_id
        WHERE wf.week_start >= %(start)s AND wf.week_start < %(end)s
    """
    start = week_start - timedelta(weeks=n_weeks)
    df = pd.read_sql(sql, engine, params={"start": start, "end": week_start})
    return df


def load_current_week(week_start: date | None = None) -> tuple[pd.DataFrame, date]:
    engine = get_engine()
    if week_start is None:
        row = pd.read_sql(
            "SELECT MAX(week_start) AS w FROM weekly_features", engine
        )
        week_start = row["w"].iloc[0]
    sql = """
        SELECT wf.*, s.cohort_id
        FROM weekly_features wf
        JOIN students s ON s.student_id = wf.student_id
        WHERE wf.week_start = %(w)s
    """
    df = pd.read_sql(sql, engine, params={"w": week_start})
    return df, week_start


# ── Calcul drift fallback (scipy KS) ─────────────────────────────────────────

def _ks_drift(ref: pd.DataFrame, cur: pd.DataFrame,
              cols: list[str], alpha: float = 0.05) -> dict[str, bool]:
    from scipy.stats import ks_2samp
    drift = {}
    for col in cols:
        r = ref[col].dropna()
        c = cur[col].dropna()
        if len(r) < 5 or len(c) < 5:
            drift[col] = False
            continue
        _, pval = ks_2samp(r, c)
        drift[col] = pval < alpha
    return drift


# ── Rapport Evidently ─────────────────────────────────────────────────────────

def _run_evidently(ref: pd.DataFrame, cur: pd.DataFrame,
                   out_path: Path) -> dict:
    col_map = ColumnMapping(
        numerical_features=FEATURE_COLS_NO_STATIC,
        target="label_dropout",
    )
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])
    report.run(reference_data=ref, current_data=cur,
               column_mapping=col_map)
    report.save_html(str(out_path))

    result = report.as_dict()
    metrics = result.get("metrics", [])

    drift_cols: dict[str, bool] = {}
    for m in metrics:
        r = m.get("result", {})
        if "drift_by_columns" in r:
            for col, info in r["drift_by_columns"].items():
                drift_cols[col] = info.get("drift_detected", False)

    return drift_cols


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_drift_report(
    week_str:       str | None = None,
    baseline_weeks: int        = 4,
    mlflow_uri:     str        = "http://localhost:5000",
    experiment:     str        = "edtech-monitoring",
) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    current_df, week_start = load_current_week(
        date.fromisoformat(week_str) if week_str else None
    )
    reference_df = load_features_window(week_start, baseline_weeks)

    if current_df.empty or reference_df.empty:
        logger.warning("Données insuffisantes pour le rapport drift.")
        return {}

    week_label = week_start.strftime("%Y-W%V") if hasattr(week_start, "strftime") \
                 else str(week_start)
    out_path = REPORTS_DIR / f"drift_{week_label}.html"

    logger.info(f"Drift report : référence {len(reference_df):,} pts  "
                f"| courant {len(current_df):,} pts")

    if EVIDENTLY_OK:
        drift_cols = _run_evidently(
            reference_df[FEATURE_COLS_NO_STATIC + ["label_dropout"]].dropna(),
            current_df  [FEATURE_COLS_NO_STATIC + ["label_dropout"]].dropna(),
            out_path,
        )
    else:
        drift_cols = _ks_drift(reference_df, current_df, FEATURE_COLS_NO_STATIC)
        _export_fallback_html(drift_cols, out_path, week_label)

    n_drift = sum(drift_cols.values())
    pct_drift = round(n_drift / max(len(drift_cols), 1) * 100, 1)
    drifted = [c for c, d in drift_cols.items() if d]

    logger.info(f"  Features en drift : {n_drift}/{len(drift_cols)} ({pct_drift}%)")
    if drifted:
        logger.warning(f"  En drift : {drifted}")
    logger.success(f"  Rapport → {out_path}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"drift_{week_label}"):
            mlflow.log_metric("pct_features_drift", pct_drift)
            mlflow.log_metric("n_features_drift",   n_drift)
            mlflow.log_param("week",                week_label)
            mlflow.log_param("baseline_weeks",      baseline_weeks)
            mlflow.log_artifact(str(out_path))
    except Exception as e:
        logger.warning(f"MLflow indisponible : {e}")

    return {
        "week":       week_label,
        "pct_drift":  pct_drift,
        "n_drift":    n_drift,
        "drifted":    drifted,
        "report_path": str(out_path),
    }


def _export_fallback_html(drift_cols: dict[str, bool],
                          out_path: Path, week_label: str) -> None:
    rows = "".join(
        f"<tr><td>{col}</td>"
        f"<td style='color:{'red' if d else 'green'}'>"
        f"{'DRIFT' if d else 'OK'}</td></tr>"
        for col, d in drift_cols.items()
    )
    html = f"""<!DOCTYPE html><html><head>
    <title>Drift Report {week_label}</title>
    <style>body{{font-family:sans-serif;padding:2rem}}
    table{{border-collapse:collapse;width:60%}}
    td,th{{border:1px solid #ccc;padding:.5rem}}</style></head><body>
    <h1>Data Drift Report — {week_label}</h1>
    <p>{sum(drift_cols.values())}/{len(drift_cols)} features en drift</p>
    <table><tr><th>Feature</th><th>Statut</th></tr>{rows}</table>
    </body></html>"""
    out_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week",           default=None)
    parser.add_argument("--baseline-weeks", type=int, default=4)
    parser.add_argument("--mlflow-uri",     default="http://localhost:5000")
    args = parser.parse_args()
    run_drift_report(args.week, args.baseline_weeks, args.mlflow_uri)
