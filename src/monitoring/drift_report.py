"""
Rapport de data drift avec Evidently AI — ColumnDriftMetric par feature.

Compare la semaine courante vs la baseline (8 premières semaines du dataset).
Alerte si drift score > 0.15.
Exporte le rapport HTML et logue le résumé dans MLflow.

Usage:
    python -m src.monitoring.drift_report [--week 2025-01-06] [--baseline-weeks 8]
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

import mlflow

try:
    from evidently import ColumnMapping
    from evidently.metrics import ColumnDriftMetric
    from evidently.report import Report
    EVIDENTLY_OK = True
except ImportError:
    EVIDENTLY_OK = False
    logger.warning("evidently non installé — fallback scipy KS-test")

from src.data.db import get_engine
from src.models.constants import FEATURE_COLS

REPORTS_DIR = Path("data/reports")
DRIFT_ALERT_THRESHOLD = 0.15

FEATURE_COLS_NO_STATIC = [
    c for c in FEATURE_COLS
    if c not in ("entry_grade", "has_job", "scholarship",
                 "distance_km", "age", "gender_enc")
]


# ── Chargement données ────────────────────────────────────────────────────────

def load_baseline(n_weeks: int = 8) -> pd.DataFrame:
    """Charge les n_weeks premières semaines du dataset comme référence fixe."""
    sql = """
        SELECT wf.*, s.cohort_id
        FROM weekly_features wf
        JOIN students s ON s.student_id = wf.student_id
        WHERE wf.week_start IN (
            SELECT DISTINCT week_start FROM weekly_features
            ORDER BY week_start ASC
            LIMIT %(n)s
        )
    """
    return pd.read_sql(sql, get_engine(), params={"n": n_weeks})


def load_current_week(week_start: date | None = None) -> tuple[pd.DataFrame, date]:
    if week_start is None:
        row = pd.read_sql(
            "SELECT MAX(week_start) AS w FROM weekly_features", get_engine()
        )
        week_start = row["w"].iloc[0]
    sql = """
        SELECT wf.*, s.cohort_id
        FROM weekly_features wf
        JOIN students s ON s.student_id = wf.student_id
        WHERE wf.week_start = %(w)s
    """
    df = pd.read_sql(sql, get_engine(), params={"w": week_start})
    return df, week_start


# ── Calcul drift fallback (scipy KS) ─────────────────────────────────────────

def _ks_drift(ref: pd.DataFrame, cur: pd.DataFrame,
              cols: list[str], alpha: float = 0.05) -> dict[str, dict]:
    """Retourne {col: {"drift_detected": bool, "drift_score": float}}."""
    from scipy.stats import ks_2samp
    result = {}
    for col in cols:
        r = ref[col].dropna()
        c = cur[col].dropna()
        if len(r) < 5 or len(c) < 5:
            result[col] = {"drift_detected": False, "drift_score": 0.0}
            continue
        stat, pval = ks_2samp(r, c)
        result[col] = {
            "drift_detected": bool(pval < alpha),
            "drift_score":    round(float(stat), 4),
        }
    return result


# ── Rapport Evidently ─────────────────────────────────────────────────────────

def _run_evidently(ref: pd.DataFrame, cur: pd.DataFrame,
                   out_path: Path) -> dict[str, dict]:
    """Retourne {col: {"drift_detected": bool, "drift_score": float}}."""
    col_map = ColumnMapping(numerical_features=FEATURE_COLS_NO_STATIC)
    metrics = [ColumnDriftMetric(column_name=col) for col in FEATURE_COLS_NO_STATIC]
    report  = Report(metrics=metrics)
    report.run(reference_data=ref, current_data=cur, column_mapping=col_map)
    report.save_html(str(out_path))

    drift_info: dict[str, dict] = {}
    for m in report.as_dict().get("metrics", []):
        r = m.get("result", {})
        col = r.get("column_name")
        if col:
            drift_info[col] = {
                "drift_detected": bool(r.get("drift_detected", False)),
                "drift_score":    round(float(r.get("drift_score", 0.0)), 4),
            }
    return drift_info


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_drift_report(
    week_str:       str | None = None,
    baseline_weeks: int        = 8,
    mlflow_uri:     str        = "http://localhost:5000",
    experiment:     str        = "edtech-monitoring",
) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    current_df, week_start = load_current_week(
        date.fromisoformat(week_str) if week_str else None
    )
    reference_df = load_baseline(baseline_weeks)

    if current_df.empty or reference_df.empty:
        logger.warning("Données insuffisantes pour le rapport drift.")
        return {}

    week_label = week_start.strftime("%Y-W%V") if hasattr(week_start, "strftime") \
                 else str(week_start)
    out_path = REPORTS_DIR / f"drift_{week_label}.html"

    logger.info(f"Drift report : référence {len(reference_df):,} pts "
                f"| courant {len(current_df):,} pts")

    ref_clean = reference_df[FEATURE_COLS_NO_STATIC].dropna()
    cur_clean = current_df  [FEATURE_COLS_NO_STATIC].dropna()

    if EVIDENTLY_OK:
        drift_info = _run_evidently(ref_clean, cur_clean, out_path)
    else:
        drift_info = _ks_drift(reference_df, current_df, FEATURE_COLS_NO_STATIC)
        _export_fallback_html(drift_info, out_path, week_label)

    n_drift    = sum(1 for v in drift_info.values() if v["drift_detected"])
    pct_drift  = round(n_drift / max(len(drift_info), 1) * 100, 1)
    scores     = [v["drift_score"] for v in drift_info.values()]
    avg_score  = round(sum(scores) / max(len(scores), 1), 4)
    drifted    = [c for c, v in drift_info.items() if v["drift_detected"]]
    alerted    = [c for c, v in drift_info.items()
                  if v["drift_score"] > DRIFT_ALERT_THRESHOLD]

    logger.info(f"  Features en drift : {n_drift}/{len(drift_info)} ({pct_drift}%)")
    logger.info(f"  Score moyen drift : {avg_score:.4f}")
    if alerted:
        logger.warning(
            f"  ALERTE — {len(alerted)} feature(s) avec score > {DRIFT_ALERT_THRESHOLD} : "
            f"{alerted}"
        )
    logger.success(f"  Rapport → {out_path}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"drift_{week_label}"):
            mlflow.log_metric("pct_features_drift",  pct_drift)
            mlflow.log_metric("n_features_drift",     n_drift)
            mlflow.log_metric("avg_drift_score",      avg_score)
            mlflow.log_param("week",                  week_label)
            mlflow.log_param("baseline_weeks",        baseline_weeks)
            mlflow.log_artifact(str(out_path))
    except Exception as e:
        logger.warning(f"MLflow indisponible : {e}")

    return {
        "week":        week_label,
        "pct_drift":   pct_drift,
        "n_drift":     n_drift,
        "avg_score":   avg_score,
        "drifted":     drifted,
        "alerted":     alerted,
        "report_path": str(out_path),
    }


def _export_fallback_html(drift_info: dict[str, dict],
                          out_path: Path, week_label: str) -> None:
    rows = "".join(
        f"<tr><td>{col}</td>"
        f"<td style='color:{'red' if v['drift_detected'] else 'green'}'>"
        f"{'DRIFT' if v['drift_detected'] else 'OK'}</td>"
        f"<td style='color:{'orange' if v['drift_score'] > DRIFT_ALERT_THRESHOLD else 'inherit'}'>"
        f"{v['drift_score']:.4f}</td></tr>"
        for col, v in drift_info.items()
    )
    n_drift = sum(1 for v in drift_info.values() if v["drift_detected"])
    html = f"""<!DOCTYPE html><html><head>
    <title>Drift Report {week_label}</title>
    <style>body{{font-family:sans-serif;padding:2rem}}
    table{{border-collapse:collapse;width:70%}}
    td,th{{border:1px solid #ccc;padding:.5rem}}</style></head><body>
    <h1>Data Drift Report — {week_label}</h1>
    <p>{n_drift}/{len(drift_info)} features en drift | seuil alerte : {DRIFT_ALERT_THRESHOLD}</p>
    <table><tr><th>Feature</th><th>Statut</th><th>Score</th></tr>{rows}</table>
    </body></html>"""
    out_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week",           default=None)
    parser.add_argument("--baseline-weeks", type=int, default=8)
    parser.add_argument("--mlflow-uri",     default="http://localhost:5000")
    args = parser.parse_args()
    run_drift_report(args.week, args.baseline_weeks, args.mlflow_uri)
