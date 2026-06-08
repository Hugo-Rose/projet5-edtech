"""
Rapport d'équité (fairness) des prédictions de décrochage.

Pour chaque attribut protégé (gender, age_group, scholarship) :
  - Demographic Parity : écart de probabilité moyenne prédite entre groupes
  - Equal Opportunity : écart de TPR (recall) entre groupes
  - Predictive Equality : écart de FPR entre groupes
  - FNR : taux de faux négatifs (décrocheurs manqués)

Alerte si écart entre groupes > FAIRNESS_THRESHOLD (10 %).

Usage:
    python -m src.monitoring.fairness_report [--week 2025-01-06]
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.data.db import get_engine

REPORTS_DIR = Path("data/reports")
FAIRNESS_THRESHOLD = 0.10   # 10% d'écart max entre groupes

PROTECTED_ATTRS = ["gender", "age_group", "scholarship"]


# ── Chargement ────────────────────────────────────────────────────────────────

def load_fairness_data(week_start: date | None = None) -> pd.DataFrame:
    """Fusionne prédictions + weekly_features + students pour la semaine donnée."""
    engine = get_engine()

    week_clause = (
        "AND wf.week_start = %(w)s" if week_start
        else "AND wf.week_start = (SELECT MAX(week_start) FROM weekly_features)"
    )
    sql = f"""
        SELECT
            p.student_id,
            p.dropout_prob,
            p.risk_label,
            wf.label_dropout,
            wf.week_start,
            s.gender,
            s.age,
            s.scholarship,
            s.has_job
        FROM predictions p
        JOIN weekly_features wf ON wf.student_id = p.student_id
                                {week_clause}
        JOIN students s ON s.student_id = p.student_id
        WHERE p.predicted_at = (
            SELECT MAX(p2.predicted_at) FROM predictions p2
            WHERE p2.student_id = p.student_id
        )
    """
    params = {"w": week_start} if week_start else {}
    df = pd.read_sql(sql, engine, params=params)
    return df


# ── Préparation ───────────────────────────────────────────────────────────────

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins   = [0, 21, 25, 30, 99]
    labels = ["≤21", "22-25", "26-30", "30+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    return df


def add_threshold_prediction(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df["y_pred"] = (df["dropout_prob"] >= threshold).astype(int)
    return df


# ── Calcul métriques par groupe ───────────────────────────────────────────────

def _safe_rate(num: int, den: int) -> float:
    return round(num / den, 4) if den > 0 else float("nan")


def compute_group_metrics(
    df: pd.DataFrame,
    group_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    df = df.dropna(subset=[group_col, "dropout_prob"])
    df = add_threshold_prediction(df, threshold)

    for group_val, grp in df.groupby(group_col, observed=True):
        n = len(grp)
        avg_prob = round(float(grp["dropout_prob"].mean()), 4)

        has_truth = grp["label_dropout"].notna().any()
        if has_truth:
            y_true = grp["label_dropout"].fillna(0).astype(int)
            y_pred = grp["y_pred"]
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tpr = _safe_rate(tp, tp + fn)   # recall
            fpr = _safe_rate(fp, fp + tn)
            fnr = _safe_rate(fn, fn + tp)
        else:
            tp = fp = tn = fn = 0
            tpr = fpr = fnr = float("nan")

        rows.append({
            "group":     group_col,
            "value":     str(group_val),
            "n":         n,
            "avg_prob":  avg_prob,
            "tpr":       tpr,
            "fpr":       fpr,
            "fnr":       fnr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    return pd.DataFrame(rows)


def compute_all_groups(df: pd.DataFrame,
                       threshold: float = 0.5) -> dict[str, pd.DataFrame]:
    results = {}
    for attr in PROTECTED_ATTRS:
        if attr not in df.columns:
            continue
        gdf = compute_group_metrics(df, attr, threshold)
        if not gdf.empty:
            results[attr] = gdf
    return results


# ── Détection d'alertes ───────────────────────────────────────────────────────

def detect_violations(
    group_results: dict[str, pd.DataFrame],
    threshold: float = FAIRNESS_THRESHOLD,
) -> list[dict]:
    violations = []
    for attr, gdf in group_results.items():
        # Demographic Parity
        prob_range = gdf["avg_prob"].max() - gdf["avg_prob"].min()
        if prob_range > threshold:
            violations.append({
                "attr":    attr,
                "metric":  "Demographic Parity",
                "gap":     round(float(prob_range), 4),
                "details": gdf[["value", "avg_prob"]].to_dict("records"),
            })

        # Equalized Odds (FPR gap)
        fpr_vals = gdf["fpr"].dropna()
        if len(fpr_vals) > 1:
            fpr_gap = fpr_vals.max() - fpr_vals.min()
            if fpr_gap > threshold:
                violations.append({
                    "attr":    attr,
                    "metric":  "Predictive Equality (FPR)",
                    "gap":     round(float(fpr_gap), 4),
                    "details": gdf[["value", "fpr"]].to_dict("records"),
                })

        # Equal Opportunity (TPR gap)
        tpr_vals = gdf["tpr"].dropna()
        if len(tpr_vals) > 1:
            tpr_gap = tpr_vals.max() - tpr_vals.min()
            if tpr_gap > threshold:
                violations.append({
                    "attr":    attr,
                    "metric":  "Equal Opportunity (TPR)",
                    "gap":     round(float(tpr_gap), 4),
                    "details": gdf[["value", "tpr"]].to_dict("records"),
                })

    return violations


# ── Export HTML ───────────────────────────────────────────────────────────────

def _table_html(gdf: pd.DataFrame, attr: str) -> str:
    display_cols = ["value", "n", "avg_prob", "tpr", "fpr", "fnr"]
    cols = [c for c in display_cols if c in gdf.columns]
    headers = "".join(f"<th>{c}</th>" for c in cols)
    rows_html = ""
    for _, row in gdf[cols].iterrows():
        cells = ""
        for c in cols:
            v = row[c]
            if isinstance(v, float) and not np.isnan(v):
                v = f"{v:.4f}"
            cells += f"<td>{v}</td>"
        rows_html += f"<tr>{cells}</tr>"
    return (
        f"<h3>Attribut : {attr}</h3>"
        f"<table><thead><tr>{headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def export_html(
    group_results: dict[str, pd.DataFrame],
    violations: list[dict],
    out_path: Path,
    week_label: str,
) -> None:
    viol_html = ""
    if violations:
        items = "".join(
            f"<li><b>{v['attr']}</b> — {v['metric']} : "
            f"écart {v['gap']:.1%} &gt; seuil {FAIRNESS_THRESHOLD:.0%}</li>"
            for v in violations
        )
        viol_html = (
            f"<div class='alert'><strong>⚠ {len(violations)} violation(s) d'équité "
            f"détectée(s)</strong><ul>{items}</ul></div>"
        )
    else:
        viol_html = "<div class='ok'>✔ Aucune violation d'équité détectée.</div>"

    tables = "".join(_table_html(gdf, attr)
                     for attr, gdf in group_results.items())
    html = f"""<!DOCTYPE html><html lang="fr"><head>
    <meta charset="utf-8">
    <title>Rapport Équité — {week_label}</title>
    <style>
      body{{font-family:sans-serif;max-width:900px;margin:2rem auto;color:#333}}
      h1{{color:#2c3e50}} h3{{color:#34495e;margin-top:2rem}}
      table{{border-collapse:collapse;width:100%;margin:1rem 0}}
      th{{background:#2c3e50;color:#fff;padding:.6rem .8rem;text-align:left}}
      td{{border:1px solid #ddd;padding:.5rem .8rem}}
      tr:nth-child(even){{background:#f9f9f9}}
      .alert{{background:#fff3cd;border-left:4px solid #ffc107;padding:1rem;margin:1rem 0}}
      .ok{{background:#d4edda;border-left:4px solid #28a745;padding:1rem;margin:1rem 0}}
      ul{{margin:.5rem 0}}
    </style></head><body>
    <h1>Rapport d'Équité des Prédictions</h1>
    <p>Semaine : <strong>{week_label}</strong> | Seuil d'alerte : {FAIRNESS_THRESHOLD:.0%}</p>
    {viol_html}
    {tables}
    </body></html>"""
    out_path.write_text(html, encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_fairness_report(
    week_str:  str | None = None,
    threshold: float      = 0.5,
) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_fairness_data(
        date.fromisoformat(week_str) if week_str else None
    )
    if df.empty:
        logger.warning("Aucune donnée pour le rapport fairness.")
        return {"violations": [], "report_path": None}

    df = add_age_group(df)
    week_label = str(df["week_start"].iloc[0]) if "week_start" in df.columns else "latest"
    out_path = REPORTS_DIR / f"fairness_{week_label}.html"

    group_results = compute_all_groups(df, threshold)
    violations    = detect_violations(group_results)
    export_html(group_results, violations, out_path, week_label)

    if violations:
        for v in violations:
            logger.warning(
                f"FAIRNESS VIOLATION — {v['attr']} / {v['metric']} : "
                f"écart={v['gap']:.1%}"
            )
    else:
        logger.success("Aucune violation d'équité détectée.")

    logger.success(f"Rapport équité → {out_path}")
    return {
        "violations":  violations,
        "report_path": str(out_path),
        "groups":      {k: v.to_dict("records") for k, v in group_results.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week",      default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_fairness_report(args.week, args.threshold)
