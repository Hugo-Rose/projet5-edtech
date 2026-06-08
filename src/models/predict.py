"""
Inférence : scoring batch + enregistrement des prédictions en base.

Usage:
    python -m src.models.predict [--features data/features/weekly_features.parquet]
                                 [--run-id <mlflow_run_id>]
                                 [--week 2025-01-06]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

import mlflow
import mlflow.xgboost
from src.data.db import engine
from src.models.constants import FEATURE_COLS, MODELS_DIR_STR, TARGET
from src.models.evaluate import score_dataframe

MODELS_DIR = Path(MODELS_DIR_STR)

MLFLOW_URI = "http://localhost:5000"


# ── Chargement modèle ─────────────────────────────────────────────────────────

def load_artifacts(run_id: str | None = None) -> tuple:
    """
    Charge le modèle XGBoost, le preprocesseur et les métadonnées.
    Si run_id est None, charge depuis data/models/ (mode offline).
    """
    if run_id:
        mlflow.set_tracking_uri(MLFLOW_URI)
        model = mlflow.xgboost.load_model(f"runs:/{run_id}/xgb_model")
        client = mlflow.MlflowClient()
        local = client.download_artifacts(run_id, "preprocessor.joblib")
        preprocessor = joblib.load(local)
        meta_path = client.download_artifacts(run_id, "model_meta.json")
        meta = json.loads(Path(meta_path).read_text())
    else:
        model_path = MODELS_DIR / "preprocessor.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Aucun modèle local trouvé dans {MODELS_DIR}. "
                "Lancez d'abord src.models.train_dropout."
            )
        preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
        meta = json.loads((MODELS_DIR / "model_meta.json").read_text())
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(MODELS_DIR / "model.xgb"))
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model._Booster = booster

    threshold = meta.get("threshold", 0.5)
    model_version = meta.get("run_id", run_id or "local")
    return model, preprocessor, threshold, model_version


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_batch(
    df: pd.DataFrame,
    model,
    preprocessor,
    threshold: float,
    model_version: str,
) -> pd.DataFrame:
    """Calcule dropout_prob + risk_label pour chaque ligne."""
    X = df[FEATURE_COLS].copy()
    X_pp = preprocessor.transform(X)
    probas = model.predict_proba(X_pp)[:, 1]

    result = df[["student_id", "week_start"]].copy()
    result["dropout_prob"]  = np.round(probas, 4)
    result["model_version"] = model_version
    result = score_dataframe(result, "dropout_prob", threshold)
    return result


# ── Persistance en base ───────────────────────────────────────────────────────

def save_predictions(preds: pd.DataFrame) -> None:
    preds_db = preds[["student_id", "model_version", "dropout_prob", "risk_label"]].copy()
    preds_db["predicted_at"] = pd.Timestamp.now()

    preds_db.to_sql(
        "predictions", engine,
        if_exists="append", index=False,
        chunksize=10_000, method="multi",
    )
    logger.success(f"  {len(preds_db):,} prédictions sauvegardées en base.")


def update_weekly_features_risk(preds: pd.DataFrame) -> None:
    """Met à jour dropout_risk_score dans weekly_features."""
    with engine.connect() as conn:
        for _, row in preds.iterrows():
            conn.execute(text("""
                UPDATE weekly_features
                   SET dropout_risk_score = :score
                 WHERE student_id = :sid AND week_start = :week
            """), {
                "score": float(row["dropout_prob"]),
                "sid":   int(row["student_id"]),
                "week":  row["week_start"],
            })
        conn.commit()


def generate_alerts(preds: pd.DataFrame) -> None:
    """Crée des alertes pour les étudiants high/critical."""
    alerts = preds[preds["risk_label"].isin(["high", "critical"])].copy()
    if alerts.empty:
        return

    alerts_db = pd.DataFrame({
        "student_id":   alerts["student_id"],
        "alert_type":   "dropout_risk",
        "severity":     alerts["risk_label"].map(
                            {"high": "warning", "critical": "critical"}
                        ),
        "message":      alerts.apply(
                            lambda r: f"Risque de décrochage {r['risk_label'].upper()} "
                                      f"(prob={r['dropout_prob']:.0%}) — semaine {r['week_start']}",
                            axis=1,
                        ),
        "triggered_at": pd.Timestamp.now(),
    })
    alerts_db.to_sql(
        "alerts", engine,
        if_exists="append", index=False,
        chunksize=5_000, method="multi",
    )
    logger.info(f"  {len(alerts_db):,} alertes générées "
                f"({(alerts['risk_label']=='critical').sum()} critiques).")


# ── Main ──────────────────────────────────────────────────────────────────────

def predict(
    features_path: str = "data/features/weekly_features.parquet",
    run_id: str | None = None,
    week: str | None = None,
    save_to_db: bool = True,
) -> pd.DataFrame:
    logger.info("=== Inférence — Prédiction Décrochage ===")

    df = pd.read_parquet(features_path)

    if week:
        df = df[df["week_start"] == pd.Timestamp(week)]
        logger.info(f"Scoring semaine {week} : {len(df):,} étudiants")
    else:
        last_week = df["week_start"].max()
        df = df[df["week_start"] == last_week]
        logger.info(f"Scoring dernière semaine ({last_week}) : {len(df):,} étudiants")

    model, preprocessor, threshold, model_version = load_artifacts(run_id)
    logger.info(f"Modèle chargé (version={model_version}, seuil={threshold:.2f})")

    preds = score_batch(df, model, preprocessor, threshold, model_version)

    dist = preds["risk_label"].value_counts()
    for label, count in dist.items():
        logger.info(f"  {label:<10} : {count:>6,} ({count/len(preds)*100:.1f}%)")

    if save_to_db:
        save_predictions(preds)
        update_weekly_features_risk(preds)
        generate_alerts(preds)

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/weekly_features.parquet")
    parser.add_argument("--run-id",   default=None)
    parser.add_argument("--week",     default=None, help="YYYY-MM-DD")
    parser.add_argument("--no-db",    action="store_true")
    args = parser.parse_args()
    predict(
        features_path=args.features,
        run_id=args.run_id,
        week=args.week,
        save_to_db=not args.no_db,
    )
