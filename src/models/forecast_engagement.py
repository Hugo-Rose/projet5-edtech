"""
Prévision de l'engagement hebdomadaire par cohorte avec Prophet.

Pour chaque cohorte × métrique :
  - Agrège les données weekly_features (moyenne par étudiant actif)
  - Entraîne un modèle Prophet avec saisonnalité annuelle
  - Prédit sur `horizon` semaines
  - Valide par cross-validation glissante (optionnel)
  - Logue dans MLflow + exporte Parquet

Usage:
    python -m src.models.forecast_engagement
           [--horizon 4] [--cv] [--output data/features]
           [--experiment edtech-forecast] [--mlflow-uri http://localhost:5000]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

import mlflow
from src.data.db import get_engine
from src.models.forecast_store import save_forecasts_db, save_forecasts_parquet

warnings.filterwarnings("ignore")

# ── Métriques à prévoir ───────────────────────────────────────────────────────

METRICS = {
    "avg_logins":        "Connexions moyennes / étudiant / semaine",
    "avg_time_min":      "Temps connecté moyen (min) / étudiant / semaine",
    "avg_quiz_pass_rate": "Taux de réussite quiz moyen",
    "dropout_risk_p50":  "Médiane du score de risque de décrochage",
}

SQL_COHORT_WEEKLY = """
SELECT
    DATE_TRUNC('week', wf.week_start)::DATE          AS ds,
    AVG(wf.login_count)                              AS avg_logins,
    AVG(wf.total_time_min)                           AS avg_time_min,
    AVG(wf.quiz_pass_rate)                           AS avg_quiz_pass_rate,
    PERCENTILE_CONT(0.5)
        WITHIN GROUP (ORDER BY wf.dropout_risk_score) AS dropout_risk_p50,
    COUNT(wf.student_id)                             AS n_students
FROM weekly_features wf
JOIN students s ON s.student_id = wf.student_id
WHERE s.cohort_id = %(cohort_id)s
  AND wf.dropout_risk_score IS NOT NULL
GROUP BY 1
ORDER BY 1
"""

SQL_COHORTS = "SELECT cohort_id, name FROM cohorts ORDER BY cohort_id"


# ── Préparation données ───────────────────────────────────────────────────────

def load_cohort_series(cohort_id: int) -> pd.DataFrame:
    """Charge la série temporelle hebdo d'une cohorte depuis PostgreSQL."""
    engine = get_engine()
    df = pd.read_sql(SQL_COHORT_WEEKLY, engine, params={"cohort_id": cohort_id})
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def prepare_series(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Retourne un DataFrame Prophet (ds, y) sans NaN."""
    series = df[["ds", metric]].rename(columns={metric: "y"}).dropna()
    series = series[series["y"] > 0].reset_index(drop=True)
    return series


# ── Modèle Prophet ────────────────────────────────────────────────────────────

PROPHET_CONFIG = {
    "avg_logins": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "yearly_seasonality": True,
    },
    "avg_time_min": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "yearly_seasonality": True,
    },
    "avg_quiz_pass_rate": {
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 5.0,
        "yearly_seasonality": True,
    },
    "dropout_risk_p50": {
        "changepoint_prior_scale": 0.10,
        "seasonality_prior_scale": 10.0,
        "yearly_seasonality": True,
    },
}


def build_model(metric: str) -> Prophet:
    cfg = PROPHET_CONFIG.get(metric, {})
    return Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
        **cfg,
    )


def fit_and_forecast(
    series: pd.DataFrame,
    metric: str,
    horizon: int,
    cohort_id: int,
    cohort_name: str,
) -> tuple[Prophet, pd.DataFrame]:
    """Entraîne Prophet et retourne (modèle, forecast_df)."""
    model = build_model(metric)
    model.fit(series)

    future = model.make_future_dataframe(periods=horizon, freq="W")
    forecast = model.predict(future)

    last_hist = series["ds"].max()
    forecast["is_future"]   = forecast["ds"] > last_hist
    forecast["cohort_id"]   = cohort_id
    forecast["cohort_name"] = cohort_name
    forecast["metric"]      = metric

    return model, forecast[
        ["ds", "cohort_id", "cohort_name", "metric",
         "yhat", "yhat_lower", "yhat_upper", "is_future"]
    ]


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(model: Prophet, series: pd.DataFrame, horizon: int) -> dict:
    """Cross-validation glissante — requiert ≥ 2×horizon points."""
    n = len(series)
    if n < 2 * horizon + 4:
        return {}

    initial_weeks = max(horizon * 3, n // 2)
    try:
        df_cv = cross_validation(
            model,
            initial=f"{initial_weeks * 7} days",
            period=f"{max(horizon, 4) * 7} days",
            horizon=f"{horizon * 7} days",
            disable_tqdm=True,
        )
        perf = performance_metrics(df_cv, rolling_window=1)
        return {
            "mae":  round(float(perf["mae"].mean()),  4),
            "rmse": round(float(perf["rmse"].mean()), 4),
            "mape": round(float(perf["mape"].mean()), 4),
        }
    except Exception as e:
        logger.warning(f"CV échouée : {e}")
        return {}


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(
    cohort_id: int,
    cohort_name: str,
    metric: str,
    forecast: pd.DataFrame,
    cv_metrics: dict,
    n_train: int,
    horizon: int,
) -> None:
    with mlflow.start_run(
        run_name=f"prophet_{cohort_id}_{metric}",
        nested=True,
    ):
        mlflow.set_tag("cohort_id",   str(cohort_id))
        mlflow.set_tag("cohort_name", cohort_name)
        mlflow.set_tag("metric",      metric)
        mlflow.log_param("horizon_weeks", horizon)
        mlflow.log_param("n_train_points", n_train)

        if cv_metrics:
            mlflow.log_metrics(cv_metrics)

        # Valeurs prévues pour les semaines futures
        future_rows = forecast[forecast["is_future"]]
        for _, row in future_rows.iterrows():
            week_label = row["ds"].strftime("%Y-W%V")
            mlflow.log_metric(f"forecast_{week_label}", round(float(row["yhat"]), 4))


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_forecasts(
    horizon: int = 4,
    run_cv_flag: bool = False,
    output_dir: str = "data/features",
    experiment_name: str = "edtech-forecast",
    mlflow_uri: str = "http://localhost:5000",
    save_db: bool = True,
) -> pd.DataFrame:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    engine = get_engine()
    cohorts = pd.read_sql(SQL_COHORTS, engine)
    if cohorts.empty:
        logger.error("Aucune cohorte en base. Chargez d'abord les données.")
        raise SystemExit(1)

    all_forecasts: list[pd.DataFrame] = []

    with mlflow.start_run(run_name="prophet_all_cohorts"):
        mlflow.log_param("horizon_weeks", horizon)
        mlflow.log_param("n_cohorts",     len(cohorts))
        mlflow.log_param("metrics",       list(METRICS.keys()))

        for _, cohort in cohorts.iterrows():
            cid   = int(cohort["cohort_id"])
            cname = cohort["name"]
            logger.info(f"\n[Cohorte {cid}] {cname}")

            raw = load_cohort_series(cid)
            if len(raw) < 8:
                logger.warning(f"  Trop peu de points ({len(raw)}) — cohorte ignorée.")
                continue

            for metric, description in METRICS.items():
                series = prepare_series(raw, metric)
                if len(series) < 6:
                    logger.warning(f"  {metric} : série trop courte ({len(series)} pts)")
                    continue

                logger.info(f"  {metric} ({len(series)} pts)...")
                model, forecast = fit_and_forecast(
                    series, metric, horizon, cid, cname
                )

                cv_metrics = run_cv(model, series, horizon) if run_cv_flag else {}
                if cv_metrics:
                    logger.info(f"    CV → MAE={cv_metrics['mae']:.4f}  "
                                f"RMSE={cv_metrics['rmse']:.4f}")

                log_to_mlflow(cid, cname, metric, forecast, cv_metrics,
                              len(series), horizon)
                all_forecasts.append(forecast)

    if not all_forecasts:
        logger.error("Aucun forecast produit.")
        raise SystemExit(1)

    combined = pd.concat(all_forecasts, ignore_index=True)

    save_forecasts_parquet(combined, Path(output_dir))
    if save_db:
        save_forecasts_db(combined)

    future = combined[combined["is_future"]]
    logger.success(
        f"\n=== Forecast terminé ===\n"
        f"  Cohortes : {combined['cohort_id'].nunique()}\n"
        f"  Métriques : {combined['metric'].nunique()}\n"
        f"  Semaines futures : {future['ds'].nunique()}\n"
        f"  Lignes totales  : {len(combined):,}"
    )
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",     type=int,  default=4)
    parser.add_argument("--cv",          action="store_true")
    parser.add_argument("--output",      default="data/features")
    parser.add_argument("--experiment",  default="edtech-forecast")
    parser.add_argument("--mlflow-uri",  default="http://localhost:5000")
    parser.add_argument("--no-db",       action="store_true")
    args = parser.parse_args()
    run_forecasts(
        horizon=args.horizon,
        run_cv_flag=args.cv,
        output_dir=args.output,
        experiment_name=args.experiment,
        mlflow_uri=args.mlflow_uri,
        save_db=not args.no_db,
    )
