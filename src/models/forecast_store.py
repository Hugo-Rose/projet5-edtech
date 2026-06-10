"""Persistance des prévisions Prophet (Parquet + PostgreSQL)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.data.db import get_engine

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS engagement_forecasts (
    id              BIGSERIAL PRIMARY KEY,
    cohort_id       INTEGER       NOT NULL,
    cohort_name     VARCHAR(200),
    metric          VARCHAR(80)   NOT NULL,
    ds              DATE          NOT NULL,
    yhat            NUMERIC(12,4),
    yhat_lower      NUMERIC(12,4),
    yhat_upper      NUMERIC(12,4),
    is_future       BOOLEAN       DEFAULT TRUE,
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (cohort_id, metric, ds)
);
CREATE INDEX IF NOT EXISTS idx_ef_cohort  ON engagement_forecasts(cohort_id);
CREATE INDEX IF NOT EXISTS idx_ef_metric  ON engagement_forecasts(metric);
CREATE INDEX IF NOT EXISTS idx_ef_ds      ON engagement_forecasts(ds);
"""


def ensure_table() -> None:
    engine = get_engine()
    with get_engine().connect() as conn:
        for stmt in CREATE_TABLE_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()


def save_forecasts_parquet(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "engagement_forecasts.parquet"
    df.to_parquet(path, index=False)
    sz = path.stat().st_size / 1e6
    logger.success(f"Parquet → {path}  ({sz:.2f} Mo, {len(df):,} lignes)")


def save_forecasts_db(df: pd.DataFrame) -> None:
    ensure_table()
    engine = get_engine()

    db_cols = ["cohort_id", "cohort_name", "metric",
               "ds", "yhat", "yhat_lower", "yhat_upper", "is_future"]
    subset = df[db_cols].copy()
    subset["ds"] = pd.to_datetime(subset["ds"]).dt.date

    # Upsert : on conflict → update yhat
    with get_engine().connect() as conn:
        conn.execute(text("""
            DELETE FROM engagement_forecasts
            WHERE (cohort_id, metric) IN (
                SELECT DISTINCT cohort_id, metric FROM engagement_forecasts
                WHERE cohort_id = ANY(:cids)
            )
        """), {"cids": list(map(int, subset["cohort_id"].unique()))})
        conn.commit()

    subset.to_sql(
        "engagement_forecasts",
        get_engine(),
        if_exists="append", index=False,
        chunksize=5_000, method="multi",
    )
    logger.success(f"DB engagement_forecasts : {len(subset):,} lignes insérées")


def load_forecasts(
    cohort_id: int | None = None,
    metric: str | None = None,
    future_only: bool = True,
) -> pd.DataFrame:
    """Charge les prévisions depuis PostgreSQL."""
    engine = get_engine()
    filters = ["1=1"]
    params: dict = {}
    if cohort_id:
        filters.append("cohort_id = %(cohort_id)s")
        params["cohort_id"] = cohort_id
    if metric:
        filters.append("metric = %(metric)s")
        params["metric"] = metric
    if future_only:
        filters.append("is_future = TRUE")

    sql = f"""
        SELECT cohort_id, cohort_name, metric, ds,
               yhat, yhat_lower, yhat_upper, is_future
        FROM engagement_forecasts
        WHERE {" AND ".join(filters)}
        ORDER BY cohort_id, metric, ds
    """
    return pd.read_sql(sql, get_engine(), params=params)
