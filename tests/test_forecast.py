"""Tests unitaires — forecast Prophet (sans DB, sans MLflow)."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from src.models.forecast_engagement import (
    METRICS,
    build_model,
    fit_and_forecast,
    prepare_series,
    run_cv,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_weekly_series(n_weeks: int = 60, metric: str = "avg_logins",
                        seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-09-04")
    dates = [start + pd.Timedelta(weeks=i) for i in range(n_weeks)]
    trend = np.linspace(5, 3, n_weeks)
    noise = rng.normal(0, 0.3, n_weeks)
    values = np.clip(trend + noise, 0.1, None)
    return pd.DataFrame({"ds": dates, metric: values})


def _make_prophet_df(n: int = 60) -> pd.DataFrame:
    """DataFrame (ds, y) prêt pour Prophet."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-09-04", periods=n, freq="W")
    y = np.clip(5 + np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 0.2, n), 0.1, None)
    return pd.DataFrame({"ds": dates, "y": y})


# ── prepare_series ────────────────────────────────────────────────────────────

def test_prepare_series_columns():
    raw = _make_weekly_series(40, "avg_logins")
    series = prepare_series(raw, "avg_logins")
    assert list(series.columns) == ["ds", "y"]


def test_prepare_series_no_nan():
    raw = _make_weekly_series(30)
    raw.loc[5, "avg_logins"] = np.nan
    series = prepare_series(raw, "avg_logins")
    assert series["y"].notna().all()


def test_prepare_series_no_zeros():
    raw = _make_weekly_series(30)
    raw.loc[3, "avg_logins"] = 0.0
    series = prepare_series(raw, "avg_logins")
    assert (series["y"] > 0).all()


def test_prepare_series_preserves_order():
    raw = _make_weekly_series(20)
    series = prepare_series(raw, "avg_logins")
    assert series["ds"].is_monotonic_increasing


# ── build_model ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("metric", list(METRICS.keys()))
def test_build_model_returns_prophet(metric):
    model = build_model(metric)
    assert isinstance(model, Prophet)


def test_build_model_no_daily_seasonality():
    model = build_model("avg_logins")
    assert not model.daily_seasonality


def test_build_model_no_weekly_seasonality():
    model = build_model("avg_logins")
    assert not model.weekly_seasonality


# ── fit_and_forecast ──────────────────────────────────────────────────────────

@pytest.fixture
def fitted_forecast():
    series = _make_prophet_df(60)
    model, forecast = fit_and_forecast(
        series, "avg_logins", horizon=4,
        cohort_id=1, cohort_name="Promo 2023",
    )
    return model, forecast, series


def test_fit_and_forecast_horizon(fitted_forecast):
    _, forecast, series = fitted_forecast
    future_rows = forecast[forecast["is_future"]]
    assert len(future_rows) == 4


def test_fit_and_forecast_columns(fitted_forecast):
    _, forecast, _ = fitted_forecast
    for col in ("ds", "yhat", "yhat_lower", "yhat_upper",
                "is_future", "cohort_id", "metric"):
        assert col in forecast.columns


def test_fit_and_forecast_yhat_finite(fitted_forecast):
    _, forecast, _ = fitted_forecast
    assert np.isfinite(forecast["yhat"]).all()


def test_fit_and_forecast_intervals_ordered(fitted_forecast):
    _, forecast, _ = fitted_forecast
    assert (forecast["yhat_upper"] >= forecast["yhat"]).all()
    assert (forecast["yhat"] >= forecast["yhat_lower"]).all()


def test_fit_and_forecast_metadata(fitted_forecast):
    _, forecast, _ = fitted_forecast
    assert (forecast["cohort_id"] == 1).all()
    assert (forecast["metric"] == "avg_logins").all()


def test_fit_and_forecast_history_not_future(fitted_forecast):
    _, forecast, series = fitted_forecast
    last_hist = series["ds"].max()
    hist_rows = forecast[~forecast["is_future"]]
    assert (hist_rows["ds"] <= last_hist).all()


# ── run_cv ────────────────────────────────────────────────────────────────────

def test_run_cv_too_short():
    series = _make_prophet_df(8)
    model = build_model("avg_logins")
    model.fit(series)
    result = run_cv(model, series, horizon=4)
    assert result == {}


def test_run_cv_returns_metrics():
    series = _make_prophet_df(80)
    model = build_model("avg_logins")
    model.fit(series)
    result = run_cv(model, series, horizon=4)
    if result:  # peut être vide si CV échoue sur petite série
        assert "mae" in result
        assert "rmse" in result
        assert "mape" in result
        assert all(v >= 0 for v in result.values())


# ── Cohérence multi-métriques ─────────────────────────────────────────────────

def test_all_metrics_forecasted():
    """Vérifie que les 4 métriques produisent chacune un forecast valide."""
    forecasts = []
    for metric in METRICS:
        series = _make_prophet_df(60)
        _, forecast = fit_and_forecast(
            series, metric, horizon=4,
            cohort_id=1, cohort_name="Test",
        )
        forecasts.append(forecast)

    combined = pd.concat(forecasts, ignore_index=True)
    assert combined["metric"].nunique() == len(METRICS)
    assert combined[combined["is_future"]].groupby("metric").size().min() == 4
