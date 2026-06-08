"""Tests unitaires — monitoring (drift + fairness), sans DB ni Evidently."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dashboard.data_loader import (
    assign_clusters,
    compute_engagement_score,
)
from src.monitoring.drift_report import _ks_drift
from src.monitoring.fairness_report import (
    FAIRNESS_THRESHOLD,
    add_age_group,
    add_threshold_prediction,
    compute_all_groups,
    compute_group_metrics,
    detect_violations,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_predictions_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genders = rng.choice(["M", "F", "Other"], p=[0.48, 0.48, 0.04], size=n)
    ages    = rng.integers(18, 40, n)
    schol   = rng.random(n) < 0.30
    probs   = np.clip(rng.normal(0.3, 0.2, n), 0, 1)
    labels  = (probs > 0.5).astype(int)
    # Biais artificiel sur M pour les tests de violations
    probs[genders == "M"] += 0.20
    probs = np.clip(probs, 0, 1)
    labels = (probs > 0.5).astype(int)

    return pd.DataFrame({
        "student_id":   range(n),
        "dropout_prob": probs,
        "risk_label":   pd.cut(probs, bins=[0,.2,.5,.65,1],
                               labels=["low","medium","high","critical"]).astype(str),
        "label_dropout": labels,
        "gender":       genders,
        "age":          ages,
        "scholarship":  schol,
        "week_start":   "2025-01-06",
    })


def _make_weekly_df(n_students: int = 50, n_weeks: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for sid in range(1, n_students + 1):
        for w in range(n_weeks):
            rows.append({
                "student_id":       sid,
                "week_start":       pd.Timestamp("2025-01-06") - pd.Timedelta(weeks=w),
                "login_count":      int(rng.integers(0, 12)),
                "total_time_min":   float(rng.uniform(0, 300)),
                "quiz_pass_rate":   float(rng.uniform(0, 1)),
                "avg_score":        float(rng.uniform(0, 100)),
                "quiz_attempts":    int(rng.integers(0, 5)),
                "forum_posts":      int(rng.integers(0, 5)),
            })
    return pd.DataFrame(rows)


# ── fairness_report ───────────────────────────────────────────────────────────

def test_add_age_group_columns():
    df = _make_predictions_df(50)
    out = add_age_group(df)
    assert "age_group" in out.columns


def test_add_age_group_no_nulls_for_valid_ages():
    df = pd.DataFrame({"age": [18, 22, 26, 35]})
    out = add_age_group(df)
    assert out["age_group"].notna().all()


def test_add_threshold_prediction():
    df = pd.DataFrame({"dropout_prob": [0.1, 0.5, 0.7, 0.9]})
    out = add_threshold_prediction(df, threshold=0.5)
    assert list(out["y_pred"]) == [0, 1, 1, 1]


def test_add_threshold_prediction_zero():
    df = pd.DataFrame({"dropout_prob": [0.0, 0.49, 0.499]})
    out = add_threshold_prediction(df, 0.5)
    assert (out["y_pred"] == 0).all()


def test_compute_group_metrics_shape():
    df = _make_predictions_df(100)
    result = compute_group_metrics(df, "gender")
    assert "group" in result.columns
    assert "avg_prob" in result.columns
    assert len(result) == df["gender"].nunique()


def test_compute_group_metrics_prob_range():
    df = _make_predictions_df(200)
    result = compute_group_metrics(df, "gender")
    assert (result["avg_prob"] >= 0).all()
    assert (result["avg_prob"] <= 1).all()


def test_compute_group_metrics_n_sum():
    df = _make_predictions_df(100)
    result = compute_group_metrics(df, "gender")
    assert result["n"].sum() == len(df)


def test_compute_group_metrics_fpr_fnr_valid():
    df = _make_predictions_df(200)
    result = compute_group_metrics(df, "gender")
    # tpr/fpr peuvent être NaN si pas de y_true, mais si label_dropout présent
    valid = result[result["tpr"].notna()]
    if not valid.empty:
        assert (valid["tpr"].between(0, 1)).all()
        assert (valid["fpr"].between(0, 1)).all()


def test_compute_all_groups_keys():
    df = _make_predictions_df(200)
    df = add_age_group(df)
    result = compute_all_groups(df)
    assert "gender" in result
    assert "scholarship" in result


def test_detect_violations_finds_gender_bias():
    """Le biais M+0.2 dans le fixture doit déclencher une violation."""
    df = _make_predictions_df(300, seed=1)
    df = add_age_group(df)
    groups = compute_all_groups(df)
    violations = detect_violations(groups, threshold=0.10)
    # Au moins une violation détectée sur gender
    v_attrs = [v["attr"] for v in violations]
    assert "gender" in v_attrs


def test_detect_violations_no_false_positive():
    """Données équitables → pas de violation."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "student_id":    range(n),
        "dropout_prob":  np.clip(rng.normal(0.3, 0.05, n), 0, 1),
        "label_dropout": rng.integers(0, 2, n),
        "gender":        rng.choice(["M", "F"], size=n),
        "age":           rng.integers(20, 30, n),
        "scholarship":   rng.random(n) < 0.3,
        "week_start":    "2025-01-06",
    })
    df = add_age_group(df)
    groups = compute_all_groups(df)
    violations = detect_violations(groups, threshold=0.25)
    assert violations == []


def test_detect_violations_structure():
    df = _make_predictions_df(300)
    df = add_age_group(df)
    violations = detect_violations(compute_all_groups(df))
    for v in violations:
        assert "attr"   in v
        assert "metric" in v
        assert "gap"    in v
        assert v["gap"] > 0


# ── drift_report (KS fallback) ────────────────────────────────────────────────

def test_ks_drift_no_drift_same_distribution():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(5, 1, 200),
                         "b": rng.uniform(0, 1, 200)})
    cur = pd.DataFrame({"a": rng.normal(5, 1, 200),
                         "b": rng.uniform(0, 1, 200)})
    result = _ks_drift(ref, cur, ["a", "b"])
    assert isinstance(result, dict)
    assert set(result.keys()) == {"a", "b"}


def test_ks_drift_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": rng.normal(5, 1, 500)})   # forte dérive
    result = _ks_drift(ref, cur, ["x"])
    assert result["x"]  # np.True_ ou True


def test_ks_drift_too_short_returns_false():
    ref = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    cur = pd.DataFrame({"x": [4.0, 5.0, 6.0]})
    result = _ks_drift(ref, cur, ["x"])
    assert result["x"] is False


# ── data_loader ───────────────────────────────────────────────────────────────

def test_engagement_score_range():
    df = _make_weekly_df(20, 4)
    scores = compute_engagement_score(df)
    assert (scores >= 0).all()
    assert (scores <= 100).all()


def test_engagement_score_zero_activity():
    df = pd.DataFrame({
        "login_count":    [0],
        "total_time_min": [0],
        "quiz_pass_rate": [0],
    })
    assert float(compute_engagement_score(df).iloc[0]) == 0.0


def test_engagement_score_full_activity():
    df = pd.DataFrame({
        "login_count":    [15],
        "total_time_min": [300],
        "quiz_pass_rate": [1.0],
    })
    assert float(compute_engagement_score(df).iloc[0]) == 100.0


def test_assign_clusters_shape():
    df = _make_weekly_df(30, 8)
    clusters = assign_clusters(df, n_clusters=4)
    assert "cluster_id"   in clusters.columns
    assert "cluster_name" in clusters.columns
    assert len(clusters) == df["student_id"].nunique()


def test_assign_clusters_names():
    df = _make_weekly_df(40, 8)
    clusters = assign_clusters(df, n_clusters=4)
    valid_names = {"Très engagé", "Engagé", "Passif", "À risque"}
    assert set(clusters["cluster_name"].dropna()).issubset(valid_names)


def test_assign_clusters_all_students_assigned():
    df = _make_weekly_df(20, 6)
    clusters = assign_clusters(df, n_clusters=4)
    assert clusters["cluster_id"].notna().all()
