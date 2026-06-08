"""Tests unitaires — évaluation et inférence (sans DB, sans MLflow)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from src.models.evaluate import (
    find_best_threshold,
    full_report,
    risk_label,
    score_dataframe,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_preds():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=1000)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.2, 1000), 0, 1)
    return y_true, y_proba


@pytest.fixture
def trained_xgb():
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6,
        weights=[0.8, 0.2], random_state=42,
    )
    model = XGBClassifier(n_estimators=50, random_state=42, tree_method="hist")
    model.fit(X, y)
    return model, X, y


# ── evaluate.py ───────────────────────────────────────────────────────────────

def test_find_best_threshold_returns_tuple(binary_preds):
    y_true, y_proba = binary_preds
    thr, rec = find_best_threshold(y_true, y_proba)
    assert 0.0 <= thr <= 1.0
    assert 0.0 <= rec <= 1.0


def test_find_best_threshold_precision_constraint(binary_preds):
    y_true, y_proba = binary_preds
    thr, _ = find_best_threshold(y_true, y_proba, min_precision=0.50)
    y_pred = (y_proba >= thr).astype(int)
    from sklearn.metrics import precision_score
    prec = precision_score(y_true, y_pred, zero_division=0)
    assert prec >= 0.45  # tolérance légère


def test_full_report_keys(binary_preds):
    y_true, y_proba = binary_preds
    metrics, thr = full_report(y_true, y_proba)
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        assert key in metrics
    assert 0.0 <= thr <= 1.0


def test_full_report_roc_auc_range(binary_preds):
    y_true, y_proba = binary_preds
    metrics, _ = full_report(y_true, y_proba)
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_full_report_confusion_matrix_sum(binary_preds):
    y_true, y_proba = binary_preds
    metrics, _ = full_report(y_true, y_proba)
    total = metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]
    assert total == len(y_true)


def test_full_report_perfect_classifier():
    y_true  = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics, _ = full_report(y_true, y_proba)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"]  == 1.0


def test_full_report_all_zeros():
    y_true  = np.zeros(100, dtype=int)
    y_proba = np.random.rand(100)
    # Ne doit pas lever d'exception
    metrics, thr = full_report(y_true, y_proba)
    assert "roc_auc" in metrics


# ── risk_label ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("proba,thr,expected", [
    (0.05, 0.5, "low"),
    (0.30, 0.5, "medium"),
    (0.60, 0.5, "high"),
    (0.95, 0.5, "critical"),
])
def test_risk_label_levels(proba, thr, expected):
    assert risk_label(proba, thr) == expected


def test_risk_label_boundary_at_threshold():
    thr = 0.4
    assert risk_label(thr, thr) in ("high", "critical")
    assert risk_label(thr - 0.01, thr) in ("low", "medium")


# ── score_dataframe ───────────────────────────────────────────────────────────

def test_score_dataframe_adds_column():
    df = pd.DataFrame({"student_id": [1, 2, 3],
                       "dropout_prob": [0.1, 0.55, 0.85]})
    out = score_dataframe(df, threshold=0.5)
    assert "risk_label" in out.columns
    assert set(out["risk_label"]).issubset({"low", "medium", "high", "critical"})


def test_score_dataframe_no_mutation():
    df = pd.DataFrame({"student_id": [1], "dropout_prob": [0.7]})
    out = score_dataframe(df, threshold=0.5)
    assert "risk_label" not in df.columns   # original non muté


# ── XGBoost end-to-end (sans MLflow) ─────────────────────────────────────────

def test_xgb_predict_proba_shape(trained_xgb):
    model, X, _ = trained_xgb
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_xgb_roc_auc_above_baseline(trained_xgb):
    model, X, y = trained_xgb
    proba = model.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, proba)
    assert auc > 0.70, f"AUC trop faible : {auc:.3f}"
