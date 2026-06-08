"""
Métriques d'évaluation et sélection de seuil optimal.

Priorité : maximiser le Recall classe décrochage (ne pas manquer les étudiants
à risque) sous contrainte Precision >= 0.50.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.50,
) -> tuple[float, float]:
    """
    Retourne (threshold, recall) maximisant le recall
    sous contrainte precision >= min_precision.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve retourne n+1 valeurs pour précision/recall
    for prec, rec, thr in zip(precisions[:-1], recalls[:-1], thresholds):
        if prec >= min_precision:
            return float(thr), float(rec)
    return 0.5, float(recalls[np.argmax(recalls)])


def full_report(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    min_precision: float = 0.50,
) -> tuple[dict[str, float], float]:
    """
    Calcule les métriques complètes et retourne (metrics_dict, optimal_threshold).
    """
    y_true = np.asarray(y_true)

    threshold, _ = find_best_threshold(y_true, y_proba, min_precision)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "roc_auc":          round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc":           round(average_precision_score(y_true, y_proba), 4),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_true, y_pred, zero_division=0), 4),
        "specificity":      round(tn / max(tn + fp, 1), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return metrics, threshold


def risk_label(proba: float, threshold: float) -> str:
    """Transforme une probabilité en label de risque à 4 niveaux."""
    if proba < threshold * 0.4:
        return "low"
    if proba < threshold:
        return "medium"
    if proba < threshold + (1 - threshold) * 0.5:
        return "high"
    return "critical"


def score_dataframe(
    df: pd.DataFrame,
    proba_col: str = "dropout_prob",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Ajoute la colonne risk_label à un DataFrame de prédictions."""
    df = df.copy()
    df["risk_label"] = df[proba_col].apply(lambda p: risk_label(p, threshold))
    return df
