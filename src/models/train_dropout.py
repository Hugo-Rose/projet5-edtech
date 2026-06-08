"""
Entraînement du modèle de prédiction de décrochage.

Pipeline : chargement Parquet → split temporel → preprocessing →
           Optuna HPO (XGBoost) → MLflow tracking → SHAP → sauvegarde.

Usage:
    python -m src.models.train_dropout [--features data/features/weekly_features.parquet]
                                       [--trials 30] [--experiment edtech-dropout]
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
import shap
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.constants import FEATURE_COLS, MODELS_DIR_STR, TARGET
from src.models.evaluate import full_report

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = Path(MODELS_DIR_STR)


# ── Split temporel ────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame,
                   val_frac: float = 0.15,
                   test_frac: float = 0.15):
    """Découpe chronologique pour éviter la fuite de données."""
    df = df.sort_values("week_start")
    n = len(df)
    i_val  = int(n * (1 - val_frac - test_frac))
    i_test = int(n * (1 - test_frac))
    train = df.iloc[:i_val]
    val   = df.iloc[i_val:i_test]
    test  = df.iloc[i_test:]
    logger.info(f"Split → train {len(train):,} | val {len(val):,} | test {len(test):,}")
    return train, val, test


def split_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].astype(int)
    return X, y


# ── Preprocessing ─────────────────────────────────────────────────────────────

def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_tr, y_tr, X_val, y_val, preprocessor):
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":  pos_weight,
            "tree_method":       "hist",
            "eval_metric":       "auc",
            "random_state":      42,
        }
        X_tr_pp  = preprocessor.fit_transform(X_tr)
        X_val_pp = preprocessor.transform(X_val)

        model = XGBClassifier(**params)
        model.fit(
            X_tr_pp, y_tr,
            eval_set=[(X_val_pp, y_val)],
            verbose=False,
        )
        proba = model.predict_proba(X_val_pp)[:, 1]
        return roc_auc_score(y_val, proba)

    return objective


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(model: XGBClassifier, X_test_pp: np.ndarray,
                 out_dir: Path) -> np.ndarray:
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_pp)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test_pp,
        feature_names=FEATURE_COLS,
        show=False, plot_type="dot",
    )
    plt.tight_layout()
    shap_path = out_dir / "shap_summary.png"
    plt.savefig(shap_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary → {shap_path}")
    return shap_values


# ── Main ──────────────────────────────────────────────────────────────────────

def train(
    features_path: str = "data/features/weekly_features.parquet",
    n_trials: int = 30,
    experiment_name: str = "edtech-dropout",
    mlflow_uri: str = "http://localhost:5000",
) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(f"Chargement features : {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"  {len(df):,} snapshots | taux décrochage : {df[TARGET].mean()*100:.1f}%")

    train_df, val_df, test_df = temporal_split(df)
    X_tr, y_tr   = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_te, y_te   = split_xy(test_df)

    preprocessor = make_preprocessor()

    # ── Optuna HPO ────────────────────────────────────────────────────────────
    logger.info(f"Optimisation Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize",
                                study_name="xgb_dropout_auc")
    study.optimize(
        make_objective(X_tr, y_tr, X_val, y_val, preprocessor),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = study.best_params
    logger.success(f"Meilleurs params : {best_params}")
    logger.success(f"Val ROC-AUC      : {study.best_value:.4f}")

    # ── Entraînement final (train+val) ────────────────────────────────────────
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    best_params.update({
        "scale_pos_weight": pos_weight,
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": 42,
    })

    X_full = pd.concat([X_tr, X_val])
    y_full = pd.concat([y_tr, y_val])

    preprocessor.fit(X_full)
    X_full_pp = preprocessor.transform(X_full)
    X_te_pp   = preprocessor.transform(X_te)

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_full_pp, y_full)

    # ── Évaluation test ───────────────────────────────────────────────────────
    y_proba = final_model.predict_proba(X_te_pp)[:, 1]
    metrics, threshold = full_report(y_te, y_proba)
    logger.info(f"Test ROC-AUC : {metrics['roc_auc']:.4f}")
    logger.info(f"Test Recall  : {metrics['recall']:.4f}  (seuil {threshold:.2f})")

    # ── MLflow logging ────────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        mlflow.log_param("n_features",  len(FEATURE_COLS))
        mlflow.log_param("n_train",     len(X_full))
        mlflow.log_param("n_test",      len(X_te))
        mlflow.log_param("threshold",   round(threshold, 4))
        mlflow.log_params({f"val_trials_{k}": v
                           for k, v in {"best_val_auc": study.best_value}.items()})
        mlflow.log_metrics(metrics)

        # SHAP
        compute_shap(final_model, X_te_pp, MODELS_DIR)
        mlflow.log_artifact(str(MODELS_DIR / "shap_summary.png"))

        # Modèle XGBoost
        mlflow.xgboost.log_model(
            final_model,
            artifact_path="xgb_model",
            registered_model_name="edtech-dropout-xgb",
        )

        # Preprocessor
        pp_path = MODELS_DIR / "preprocessor.joblib"
        joblib.dump(preprocessor, pp_path)
        mlflow.log_artifact(str(pp_path))

        # Metadata
        meta = {
            "threshold": threshold,
            "feature_cols": FEATURE_COLS,
            "metrics": metrics,
        }
        import json
        meta_path = MODELS_DIR / "model_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(meta_path))

        run_id = run.info.run_id
        logger.success(f"MLflow run : {run_id}")

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   default="data/features/weekly_features.parquet")
    parser.add_argument("--trials",     type=int, default=30)
    parser.add_argument("--experiment", default="edtech-dropout")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    args = parser.parse_args()
    train(
        features_path=args.features,
        n_trials=args.trials,
        experiment_name=args.experiment,
        mlflow_uri=args.mlflow_uri,
    )
