"""
Clustering K-Means des profils étudiants.

Pipeline :
  - Agrège weekly_features par étudiant (moyenne sur toutes les semaines)
  - StandardScaler + KMeans (k=2..6, choix par silhouette)
  - Log MLflow : silhouette score, inertia, composition des clusters
  - Sauvegarde data/models/kmeans.joblib

Usage:
    python -m src.models.train_clustering
           [--features data/features/weekly_features_uci.parquet]
           [--k 4]  # 0 = auto-select via silhouette
           [--mlflow-uri http://localhost:5000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path("data/models")

# Features utilisées pour le clustering (disponibles dans toutes les sources)
CLUSTER_FEATURES = [
    "login_count",
    "total_time_min",
    "videos_watched",
    "quiz_attempts",
    "quiz_pass_rate",
    "avg_score",
    "forum_posts",
    "assignments_on_time",
]

CLUSTER_NAMES = {
    0: "Très engagé",
    1: "Engagé",
    2: "Passif",
    3: "À risque",
}


# ── Agrégation par étudiant ───────────────────────────────────────────────────

def aggregate_per_student(df: pd.DataFrame) -> pd.DataFrame:
    """Moyenne des features hebdomadaires par étudiant."""
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    if not available:
        raise ValueError(f"Aucune feature de clustering trouvée dans {df.columns.tolist()}")

    agg = (
        df.groupby("student_id")[available]
        .mean()
        .fillna(0)
        .reset_index()
    )
    logger.info(f"Agrégation : {len(agg):,} étudiants × {len(available)} features")
    return agg, available


# ── Sélection du k optimal ────────────────────────────────────────────────────

def find_best_k(X: np.ndarray, k_range: range = range(2, 7)) -> tuple[int, dict]:
    """Teste k=2..6 et retourne le k avec le meilleur silhouette score."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sc = silhouette_score(X, labels, sample_size=min(3000, len(X)))
        scores[k] = round(float(sc), 4)
        logger.info(f"  k={k}  silhouette={sc:.4f}")

    best_k = max(scores, key=scores.get)
    logger.success(f"k optimal : {best_k}  (silhouette={scores[best_k]:.4f})")
    return best_k, scores


# ── Nommage des clusters ──────────────────────────────────────────────────────

def name_clusters(km: KMeans, feature_names: list[str],
                  n_clusters: int) -> dict[int, str]:
    """
    Nomme les clusters par engagement décroissant (engagement = score moyen
    normalisé des features de connexion + performance).
    """
    centroids = pd.DataFrame(km.cluster_centers_, columns=feature_names)
    engagement_cols = [c for c in
                       ["login_count", "total_time_min", "avg_score", "quiz_pass_rate"]
                       if c in centroids.columns]

    if not engagement_cols:
        return {i: f"Cluster {i}" for i in range(n_clusters)}

    scores = centroids[engagement_cols].mean(axis=1)
    rank   = scores.rank(ascending=False).astype(int) - 1
    names  = {
        cluster_id: CLUSTER_NAMES.get(rank_val, f"Cluster {rank_val}")
        for cluster_id, rank_val in rank.items()
    }
    return names


# ── Pipeline principal ────────────────────────────────────────────────────────

def train(
    features_path: str = "data/features/weekly_features.parquet",
    k:             int = 0,
    mlflow_uri:    str = "http://localhost:5000",
    experiment:    str = "edtech-clustering",
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    logger.info(f"Chargement features : {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"  {len(df):,} snapshots, {df['student_id'].nunique():,} étudiants")

    agg, used_features = aggregate_per_student(df)
    X_raw = agg[used_features].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Sélection du k
    if k == 0:
        logger.info("Recherche du k optimal (silhouette)...")
        best_k, sil_scores = find_best_k(X)
    else:
        best_k = k
        sil_scores = {}

    # Entraînement final
    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    sil = float(silhouette_score(X, labels, sample_size=min(3000, len(X))))
    inertia = float(km.inertia_)

    logger.success(f"KMeans k={best_k}  silhouette={sil:.4f}  inertia={inertia:.0f}")

    # Nommer les clusters
    cluster_names = name_clusters(km, used_features, best_k)

    # Distribution
    agg["cluster_id"]   = labels
    agg["cluster_name"] = [cluster_names[c] for c in labels]
    distribution = agg["cluster_name"].value_counts().to_dict()

    for name, count in sorted(distribution.items(), key=lambda x: -x[1]):
        logger.info(f"  {name:<15} : {count:>5,} ({count/len(agg)*100:.1f}%)")

    # FEATURE_COLS importance : distance centroïde moyen par feature
    feature_importance = pd.DataFrame(
        np.abs(km.cluster_centers_).mean(axis=0),
        index=used_features, columns=["importance"],
    ).sort_values("importance", ascending=False)

    # ── MLflow ────────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"kmeans_k{best_k}"):
        mlflow.log_param("n_clusters",      best_k)
        mlflow.log_param("features_used",   used_features)
        mlflow.log_param("n_students",      len(agg))
        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("inertia",          inertia)
        for k_val, sc in sil_scores.items():
            mlflow.log_metric(f"silhouette_k{k_val}", sc)

        for name, count in distribution.items():
            safe_name = name.replace(" ", "_").replace("é", "e").replace("è", "e")
            mlflow.log_metric(f"cluster_{safe_name}_count", count)

        # Sauvegarde artefacts
        bundle = {
            "kmeans":        km,
            "scaler":        scaler,
            "feature_names": used_features,
            "cluster_names": cluster_names,
            "n_clusters":    best_k,
            "silhouette":    sil,
        }
        model_path = MODELS_DIR / "kmeans.joblib"
        joblib.dump(bundle, model_path)
        mlflow.log_artifact(str(model_path))

        meta = {
            "n_clusters":    best_k,
            "silhouette":    sil,
            "inertia":       inertia,
            "feature_names": used_features,
            "cluster_names": {str(k): v for k, v in cluster_names.items()},
            "distribution":  distribution,
        }
        meta_path = MODELS_DIR / "kmeans_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        mlflow.log_artifact(str(meta_path))

        run_id = mlflow.active_run().info.run_id
        logger.success(f"MLflow run : {run_id}")

    return {
        "k":           best_k,
        "silhouette":  round(sil, 4),
        "inertia":     round(inertia, 1),
        "distribution": distribution,
        "cluster_names": cluster_names,
        "run_id":      run_id,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   default="data/features/weekly_features.parquet")
    parser.add_argument("--k",          type=int, default=0,
                        help="Nombre de clusters (0 = auto via silhouette, k=2..6)")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--experiment", default="edtech-clustering")
    args = parser.parse_args()
    result = train(
        features_path=args.features,
        k=args.k,
        mlflow_uri=args.mlflow_uri,
        experiment=args.experiment,
    )
    print(f"\nRésultat : k={result['k']}, silhouette={result['silhouette']}")
