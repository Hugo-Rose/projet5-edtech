"""
Système de recommandation content-based par cluster K-Means.

Algorithme :
  1. Pour chaque cluster, calcule la corrélation lead-lag entre chaque
     comportement LMS (semaine N) et l'amélioration du score (semaine N+1)
  2. Sélectionne les 3 comportements les plus positivement corrélés
  3. Génère des recommandations personnalisées avec valeur actuelle vs cible

Usage standalone :
    from src.models.recommender import build_cluster_recommendations, recommend
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PARQUET_PATH = Path("data/features/weekly_features.parquet")

# ── Métadonnées des comportements analysables ─────────────────────────────────

FEATURE_META: dict[str, dict] = {
    "login_count": {
        "action_type":  "connections",
        "title":        "Se connecter plus régulièrement",
        "description":  "Une présence régulière est le premier facteur de réussite dans votre profil.",
        "unit":         "connexions/semaine",
        "target_pct":   0.75,
    },
    "total_time_min": {
        "action_type":  "time",
        "title":        "Augmenter le temps de travail hebdomadaire",
        "description":  "Les étudiants de votre profil qui passent plus de temps progressent significativement.",
        "unit":         "minutes/semaine",
        "target_pct":   0.70,
    },
    "videos_watched": {
        "action_type":  "videos",
        "title":        "Visionner les vidéos de cours",
        "description":  "Les ressources vidéo sont fortement corrélées à la progression dans votre groupe.",
        "unit":         "vidéos/semaine",
        "target_pct":   0.75,
    },
    "quiz_attempts": {
        "action_type":  "quiz",
        "title":        "Multiplier les tentatives de quiz",
        "description":  "S'entraîner régulièrement aux quiz renforce la mémorisation.",
        "unit":         "tentatives/semaine",
        "target_pct":   0.75,
    },
    "quiz_pass_rate": {
        "action_type":  "quiz_quality",
        "title":        "Retravailler les quiz échoués",
        "description":  "Améliorer votre taux de réussite aux quiz a un fort impact sur vos notes.",
        "unit":         "taux réussite",
        "target_pct":   0.80,
    },
    "forum_posts": {
        "action_type":  "forum",
        "title":        "Participer aux forums de discussion",
        "description":  "Poser des questions et répondre à vos pairs améliore la compréhension.",
        "unit":         "posts/semaine",
        "target_pct":   0.75,
    },
    "assignments_on_time": {
        "action_type":  "assignments",
        "title":        "Rendre les travaux dans les délais",
        "description":  "Le respect des échéances est fortement corrélé à la réussite dans votre profil.",
        "unit":         "devoirs rendus à temps/semaine",
        "target_pct":   0.90,
    },
}

BEHAVIOR_FEATURES = list(FEATURE_META.keys())


# ── Structure de données ──────────────────────────────────────────────────────

@dataclass
class RecommendationItem:
    rank:                 int
    action_type:          str
    title:                str
    description:          str
    current_value:        Optional[float]
    target_value:         Optional[float]
    correlation:          float
    unit:                 str
    expected_improvement: Optional[str] = None


@dataclass
class RecommendationsResult:
    student_id:      int
    cluster_id:      int
    cluster_name:    str
    recommendations: list[RecommendationItem] = field(default_factory=list)
    computed_at:     datetime = field(default_factory=datetime.now)


# ── Calcul des corrélations par cluster ───────────────────────────────────────

def _lead_correlations(
    df: pd.DataFrame,
    features: list[str],
    target: str = "avg_score",
) -> dict[str, float]:
    """Corrélation de chaque feature(t) avec target(t+1) - target(t)."""
    df = df.sort_values(["student_id", "week_start"]).copy()
    df["_next_score"] = df.groupby("student_id")[target].shift(-1)
    df["_improvement"] = df["_next_score"] - df[target]

    corrs = {}
    valid = df.dropna(subset=["_improvement"])
    for feat in features:
        if feat not in valid.columns:
            continue
        col = valid[feat].fillna(0)
        if col.std() == 0:
            continue
        corrs[feat] = float(col.corr(valid["_improvement"]))
    return corrs


def build_cluster_recommendations(
    weekly_df: pd.DataFrame,
    n_top: int = 3,
) -> dict[int, list[tuple[str, float]]]:
    """
    Retourne pour chaque cluster_id les n_top comportements les plus corrélés
    avec l'amélioration du score.

    Returns: {cluster_id: [(feature_name, correlation), ...]}
    """
    if "cluster_id" not in weekly_df.columns:
        return {}

    result: dict[int, list[tuple[str, float]]] = {}

    for cid, grp in weekly_df.groupby("cluster_id"):
        corrs = _lead_correlations(grp, BEHAVIOR_FEATURES)
        # Ne garder que les corrélations positives, triées décroissantes
        positive = sorted(
            ((f, c) for f, c in corrs.items() if c > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        result[int(cid)] = positive[:n_top]

    return result


# ── Cible personnalisée par cluster ──────────────────────────────────────────

def _cluster_target(
    weekly_df: pd.DataFrame,
    cluster_id: int,
    feature: str,
    percentile: float,
) -> Optional[float]:
    cluster_data = weekly_df[weekly_df["cluster_id"] == cluster_id][feature].dropna()
    if cluster_data.empty:
        return None
    return round(float(np.percentile(cluster_data, percentile * 100)), 2)


# ── Recommandation pour un étudiant ──────────────────────────────────────────

def recommend(
    student_id:   int,
    weekly_df:    pd.DataFrame,
    cluster_recs: dict[int, list[tuple[str, float]]],
    cluster_map:  Optional[dict[int, str]] = None,
    n_top:        int = 3,
) -> RecommendationsResult:
    """
    Génère les recommandations personnalisées pour un étudiant.

    cluster_map : {cluster_id → cluster_name}
    """
    student_rows = weekly_df[weekly_df["student_id"] == student_id]

    # Cluster de l'étudiant (dernière semaine connue)
    if student_rows.empty or "cluster_id" not in student_rows.columns:
        cluster_id   = 0
        cluster_name = "Inconnu"
    else:
        last = student_rows.sort_values("week_start").iloc[-1]
        cluster_id   = int(last.get("cluster_id", 0))
        cluster_name = (cluster_map or {}).get(cluster_id, f"Cluster {cluster_id}")

    # Stats actuelles de l'étudiant (moyenne 4 dernières semaines)
    recent = student_rows.sort_values("week_start").tail(4)
    current_stats: dict[str, float] = {
        feat: round(float(recent[feat].mean()), 2)
        for feat in BEHAVIOR_FEATURES
        if feat in recent.columns and recent[feat].notna().any()
    }

    # Top comportements pour ce cluster
    top_behaviors = cluster_recs.get(cluster_id, [])
    if not top_behaviors:
        # Fallback : recommandations génériques si cluster inconnu
        top_behaviors = [
            ("login_count",    0.4),
            ("videos_watched", 0.35),
            ("quiz_attempts",  0.3),
        ]

    items: list[RecommendationItem] = []
    for rank, (feat, corr) in enumerate(top_behaviors[:n_top], start=1):
        meta       = FEATURE_META.get(feat, {})
        current    = current_stats.get(feat)
        target     = _cluster_target(weekly_df, cluster_id, feat,
                                     meta.get("target_pct", 0.75))
        delta_pct  = round(abs(corr) * 20, 0)  # indicateur approximatif
        improvement = f"+{delta_pct:.0f}% score moyen estimé" if delta_pct > 0 else None

        items.append(RecommendationItem(
            rank                 = rank,
            action_type          = meta.get("action_type", feat),
            title                = meta.get("title", feat),
            description          = meta.get("description", ""),
            current_value        = current,
            target_value         = target,
            correlation          = round(corr, 4),
            unit                 = meta.get("unit", ""),
            expected_improvement = improvement,
        ))

    return RecommendationsResult(
        student_id      = student_id,
        cluster_id      = cluster_id,
        cluster_name    = cluster_name,
        recommendations = items,
    )


# ── Chargement depuis parquet (mode standalone) ───────────────────────────────

def load_weekly_with_clusters(path: Path = PARQUET_PATH) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Les clusters sont dans le parquet si build_features a été lancé avec data_loader
    if "cluster_id" not in df.columns:
        from src.dashboard.data_loader import assign_clusters
        clusters = assign_clusters(df)
        df = df.merge(clusters, on="student_id", how="left")
    return df


@lru_cache(maxsize=1)
def _cached_recommendations(
    path_str: str,
) -> tuple[pd.DataFrame, dict[int, list[tuple[str, float]]], dict[int, str]]:
    """Cache LRU pour éviter de recalculer à chaque requête API."""
    df = load_weekly_with_clusters(Path(path_str))
    if df is None or df.empty:
        return pd.DataFrame(), {}, {}

    cluster_recs = build_cluster_recommendations(df)

    # Mapping cluster_id → cluster_name
    cluster_map: dict[int, str] = {}
    if "cluster_name" in df.columns:
        cluster_map = (
            df.dropna(subset=["cluster_id", "cluster_name"])
            .drop_duplicates("cluster_id")
            .set_index("cluster_id")["cluster_name"]
            .to_dict()
        )

    return df, cluster_recs, cluster_map


def get_recommendations_for_student(
    student_id: int,
    weekly_df:  Optional[pd.DataFrame] = None,
) -> Optional[RecommendationsResult]:
    """
    Point d'entrée principal — utilisé par l'API et le dashboard.
    Charge les données si non fournies.
    """
    if weekly_df is None:
        df, cluster_recs, cluster_map = _cached_recommendations(str(PARQUET_PATH))
        if df.empty:
            return None
    else:
        from src.dashboard.data_loader import assign_clusters
        if "cluster_id" not in weekly_df.columns:
            clusters = assign_clusters(weekly_df)
            weekly_df = weekly_df.merge(clusters, on="student_id", how="left")
        df = weekly_df
        cluster_recs = build_cluster_recommendations(df)
        cluster_map = {}
        if "cluster_name" in df.columns:
            cluster_map = (
                df.dropna(subset=["cluster_id", "cluster_name"])
                .drop_duplicates("cluster_id")
                .set_index("cluster_id")["cluster_name"]
                .to_dict()
            )

    if student_id not in df["student_id"].values:
        return None

    return recommend(student_id, df, cluster_recs, cluster_map)
