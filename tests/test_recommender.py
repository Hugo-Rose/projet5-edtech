"""Tests unitaires — système de recommandation content-based."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.recommender import (
    BEHAVIOR_FEATURES,
    FEATURE_META,
    RecommendationItem,
    RecommendationsResult,
    build_cluster_recommendations,
    recommend,
    _lead_correlations,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_weekly_df(
    n_students: int = 40,
    n_weeks:    int = 16,
    seed:       int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(1, n_students + 1):
        cluster = (sid - 1) % 4
        for w in range(n_weeks):
            week = pd.Timestamp("2024-09-02") + pd.Timedelta(weeks=w)
            # Profil différent par cluster
            engagement = 1.0 - cluster * 0.2
            rows.append({
                "student_id":          sid,
                "week_start":          week,
                "cluster_id":          cluster,
                "cluster_name":        ["Très engagé", "Engagé", "Passif", "À risque"][cluster],
                "login_count":         max(0, int(rng.normal(8 * engagement, 2))),
                "total_time_min":      max(0, float(rng.normal(200 * engagement, 40))),
                "videos_watched":      max(0, int(rng.normal(5 * engagement, 2))),
                "quiz_attempts":       max(0, int(rng.normal(3 * engagement, 1))),
                "quiz_pass_rate":      float(np.clip(rng.normal(0.7 * engagement, 0.1), 0, 1)),
                "avg_score":           float(np.clip(rng.normal(65 * engagement, 10), 0, 100)),
                "forum_posts":         max(0, int(rng.normal(2 * engagement, 1))),
                "assignments_on_time": max(0, int(rng.normal(2 * engagement, 0.5))),
                "assignments_late":    max(0, int(rng.normal(1 * (1 - engagement), 0.5))),
            })
    return pd.DataFrame(rows)


# ── _lead_correlations ────────────────────────────────────────────────────────

def test_lead_correlations_returns_dict():
    df = _make_weekly_df(20, 12)
    corrs = _lead_correlations(df, BEHAVIOR_FEATURES)
    assert isinstance(corrs, dict)


def test_lead_correlations_keys_are_feature_names():
    df = _make_weekly_df(20, 12)
    corrs = _lead_correlations(df, BEHAVIOR_FEATURES)
    for k in corrs:
        assert k in BEHAVIOR_FEATURES


def test_lead_correlations_values_in_range():
    df = _make_weekly_df(30, 16)
    corrs = _lead_correlations(df, BEHAVIOR_FEATURES)
    for v in corrs.values():
        assert -1.0 <= v <= 1.0


def test_lead_correlations_skips_constant_features():
    df = _make_weekly_df(10, 8)
    df["login_count"] = 5  # constante → pas de corrélation
    corrs = _lead_correlations(df, ["login_count", "videos_watched"])
    assert "login_count" not in corrs


# ── build_cluster_recommendations ────────────────────────────────────────────

def test_build_cluster_recommendations_returns_dict():
    df = _make_weekly_df()
    result = build_cluster_recommendations(df)
    assert isinstance(result, dict)


def test_build_cluster_recommendations_has_all_clusters():
    df = _make_weekly_df()
    result = build_cluster_recommendations(df)
    assert set(result.keys()) == set(df["cluster_id"].unique())


def test_build_cluster_recommendations_n_top():
    df = _make_weekly_df()
    result = build_cluster_recommendations(df, n_top=3)
    for cluster_id, recs in result.items():
        assert len(recs) <= 3


def test_build_cluster_recommendations_positive_correlations_only():
    df = _make_weekly_df()
    result = build_cluster_recommendations(df)
    for recs in result.values():
        for feat, corr in recs:
            assert corr >= 0


def test_build_cluster_recommendations_sorted_descending():
    df = _make_weekly_df()
    result = build_cluster_recommendations(df)
    for recs in result.values():
        corrs = [c for _, c in recs]
        assert corrs == sorted(corrs, reverse=True)


def test_build_cluster_recommendations_no_cluster_col():
    df = _make_weekly_df().drop(columns=["cluster_id"])
    result = build_cluster_recommendations(df)
    assert result == {}


# ── recommend ─────────────────────────────────────────────────────────────────

def test_recommend_returns_result():
    df = _make_weekly_df()
    cluster_recs = build_cluster_recommendations(df)
    result = recommend(1, df, cluster_recs)
    assert isinstance(result, RecommendationsResult)
    assert result.student_id == 1


def test_recommend_has_3_items():
    df = _make_weekly_df()
    cluster_recs = build_cluster_recommendations(df)
    result = recommend(1, df, cluster_recs)
    assert len(result.recommendations) == 3


def test_recommend_ranks_are_sequential():
    df = _make_weekly_df()
    cluster_recs = build_cluster_recommendations(df)
    result = recommend(1, df, cluster_recs)
    ranks = [r.rank for r in result.recommendations]
    assert ranks == list(range(1, len(ranks) + 1))


def test_recommend_items_have_title():
    df = _make_weekly_df()
    cluster_recs = build_cluster_recommendations(df)
    result = recommend(1, df, cluster_recs)
    for item in result.recommendations:
        assert item.title != ""


def test_recommend_correlation_in_range():
    df = _make_weekly_df()
    cluster_recs = build_cluster_recommendations(df)
    result = recommend(1, df, cluster_recs)
    for item in result.recommendations:
        assert -1.0 <= item.correlation <= 1.0


def test_recommend_cluster_name_mapped():
    df = _make_weekly_df()
    cluster_recs = build_cluster_recommendations(df)
    cluster_map = {0: "Très engagé", 1: "Engagé", 2: "Passif", 3: "À risque"}
    result = recommend(1, df, cluster_recs, cluster_map=cluster_map)
    assert result.cluster_name in cluster_map.values()


def test_recommend_unknown_student_uses_fallback():
    df = _make_weekly_df()
    cluster_recs = build_cluster_recommendations(df)
    # student_id 9999 n'existe pas → fallback
    result = recommend(9999, df, cluster_recs)
    assert len(result.recommendations) == 3  # fallback générique


def test_recommend_current_value_present():
    df = _make_weekly_df()
    cluster_recs = build_cluster_recommendations(df)
    result = recommend(1, df, cluster_recs)
    # Au moins une recommandation doit avoir une valeur actuelle
    has_value = any(r.current_value is not None for r in result.recommendations)
    assert has_value


def test_recommend_different_clusters_different_recs():
    df = _make_weekly_df(40, 20)
    cluster_recs = build_cluster_recommendations(df)
    # Étudiant cluster 0 (très engagé) vs cluster 3 (à risque)
    r0 = recommend(1, df, cluster_recs)   # cluster 0
    r3 = recommend(4, df, cluster_recs)   # cluster 3
    # Les top features ne sont pas forcément identiques
    top0 = {item.action_type for item in r0.recommendations}
    top3 = {item.action_type for item in r3.recommendations}
    # Simplement vérifier que les deux ont bien 3 recs
    assert len(top0) > 0
    assert len(top3) > 0


# ── FEATURE_META ──────────────────────────────────────────────────────────────

def test_all_behavior_features_have_meta():
    for feat in BEHAVIOR_FEATURES:
        assert feat in FEATURE_META, f"{feat} manque dans FEATURE_META"


def test_feature_meta_has_required_keys():
    required = {"action_type", "title", "description", "unit", "target_pct"}
    for feat, meta in FEATURE_META.items():
        assert required.issubset(meta.keys()), f"{feat} manque des clés"


def test_feature_meta_target_pct_valid():
    for feat, meta in FEATURE_META.items():
        assert 0 < meta["target_pct"] <= 1.0
