"""Vue Enseignant — heatmap, top risques, clusters, forecast."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RISK_COLORS = {
    "low":      "#28a745",
    "medium":   "#ffc107",
    "high":     "#fd7e14",
    "critical": "#dc3545",
}


def render(data: dict[str, pd.DataFrame]) -> None:
    weekly   = data["weekly"]
    students = data["students"]
    forecasts = data.get("forecasts", pd.DataFrame())

    st.header("Vue Enseignant")

    # ── Sélecteur cohorte ─────────────────────────────────────────────────────
    cohorts   = data["cohorts"]
    cohort_opts = ["Toutes"] + cohorts["name"].tolist()
    selected  = st.selectbox("Cohorte", cohort_opts)

    if selected != "Toutes":
        cid = int(cohorts.loc[cohorts["name"] == selected, "cohort_id"].iloc[0])
        sid_filter = students.loc[students["cohort_id"] == cid, "student_id"]
        weekly   = weekly[weekly["student_id"].isin(sid_filter)]
        students = students[students["cohort_id"] == cid]

    st.divider()

    # ── Heatmap engagement ────────────────────────────────────────────────────
    st.subheader("Heatmap d'activité hebdomadaire")

    last_8 = sorted(weekly["week_start"].unique())[-8:]
    hw = weekly[weekly["week_start"].isin(last_8)]

    # Sous-échantillon : top 40 étudiants par risque
    top_ids = (
        students.sort_values("dropout_prob", ascending=False)
        ["student_id"].head(40).tolist()
    )
    hw = hw[hw["student_id"].isin(top_ids)]

    pivot = hw.pivot_table(
        index="student_id", columns="week_start",
        values="engagement_score", aggfunc="mean",
    )
    pivot.columns = [str(c)[:10] for c in pivot.columns]
    pivot = pivot.fillna(0)

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=100,
        labels={"x": "Semaine", "y": "Étudiant", "color": "Engagement"},
        aspect="auto",
    )
    fig_heat.update_layout(
        height=420,
        margin=dict(l=60, r=20, t=40, b=40),
        coloraxis_colorbar_title="Score",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Top 10 à risque ───────────────────────────────────────────────────────
    with col1:
        st.subheader("Top 10 étudiants critiques")
        top10 = (
            students[students["risk_label"].isin(["high", "critical"])]
            .sort_values("dropout_prob", ascending=False)
            .head(10)
        )
        if top10.empty:
            st.info("Aucun étudiant critique dans cette cohorte.")
        else:
            display = top10[["student_id", "first_name", "last_name",
                              "dropout_prob", "risk_label"]].copy()
            display["dropout_prob"] = display["dropout_prob"].map("{:.1%}".format)
            display.columns = ["ID", "Prénom", "Nom", "Prob. décrochage", "Risque"]

            def color_risk(val):
                color = RISK_COLORS.get(val, "#333")
                return f"color: {color}; font-weight: bold"

            st.dataframe(
                display.style.applymap(color_risk, subset=["Risque"]),
                use_container_width=True, hide_index=True,
            )

    # ── Distribution clusters ─────────────────────────────────────────────────
    with col2:
        st.subheader("Distribution des profils (K-Means)")
        if "cluster_name" in students.columns:
            cluster_counts = (
                students["cluster_name"].value_counts().reset_index()
            )
            cluster_counts.columns = ["Profil", "Effectif"]
            fig_clust = px.pie(
                cluster_counts, names="Profil", values="Effectif",
                color="Profil",
                color_discrete_map={
                    "Très engagé": "#28a745",
                    "Engagé":      "#17a2b8",
                    "Passif":      "#ffc107",
                    "À risque":    "#dc3545",
                },
                hole=0.35,
            )
            fig_clust.update_traces(textinfo="percent+label")
            fig_clust.update_layout(
                showlegend=False, height=340,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_clust, use_container_width=True)
        else:
            st.info("Clustering non disponible.")

    # ── Forecast Prophet ──────────────────────────────────────────────────────
    st.subheader("Prévision d'engagement — Prophet (4 semaines)")
    _render_forecast(weekly, forecasts, selected, cohorts)


def _render_forecast(
    weekly: pd.DataFrame,
    forecasts: pd.DataFrame,
    selected_cohort: str,
    cohorts: pd.DataFrame,
) -> None:
    metric_labels = {
        "avg_logins":         "Connexions moyennes",
        "avg_time_min":       "Temps connecté (min)",
        "avg_quiz_pass_rate": "Taux réussite quiz",
        "dropout_risk_p50":   "Risque décrochage (médiane)",
    }
    metric = st.selectbox("Métrique", list(metric_labels.keys()),
                          format_func=lambda k: metric_labels[k])

    # Historique
    hist = (
        weekly.groupby("week_start")
        .agg(
            avg_logins        =("login_count",      "mean"),
            avg_time_min      =("total_time_min",   "mean"),
            avg_quiz_pass_rate=("quiz_pass_rate",   "mean"),
            dropout_risk_p50  =("dropout_risk_score","median"),
        )
        .reset_index()
        .rename(columns={"week_start": "ds"})
    )
    hist["ds"] = pd.to_datetime(hist["ds"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist[metric],
        mode="lines+markers", name="Historique",
        line=dict(color="#2c3e50", width=2),
    ))

    if not forecasts.empty:
        cid = None
        if selected_cohort != "Toutes" and not cohorts.empty:
            row = cohorts[cohorts["name"] == selected_cohort]
            if not row.empty:
                cid = int(row["cohort_id"].iloc[0])

        fc = forecasts[forecasts["metric"] == metric]
        if cid:
            fc = fc[fc["cohort_id"] == cid]
        fc = fc[fc["is_future"]].sort_values("ds")

        if not fc.empty:
            fc["ds"] = pd.to_datetime(fc["ds"])
            fig.add_trace(go.Scatter(
                x=pd.concat([hist["ds"].iloc[[-1]], fc["ds"]]),
                y=pd.concat([hist[metric].iloc[[-1]], fc["yhat"]]),
                mode="lines+markers", name="Prévision",
                line=dict(color="#e74c3c", width=2, dash="dash"),
                marker=dict(symbol="diamond"),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
                y=pd.concat([fc["yhat_upper"], fc["yhat_lower"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(231,76,60,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=True, name="IC 80%",
            ))

    fig.update_layout(
        height=360, hovermode="x unified",
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Semaine", yaxis_title=metric_labels[metric],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
