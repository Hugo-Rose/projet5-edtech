"""Vue Étudiant — gauge engagement, progression, recommandations."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ACTION_ICONS = {
    "connections":   "🔗",
    "time":          "⏱️",
    "videos":        "🎬",
    "quiz":          "📝",
    "quiz_quality":  "🎯",
    "forum":         "💬",
    "assignments":   "📋",
}

GAUGE_COLORS = {
    (0,   30):  "#dc3545",
    (30,  55):  "#fd7e14",
    (55,  75):  "#ffc107",
    (75, 100):  "#28a745",
}


def _gauge_color(score: float) -> str:
    for (lo, hi), color in GAUGE_COLORS.items():
        if lo <= score < hi:
            return color
    return "#28a745"


def render(data: dict[str, pd.DataFrame]) -> None:
    weekly   = data["weekly"]
    students = data["students"]
    preds    = data["predictions"]

    st.header("Vue Étudiant")

    # ── Sélecteur étudiant ────────────────────────────────────────────────────
    student_ids = sorted(students["student_id"].unique())
    sid = st.selectbox(
        "Sélectionner un étudiant (ID)",
        student_ids,
        format_func=lambda i: (
            f"{i} — "
            + str(students.loc[students["student_id"]==i, "first_name"].iloc[0])
            + " "
            + str(students.loc[students["student_id"]==i, "last_name"].iloc[0])
        ),
    )

    s_row  = students[students["student_id"] == sid].iloc[0]
    sw     = weekly[weekly["student_id"] == sid].sort_values("week_start")
    sp     = preds[preds["student_id"] == sid]

    cluster_name = s_row.get("cluster_name", "Engagé")
    if pd.isna(cluster_name):
        cluster_name = "Engagé"

    st.divider()

    # ── Infos étudiant ────────────────────────────────────────────────────────
    col_info, col_gauge = st.columns([2, 1])

    with col_info:
        st.subheader(f"{s_row.get('first_name','')} {s_row.get('last_name','')}")
        cid   = int(s_row["cohort_id"]) if pd.notna(s_row.get("cohort_id")) else None
        cname = ""
        if cid and not data["cohorts"].empty:
            c = data["cohorts"][data["cohorts"]["cohort_id"] == cid]
            if not c.empty:
                cname = c["name"].iloc[0]
        st.caption(f"Cohorte : {cname or '—'}  |  Statut : {s_row.get('status','—')}")
        st.caption(f"Profil K-Means : **{cluster_name}**")

        if not sp.empty:
            prob = float(sp["dropout_prob"].iloc[0])
            label = sp["risk_label"].iloc[0]
            color = {"low":"green","medium":"orange",
                     "high":"darkorange","critical":"red"}.get(label,"black")
            st.markdown(
                f"Risque de décrochage : "
                f"<span style='color:{color};font-weight:bold'>"
                f"{prob:.1%} ({label.upper()})</span>",
                unsafe_allow_html=True,
            )

    with col_gauge:
        last_score = float(sw["engagement_score"].iloc[-1]) if not sw.empty else 50.0
        color = _gauge_color(last_score)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=last_score,
            number={"suffix": "/100", "font": {"size": 28}},
            title={"text": "Score d'engagement", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": color, "thickness": 0.3},
                "bgcolor": "white",
                "steps": [
                    {"range": [0,  30], "color": "#ffeaea"},
                    {"range": [30, 55], "color": "#fff3cd"},
                    {"range": [55, 75], "color": "#fff8e1"},
                    {"range": [75,100], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.85,
                    "value": last_score,
                },
            },
        ))
        fig_gauge.update_layout(
            height=230, margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Progression temporelle ────────────────────────────────────────────────
    st.subheader("Progression sur les 12 dernières semaines")

    if sw.empty:
        st.info("Pas de données hebdomadaires pour cet étudiant.")
    else:
        tab1, tab2, tab3 = st.tabs(["Engagement", "Scores Quiz", "Temps connecté"])

        with tab1:
            fig = px.area(
                sw, x="week_start", y="engagement_score",
                labels={"week_start": "Semaine", "engagement_score": "Score /100"},
                color_discrete_sequence=["#2c3e50"],
            )
            fig.add_hline(y=55, line_dash="dot", line_color="#ffc107",
                          annotation_text="Seuil attention")
            fig.add_hline(y=30, line_dash="dot", line_color="#dc3545",
                          annotation_text="Seuil critique")
            fig.update_layout(height=280, margin=dict(l=20,r=20,t=20,b=40))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sw["week_start"], y=sw["avg_score"].fillna(0),
                mode="lines+markers", name="Score moyen",
                line=dict(color="#3498db"),
            ))
            if "quiz_pass_rate" in sw.columns:
                fig2.add_trace(go.Scatter(
                    x=sw["week_start"],
                    y=(sw["quiz_pass_rate"].fillna(0) * 100),
                    mode="lines+markers", name="Taux réussite (%)",
                    line=dict(color="#e67e22", dash="dash"),
                    yaxis="y2",
                ))
            fig2.update_layout(
                height=280, margin=dict(l=20,r=20,t=20,b=40),
                yaxis=dict(title="Score /100"),
                yaxis2=dict(title="Taux réussite (%)", overlaying="y",
                            side="right", range=[0,100]),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            fig3 = px.bar(
                sw, x="week_start", y="total_time_min",
                labels={"week_start": "Semaine", "total_time_min": "Minutes"},
                color="total_time_min",
                color_continuous_scale="Blues",
            )
            fig3.update_layout(height=280, margin=dict(l=20,r=20,t=20,b=40),
                               showlegend=False,
                               coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    # ── Recommandations ───────────────────────────────────────────────────────
    st.subheader("Recommandations personnalisées")
    _render_recommendations(sid, data["weekly"], cluster_name)


# ── Rendu recommandations ─────────────────────────────────────────────────────

def _render_recommendations(
    student_id:   int,
    weekly_df:    pd.DataFrame,
    cluster_name: str,
) -> None:
    from src.models.recommender import get_recommendations_for_student

    with st.spinner("Calcul des recommandations..."):
        result = get_recommendations_for_student(student_id, weekly_df)

    if result is None or not result.recommendations:
        st.info("Recommandations non disponibles — données insuffisantes.")
        return

    for rec in result.recommendations:
        icon = ACTION_ICONS.get(rec.action_type, "📌")
        with st.container():
            col_icon, col_content, col_stats = st.columns([0.5, 4, 2])

            with col_icon:
                st.markdown(f"<h2 style='text-align:center;margin-top:8px'>{icon}</h2>",
                            unsafe_allow_html=True)

            with col_content:
                st.markdown(f"**{rec.title}**")
                st.caption(rec.description)
                if rec.expected_improvement:
                    st.markdown(
                        f"<span style='color:#28a745;font-size:0.85em'>"
                        f"📈 {rec.expected_improvement}</span>",
                        unsafe_allow_html=True,
                    )

            with col_stats:
                if rec.current_value is not None and rec.target_value is not None:
                    st.metric(
                        label=rec.unit,
                        value=f"{rec.current_value:.1f}",
                        delta=f"cible : {rec.target_value:.1f}",
                        delta_color="off",
                    )
                elif rec.current_value is not None:
                    st.metric(label=rec.unit, value=f"{rec.current_value:.1f}")

            st.markdown(
                "<hr style='margin:4px 0;border-color:#eee'>",
                unsafe_allow_html=True,
            )
