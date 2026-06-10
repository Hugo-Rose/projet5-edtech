"""Vue Administrateur — KPIs globaux, fairness, MLflow runs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPORTS_DIR = Path("data/reports")


def render(data: dict[str, pd.DataFrame]) -> None:
    students  = data["students"]
    weekly    = data["weekly"]
    alerts    = data["alerts"]

    st.header("Vue Administrateur")
    st.divider()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.subheader("Indicateurs globaux")
    _render_kpis(students, weekly, alerts)
    st.divider()

    col_left, col_right = st.columns([3, 2])

    # ── Décrochage par cohorte ────────────────────────────────────────────────
    with col_left:
        st.subheader("Taux de décrochage par cohorte")
        _render_cohort_dropout(students, data["cohorts"])

    # ── Distribution des risques ──────────────────────────────────────────────
    with col_right:
        st.subheader("Distribution des risques")
        _render_risk_distribution(students)

    st.divider()

    # ── Rapport Fairness ──────────────────────────────────────────────────────
    st.subheader("Rapport d'équité (Evidently / Fairness)")
    _render_fairness(students)

    st.divider()

    # ── Génération rapport drift ──────────────────────────────────────────────
    st.subheader("Rapport de drift")
    _render_drift_button()

    st.divider()

    # ── MLflow runs ───────────────────────────────────────────────────────────
    st.subheader("Derniers runs MLflow")
    _render_mlflow_runs()


def _render_kpis(students: pd.DataFrame,
                 weekly: pd.DataFrame,
                 alerts: pd.DataFrame) -> None:
    n_total    = len(students)
    n_active   = (students["status"] != "dropped_out").sum()
    n_dropout  = (students["status"] == "dropped_out").sum()
    dr_rate    = n_dropout / max(n_total, 1)

    last_week = weekly.groupby("student_id").last().reset_index()
    avg_eng   = last_week["engagement_score"].mean() if "engagement_score" in last_week else 0
    n_alerts  = len(alerts)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Étudiants actifs",    f"{n_active:,}",
              delta=f"-{n_dropout} décrochages",
              delta_color="inverse")
    c2.metric("Taux décrochage",     f"{dr_rate:.1%}",
              delta_color="inverse")
    c3.metric("Engagement moyen",    f"{avg_eng:.0f}/100",
              delta_color="normal")
    c4.metric("Alertes ouvertes",    str(n_alerts),
              delta_color="inverse")

    # Sous-indicateurs
    if "risk_label" in students.columns:
        risk_counts = students["risk_label"].value_counts()
        n_crit = int(risk_counts.get("critical", 0))
        n_high = int(risk_counts.get("high", 0))
        cols = st.columns(4)
        cols[0].metric("Risque Critique", n_crit, delta_color="inverse")
        cols[1].metric("Risque Élevé",    n_high, delta_color="inverse")
        cols[2].metric("Risque Moyen",    int(risk_counts.get("medium", 0)))
        cols[3].metric("Risque Faible",   int(risk_counts.get("low", 0)))


def _render_cohort_dropout(students: pd.DataFrame,
                            cohorts: pd.DataFrame) -> None:
    if cohorts.empty:
        st.info("Aucune donnée cohorte.")
        return

    merged = students.merge(cohorts[["cohort_id", "name"]], on="cohort_id", how="left")
    stats = (
        merged.groupby("name")
        .agg(
            total     =("student_id", "count"),
            dropouts  =("status",
                        lambda x: (x == "dropped_out").sum()),
        )
        .reset_index()
    )
    stats["taux_decrochage"] = (stats["dropouts"] / stats["total"] * 100).round(1)

    fig = px.bar(
        stats.sort_values("taux_decrochage", ascending=True),
        x="taux_decrochage", y="name",
        orientation="h",
        text="taux_decrochage",
        labels={"taux_decrochage": "Taux décrochage (%)", "name": ""},
        color="taux_decrochage",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=280, margin=dict(l=10,r=40,t=10,b=40),
        coloraxis_showscale=False, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_distribution(students: pd.DataFrame) -> None:
    if "risk_label" not in students.columns:
        st.info("Prédictions non disponibles.")
        return

    risk_order  = ["critical", "high", "medium", "low"]
    risk_colors = {
        "critical": "#dc3545", "high": "#fd7e14",
        "medium": "#ffc107",   "low": "#28a745",
    }
    counts = (
        students["risk_label"].value_counts()
        .reindex(risk_order).fillna(0).reset_index()
    )
    counts.columns = ["Niveau", "Effectif"]

    fig = px.pie(
        counts, names="Niveau", values="Effectif",
        color="Niveau", color_discrete_map=risk_colors,
        hole=0.45,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        height=280, margin=dict(l=10,r=10,t=10,b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_fairness(students: pd.DataFrame) -> None:
    # Cherche le dernier rapport HTML fairness
    reports = sorted(REPORTS_DIR.glob("fairness_*.html"), reverse=True)
    if reports:
        with open(reports[0], encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=500, scrolling=True)
        st.caption(f"Source : `{reports[0].name}`")
    else:
        # Calcul inline simplifié
        st.info("Aucun rapport HTML disponible — calcul inline.")
        _inline_fairness(students)


def _inline_fairness(students: pd.DataFrame) -> None:
    if "dropout_prob" not in students.columns:
        st.warning("Prédictions absentes.")
        return

    for attr in ["gender", "scholarship"]:
        if attr not in students.columns:
            continue
        grp = (
            students.groupby(attr)["dropout_prob"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "Prob. moy.", "count": "N"})
        )
        grp["Prob. moy."] = grp["Prob. moy."].map("{:.1%}".format)
        st.markdown(f"**{attr}**")
        st.dataframe(grp, use_container_width=True, hide_index=True)


def _render_drift_button() -> None:
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("⚡ Générer rapport drift", type="primary")
    with col_status:
        reports = sorted(REPORTS_DIR.glob("drift_*.html"), reverse=True)
        if reports:
            st.caption(f"Dernier rapport : `{reports[0].name}`")

    if run_clicked:
        try:
            from src.monitoring.drift_report import run_drift_report
            with st.spinner("Génération du rapport drift en cours…"):
                result = run_drift_report()
            if result:
                st.success(
                    f"Rapport généré — {result.get('n_drift', 0)} feature(s) "
                    f"en drift ({result.get('pct_drift', 0):.1f}%) | "
                    f"score moyen : {result.get('avg_score', 0):.4f}"
                )
                report_path = Path(result.get("report_path", ""))
                if report_path.exists():
                    with open(report_path, encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=500, scrolling=True)
            else:
                st.warning("Rapport vide — données insuffisantes ou base inaccessible.")
        except Exception as e:
            st.error(f"Erreur lors de la génération : {e}")


def _render_mlflow_runs() -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        client = mlflow.MlflowClient()
        exps   = client.search_experiments()
        if not exps:
            st.info("Aucune expérience MLflow trouvée.")
            return

        all_runs = []
        for exp in exps[:3]:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=5,
                order_by=["start_time DESC"],
            )
            for r in runs:
                all_runs.append({
                    "Expérience": exp.name,
                    "Run":        r.info.run_name or r.info.run_id[:8],
                    "Statut":     r.info.status,
                    "Démarré":    pd.Timestamp(r.info.start_time, unit="ms")
                                  .strftime("%Y-%m-%d %H:%M"),
                    "ROC-AUC":   r.data.metrics.get("roc_auc", "—"),
                    "Drift %":   r.data.metrics.get("pct_features_drift", "—"),
                })
        if all_runs:
            st.dataframe(pd.DataFrame(all_runs), use_container_width=True,
                         hide_index=True)
    except Exception as e:
        st.warning(f"MLflow inaccessible ({e}).")
        st.caption("Lancez `docker compose up mlflow` pour activer le suivi.")
