"""
Dashboard EdTech — point d'entrée Streamlit multi-rôles.

3 vues : Enseignant | Étudiant | Administrateur
Données : PostgreSQL (si dispo) ou mode démo automatique.
"""
import streamlit as st

st.set_page_config(
    page_title="EdTech Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header logo ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        display:flex;align-items:center;gap:14px;
        padding:.6rem 0 .2rem 0;margin-bottom:.5rem;
        border-bottom:1px solid #2d3139;
    ">
      <span style="font-size:2.4rem;line-height:1">🎓</span>
      <div>
        <span style="font-size:1.45rem;font-weight:700;color:#fafafa;
                     letter-spacing:.5px">EdTech Analytics</span><br>
        <span style="font-size:.78rem;color:#8b949e;letter-spacing:.3px">
          Plateforme d'analyse pédagogique &amp; prédiction décrochage
        </span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Chargement données (cache 5 min) ──────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Chargement des données…")
def _get_data():
    from src.dashboard.data_loader import load_all_data
    return load_all_data()


data, is_demo = _get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EdTech Analytics")
    st.divider()

    role = st.radio(
        "Rôle",
        ["Enseignant", "Étudiant", "Administrateur"],
        index=0,
    )

    st.divider()
    if is_demo:
        st.warning("⚠ Mode démo\nDonnées simulées — base de données inaccessible.")
    else:
        st.success("✔ Connecté à PostgreSQL")

    n_students = len(data["students"])
    n_alerts   = len(data["alerts"])
    st.metric("Étudiants", n_students)
    st.metric("Alertes ouvertes", n_alerts)
    st.divider()
    st.caption("Projet 5 — Analytics Pédagogique")

# ── Contenu principal ─────────────────────────────────────────────────────────
if role == "Enseignant":
    from src.dashboard.pages.teacher import render
    render(data)

elif role == "Étudiant":
    from src.dashboard.pages.student import render
    render(data)

else:
    from src.dashboard.pages.admin import render
    render(data)
