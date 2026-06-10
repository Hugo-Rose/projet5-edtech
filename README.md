# Analytics Pédagogique & Prédiction du Décrochage Scolaire

![pytest](https://img.shields.io/badge/pytest-155%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-compose-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B)

Projet 5 EdTech — Pipeline complet de bout en bout : ingestion de données LMS,
feature engineering, modèles ML (classification, séries temporelles, clustering),
API de prédiction, monitoring Evidently AI et dashboard analytique multi-rôles.

---

## Lancer le projet en 3 commandes

```bash
# 1. Démarrer l'infrastructure (PostgreSQL · MLflow · API · Dashboard)
docker compose up -d

# 2. Exécuter le pipeline complet (données → modèles → dashboard)
python run_all.py

# 3. Ouvrir le dashboard
#    → http://localhost:8501
```

> **Mode démo sans Docker :** `python run_all.py --skip-monitoring --no-streamlit`
> puis `streamlit run src/dashboard/app.py`

---

## Résultats finaux — Dataset UCI (n = 4 424, 32,1 % décrochage)

### Prédiction du décrochage — XGBoost + Optuna HPO

| Métrique | Valeur | Cible |
|----------|--------|-------|
| **ROC-AUC** | **0.922** | ≥ 0.85 ✅ |
| **PR-AUC** | 0.887 | — |
| **Recall** | **0.961** | maximiser ✅ |
| **F1-Score** | 0.658 | — |
| **Precision** | 0.500 | ≥ 50 % ✅ |
| **Spécificité** | 0.525 | — |

> Seuil optimal : **0.047** (maximise recall sous contrainte precision ≥ 50 %)
> Algorithme : XGBoost (n_estimators=778, max_depth=8, lr=0.16)

### Clustering K-Means (profils étudiants)

| Métrique | Valeur |
|----------|--------|
| **k optimal** | **2** (sélection auto via silhouette) |
| **Silhouette score** | **0.497** |
| **Inertia** | 18 233 |

---

## Objectifs par module

| Module | Description | Statut |
|--------|-------------|--------|
| **Ingestion** | ~10 000 étudiants + événements LMS sur 2 ans | ✅ |
| **Feature Engineering** | Agrégats hebdomadaires + rolling 4 semaines | ✅ |
| **ML — Décrochage** | XGBoost + Optuna HPO + SHAP | ✅ |
| **ML — Forecast** | Prophet × 4 métriques × 4 cohortes | ✅ |
| **Clustering** | K-Means auto-k + recommandations content-based | ✅ |
| **Monitoring** | Drift (Evidently AI) + Fairness (3 attributs) | ✅ |
| **API** | FastAPI : prédiction temps réel + batch | ✅ |
| **Dashboard** | Streamlit multi-rôles + thème sombre | ✅ |

---

## Stack technique

```
Python 3.11 · PostgreSQL 15 · Pandas · scikit-learn · XGBoost · LightGBM
Prophet · FastAPI · Streamlit · Plotly · MLflow · Evidently AI · Docker
```

---

## Architecture

```
projet5-edtech/
├── data/
│   ├── raw/                 # CSV bruts (UCI, OULAD, synthétiques)
│   ├── processed/           # Données nettoyées
│   ├── features/            # Feature stores (parquet)
│   ├── models/              # Modèles joblib sérialisés
│   └── reports/             # Rapports HTML Evidently (drift, fairness)
├── src/
│   ├── data/                # Ingestion, ETL, génération synthétique
│   ├── features/            # Feature engineering hebdomadaire
│   ├── models/              # Entraînement, évaluation, inférence
│   ├── api/                 # FastAPI — endpoints REST
│   ├── monitoring/          # Drift report + Fairness report
│   └── dashboard/           # Streamlit multi-rôles
├── tests/                   # 155 tests pytest
├── docker/
│   ├── init.sql             # Schéma PostgreSQL
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── .streamlit/config.toml   # Thème sombre
├── docker-compose.yml
├── run_all.py               # Orchestrateur pipeline complet
└── requirements_lock.txt    # Dépendances figées (pip freeze)
```

---

## Sources de données

| Source | Commande | Étudiants | Taux décrochage |
|--------|----------|-----------|-----------------|
| **UCI Dropout** (ID=697) | `python -m src.data.load_uci_dropout` | 4 424 | 32,1 % |
| **OULAD** (HuggingFace) | `python -m src.data.load_oulad` | ~32 000 | variable |
| **Synthétique** (fallback) | `python -m src.data.generate_synthetic_data` | 2 000–10 000 | ~16 % |

```bash
# Sélection de la source au moment du feature engineering
python -m src.features.build_features --source uci
python -m src.features.build_features --source oulad
python -m src.features.build_features --source synthetic
python -m src.features.build_features --source db        # PostgreSQL (défaut)
```

---

## Dashboard — captures d'écran

Le dashboard tourne sur **http://localhost:8501** après `python run_all.py`.

Pour faire des captures d'écran pour votre rapport/soutenance :

```bash
# 1. Lancer le dashboard
streamlit run src/dashboard/app.py

# 2. Naviguer dans les 3 vues depuis la sidebar :
#    - Enseignant : heatmap engagement, top 10 critiques, clusters, forecast
#    - Étudiant   : gauge score, progression avg_score, recommandations
#    - Admin      : KPIs globaux, fairness, runs MLflow, bouton drift

# 3. Capture via navigateur : F12 → Device toolbar → 1280×800
#    Ou outil système : Windows Snipping Tool (Win+Shift+S)
```

**Vues disponibles :**

| Vue | Contenus clés |
|-----|--------------|
| **Enseignant** | Heatmap activité (px.imshow), top 10 risque critique, pie chart K-Means, courbe forecast Prophet |
| **Étudiant** | Gauge engagement (go.Indicator), progression avg_score, badge risque coloré, recommandations cluster |
| **Admin** | KPIs + taux décrochage, rapport fairness HTML, runs MLflow, bouton "Générer rapport drift" |

---

## Pipeline détaillé

```bash
# ── Données ────────────────────────────────────────────────────────────────
python -m src.data.load_uci_dropout
python -m src.features.build_features --source uci

# ── Modèles ────────────────────────────────────────────────────────────────
python -m src.models.train_dropout \
  --features data/features/weekly_features_uci.parquet --trials 30
python -m src.models.train_clustering \
  --features data/features/weekly_features_uci.parquet
python -m src.models.predict --features data/features/weekly_features_uci.parquet --no-db

# ── Monitoring ─────────────────────────────────────────────────────────────
python -m src.monitoring.drift_report     # Evidently ColumnDriftMetric, baseline 8 sem.
python -m src.monitoring.fairness_report  # gender / age_group / scholarship

# ── Dashboard ──────────────────────────────────────────────────────────────
streamlit run src/dashboard/app.py
```

---

## Monitoring (Evidently AI)

- **Drift** : `ColumnDriftMetric` par feature vs baseline (8 premières semaines)
  - Alerte console si score > 0.15
  - Log MLflow : `pct_features_drift`, `avg_drift_score`, `n_features_drift`
  - Export HTML → `data/reports/drift_YYYY-WW.html`
- **Fairness** : taux FP/FN par groupe (gender, age_group, scholarship)
  - Demographic Parity, Equal Opportunity, Predictive Equality
  - Alerte si écart entre groupes > 10 %
  - Export HTML → `data/reports/fairness_YYYY-MM-DD.html`

---

## API (FastAPI)

```bash
# Lancer l'API seule
uvicorn src.api.main:app --reload --port 8000

# Endpoints principaux
GET  /students          # Liste des étudiants
GET  /students/{id}     # Détail étudiant
GET  /cohorts           # Cohortes
POST /predictions       # Prédiction décrochage (temps réel)
POST /predictions/batch # Batch scoring
GET  /alerts            # Alertes ouvertes
```

Swagger UI → **http://localhost:8000/docs**

---

## Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
# 155 tests, 0 échec
```

| Fichier | Tests | Couverture |
|---------|-------|------------|
| `test_monitoring.py` | 23 | drift KS, fairness, seuil 0.15 |
| `test_models.py` | 16 | XGBoost, SHAP, évaluation |
| `test_api.py` | 16 | FastAPI endpoints |
| `test_recommender.py` | 22 | recommandations content-based |
| `test_forecast.py` | 19 | Prophet, fallback |
| `test_build_features.py` | 8 | feature engineering |
| `test_generate_data.py` | 10 | synthétique |
| `test_real_datasets.py` | 41 | UCI Dropout, OULAD |

---

## Variables d'environnement

Copier `.env.example` → `.env` :

```bash
cp .env.example .env
```

| Variable | Défaut | Description |
|----------|--------|-------------|
| `DATABASE_URL` | `postgresql://edtech:edtech_pass@localhost:5432/edtech_db` | PostgreSQL |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow |
| `API_URL` | `http://localhost:8000` | FastAPI (pour le dashboard) |
