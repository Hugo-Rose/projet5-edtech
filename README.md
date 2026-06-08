# Analytics Pédagogique & Prédiction du Décrochage Scolaire

Projet 5 EdTech — Pipeline complet de bout en bout : ingestion de données LMS,
feature engineering, modèles ML (classification, séries temporelles), API de
prédiction et dashboard analytique interactif.

## Objectifs

| Module | Description |
|--------|-------------|
| **Ingestion** | Génération / chargement de ~10 000 étudiants + événements LMS sur 2 ans |
| **Feature Engineering** | Agrégats hebdomadaires d'engagement, performance, assiduité |
| **ML — Décrochage** | Classifieur XGBoost/LightGBM : probabilité de décrochage par semaine |
| **ML — Prévision** | Prophet : prévision du taux d'engagement à 4 semaines par cohorte |
| **MLOps** | Tracking MLflow, monitoring de drift Evidently AI, versioning des modèles |
| **API** | FastAPI : endpoint de prédiction temps réel + batch scoring |
| **Dashboard** | Streamlit + Plotly : KPIs, heatmaps, alertes pédagogiques |

## Stack technique

```
Python 3.11 · PostgreSQL 15 · Pandas · scikit-learn · XGBoost · LightGBM
Prophet · FastAPI · Streamlit · Plotly · MLflow · Evidently AI · Docker
```

## Démarrage rapide

```bash
# 1. Copier les variables d'environnement
cp .env.example .env

# 2. Lancer tous les services
docker compose up -d

# 3. Générer les données synthétiques
python -m src.data.generate_synthetic_data --students 10000

# 4. Charger dans PostgreSQL
python -m src.data.load_to_db

# Services disponibles :
#   PostgreSQL  → localhost:5432
#   MLflow UI   → http://localhost:5000
#   API FastAPI → http://localhost:8000/docs
#   Dashboard   → http://localhost:8501
```

## Architecture

```
projet5-edtech/
├── data/
│   ├── raw/            # CSV synthétiques générés
│   ├── processed/      # Données nettoyées
│   └── features/       # Feature stores (parquet)
├── src/
│   ├── data/           # Génération, chargement, ETL
│   ├── features/       # Feature engineering (weekly_features)
│   ├── models/         # Entraînement, évaluation, inférence
│   ├── api/            # FastAPI — endpoints de prédiction
│   └── dashboard/      # Streamlit — visualisation
├── docker/
│   ├── init.sql        # Schéma PostgreSQL
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── mlflow/artifacts/   # Modèles MLflow
├── notebooks/          # EDA, expérimentations
├── tests/              # pytest
└── docker-compose.yml
```

## Modèles ML

### Prédiction du décrochage (classification)
- **Features** : logins/semaine, temps connecté, taux de réussite quiz, posts forum,
  retards de rendu, tendance sur 4 semaines glissantes, profil socio-démographique
- **Algorithme** : XGBoost + SHAP pour l'explicabilité
- **Métrique cible** : ROC-AUC ≥ 0.85, Recall classe décrochage ≥ 0.80

### Prévision de l'engagement (séries temporelles)
- **Algorithme** : Prophet par cohorte
- **Horizon** : 4 semaines
- **Signal** : sessions hebdomadaires, taux de connexion

## Monitoring (Evidently AI)
- Drift des features d'engagement (référence : semaines 1-8)
- Distribution des scores de prédiction (alerte si drift > 0.15)
- Rapport HTML généré hebdomadairement

## Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```
