## Résumé

<!-- Décrivez ce que fait cette PR et pourquoi elle est nécessaire. -->

## Type de changement

- [ ] Bugfix
- [ ] Nouvelle fonctionnalité
- [ ] Refactoring / nettoyage
- [ ] Documentation
- [ ] CI/CD / configuration

## Modules impactés

<!-- Cochez les modules modifiés. -->
- [ ] `src/data/` — ingestion, génération
- [ ] `src/features/` — feature engineering
- [ ] `src/models/` — entraînement, évaluation, inférence
- [ ] `src/api/` — FastAPI
- [ ] `src/dashboard/` — Streamlit
- [ ] `src/monitoring/` — drift, fairness
- [ ] `docker/` / `docker-compose.yml`
- [ ] Tests

## Plan de test

- [ ] `pytest tests/` passe sans erreur
- [ ] Mode démo dashboard fonctionnel (`streamlit run src/dashboard/app.py`)
- [ ] Pas de secrets ni de données réelles committées

## Checklist

- [ ] Les tests existants passent (`pytest tests/ -q`)
- [ ] Des tests ont été ajoutés pour le nouveau code
- [ ] Le README est à jour si nécessaire
- [ ] Aucune dépendance non versionnée ajoutée
