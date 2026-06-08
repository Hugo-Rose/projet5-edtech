.PHONY: install install-ci test test-cov lint \
        generate-data load-db build-features \
        train forecast predict \
        drift fairness pipeline \
        run-api run-dashboard \
        docker-up docker-down docker-logs

# ── Installation ───────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-ci:
	pip install -r requirements-ci.txt

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Rapport HTML : htmlcov/index.html"

lint:
	ruff check src/ tests/

# ── Pipeline données ───────────────────────────────────────────────────────────
generate-data:
	python -m src.data.generate_synthetic_data --students 10000

load-db:
	python -m src.data.load_to_db --truncate

build-features:
	python -m src.features.build_features

# ── Modèles ────────────────────────────────────────────────────────────────────
train:
	python -m src.models.train_dropout --trials 30

forecast:
	python -m src.models.forecast_engagement

predict:
	python -m src.models.predict

# ── Monitoring ─────────────────────────────────────────────────────────────────
drift:
	python -m src.monitoring.drift_report

fairness:
	python -m src.monitoring.fairness_report

# ── Pipeline complet (hors Docker) ────────────────────────────────────────────
pipeline: generate-data load-db build-features train forecast predict drift fairness
	@echo "Pipeline terminé."

# ── Serveurs locaux ────────────────────────────────────────────────────────────
run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run src/dashboard/app.py

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-pipeline: docker-up
	@echo "Attente PostgreSQL..."
	sleep 10
	$(MAKE) generate-data load-db build-features train forecast predict
