#!/usr/bin/env python3
"""
run_all.py — Pipeline complet EdTech, du chargement des données au dashboard.

Enchaîne :
  1. Chargement UCI Dropout
  2. Feature engineering (source=uci)
  3. Entraînement modèle de décrochage (XGBoost + Optuna)
  4. Entraînement clustering K-Means
  5. Inférence (prédictions)
  6. Rapport de drift (Evidently AI / KS fallback)
  7. Rapport fairness
  8. Lancement Streamlit en arrière-plan

Usage:
    python run_all.py
    python run_all.py --source synthetic --trials 5
    python run_all.py --skip-monitoring --no-streamlit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

# ── Constantes ────────────────────────────────────────────────────────────────
FEATURES_UCI  = Path("data/features/weekly_features_uci.parquet")
MLFLOW_URI    = "http://localhost:5000"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(label: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Lance une commande Python et loggue durée + code de retour."""
    cmd = [sys.executable, "-m"] + args
    logger.info(f"  ▶  {' '.join(args)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        logger.error(f"  ✗  {label} — code {result.returncode} ({elapsed:.1f}s)")
        if check:
            sys.exit(result.returncode)
    else:
        logger.success(f"  ✔  {label} ({elapsed:.1f}s)")
    return result


def _section(title: str) -> None:
    logger.info("")
    logger.info(f"{'─'*55}")
    logger.info(f"  {title}")
    logger.info(f"{'─'*55}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline complet EdTech")
    parser.add_argument(
        "--source", choices=["uci", "synthetic"], default="uci",
        help="Source des données (défaut: uci)",
    )
    parser.add_argument(
        "--trials", type=int, default=15,
        help="Nombre de trials Optuna pour train_dropout (défaut: 15)",
    )
    parser.add_argument(
        "--skip-monitoring", action="store_true",
        help="Ne pas générer les rapports drift/fairness (requiert DB)",
    )
    parser.add_argument(
        "--no-streamlit", action="store_true",
        help="Ne pas lancer Streamlit en arrière-plan",
    )
    parser.add_argument(
        "--mlflow-uri", default=MLFLOW_URI,
        help=f"URI MLflow (défaut: {MLFLOW_URI})",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    logger.info("=" * 55)
    logger.info("  EdTech Analytics — Pipeline complet")
    logger.info("=" * 55)

    # ── 1. Données ────────────────────────────────────────────────────────────
    _section("Étape 1/7 — Chargement des données")
    if args.source == "uci":
        _run("Chargement UCI Dropout", ["src.data.load_uci_dropout"])
    else:
        _run("Génération données synthétiques",
             ["src.data.generate_synthetic_data", "--students", "2000"])

    # ── 2. Feature engineering ────────────────────────────────────────────────
    _section("Étape 2/7 — Feature engineering")
    _run(
        "build_features",
        ["src.features.build_features", "--source", args.source],
    )

    # Vérification parquet produit
    if not FEATURES_UCI.exists():
        # Fallback : cherche un parquet alternatif
        candidates = list(Path("data/features").glob("weekly_features_*.parquet"))
        if not candidates:
            logger.error("Aucun fichier de features trouvé dans data/features/")
            sys.exit(1)
        features_path = str(candidates[0])
        logger.warning(f"  Parquet utilisé : {features_path}")
    else:
        features_path = str(FEATURES_UCI)

    # ── 3. Entraînement décrochage ────────────────────────────────────────────
    _section("Étape 3/7 — Entraînement modèle de décrochage")
    _run(
        "train_dropout",
        [
            "src.models.train_dropout",
            "--features", features_path,
            "--trials",   str(args.trials),
            "--mlflow-uri", args.mlflow_uri,
        ],
    )

    # ── 4. Clustering ─────────────────────────────────────────────────────────
    _section("Étape 4/7 — Clustering K-Means")
    _run(
        "train_clustering",
        [
            "src.models.train_clustering",
            "--features",   features_path,
            "--k",          "0",
            "--mlflow-uri", args.mlflow_uri,
        ],
    )

    # ── 5. Prédictions ────────────────────────────────────────────────────────
    _section("Étape 5/7 — Inférence (prédictions)")
    predict_args = ["src.models.predict", "--features", features_path, "--no-db"]
    _run("predict", predict_args)

    # ── 6 & 7. Monitoring (optionnel — requiert DB) ───────────────────────────
    if args.skip_monitoring:
        logger.info("")
        logger.info("  ⏭  Monitoring ignoré (--skip-monitoring).")
    else:
        _section("Étape 6/7 — Rapport de drift")
        _run(
            "drift_report",
            ["src.monitoring.drift_report",
             "--mlflow-uri", args.mlflow_uri],
            check=False,   # non bloquant si DB absente
        )

        _section("Étape 7/7 — Rapport fairness")
        _run(
            "fairness_report",
            ["src.monitoring.fairness_report"],
            check=False,
        )

    # ── Streamlit ─────────────────────────────────────────────────────────────
    if args.no_streamlit:
        logger.info("")
        logger.info("  ⏭  Streamlit non lancé (--no-streamlit).")
    else:
        _section("Lancement Streamlit")
        logger.info("  Démarrage en arrière-plan sur http://localhost:8501 …")
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             "src/dashboard/app.py",
             "--server.headless", "true",
             "--server.port", "8501"],
        )
        logger.success("  ✔  Streamlit lancé — http://localhost:8501")

    # ── Résumé ────────────────────────────────────────────────────────────────
    total = time.perf_counter() - t_start
    logger.info("")
    logger.info("=" * 55)
    logger.success(f"  Pipeline terminé en {total:.1f}s")
    logger.info("  Services :")
    logger.info("    MLflow UI  → http://localhost:5000")
    logger.info("    API FastAPI → http://localhost:8000/docs")
    logger.info("    Dashboard  → http://localhost:8501")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
