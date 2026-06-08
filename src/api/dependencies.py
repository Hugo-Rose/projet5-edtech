"""Dépendances FastAPI : session DB, chargement modèle."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Generator

import joblib
from loguru import logger
from sqlalchemy.orm import Session, sessionmaker

from src.data.db import get_engine, get_session

MODELS_DIR = Path("data/models")


# ── Session DB ────────────────────────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Modèle ML ─────────────────────────────────────────────────────────────────

class ModelBundle:
    """Conteneur thread-safe pour le modèle XGBoost + preprocesseur."""

    def __init__(self):
        self.model        = None
        self.preprocessor = None
        self.threshold    = 0.5
        self.version      = "not_loaded"
        self._loaded      = False

    def load(self) -> bool:
        pp_path   = MODELS_DIR / "preprocessor.joblib"
        meta_path = MODELS_DIR / "model_meta.json"

        if not pp_path.exists() or not meta_path.exists():
            logger.warning(
                f"Modèle introuvable dans {MODELS_DIR}. "
                "Lancez src.models.train_dropout d'abord."
            )
            return False

        self.preprocessor = joblib.load(pp_path)
        meta = json.loads(meta_path.read_text())
        self.threshold    = meta.get("threshold", 0.5)
        self.version      = meta.get("run_id", "local")
        self._loaded      = True
        logger.info(f"Modèle chargé (seuil={self.threshold:.2f}, v={self.version})")
        return True

    @property
    def ready(self) -> bool:
        return self._loaded


_model_bundle: ModelBundle | None = None


def get_model() -> ModelBundle:
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = ModelBundle()
        _model_bundle.load()
    return _model_bundle
