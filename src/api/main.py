"""Application FastAPI — point d'entrée principal."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.dependencies import get_model
from src.api.routes import alerts, cohorts, predictions, students


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = get_model()
    if model.ready:
        logger.info("Modèle ML chargé au démarrage.")
    else:
        logger.warning("Modèle ML absent — /predict retournera 503.")
    yield


app = FastAPI(
    title="EdTech Analytics API",
    description=(
        "API de prédiction du décrochage scolaire et d'analytics pédagogique.\n\n"
        "**Modules** : étudiants, cohortes, prédictions (unitaire & batch), alertes."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Middleware latence ────────────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    return response

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(students.router)
app.include_router(cohorts.router)
app.include_router(predictions.router)
app.include_router(alerts.router)


# ── Routes utilitaires ────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root():
    return {"message": "EdTech Analytics API", "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health():
    from src.data.db import check_connection
    db_ok    = check_connection()
    model_ok = get_model().ready
    return {
        "status":  "ok" if (db_ok and model_ok) else "degraded",
        "db":      "ok" if db_ok    else "unavailable",
        "model":   "ok" if model_ok else "not_loaded",
        "version": app.version,
    }
