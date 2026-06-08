"""Connexion PostgreSQL partagée (SQLAlchemy) — initialisation lazy."""
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://edtech:edtech_pass@localhost:5432/edtech_db",
)

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


# Alias module-level pour compatibilité pandas read_sql / to_sql
class _LazyEngine:
    """Proxy qui reporte la création de l'engine au premier accès."""
    def __getattr__(self, name):
        return getattr(get_engine(), name)

    def connect(self):
        return get_engine().connect()

    def execute(self, *a, **kw):
        return get_engine().execute(*a, **kw)


engine = _LazyEngine()

@contextmanager
def get_session():
    session = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        print(f"Connexion DB échouée : {exc}")
        return False
