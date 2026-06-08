"""
Charge les CSV générés dans PostgreSQL.

Usage:
    python -m src.data.load_to_db [--input data/raw] [--truncate]
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.db import check_connection, engine

# Colonnes à exclure du CSV (métadonnées de simulation)
STUDENTS_DROP = {"profile", "cohort_start", "cohort_end"}

# Ordre de chargement respectant les FK
LOAD_ORDER = [
    ("cohorts",    "cohorts.csv",    None),
    ("modules",    "modules.csv",    None),
    ("students",   "students.csv",   STUDENTS_DROP),
    ("lms_events", "lms_events.csv", None),
]

CHUNK_SIZE = 50_000


def _load_table(table: str, csv_path: Path, drop_cols: set | None,
                truncate: bool) -> int:
    if not csv_path.exists():
        logger.warning(f"{csv_path} introuvable — table {table} ignorée.")
        return 0

    df = pd.read_csv(csv_path, low_memory=False)

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Nettoyage types
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].where(df[col].notna(), other=None)

    # Dates
    for col in df.columns:
        if "date" in col or col.endswith("_ts") or col == "event_ts":
            df[col] = pd.to_datetime(df[col], errors="coerce")

    with engine.connect() as conn:
        if truncate:
            conn.execute(
                __import__("sqlalchemy").text(f"TRUNCATE TABLE {table} CASCADE")
            )
            conn.commit()
            logger.info(f"TRUNCATE {table}")

    t0 = time.time()
    df.to_sql(
        table,
        engine,
        if_exists="append",
        index=False,
        chunksize=CHUNK_SIZE,
        method="multi",
    )
    elapsed = time.time() - t0
    logger.success(f"  {table:<15} {len(df):>10,} lignes  ({elapsed:.1f}s)")
    return len(df)


def main(input_dir: str = "data/raw", truncate: bool = False) -> None:
    logger.info("=== Chargement CSV → PostgreSQL ===")

    if not check_connection():
        logger.error("PostgreSQL inaccessible. Vérifiez docker compose up.")
        raise SystemExit(1)

    raw = Path(input_dir)
    total_rows = 0

    # Ordre inversé pour TRUNCATE CASCADE (évite les erreurs FK)
    if truncate:
        for table, _, _ in reversed(LOAD_ORDER):
            with engine.connect() as conn:
                conn.execute(
                    __import__("sqlalchemy").text(f"TRUNCATE TABLE {table} CASCADE")
                )
                conn.commit()
                logger.info(f"TRUNCATE {table}")

    for table, filename, drop_cols in LOAD_ORDER:
        n = _load_table(table, raw / filename, drop_cols, truncate=False)
        total_rows += n

    logger.success(f"\nTotal chargé : {total_rows:,} lignes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="data/raw")
    parser.add_argument("--truncate", action="store_true",
                        help="Vider les tables avant insertion")
    args = parser.parse_args()
    main(input_dir=args.input, truncate=args.truncate)
