#!/usr/bin/env python3
"""Initialize the local Postgres with PostGIS schema for construction detection.

Usage:
  python db/init_db.py               # uses env (.env) DB_* to connect and apply schema
  POSTGRES_DSN='host=... user=... password=... dbname=...' python db/init_db.py

Dependencies:
  - psycopg (preferred) or psycopg2
  - python-dotenv (optional; for loading .env)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .config import build_dsn


def _connect(dsn: str):
    """Return a DB-API connection using psycopg (v3) or psycopg2 as fallback."""
    try:
        import psycopg  # type: ignore

        return psycopg.connect(dsn)
    except Exception:
        try:
            import psycopg2  # type: ignore

            return psycopg2.connect(dsn)
        except Exception as e:
            raise SystemExit(
                "Unable to import psycopg or psycopg2. Install with:\n"
                "  pip install psycopg[binary]\n"
                "or pip install psycopg2-binary\n"
                f"Original error: {e}"
            )


def _read_schema_sql() -> str:
    here = Path(__file__).resolve().parent
    schema_path = here / "schema.sql"
    if not schema_path.exists():
        raise SystemExit(f"schema.sql not found at {schema_path}")
    return schema_path.read_text(encoding="utf-8")


def ensure_extensions(cur) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")


def apply_schema(cur, sql: str) -> None:
    cur.execute(sql)


def main() -> None:
    dsn_env = os.getenv("POSTGRES_DSN")
    dsn = dsn_env or build_dsn()
    print("Connecting with DSN:", dsn.replace(os.getenv("DB_PASS") or "", "******"))

    conn = _connect(dsn)
    # autocommit to allow CREATE EXTENSION
    try:
        # psycopg (v3) uses .execute with context manager; psycopg2 similar
        conn.autocommit = True  # type: ignore[attr-defined]
    except Exception:
        pass

    with conn:
        with conn.cursor() as cur:
            print("Ensuring extensions (postgis, pgcrypto)...")
            ensure_extensions(cur)
            print("Applying schema.sql ...")
            schema_sql = _read_schema_sql()
            apply_schema(cur, schema_sql)
            print("Schema applied.")

    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

