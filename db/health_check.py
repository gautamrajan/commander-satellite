#!/usr/bin/env python3
"""Quick health check for Postgres/PostGIS setup.

Prints available extensions and verifies presence of core tables.
"""

from __future__ import annotations

import os
from typing import List

from .config import build_dsn


def _connect(dsn: str):
    try:
        import psycopg  # type: ignore

        return psycopg.connect(dsn)
    except Exception:
        import psycopg2  # type: ignore

        return psycopg2.connect(dsn)


def main():
    dsn = os.getenv("POSTGRES_DSN") or build_dsn()
    conn = _connect(dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                print("Extensions:")
                cur.execute("SELECT extname FROM pg_extension ORDER BY 1;")
                for (name,) in cur.fetchall():
                    print(" -", name)

                print("\nTables present:")
                cur.execute(
                    """
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname='public'
                    ORDER BY tablename
                    """
                )
                tables = [r[0] for r in cur.fetchall()]
                for t in tables:
                    print(" -", t)

                expected = [
                    "aoi_runs",
                    "imagery_tiles",
                    "detections",
                    "sites",
                    "site_detections",
                ]
                missing = [t for t in expected if t not in tables]
                if missing:
                    print("\nMissing tables:", ", ".join(missing))
                else:
                    print("\nAll expected tables are present.")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

