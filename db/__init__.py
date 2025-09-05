"""Database utilities package for PostGIS integration.

Provides minimal connection helpers and constants. See db/init_db.py
to initialize schema on a local Postgres instance.
"""

from .config import build_dsn

__all__ = ["build_dsn"]

