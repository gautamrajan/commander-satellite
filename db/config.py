import os
from typing import Optional

# Optional: auto-load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def build_dsn(
    *,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    dbname: Optional[str] = None,
    port: Optional[str] = None,
) -> str:
    """Build a PostgreSQL DSN from env vars or provided overrides.

    Expected env vars (already present in .env):
    - DB_DIALECT (must be 'postgres')
    - DB_HOST, DB_USER, DB_PASS, DB_NAME, optional DB_PORT
    """

    dialect = (os.getenv("DB_DIALECT") or "postgres").strip()
    if dialect not in {"postgres", "postgresql"}:
        raise ValueError(f"Unsupported DB_DIALECT: {dialect}")

    def _clean(v: Optional[str], default: str) -> str:
        s = (v if v is not None else default)
        s = str(s)
        # Strip matching single or double quotes around entire value
        if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
            s = s[1:-1]
        return s

    host = _clean(host or os.getenv("DB_HOST"), "localhost")
    user = _clean(user or os.getenv("DB_USER"), "postgres")
    password = _clean(password or os.getenv("DB_PASS"), "")
    dbname = _clean(dbname or os.getenv("DB_NAME"), "postgres")
    port = _clean(port or os.getenv("DB_PORT"), "5432")

    # psycopg/psycopg2 compatible DSN
    # Note: password may be empty; quoting not necessary for simple values
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"
