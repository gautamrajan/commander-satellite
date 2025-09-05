# DB Setup (Local PostGIS)

This project now persists detections and sites in PostGIS. The JSONL files remain available as audit artifacts during the transition.

## Prereqs
- PostgreSQL 13+ with the ability to install extensions
- PostGIS 3+ available on the server (installed via your package manager)
- Python deps: `psycopg[binary]` (or `psycopg2-binary`), optional `python-dotenv`

## Configure connection
Populate `.env` (already present):

```
DB_DIALECT=postgres
DB_HOST=localhost
DB_USER=shan
DB_PASS=commander
DB_NAME=prd_dump
# Optional: DB_PORT=5432
```

Alternatively, provide a full DSN at runtime via `POSTGRES_DSN`, e.g.:

```
POSTGRES_DSN="host=localhost port=5432 dbname=prd_dump user=shan password=commander"
```

## Initialize schema

```
# Install driver in your active venv if needed
pip install 'psycopg[binary]'  # or: pip install psycopg2-binary

# Apply extensions and schema
python db/init_db.py

# Verify
python db/health_check.py
```

This creates tables:
- `aoi_runs`, `imagery_tiles`, `detections`, `sites`, `site_detections`
- Ensures extensions: `postgis`, `pgcrypto`

## Load ZCTA (optional, for ZCTA AOIs)
Use `ogr2ogr` or `shp2pgsql` to load a ZCTA shapefile (e.g., Census ZCTA5) into a `zcta` table, then materialize a 3857 column.

Example (indicative, adjust paths and SRIDs as needed):

```
# Load WGS84 ZCTA geometry into public.zcta (columns: zcta5ce10, geom)
ogr2ogr -f PostgreSQL \
  PG:"host=localhost dbname=prd_dump user=shan password=commander" \
  /path/to/tl_2020_us_zcta510.shp -nln zcta -nlt MULTIPOLYGON -lco GEOMETRY_NAME=geom

# Add 3857 geometry and index
psql "host=localhost dbname=prd_dump user=shan password=commander" -c \
  "ALTER TABLE zcta ADD COLUMN IF NOT EXISTS geom_3857 geometry(MULTIPOLYGON,3857); \
    UPDATE zcta SET geom_3857 = ST_Transform(geom,3857) WHERE geom IS NOT NULL; \
    CREATE INDEX IF NOT EXISTS zcta_geom3857_gix ON zcta USING GIST(geom_3857);"
```

## Next steps
- Wire scan pipeline to write `detections` (bbox_3857, centroid_4326, raw JSON).
- Add clustering + promotion to maintain `sites` and `site_detections`.
- Add a minimal REST layer to query detections/sites by bbox, date, and ZCTA.

## Troubleshooting

- Using a venv? Ensure the driver is installed in that venv, then restart the Flask app from the same venv:
  - macOS/Linux: `source myenv/bin/activate && python review_app.py`
- Connection string: the app builds a DSN from `.env` (`DB_*` vars). Alternatively set `POSTGRES_DSN` directly.
- Quick connectivity test (runs a `SELECT 1`):
  ```bash
  python - <<'PY'
  import os
  from db.config import build_dsn
  import psycopg
  dsn = build_dsn()
  with psycopg.connect(dsn) as conn:
      with conn.cursor() as cur:
          cur.execute('SELECT 1')
          print('OK', cur.fetchone())
  PY
  ```

## Logging

- AOI scan launch writes a banner to `runs/<AREA_ID>/logs/scan.err.log` with DB status (`DB: enabled/disabled/failed`).
- Set `SCANNER_DEBUG=1` and/or `SCANNER_TRACE=1` in `.env` for deeper visibility.
