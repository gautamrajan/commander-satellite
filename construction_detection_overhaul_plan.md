# Construction Detection Pipeline Overhaul Plan (R&D v1)

## Purpose
- Modernize the prototype into a production‑aligned pipeline focused on construction site detection.
- Move storage from JSONL to PostGIS; keep JSONL as optional audit artifacts.
- Accept ZCTA (Census ZIP) and point+radius AOIs; keep KML polygons.
- Shift model output to bounding boxes (and optional masks) for construction sites only.
- Add multi‑tile logic to merge detections across tile boundaries and promote persistent Sites.

## Current State (Repo Survey)
- `grab_imagery.py`: Fetches Esri World_Imagery tiles and optionally writes a GeoTIFF + `tiles/z/x/y.jpg`. Supports KML/geometry JSON.
- `scan_objects.py`: LLM scanning for `dumpsters|construction` with coarse/context stitching; writes `all_results.jsonl`, positives JSONL; no box/mask extraction.
- `review_app.py`: Flask UI for AOI orchestration and review. Area‑scoped runs live under `runs/<area_id>/` and use JSONL files.
- `prompts.py`: Detection prompts for dumpsters and construction (base/context/coarse).
- `detection_types.py`: Two detection types; thresholds differ; file naming helpers.
- `kml_utils.py`: KML polygon parsing and simple geometry helpers.

Constraints and gaps:
- JSONL‑only persistence; no spatial queries or deduplication across runs.
- No bounding boxes; only tile‑level positives.
- Cross‑tile objects not merged; context stitching exists but not persisted geometrically.
- AOI inputs are point+radius and KML; no structured ZCTA lookup.

## Target Architecture (v1)
- **AOI inputs**: ZCTA (Census ZIP) via PostGIS lookup; point+radius; KML polygons. All normalized to polygons in EPSG:3857.
- **Storage**: PostGIS for detections and sites. Boxes as polygons in 3857; centroids in 4326 for display.
- **Model IO**: Gemini‑based vision with JSON output carrying box/mask/label. Boxes measured in tile pixels, remapped to 3857 geometry.
- **Multi‑tile**: Heuristic edge detection → re‑run with stitched neighborhood; merge boxes in 3857 with IoU clustering.
- **Entities**: Detection (per inference) and Site (promoted persistent object). Immediate promotion policy in v1 with thresholds.
- **Serving/API**: Minimal REST to enqueue runs and query detections/sites by bbox/date/zcta.
- **Artifacts**: Optional mask chips saved (local or S3) for later polygonization.

## Database Schema (PostGIS)
Connection: `POSTGRES_DSN` from `.env` (do not commit secrets).

Key tables (EPSG codes noted):
- `zcta` (reference): `zcta5ce10 text primary key`, `geom geometry(MULTIPOLYGON, 4326)`; add `geom_3857 geometry(MULTIPOLYGON, 3857)` materialized.
- `aoi_runs`: AOI submission and execution metadata.
  - `id uuid pk`, `created_at timestamptz`, `mode text` ('zcta'|'point_radius'|'kml'), `params jsonb`,
    `geom_3857 geometry(MULTIPOLYGON, 3857)`, `status text`, `stats jsonb`.
- `imagery_tiles`: catalog of fetched tiles.
  - `id bigserial pk`, `z int`, `x int`, `y int`, `path text`,
    `center_4326 geometry(POINT, 4326)`, `geom_3857 geometry(POLYGON, 3857)`, unique `(z,x,y)`.
- `detections`: raw model hits.
  - `id uuid pk default gen_random_uuid()`, `run_id uuid references aoi_runs(id)`, `tile_id bigint references imagery_tiles(id)`,
    `model text`, `model_ver text`, `confidence numeric(4,3)`, `label text`,
    `bbox_px int[4]` (x0,y0,x1,y1 in tile/stitched coords), `bbox_3857 geometry(POLYGON,3857)`, `centroid_4326 geometry(POINT,4326)`,
    `edge_touch bool`, `raw jsonb`, `created_at timestamptz default now()`;
    GiST index on `bbox_3857`.
- `sites`: promoted/persistent objects.
  - `id uuid pk`, `zcta text`, `aabb_3857 geometry(POLYGON,3857)`, `centroid_4326 geometry(POINT,4326)`,
    `first_seen date`, `last_seen date`, `status text`, `conf numeric(4,3)`, `detections_count int`;
    GiST on `aabb_3857`, btree on `zcta`.
- `site_detections`: link table `(site_id uuid fk, detection_id uuid fk, primary key(site_id,detection_id))`.

Notes:
- Compute centroids with generated columns or triggers in v2; in v1, compute in code.
- Store `bbox_px` for provenance; `bbox_3857` is the canonical geometry for spatial ops.

DDL sketch (indicative):
```sql
create extension if not exists postgis;

create table aoi_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz default now(),
  mode text not null,
  params jsonb,
  geom_3857 geometry(multipolygon, 3857) not null,
  status text default 'queued',
  stats jsonb
);

create table imagery_tiles (
  id bigserial primary key,
  z int not null, x int not null, y int not null,
  path text,
  center_4326 geometry(point,4326),
  geom_3857 geometry(polygon,3857),
  unique (z,x,y)
);
create index imagery_tiles_gix on imagery_tiles using gist(geom_3857);

create table detections (
  id uuid primary key default gen_random_uuid(),
  run_id uuid references aoi_runs(id),
  tile_id bigint references imagery_tiles(id),
  model text, model_ver text,
  confidence numeric(4,3), label text,
  bbox_px int[],
  bbox_3857 geometry(polygon,3857) not null,
  centroid_4326 geometry(point,4326) not null,
  edge_touch boolean default false,
  raw jsonb,
  created_at timestamptz default now()
);
create index detections_gix on detections using gist(bbox_3857);

create table sites (
  id uuid primary key default gen_random_uuid(),
  zcta text,
  aabb_3857 geometry(polygon,3857) not null,
  centroid_4326 geometry(point,4326) not null,
  first_seen date, last_seen date,
  status text, conf numeric(4,3), detections_count int default 0
);
create index sites_gix on sites using gist(aabb_3857);
create index sites_zcta_idx on sites(zcta);

create table site_detections (
  site_id uuid references sites(id) on delete cascade,
  detection_id uuid references detections(id) on delete cascade,
  primary key (site_id, detection_id)
);
```

ZCTA load: Use `shp2pgsql`/`ogr2ogr` into `zcta` then `update zcta set geom_3857 = ST_Transform(geom,3857);` Add `gist(geom_3857)`.

## Model I/O Contract (Construction Only)
- System prompt: keep concise JSON‑only contract.
- Human prompt (for per‑tile or stitched context):
  "Give the segmentation masks for construction site. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels."

Expected JSON shape:
```json
[
  {"box_2d": [x0,y0,x1,y1], "mask": "<optional-rle-or-binary>", "label": "residential construction", "confidence": 0.92}
]
```
Notes:
- In v1, store `box_2d` and `label`, `confidence`, and the raw JSON; `mask` optional. If present, optionally write a small chip to S3/local.
- When run on stitched (e.g., 3x3) images, `box_2d` is relative to the stitched canvas; remap to each tile and to 3857 geometry.

## Multi‑Tile Strategy
1) Primary pass: infer on central tile (256×256) using construction prompts; collect boxes.
2) Edge heuristic: if `box_2d` touches any tile edge (x0≈0, y0≈0, x1≈255, y1≈255), mark `edge_touch=true` and enqueue a stitched re‑run for that tile with radius R (default 1 → 3×3).
3) Stitched pass: run on stitched image; transform each predicted box from stitched pixel coords → target tile’s pixel coords → 3857 polygon using tile math at `(z,x,y)` and pixel size at latitude.
4) Merge across tiles: cluster all boxes per run in 3857 with IoU threshold (e.g., IoU ≥ 0.3) or distance/overlap union. For each cluster, compute the minimal AABB union → Detection group. Promote to Site (policy below).

Implementation notes:
- Reuse existing context stitching in `scan_objects.py` but change outputs to include structured boxes.
- Add a small geometry helper to convert `[x0,y0,x1,y1]@z/x/y` to `POLYGON(3857)` and `POINT(4326)` centroid.
- Persist both the tile‑level box and merged 3857 box (post‑cluster) if needed; in v1, persist only final 3857 boxes as detections.

## Promotion Policy (v1)
- Immediate site creation for each detection above thresholds:
  - `confidence ≥ 0.80`
  - `box area ≥ 600 m²` (compute from 3857 polygon)
- On insertion:
  - Find existing site overlapping the new detection using IoU≥0.3 or center‑within AABB; if found, update `aabb_3857 = ST_Envelope(ST_Union(aabb_3857, new_box))`, bump `last_seen`, increment `detections_count`, and upsert link in `site_detections`.
  - Else create a new site row with `first_seen=last_seen=today` and `detections_count=1`.

## AOI Ingestion & Tile Selection
- Inputs supported:
  - `zcta`: codes array (e.g., ["90012","90013"]) → query `zcta.geom_3857`; union for run geometry.
  - `point_radius`: lat/lon + radius (meters) → create buffer polygon in 3857.
  - `kml`: keep existing polygon ingestion; transform to 3857.
- Tile selection: compute tiles intersecting `geom_3857` at target zoom. Pad AOI by R tiles for stitched re‑runs.
- Catalog tiles into `imagery_tiles` with `geom_3857` (exact tile polygon in 3857) and center point.

## API (Minimal)
- `POST /runs` → `{ mode: 'zcta'|'point_radius'|'kml', params: {...}, zoom, options }` → create `aoi_runs` row; enqueue tile fetch + detection pipeline.
- `GET /detections?bbox=&date=&zcta=` → spatial filter on `detections.bbox_3857` (intersects), optional date ranges.
- `GET /sites?bbox=&date=&zcta=` → spatial filter on `sites.aabb_3857`, with `first_seen/last_seen` filters.
- Optional: `GET /zcta?code_like=900%` to help clients discover codes.

## Pipeline Changes (Code‑Level)
1) New DB module `db/` (SQLAlchemy or psycopg + simple SQL):
   - Connection via `POSTGRES_DSN`.
   - Upsert helpers for tiles/detections/sites; simple migrations (SQL files or Alembic later).

2) Scanning (`scan_objects.py` → construction‑only path):
   - Add `--construction_only` mode (or default to construction for this branch); deprecate dumpsters.
   - Add `--emit_boxes`: write structured box JSON to DB and optionally to `all_results.jsonl` as audit.
   - Replace positive‑only JSONL with DB writes: detections with `bbox_px`, `bbox_3857`, `edge_touch`.
   - For stitched context, persist the stitch radius used.

3) Geometry helpers (`geom/tiles.py`):
   - Tile pixel → lat/lon → 3857 polygon conversion.
   - Box area in m²; IoU on 3857 polygons.

4) Merger (`detect/merge.py`):
   - Cluster boxes with IoU≥0.3 (tunable) per run; produce final per‑object AABB in 3857.

5) Promotion (`detect/promote.py`):
   - Given detections, upsert/update Sites per policy; return linkage rows.

6) AOI ingestion (`aoi/ingest.py`):
   - ZCTA lookup in PostGIS; point+radius buffers; KML transformation to 3857.
   - Write `aoi_runs` and enqueue jobs.

7) Review App (transition plan):
   - Phase A: keep JSONL UI working; add an export step that mirrors DB rows to JSONL for review.
   - Phase B: add endpoints in Flask to query from PostGIS directly for `/areas/*/detections` and Mosaic overlays.

## Execution Plan (Sprints)
Sprint 0 – Setup (0.5 week)
- Provision Postgres + PostGIS locally; add `.env` `POSTGRES_DSN`.
- Load ZCTA shapefile into `zcta` and build spatial indexes.

Sprint 1 – DB + AOI normalization (1 week)
- Create schema above; add lightweight SQL migrations.
- Implement `aoi/ingest.py` and CLI `aoi_run.py` to create `aoi_runs` with `geom_3857`.
- Update `grab_imagery.py` to optionally register tiles into `imagery_tiles` when saving.

Sprint 2 – Model I/O + boxes (1–1.5 weeks)
- Extend prompts to request boxes/masks JSON for construction.
- Add stitched re‑run on edge‑touch; implement pixel→3857 mapping and DB writes to `detections`.
- Keep `all_results.jsonl` logging for audit/resume during transition.

Sprint 3 – Merge + Promotion + API (1 week)
- Implement IoU clustering over `detections` per run to create merged 3857 AABBs.
- Implement promotion into `sites` with upsert logic and linkage.
- Add minimal REST for `/runs`, `/detections`, `/sites` (could be embedded in Flask or a small FastAPI).

Sprint 4 – UI integration (0.5–1 week)
- Phase A: add DB→JSONL export so current `review_app.py` works unchanged for review.
- Phase B: swap review endpoints to query PostGIS for AOI tiles/detections.

## Validation & Metrics
- Metrics: counts by AOI/date, coarse→refine ratios, edge‑touch rates, box area distributions.
- Manual QA: sample 50 detections per AOI; track false‑positive flags (persist as a column on `detections`).
- Perf: ensure GiST indexes on geometries; validate query explain plans for typical bbox queries.

## Open Questions
- Mask storage format (RLE vs PNG chips) and retention policy.
- IoU thresholds for clustering and promotion; whether to use oriented boxes in edge cases.
- How sticky should Site geometry be across time (shrink/expand rules)?
- Cadence and retention: weekly scans vs biweekly; data lifecycle in DB vs S3.

## Immediate Next Steps
1) Stand up PostGIS and load ZCTA.
2) Add DB layer and schema migrations.
3) Modify scanning to emit box JSON and write to `detections`.
4) Implement clustering + promotion to `sites`.
5) Add minimal REST endpoints and a DB→JSONL export for the current UI.

