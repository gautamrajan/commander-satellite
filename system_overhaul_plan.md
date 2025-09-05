### Integrated Area-Based Scanning & Review: Overhaul Plan (OLD)

#### Objective
Unify imagery fetching, scanning, and human review into a single, area-driven workflow: select an area on a map, fetch imagery, scan it, and review detections for that specific area. Add a streamlined dashboard with clear job orchestration, status, and fast approvals.

---

### High-level Architecture

- **Frontend (Flask + Leaflet UI)**
  - Map view with draw tools (box/polygon) to define an area of interest (AOI)
  - Area list with statuses and quick actions (Fetch, Scan, Review)
  - Review screen filtered by AOI; Mosaic per AOI with prediction + review overlays
  - Job center panel showing progress/logs for fetch and scan

- **Backend (Flask services; existing scripts orchestrated as jobs)**
  - Area registry (lightweight JSON or SQLite) storing AOIs, metadata, and run artifacts paths
  - Job orchestrator for two-stage pipeline: Fetch → Scan (with resume)
  - Area-scoped endpoints for detections, mosaic tiles, approvals
  - Filesystem layout per AOI: `runs/<area_id>/{tiles/, all_results.jsonl, dumpsters.jsonl, reviewed_results.jsonl, logs/}`

- **Batch workers (subprocess)**
  - Reuse `grab_imagery.py` to fetch AOI tiles under `runs/<area_id>/tiles/`
  - Reuse `scan_dumpsters.py` writing logs to `runs/<area_id>/*.jsonl`
  - Background processes with resumable state and status endpoints

---

### Directory & State Layout

- `runs/<area_id>/`
  - `area.json` (AOI geometry, human label, timestamps, basemap params)
  - `tiles/` (z/x/y.jpg images)
  - `all_results.jsonl` (all scan results for AOI)
  - `dumpsters.jsonl` (positives-only for AOI)
  - `reviewed_results.jsonl` (human decisions for AOI)
  - `logs/` (stdout/err for fetch + scan)

- `areas_index.json` (top-level registry of areas: id, name, bbox, created_at, last_status)

Rationale: keep each AOI fully self-contained for portability and easy cleanup.

---

### Backend API Additions

- Area management
```
POST   /areas                     # create AOI from bbox/polygon; returns {area_id}
GET    /areas                     # list AOIs with statuses and summary counts
GET    /areas/{area_id}           # details for AOI (geometry, tiles, latest run info)
DELETE /areas/{area_id}           # archive/remove AOI (optional)
```

- Fetch (imagery) stage
```
POST   /areas/{area_id}/fetch/start   # run grab_imagery.py into runs/<area_id>/tiles
GET    /areas/{area_id}/fetch/status  # progress (tiles, logs)
```

- Scan stage
```
POST   /areas/{area_id}/scan/start    # run scan_dumpsters.py using runs/<area_id>/tiles
GET    /areas/{area_id}/scan/status   # progress (counts, logs)
```

- Review & detections (area-scoped)
```
GET    /areas/{area_id}/detections            # positives from AOI all_results.jsonl minus reviewed
POST   /areas/{area_id}/review                # {path, approved}
POST   /areas/{area_id}/review/toggle         # {path} flip approve/reject
GET    /areas/{area_id}/mosaic/zooms          # available zooms under AOI tiles
GET    /areas/{area_id}/mosaic/tiles/{z}      # tiles + prediction + review flags
```

- Global job center (optional adjunct to existing `/scan/*`)
```
GET    /jobs                                 # list of recent jobs across AOIs
GET    /jobs/{job_id}/status                 # status + live logs
```

Notes:
- Keep current `/scan/start`, `/scan/status`, etc., for backward compatibility; prefer area-scoped endpoints going forward.

---

### Orchestration & State

- Replace single `SCAN_STATE` with a small job manager that tracks:
  - `FETCH` job per AOI (cmd, pid, started_at, stdout/err files, progress)
  - `SCAN` job per AOI (cmd, pid, started_at, stdout/err files, progress)
  - Job persists into `runs/<area_id>/area.json` and in-memory cache

- Progress semantics
  - Fetch: number of tiles written, zoom levels discovered
  - Scan: incremental line counts for `all_results.jsonl`, `dumpsters.jsonl`, `coarse.jsonl`
  - Both: tail logs endpoint for last N lines

- Resumability
  - Fetch: idempotent tile writes based on path existence
  - Scan: use `--resume` with `--log_all runs/<area_id>/all_results.jsonl` to skip previously scanned tiles

---

### Frontend (Dashboard) Overhaul

1) Map-first home tab
  - Leaflet basemap (OSM + optional Esri World Imagery)
  - Leaflet.draw to create bbox/polygon AOI
  - Form: AOI name, max area, target zoom, coarse/context options
  - CTA: “Create Area & Fetch”, then “Start Scan”

2) Areas list panel
  - Cards/rows: name, created, tiles, detections, reviewed counts, status badges
  - Inline actions: Fetch, Scan, Review, Open Mosaic

3) AOI detail view
  - Summary: geometry, stats, last run
  - Job center: live progress bars + logs for fetch/scan
  - Review queue: only AOI positives (existing review UI filtered by AOI)
  - Mosaic viewer: only AOI tiles, existing overlays, click-to-toggle review

4) UX polish
  - Global header with nav: Map, Areas, Review, Mosaic, Jobs
  - Sticky controls; responsive split-pane layout for map + side panel
  - Keyboard shortcuts for review; batch navigation; filters (predicted positive/negative, reviewed/approved)

---

### Minimal Data Model (No external DB yet)

- `areas_index.json`
```
{
  "areas": [
    {
      "id": "aoi_2025_08_18_01",
      "name": "SF - Soma blocks",
      "geometry": { "type": "Polygon", "coordinates": [...] },
      "bbox": [minLon, minLat, maxLon, maxLat],
      "created_at": 1690000000,
      "last_status": { "fetch": "idle|running|done|error", "scan": "idle|running|done|error" }
    }
  ]
}
```

- `runs/<area_id>/area.json`
```
{
  "id": "aoi_2025_08_18_01",
  "name": "SF - Soma blocks",
  "geometry": { ... },
  "fetch": {
    "params": { "zoom": 22, "max_edge_px": 8192 },
    "started_at": 1690000000, "status": "running", "stdout": "logs/fetch.out.log", "stderr": "logs/fetch.err.log"
  },
  "scan": {
    "params": { "rpm": 60, "model": "google/gemini-2.5-pro" },
    "started_at": 1690001111, "status": "idle|running|done|error",
    "stdout": "logs/scan.out.log", "stderr": "logs/scan.err.log"
  }
}
```

---

### Implementation Steps (2 sprints)

Sprint 1 (Map-driven AOIs + orchestration)
- Backend
  - Add area registry + per-AOI directory structure
  - Implement `/areas` CRUD and `runs/<area_id>` scaffolding
  - Implement `fetch/start` and `scan/start` per AOI with background subprocess + status
  - Update mosaic/review endpoints to be AOI-scoped
- Frontend
  - Add Leaflet + Leaflet.draw; map UI to create AOIs and submit to `/areas`
  - Areas list and AOI detail pages
  - AOI-scoped review queue and mosaic viewer (reuse existing UI with new endpoints)
- Deliverable: Select area on map → Fetch → Scan → Review for that AOI end-to-end

Sprint 2 (UX polish + productivity)
- Frontend
  - Streamlined dashboard layout (nav, panels, filters)
  - Hover tooltips and color legend for prediction vs review states
  - Batch review helpers, keyboard shortcuts, quick-filter chips
- Backend
  - Better progress bars; cancel/retry jobs; resume scan; input validation
  - Optional: simple search and analytics per AOI (counts, densities)
- Deliverable: Production-feel dashboard with smooth review workflow

---

### Reuse & Changes to Existing Code

- `grab_imagery.py`
  - Add bbox/polygon input (lat/lon) → compute tiles for target zoom, write to `runs/<area_id>/tiles/`
  - Add `--save_tiles_dir runs/<area_id>/tiles` consistently

- `scan_dumpsters.py`
  - Use `--tiles_dir runs/<area_id>/tiles --log_all runs/<area_id>/all_results.jsonl --out runs/<area_id>/dumpsters.jsonl`
  - Keep `--resume` for idempotency

- `review_app.py`
  - Add AOI registry and new endpoints listed above
  - Generalize current `/scan/*` logic into area-scoped job manager
  - Update review + mosaic handlers to accept `{area_id}` and load from AOI paths

- `templates/index.html` and `static/style.css`
  - Add Map + AOI creation UI (Leaflet + draw)
  - Add Areas list and AOI detail views
  - Refactor mosaic/review to be AOI-scoped and add filters/shortcuts

---

### Risk & Mitigation

- Long-running jobs: detached subprocess + robust status endpoints; tail logs; kill/retry
- Large AOIs: enforce max area + zoom limits; show estimated tile count before fetch
- Race conditions: per-AOI locks; simple state machine (idle → running → done/error)
- Storage growth: per-AOI cleanup/archive; clear guidance to avoid committing large artifacts

---

### Success Criteria

- From map → define AOI → fetch → scan → review in one place
- Review queue and mosaic filtered to AOI with live counts and fast toggles
- Single, consistent filesystem layout per AOI with resumable jobs
- Operators can run multiple AOIs in parallel and monitor progress

---

### Stretch (later)

- SQLite or PostGIS for persistent catalog (when scale demands)
- Websocket live updates; multi-user roles; exports per AOI (CSV/GeoJSON)
- Heatmaps, density analytics, and prioritization layers on the map


