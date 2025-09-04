### Dumpster Detection – Agent Primer

This document orients AI agents picking up development. It summarizes architecture, data flows, endpoints, UI, and gotchas so you can ship improvements quickly and safely.

---

### 1) Project Overview

- **Scripts**
  - `grab_imagery.py`: Fetches Esri World Imagery tiles for a bbox or polygon geometry.
    - BBox mode (point + area): writes `mosaic.tif`/`mosaic_preview.png` and optionally persists `tiles/z/x/y.jpg`.
    - Geometry mode: accepts `--geometry_json` or `--kml` and fetches only tiles whose centers fall inside the polygon(s). Use `--no_mosaic` to skip GeoTIFF/PNG for large AOIs.
  - `scan_dumpsters.py` / `scan_objects.py`: Iterate tiles and call OpenRouter (LangChain) for detections. Writes `all_results.jsonl` (all outcomes) and detection-type positives (e.g., `dumpsters.jsonl`).
  - `review_app.py` (Flask): Web dashboard for scanning orchestration, AOI (Area of Interest) management, mosaic visualization, and human review.
    - Supports KML area import and address lookup (reverse geocoding) per tile, including precise click-to-pick inside a tile.

- **AOI (Area) model**
  - Each area lives under `runs/<area_id>/` with:
    - `area.json`, `tiles/`, `all_results.jsonl`, `dumpsters.jsonl`, `reviewed_results.jsonl`, `coarse.jsonl`, `logs/`, `mosaic.tif`, `mosaic_preview.png`.
  - Global legacy mode still supported: `tiles/`, `all_results.jsonl`, `reviewed_results.jsonl` in repo root.
  - Imported KML is persisted as `runs/<area_id>/source.kml`; normalized polygons are stored in `area.json.geometry`.

- **Key env**
  - `OPENROUTER_API_KEY` must be set to scan. If credits are insufficient, OpenRouter returns HTTP 402 (you’ll see this in logs).
  - Geocoding (optional):
    - No key needed for ArcGIS/Nominatim (use modest rates and descriptive UA).
    - `GOOGLE_MAPS_API_KEY` for Google Geocoding.
    - `MAPBOX_ACCESS_TOKEN` for Mapbox Geocoding.

---

### 2) Data Contracts (JSONL)

- `all_results.jsonl` (per tile; all outcomes)
  - `{ "path":"z/x/y.jpg", "z":int, "x":int, "y":int, "model":str, "result_raw":{...}, "positive":bool, "confidence":float, ... }`

- `dumpsters.jsonl` (positives only)
  - `{ "path":"z/x/y.jpg", "z":int, "x":int, "y":int, "lat"?:float, "lon"?:float, "confidence":float, "model":str }`
  - Notes: `lat`/`lon` (tile-center) are included for new scans; older files may omit and can be backfilled.

- `reviewed_results.jsonl` (human review)
  - `{ "path":"z/x/y.jpg", "approved":bool }`

- `annotations.jsonl` (human annotations)
  - Append-only; latest line for a `path` is authoritative
  - Oriented box with center/size/angle and normalized fields, plus AABB and polygon:
    - Box: `{ cx, cy, w, h, angle_deg, ncx, ncy, nw, nh, poly:[[nx,ny]×4], label }`
    - AABB fallback: `{ x, y, w_aabb, h_aabb, nx, ny, nw_aabb, nh_aabb }`
  - Location: AOI `runs/<AREA_ID>/annotations.jsonl`, legacy root `annotations.jsonl`

Notes:
- Review app hides already-reviewed positives from `/detections`.
- Mosaic overlays: AI-positive/negative (unreviewed) vs human-reviewed approved/rejected.

---

### 3) HTTP/API Surface (Flask)

Global (legacy) endpoints:
- `POST /scan/start` – start scan on a directory (`tiles/` by default)
- `GET /scan/status` – running + counts delta since start
- `GET /scan/logs?n=200` – stdout/stderr tail
- `GET /mosaic/zooms` / `GET /mosaic/tiles/{z}` / `GET /image/{path}`
 - `GET /detections` – positives minus reviewed (legacy)
 - `POST /review` / `POST /review/toggle`
 - `GET /annotations?path=z/x/y.jpg` – latest annotations for a tile
 - `POST /annotations` – append a snapshot
  - `POST /geocode` – reverse geocode `{ lat, lon }` or `{ z, x, y }` and optional `{ px, py }` (pixel inside tile) to improve accuracy. Returns `{ address, cached }`. Include `area_id` to scope the cache to an AOI.

Area-scoped endpoints (preferred):
- Areas
  - `GET /areas` – list AOIs
  - `POST /areas` – create AOI `{ name, center:{lat,lon}, area_sqmi, zoom? }`
  - `POST /areas/import_kml` – multipart upload: field `kml` + optional `name`, `zoom`. Creates AOI with `geometry` (list of [lat,lon] pairs per polygon).
  - `GET /areas/{area_id}` – AOI detail

- Fetch imagery (AOI)
  - `POST /areas/{area_id}/fetch/start` – runs `grab_imagery.py`
    - If AOI has `geometry`, writes a `geometry.json` and invokes `grab_imagery.py --geometry_json ... --no_mosaic --save_tiles_dir runs/<area_id>/tiles`.
    - Else uses bbox mode with `--lat/--lon/--area_sqmi` (+ `--auto_zoom` or `--zoom`).
  - `GET  /areas/{area_id}/fetch/status` – returns `{ running, pid, zooms, progress:{ fetched, target, percent }, stdout, stderr }`
  - Progress is computed by counting saved tiles and parsing `Tiles: … = N tiles` from `fetch.out.log`.

- Scan (AOI)
  - `POST /areas/{area_id}/scan/start` – runs `scan_dumpsters.py` with `--resume`, `--log_all` into AOI paths
  - `GET  /areas/{area_id}/scan/status` – `{ running, pid, counts:{ all_results, out, coarse }, progress:{ processed, total, percent }, stdout, stderr }`
  - Progress uses `processed = lines(all_results)` over `total = tiles count`.

- Mosaic/Images (AOI)
  - `GET /areas/{area_id}/mosaic/zooms` – zooms present under AOI tiles
  - `GET /areas/{area_id}/mosaic/tiles/{z}` – returns grid and per-tile meta + review flags
  - `GET /areas/{area_id}/image/{path}` – serves AOI tiles

- Review (AOI)
  - `GET /areas/{area_id}/detections` – positives minus reviewed
  - `POST /areas/{area_id}/review` – `{ path, approved }`
  - `POST /areas/{area_id}/review/toggle` – `{ path }` (toggle or set with `approved`)
  - `GET /areas/{area_id}/annotations?path=z/x/y.jpg` – latest annotations
  - `POST /areas/{area_id}/annotations` – append snapshot `{ path, z, x, y, context_radius, canvas_size, boxes:[...] }`
  - `POST /geocode` – same as global; include `area_id` to scope cache to AOI.

---

### 4) UI & Workflow

- Areas tab
  - Leaflet map to set center; form to create areas; list with actions (Fetch/Scan/Review/Mosaic).
  - Live status badges + progress bars for Fetch and Scan (reads new `progress` from status endpoints).

- Mosaic tab
  - Area selector → Zoom selector → Filter (AI+, AI−, Reviewed Approved/Rejected, All).
  - Sidebar shows area summary counts, legend; click tiles to toggle review in-place.
  - Selected tile panel:
    - “Lookup Address” uses tile center.
    - “Pick Address Point” enters a one-click mode (crosshair cursor + hint). Click on the image to geocode that exact point (marker shown), then mode exits automatically.
  - Annotate a selected tile: opens oriented-bbox modal (draw, move, rotate, resize, label; save to `annotations.jsonl`).

- Review tab
  - Area selector; step through unreviewed positives; Approve/Reject.
  - Annotate button opens the same modal with stitched context (0/1/2).
  - Address helpers: “Lookup” (tile center) and “Pick Address Point” (precise click-to-geocode on the large image).

---

### 5) Scanning Model & Parameters

- Model via OpenRouter (default: `google/gemini-2.5-pro`). Requires credits; otherwise 402 errors.
- `--min_confidence` gates AI-positive.
- `--context_radius` stitches neighbor tiles for more context (central tile is the focus).
- `--coarse_factor` (optional) does a coarse pass to prefilter blocks; `--coarse_log` logs coarse stage.
- `--resume` skips already-logged tiles (by `path`) using `all_results.jsonl`.

Gotchas:
- Funding: If every call returns 402, the scan “starts” but appears stalled; check AOI `logs/scan.err.log`.
- Rate: Use `--rpm` to match your credits / throttling.

---

### 6) Local Conventions (Python)

- Prefer explicit names; add type hints to function signatures where clear.
- Early returns, don’t silently swallow exceptions; include minimal context on catch.
- JSONL I/O in UTF‑8; flush on long loops.
- CLI scripts expose `main()` + `if __name__ == "__main__": main()`.
- HTTP: set a lightweight User‑Agent; respect rate limits/backoff.
- Don’t commit big outputs: `tiles/`, `*.tiff`, `*.png`, `*.jsonl`.

---

### 7) Running Locally (quick)

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt  # if present; otherwise install imports seen in scripts
pip install rasterio             # required for GeoTIFF/PNG mosaic outputs
export OPENROUTER_API_KEY=...    # required for scanning

# Start dashboard
python review_app.py

# Create area (UI) → Fetch → Scan → Review
```

API snippets:
```bash
# Create area
curl -X POST localhost:5000/areas -H 'Content-Type: application/json' \
  -d '{"name":"Test","center":{"lat":37.77,"lon":-122.42},"area_sqmi":0.25,"zoom":22}'

# Start fetch/scan
curl -X POST localhost:5000/areas/<AREA_ID>/fetch/start
curl -X POST localhost:5000/areas/<AREA_ID>/scan/start -H 'Content-Type: application/json' -d '{}'

# Import a KML as a new area (multipart)
curl -X POST localhost:5000/areas/import_kml \
  -F 'kml=@/path/to/area.kml' -F 'name=My KML AOI' -F 'zoom=19'

# Poll status
curl localhost:5000/areas/<AREA_ID>/fetch/status
curl localhost:5000/areas/<AREA_ID>/scan/status

# Reverse geocode (tile center)
curl -X POST localhost:5000/geocode -H 'Content-Type: application/json' \
  -d '{"z":19, "x":89721, "y":209373}'

# Reverse geocode (precise pixel inside tile)
curl -X POST localhost:5000/geocode -H 'Content-Type: application/json' \
  -d '{"z":19, "x":89721, "y":209373, "px":130, "py":192}'
```

---

### 8) Troubleshooting

- Scan “not starting” → check `runs/<AREA_ID>/logs/scan.err.log` for HTTP 402; add credits or change model.
- Fetch progress stuck → `fetch.out.log` may not yet include the Tiles total; progress shows only `fetched` then.
- Missing images in mosaic → verify `runs/<AREA_ID>/tiles/z/x/y.jpg` exists; check zoom directory.
- High false positives → lower `--min_confidence` + gather more labeled data; use Review to build dataset.

---

### 9) Suggested Next Work

- UI/UX
  - Batch review (multi-select, keyboard shortcuts), “next unreviewed” nav.
  - AOI Job Center panel with live log tails and cancel/retry.
  - Legend + filters chips; per-zoom stats panel.

- Backend
  - Websocket push updates for progress bars.
  - Graceful fail-fast in `scan_dumpsters.py` on repeated 402s (exit with clear message that UI can show).
  - Cancellation endpoints; job persistence across app restart.

- Dataset/Model
  - Exporters (YOLO/COCO) consuming oriented boxes (`cx,cy,w,h,angle_deg`) or `poly`.
  - Keyboard shortcuts for annotator; optional canvas rotation; index for large `annotations.jsonl`.

---

### 10) Code Hygiene & Safety

- Keep AOI isolation: always write into `runs/<AREA_ID>/...` to avoid clobbering data.
- Don’t block the Flask event loop with long jobs—use subprocess (already in place).
- When adding fields to JSONL, do so backward-compatibly; UI tolerates missing fields.
- For sizeable edits, update this primer with any new contracts or flows.
