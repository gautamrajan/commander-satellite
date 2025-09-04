# KML Ingestion + Scan Plan

Goal: Add end-to-end support to import KML-defined areas in the dashboard, fetch tiles only for the KML polygon(s), and scan them using existing flows.

## Scope
- UI: Add an “Import KML” option in Areas tab to create an Area from a KML file. Allow specifying a name and optional zoom.
- Backend (Flask):
  - New endpoint `POST /areas/import_kml` to accept a KML upload, parse polygons, create an Area with geometry, and store original KML.
  - Update `POST /areas/<id>/fetch/start` to detect geometry and fetch imagery constrained to polygon(s).
- Tile Fetcher (`grab_imagery.py`):
  - Add a geometry-based fetch mode that accepts `--geometry_json` (or `--kml`) and fetches only tiles whose centers fall inside the polygon(s).
  - Add `--no_mosaic` to skip GeoTIFF/PNG writing for very large AOIs (still prints progress and saves `z/x/y.jpg` tiles).

## Design Decisions
- Geometry format at rest: store `geometry` as a list of polygons in Area JSON, where each polygon is an array of `[lat, lon]` pairs (outer ring). Holes ignored for v1.
- Fetch coverage criterion: include tiles whose centers lie within any polygon. This is simple, robust, and sufficient for scanning.
- Zoom handling: if a zoom is provided for the Area, use it. Otherwise, auto-probe based on Area center (centroid of bbox) like existing flow.
- Progress reporting: print `Fetching N tiles` before fetching to keep `/areas/<id>/fetch/status` progress estimates useful.

## Implementation Steps
1) Add small KML parser `kml_utils.py` to extract polygons (outer rings) via `xml.etree.ElementTree` (no new deps).
2) Extend `grab_imagery.py`:
   - Args: `--geometry_json`, `--kml`, `--no_mosaic`.
   - Geometry mode: compute bounding tile range, preselect tiles by point-in-polygon test (tile centers), log `Fetching N tiles`, fetch, save to `--save_tiles_dir`, skip mosaic when `--no_mosaic`.
3) Backend: Add `POST /areas/import_kml` that:
   - Accepts multipart (field `kml`), optional `name`, `zoom`.
   - Parses polygons, estimates center as bbox mid, stores `geometry` and original KML under `runs/<area>/source.kml`.
4) Backend: Update `area_fetch_start`:
   - If area has `geometry`, write `geometry.json` and call `grab_imagery.py` with `--geometry_json` and `--no_mosaic` (plus `--zoom` or `--auto_zoom` with center for probing). Keep logs consistent.
5) UI (templates/index.html):
   - Add file input + button under Area form to import KML (with optional name + zoom). On success, refresh Areas list.
6) Sanity pass: ensure `/areas/<id>/fetch/status` can parse target from logs; ensure scan uses `runs/<area>/tiles` as before.

## Risks / Notes
- Very large KMLs could still trigger high tile counts. `grab_imagery.py`’s `--max_tiles` cap still applies.
- Holes, inner boundaries, and curved geometries are ignored in v1.
- If desired later, we can render polygon overlays on the Leaflet map for visual confirmation.

