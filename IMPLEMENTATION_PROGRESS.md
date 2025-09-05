# Implementation Progress — Construction Detection Overhaul

Last updated: 2025-09-05

## Completed
- PostGIS setup in Dockerized Postgres (`postgres-db`), extensions enabled (`postgis`, `pgcrypto`).
- Schema applied: `aoi_runs`, `imagery_tiles`, `detections`, `sites`, `site_detections`.
- Geometry helper (`geom/tile_math.py`): pixel→lon/lat, pixel-box→EPSG:3857 polygon WKT; centroid lon/lat.
- DB layer (`db/writer.py`): upsert tiles, insert detections, promote detections to sites with IoU-based linking.
- Prompts: added `get_construction_boxes_prompt` returning `{ "detections": [...] }` object for boxes/masks.
- Scanner integration (`scan_objects.py`):
  - New flags: `--emit_boxes_to_db`, `--db_dsn`.
  - For detection_type=construction, requests boxes JSON (base/context/coarse refined path) and inserts per-box detections into DB.
  - Edge-touch heuristic computed; stitched offset handled for context scans (subtract `r*256`).
- DB-backed API in Flask (`review_app.py`):
  - `GET /api/detections?bbox=&run_id=&zcta=&date_from=&date_to=&limit=`
  - `GET /api/sites?bbox=&zcta=&date_from=&date_to=&limit=`
  - Returns GeoJSON geometry and core attributes for mapping.
- run_id linkage from UI:
  - `review_app` creates `aoi_runs` for AOIs (KML geometry or synthesized point+area square) and passes `--run_id` to the scanner when starting scans from the Scan tab (`POST /scan/start`) or from the Area card (`POST /areas/<id>/scan/start`).

## In Progress
- Validate stitched-offset math against sample tiles (visual QA pass).

## Pending / Next Steps
- ZCTA tagging for `sites`: load ZCTA reference into DB (`zcta` with `geom_3857`) so automatic assignment in `writer.py` is effective.
- Box area threshold (>= 600 m²): enforced in `promote_detection_to_site`; confirm with field QA and tune as needed.
- Concurrency throughput: DB inserts work in multi-thread path; consider batching/inserts and connection pooling for higher RPM.
- Optional: store small mask chips (S3/local) and chip URLs in `detections` for future polygonization.
- UI: consider pagination on DB-backed endpoints; add bbox tile summarization for faster mosaics; wire UI overlays to `/api/sites`.

## Notes
- Gemini output: accepts both a list and an object with `detections` key; code normalizes to a list.
- JSONL artifacts (`all_results.jsonl`, positives-only files) remain for audit/resume during transition.
- For stitched context, boxes are offset by `r*256` and clamped to [0,256]; `edge_touch` is set when boxes meet borders.
