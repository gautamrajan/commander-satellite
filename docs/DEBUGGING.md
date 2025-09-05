Debugging Scans and DB Integration

Overview
- This guide shows where to find logs, how to enable tracing, and how to verify DB writes during construction scans.

Key Log Files (Area of Interest)
- `runs/<AREA_ID>/logs/scan.err.log`: launch banner (args, DB status, full command) + optional DEBUG/TRACE lines
- `runs/<AREA_ID>/coarse.jsonl`: coarse-stage results with `prompt_excerpt`, `response_excerpt`, `confidence`, `positive`
- `runs/<AREA_ID>/all_results.jsonl`: per-tile results including `result_raw` (parsed), full `response_text`, and `prompt_excerpt`

Enable Tracing
- In `.env`, add:
  - `REVIEW_SILENCE_POLL_LOGS=1` (hide frequent status request logs in the Flask console)
  - `SCANNER_SILENCE_HTTP=1` (default; silences httpx/langchain INFO logs)
  - `SCANNER_DEBUG=1` (one-line LLM call traces with prompt excerpts)
  - `SCANNER_TRACE=1` (per-tile outcome lines including z/x/y, variant, conf, positive)
- Restart the app from your venv: `source myenv/bin/activate && python review_app.py`

Sanity Scans
- If coarse returns all negatives (no refinement), set `Coarse factor = 0` and `Limit = 50` in the Scan tab to force per‑tile requests and generate `all_results.jsonl` entries.

Construction Boxes Mode
- Boxes prompt is off by default for stability. To enable boxes and DB writes:
  - Add `--construction_boxes` and `--emit_boxes_to_db` to the scanner command (UI may add `--emit_boxes_to_db` automatically when DB is available).
- Coarse stage always uses boolean+confidence prompts.

DB Verification
- Ensure the driver is installed in your venv: `pip install 'psycopg[binary]'`
- Run `python db/init_db.py` and `python db/health_check.py`.
- After starting an AOI scan, check:
  - `SELECT COUNT(*) FROM aoi_runs;` increases
  - For boxes-enabled construction scans, `detections`/`sites` counts increase if model returns boxes.

