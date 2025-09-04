# Repository Guidelines

## Project Structure & Module Organization
- `review_app.py`: Flask UI to review detections and orchestrate scans; serves `templates/` and `static/`.
- `scan_objects.py`: CLI scanner. Reads tiles under `tiles/z/x/y.(jpg|png)`, writes `all_results.jsonl` and positives (e.g., `dumpsters.jsonl`, `construction_sites.jsonl`).
- `grab_imagery.py`: Fetches basemap tiles for an area; can save a GeoTIFF (`mosaic.tif`), preview PNG, and a `tiles/` pyramid.
- Data/output: `runs/` (area-of-interest runs), `reviewed_results.jsonl`, optional `coarse.jsonl`, and `tiles/` (all git-ignored per `.gitignore`).
- Config: `.env` for `OPENROUTER_API_KEY` and local settings (do not commit secrets).

## Build, Test, and Development Commands
```bash
# Create env and install deps
python -m venv .venv && source .venv/bin/activate
pip install Flask Pillow langchain-openai langchain-core python-dotenv

# Run UI (http://localhost:5001)
python review_app.py

# Fetch imagery (example) and save tiles
python grab_imagery.py --lat 37.78 --lon -122.42 \
  --area_sqmi 0.5 --tiff mosaic.tif --png mosaic_preview.png \
  --save_tiles_dir tiles

# Scan tiles (dumpsters example)
OPENROUTER_API_KEY=... python scan_objects.py --tiles_dir tiles \
  --log_all all_results.jsonl --out dumpsters.jsonl --detection_type dumpsters \
  --rpm 60 --resume --concurrency 4 --context_radius 1 \
  --coarse_factor 2 --coarse_log coarse.jsonl
```

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; prefer type hints and docstrings.
- Files and functions: snake_case; classes/dataclasses: PascalCase.
- Keep JSONL records compact and schema-stable; one JSON object per line.

## Testing Guidelines
- No formal tests yet. Validate changes by:
  - Running `review_app.py` and exercising key routes.
  - Scanning a small sample with `--limit 50` and reviewing outputs.
- If adding tests, use `pytest` in `tests/` with `test_*.py` files.

## Commit & Pull Request Guidelines
- Commits: imperative, scoped summary (e.g., "review_app: add area API"). Include reasoning when non-obvious.
- PRs: clear description, linked issues, steps to reproduce, example commands, and UI screenshots when relevant. Note any data/migration impacts.

## Security & Configuration
- Store `OPENROUTER_API_KEY` in `.env` or env vars; never commit secrets.
- Large artifacts (`tiles/`, `*.log`, `*.jsonl`, `mosaic*.{tif,png}`, `runs/`) are git-ignored—keep it that way.

