import json
import os
import sys
import time
import threading
import subprocess
import shutil
from datetime import datetime
from flask import Flask, jsonify, render_template, request, send_from_directory, Response
from flask import stream_with_context
import logging
import math
from typing import List, Tuple

# Optional KML helpers
try:
    from kml_utils import parse_kml_polygons, bbox_center, bbox_area_sqmi
except Exception:
    parse_kml_polygons = None
    bbox_center = None
    bbox_area_sqmi = None

# Import detection type configuration
from detection_types import get_available_detection_types, get_detection_type, get_output_filename
try:
    from db.writer import DbClient
except Exception:
    DbClient = None  # optional DB integration

app = Flask(__name__)

# Reduce noisy request logs for status polling endpoints to keep console readable.
def _configure_request_logging():
    try:
        wl = logging.getLogger('werkzeug')
        # Filter frequent polling endpoints by default
        if os.getenv('REVIEW_SILENCE_POLL_LOGS', '1') == '1':
            class _StatusEndpointFilter(logging.Filter):
                def filter(self, record):
                    try:
                        msg = record.getMessage()
                        if ('/scan/status' in msg) or ('/fetch/status' in msg) or ('/scan/logs' in msg) or ('/scan/events/stream' in msg):
                            return 0
                    except Exception:
                        pass
                    return 1
            wl.addFilter(_StatusEndpointFilter())
        # Optionally suppress all werkzeug request logs below WARNING
        if os.getenv('REVIEW_SILENCE_REQUEST_LOGS', '0') == '1':
            wl.setLevel(logging.WARNING)
            wl.propagate = False
    except Exception:
        pass

_configure_request_logging()

TILES_DIR = os.path.abspath("tiles")
ALL_RESULTS_FILE = "all_results.jsonl"
REVIEWED_RESULTS_FILE = "reviewed_results.jsonl"
ANNOTATIONS_FILE = "annotations.jsonl"

# AOI runs directory and registry
RUNS_DIR = os.path.abspath("runs")
AREAS_INDEX = os.path.join(RUNS_DIR, "areas_index.json")

os.makedirs(RUNS_DIR, exist_ok=True)


def _safe_read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _safe_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _generate_area_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"aoi_{ts}_{os.getpid()}"


def load_areas_index():
    data = _safe_read_json(AREAS_INDEX, {"areas": []})
    if not isinstance(data, dict) or "areas" not in data:
        data = {"areas": []}
    return data


def save_areas_index(data) -> None:
    _safe_write_json(AREAS_INDEX, data)


def area_dir(area_id: str) -> str:
    return os.path.join(RUNS_DIR, area_id)


def area_paths(area_id: str) -> dict:
    base = area_dir(area_id)
    return {
        "base": base,
        "area_json": os.path.join(base, "area.json"),
        "tiles": os.path.join(base, "tiles"),
        "all_results": os.path.join(base, "all_results.jsonl"),
        "dumpsters": os.path.join(base, "dumpsters.jsonl"),
        "reviewed": os.path.join(base, "reviewed_results.jsonl"),
        "annotations": os.path.join(base, "annotations.jsonl"),
        "coarse": os.path.join(base, "coarse.jsonl"),
        "logs": os.path.join(base, "logs"),
        "fetch_stdout": os.path.join(base, "logs", "fetch.out.log"),
        "fetch_stderr": os.path.join(base, "logs", "fetch.err.log"),
        "scan_stdout": os.path.join(base, "logs", "scan.out.log"),
        "scan_stderr": os.path.join(base, "logs", "scan.err.log"),
        "tiff": os.path.join(base, "mosaic.tif"),
        "png": os.path.join(base, "mosaic_preview.png"),
    }


def load_results():
    results = []
    try:
        if not os.path.exists(ALL_RESULTS_FILE):
            return results
        with open(ALL_RESULTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        # Fail closed (no detections) instead of 500ing
        return []
    return results


def load_reviewed_paths():
    if not os.path.exists(REVIEWED_RESULTS_FILE):
        return set()
    with open(REVIEWED_RESULTS_FILE, "r") as f:
        return {json.loads(line).get("path") for line in f}


def load_review_status_map():
    """Return a dict mapping tile path to review status."""
    review_map = {}
    if not os.path.exists(REVIEWED_RESULTS_FILE):
        return review_map
    with open(REVIEWED_RESULTS_FILE, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = obj.get("path")
            if path:
                review_map[path] = {
                    "approved": bool(obj.get("approved")),
                    "reviewed": True
                }
    return review_map


def load_results_map():
	"""Return a dict mapping tile path (e.g., 'z/x/y.jpg') to metadata.

	Metadata keys: positive (bool), confidence (float), z, x, y
	"""
	results_map = {}
	if not os.path.exists(ALL_RESULTS_FILE):
		return results_map
	with open(ALL_RESULTS_FILE, "r") as f:
		for line in f:
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				continue
			path = obj.get("path")
			if not path:
				continue
			results_map[path] = {
				"positive": bool(obj.get("positive")),
				"confidence": float(obj.get("confidence") or 0.0),
				"z": obj.get("z"),
				"x": obj.get("x"),
				"y": obj.get("y"),
			}
	return results_map


def _build_coarse_negative_xy_from_file(coarse_log_path: str, zoom: int) -> set:
    """Return a set of (x,y) tuples at a given zoom that were marked coarse-negative.

    Reads coarse.jsonl where each line can include: {stage:"coarse", z, x, y, factor, positive:bool}.
    For any record with positive == False, we mark all children (x..x+f-1, y..y+f-1) as coarse-negative.
    """
    neg: set = set()
    try:
        if not coarse_log_path or not os.path.exists(coarse_log_path):
            return neg
        with open(coarse_log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("stage") != "coarse":
                    continue
                z = obj.get("z")
                if z != zoom:
                    continue
                if bool(obj.get("positive")):
                    continue
                x0 = obj.get("x")
                y0 = obj.get("y")
                factor = int(obj.get("factor") or 0)
                if not isinstance(x0, int) or not isinstance(y0, int) or factor <= 0:
                    continue
                for yy in range(y0, y0 + factor):
                    for xx in range(x0, x0 + factor):
                        neg.add((xx, yy))
    except Exception:
        pass
    return neg


def load_results_map_for_area(area_id: str):
    """Read an AOI's all_results.jsonl into a path->metadata map."""
    p = area_paths(area_id)
    fname = p["all_results"]
    results_map = {}
    if not os.path.exists(fname):
        return results_map
    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = obj.get("path")
            if not path:
                continue
            results_map[path] = {
                "positive": bool(obj.get("positive")),
                "confidence": float(obj.get("confidence") or 0.0),
                "z": obj.get("z"),
                "x": obj.get("x"),
                "y": obj.get("y"),
            }
    return results_map


def get_available_zooms():
	"""List available zoom levels present under tiles directory."""
	zooms = []
	if not os.path.isdir(TILES_DIR):
		return zooms
	for name in os.listdir(TILES_DIR):
		full = os.path.join(TILES_DIR, name)
		if os.path.isdir(full) and name.isdigit():
			try:
				zooms.append(int(name))
			except ValueError:
				continue
	zooms.sort()
	return zooms


def get_available_zooms_for_area(area_id: str):
	"""List available zoom levels present under runs/<area_id>/tiles directory."""
	zooms = []
	p = area_paths(area_id)
	tiles_dir = p["tiles"]
	if not os.path.isdir(tiles_dir):
		return zooms
	for name in os.listdir(tiles_dir):
		full = os.path.join(tiles_dir, name)
		if os.path.isdir(full) and name.isdigit():
			try:
				zooms.append(int(name))
			except ValueError:
				continue
	zooms.sort()
	return zooms


# -------- Scan orchestration (background subprocess) --------
SCAN_STATE = {
    "proc": None,
    "started_at": None,
    "run_dir": None,
    "args": None,
    "baseline": {
        "all_results": 0,
        "out": 0,
        "coarse": 0,
    },
    "files": {
        "all_results": ALL_RESULTS_FILE,
        "out": "dumpsters.jsonl",
        "coarse": None,
        "stdout": None,
        "stderr": None,
        "events": None,
    },
}
SCAN_LOCK = threading.Lock()
GEOCODE_LOCK = threading.Lock()

# ---- Geocode helpers (tile math + providers) ----
def _tile_to_latlon(x: int, y: int, z: int) -> tuple:
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y/n))))
    return lat, lon


def _tile_center_latlon(z: int, x: int, y: int) -> tuple:
    lat0, lon0 = _tile_to_latlon(x, y, z)
    lat1, lon1 = _tile_to_latlon(x + 1, y + 1, z)
    return (lat0 + lat1) / 2.0, (lon0 + lon1) / 2.0


def _pixel_to_latlon(z: int, x: int, y: int, px: int, py: int) -> tuple:
    """Convert a pixel within a z/x/y tile (px,py in [0,255]) to lat/lon."""
    # Clamp px/py
    px = max(0, min(int(px), 255))
    py = max(0, min(int(py), 255))
    n = 2 ** int(z)
    map_size = 256 * n
    gx = int(x) * 256 + px
    gy = int(y) * 256 + py
    lon = gx / map_size * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (gy / map_size)))))
    return (lat, lon)


def _geocode_cache_path(area_id: str = None) -> str:
    if area_id:
        return os.path.join(area_paths(area_id)["base"], "geocode_cache.json")
    return os.path.abspath("geocode_cache.json")


def _geocode_provider_arcgis(lat: float, lon: float, timeout: float = 10.0):
    from urllib.request import urlopen, Request
    from urllib.parse import urlencode
    base = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/reverseGeocode"
    qs = urlencode({"f": "pjson", "location": f"{lon},{lat}", "outSR": 4326, "langCode": "en"})
    url = f"{base}?{qs}"
    req = Request(url, headers={"User-Agent": "sat-data-fetcher/1.0"})
    with urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="ignore"))
    try:
        return (data.get("address") or {}).get("Match_addr")
    except Exception:
        return None


def _geocode_provider_nominatim(lat: float, lon: float, timeout: float = 10.0):
    from urllib.request import urlopen, Request
    from urllib.parse import urlencode
    base = "https://nominatim.openstreetmap.org/reverse"
    qs = urlencode({"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 1})
    url = f"{base}?{qs}"
    req = Request(url, headers={"User-Agent": "sat-data-fetcher/1.0 (dashboard)"})
    with urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="ignore"))
    return data.get("display_name")


def _geocode_provider_google(lat: float, lon: float, api_key: str, timeout: float = 10.0):
    from urllib.request import urlopen, Request
    from urllib.parse import urlencode
    base = "https://maps.googleapis.com/maps/api/geocode/json"
    qs = urlencode({"latlng": f"{lat},{lon}", "key": api_key})
    url = f"{base}?{qs}"
    req = Request(url, headers={"User-Agent": "sat-data-fetcher/1.0"})
    with urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="ignore"))
    if data.get("status") == "OK" and data.get("results"):
        return data["results"][0].get("formatted_address")
    return None


def _geocode_provider_mapbox(lat: float, lon: float, token: str, timeout: float = 10.0):
    from urllib.request import urlopen, Request
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json?language=en&access_token={token}"
    req = Request(url, headers={"User-Agent": "sat-data-fetcher/1.0"})
    with urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="ignore"))
    feats = data.get("features") or []
    if feats:
        return feats[0].get("place_name")
    return None


def safe_count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def proc_is_running(p: subprocess.Popen) -> bool:
    return (p is not None) and (p.poll() is None)


def tail_lines(path: str, max_lines: int = 100) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            return "".join(lines[-max_lines:])
    except Exception:
        return ""


# ---- Annotations helpers ----
def _append_jsonl(path: str, obj: dict) -> None:
    """Append a single JSON object as a line to a JSONL file (UTF-8)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _load_latest_annotation_for_path(path_to_jsonl: str, image_rel_path: str) -> dict:
    """Return the most recent annotation object for a given tile path from a JSONL file.

    If none exists, return a default structure with empty boxes.
    """
    latest = None
    try:
        with open(path_to_jsonl, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("path") == image_rel_path:
                    latest = obj
    except FileNotFoundError:
        pass
    except Exception:
        pass
    if latest is None:
        latest = {"path": image_rel_path, "boxes": []}
    # Ensure boxes array exists
    if not isinstance(latest.get("boxes"), list):
        latest["boxes"] = []
    return latest


# ---- Utility helpers for progress ----
def _count_tiles_under(root: str) -> int:
    count = 0
    try:
        for r, _d, files in os.walk(root):
            for name in files:
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    count += 1
    except Exception:
        pass
    return count


def _parse_fetch_target_total(fetch_stdout_path: str):
    try:
        import re
        with open(fetch_stdout_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # Try multiple patterns to be more robust
        patterns = [
            r"Tiles:.*=\s*(\d+)\s+tiles",  # Original pattern
            r"(\d+)\s+tiles\s+total",      # Alternative pattern
            r"Total tiles:\s*(\d+)",       # Another alternative
            r"Fetching\s+(\d+)\s+tiles"    # Yet another
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
    except Exception:
        return None
    return None


def _estimate_fetch_target(area_sqmi: float, zoom: int) -> int:
    """Estimate the number of tiles needed for a given area and zoom level."""
    try:
        import math
        # Rough estimation: tiles per square mile varies by zoom
        # At zoom 18: ~4M tiles per sq mile
        # At zoom 17: ~1M tiles per sq mile  
        # At zoom 16: ~250K tiles per sq mile
        # etc. (roughly 4x fewer tiles per zoom level down)
        tiles_per_sqmi_at_18 = 4000000
        zoom_factor = 4 ** (18 - zoom)
        estimated = int(area_sqmi * tiles_per_sqmi_at_18 / zoom_factor)
        return max(1, estimated)  # At least 1 tile
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detection_types")
def get_detection_types():
    """Return available detection types for UI selection."""
    types = []
    for type_id in get_available_detection_types():
        config = get_detection_type(type_id)
        types.append({
            "id": config.id,
            "name": config.name,
            "description": config.description,
            "color": config.ui_color,
            "icon": config.icon
        })
    return jsonify(types)


@app.route("/detections")
def get_detections():
    results = load_results()
    reviewed_paths = load_reviewed_paths()
    
    positive_results = [
        r for r in results if r.get("positive") and r["path"] not in reviewed_paths
    ]
    
    return jsonify(positive_results)


@app.route("/mosaic/zooms")
def mosaic_zooms():
	"""Return available zoom levels discovered under tiles directory."""
	zooms = get_available_zooms()
	return jsonify(zooms)


def list_tiles_at_zoom(z: int):
	"""Enumerate tiles at a given zoom from filesystem and join with results map."""
	z_path = os.path.join(TILES_DIR, str(z))
	if not os.path.isdir(z_path):
		return {
			"z": z,
			"minX": None,
			"maxX": None,
			"minY": None,
			"maxY": None,
			"tiles": [],
		}

	results_map = load_results_map()
	# derive coarse-negative set from global coarse.jsonl if present
	global_coarse_log = "coarse.jsonl"
	coarse_neg_xy = _build_coarse_negative_xy_from_file(global_coarse_log, z)
	review_map = load_review_status_map()
	tiles = []
	min_x = None
	max_x = None
	min_y = None
	max_y = None

	for x_name in os.listdir(z_path):
		x_full = os.path.join(z_path, x_name)
		if not os.path.isdir(x_full) or not x_name.isdigit():
			continue
		try:
			x_val = int(x_name)
		except ValueError:
			continue
		for y_file in os.listdir(x_full):
			if not (y_file.endswith(".jpg") or y_file.endswith(".jpeg") or y_file.endswith(".png")):
				continue
			base, _ext = os.path.splitext(y_file)
			if not base.isdigit():
				continue
			try:
				y_val = int(base)
			except ValueError:
				continue
			rel_path = os.path.join(str(z), x_name, y_file)
			meta = results_map.get(rel_path, {})
			review_status = review_map.get(rel_path, {})
			positive = bool(meta.get("positive")) if meta else False
			confidence = meta.get("confidence") if meta else None
			# mark coarse-negative when no AI meta and present in coarse-neg set
			is_meta = confidence is not None
			is_coarse_neg = (not is_meta) and ((x_val, y_val) in coarse_neg_xy)
			tiles.append({
				"z": z,
				"x": x_val,
				"y": y_val,
				"path": rel_path.replace("\\", "/"),
				"positive": positive,
				"confidence": confidence,
				"reviewed": review_status.get("reviewed", False),
				"approved": review_status.get("approved", None),
				"coarse_negative": is_coarse_neg,
				"image_url": "/image/" + rel_path.replace("\\", "/"),
			})
			min_x = x_val if min_x is None else min(min_x, x_val)
			max_x = x_val if max_x is None else max(max_x, x_val)
			min_y = y_val if min_y is None else min(min_y, y_val)
			max_y = y_val if max_y is None else max(max_y, y_val)

	# Sort tiles for stable layout
	tiles.sort(key=lambda t: (t["y"], t["x"]))

	return {
		"z": z,
		"minX": min_x,
		"maxX": max_x,
		"minY": min_y,
		"maxY": max_y,
		"tiles": tiles,
	}


def list_tiles_at_zoom_for_area(area_id: str, z: int):
	"""Enumerate tiles at a given zoom for an AOI and join with that AOI's results map and review status."""
	p = area_paths(area_id)
	z_path = os.path.join(p["tiles"], str(z))
	if not os.path.isdir(z_path):
		return {
			"z": z,
			"minX": None,
			"maxX": None,
			"minY": None,
			"maxY": None,
			"tiles": [],
		}

	results_map = load_results_map_for_area(area_id)
	# Build coarse-negative set for this AOI
	p = area_paths(area_id)
	coarse_neg_xy = _build_coarse_negative_xy_from_file(p.get("coarse"), z)

	# build review map
	review_map = {}
	if os.path.exists(p["reviewed"]):
		with open(p["reviewed"], "r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				try:
					obj = json.loads(line)
				except json.JSONDecodeError:
					continue
				path = obj.get("path")
				if path:
					review_map[path] = {"approved": bool(obj.get("approved")), "reviewed": True}

	tiles = []
	min_x = None
	max_x = None
	min_y = None
	max_y = None

	for x_name in os.listdir(z_path):
		x_full = os.path.join(z_path, x_name)
		if not os.path.isdir(x_full) or not x_name.isdigit():
			continue
		try:
			x_val = int(x_name)
		except ValueError:
			continue
		for y_file in os.listdir(x_full):
			if not (y_file.endswith(".jpg") or y_file.endswith(".jpeg") or y_file.endswith(".png")):
				continue
			base, _ext = os.path.splitext(y_file)
			if not base.isdigit():
				continue
			try:
				y_val = int(base)
			except ValueError:
				continue
			rel_path = os.path.join(str(z), x_name, y_file)
			meta = results_map.get(rel_path, {})
			rmeta = review_map.get(rel_path, {})
			positive = bool(meta.get("positive")) if meta else False
			confidence = meta.get("confidence") if meta else None
			is_meta = confidence is not None
			is_coarse_neg = (not is_meta) and ((x_val, y_val) in coarse_neg_xy)
			tiles.append({
				"z": z,
				"x": x_val,
				"y": y_val,
				"path": rel_path.replace("\\", "/"),
				"positive": positive,
				"confidence": confidence,
				"reviewed": rmeta.get("reviewed", False),
				"approved": rmeta.get("approved", None),
				"coarse_negative": is_coarse_neg,
				"image_url": "/areas/" + area_id + "/image/" + rel_path.replace("\\", "/"),
			})
			min_x = x_val if min_x is None else min(min_x, x_val)
			max_x = x_val if max_x is None else max(max_x, x_val)
			min_y = y_val if min_y is None else min(min_y, y_val)
			max_y = y_val if max_y is None else max(max_y, y_val)

	# Sort tiles
	tiles.sort(key=lambda t: (t["y"], t["x"]))

	return {
		"z": z,
		"minX": min_x,
		"maxX": max_x,
		"minY": min_y,
		"maxY": max_y,
		"tiles": tiles,
	}
@app.route("/mosaic/tiles/<int:z>")
def mosaic_tiles(z: int):
	data = list_tiles_at_zoom(z)
	return jsonify(data)


@app.route("/areas", methods=["GET"])
def list_areas():
    idx = load_areas_index()
    return jsonify(idx.get("areas", []))


@app.route("/areas", methods=["POST"])
def create_area():
    payload = request.get_json(force=True) or {}
    name = (payload.get("name") or "Unnamed Area").strip()
    center = payload.get("center")
    area_sqmi = float(payload.get("area_sqmi") or 0.25)
    zoom = payload.get("zoom")
    geometry = payload.get("geometry")

    if not center or "lat" not in center or "lon" not in center:
        return jsonify({"error": "center {lat, lon} required"}), 400

    area_id = _generate_area_id()
    p = area_paths(area_id)
    os.makedirs(p["base"], exist_ok=True)
    os.makedirs(p["tiles"], exist_ok=True)
    os.makedirs(p["logs"], exist_ok=True)

    area_obj = {
        "id": area_id,
        "name": name,
        "center": center,
        "area_sqmi": area_sqmi,
        "zoom": zoom,
        "geometry": geometry,
        "created_at": time.time(),
        "status": {"fetch": "idle", "scan": "idle"},
        "paths": {
            "tiles": p["tiles"],
            "all_results": p["all_results"],
            "dumpsters": p["dumpsters"],
            "reviewed": p["reviewed"],
        },
    }
    _safe_write_json(p["area_json"], area_obj)

    idx = load_areas_index()
    idx.setdefault("areas", []).append({
        "id": area_id,
        "name": name,
        "center": center,
        "area_sqmi": area_sqmi,
        "zoom": zoom,
        "created_at": area_obj["created_at"],
    })
    save_areas_index(idx)
    return jsonify({"area_id": area_id, "area": area_obj})


@app.route("/areas/import_kml", methods=["POST"])
def import_kml_area():
    if parse_kml_polygons is None:
        return jsonify({"error": "KML parsing not available"}), 500
    f = request.files.get("kml") or request.files.get("file")
    if not f:
        return jsonify({"error": "missing file 'kml'"}), 400
    name = (request.form.get("name") or os.path.splitext(f.filename or "KML Area")[0]).strip() or "KML Area"
    zoom_raw = (request.form.get("zoom") or "").strip()
    zoom = int(zoom_raw) if zoom_raw.isdigit() else None

    # Read file into temp path to parse
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tf:
        kml_tmp = tf.name
        data = f.read()
        tf.write(data)
    try:
        polys = parse_kml_polygons(kml_tmp)
    finally:
        try:
            os.remove(kml_tmp)
        except Exception:
            pass
    if not polys:
        return jsonify({"error": "no polygons found in KML"}), 400

    # Compute center and rough area from bbox
    try:
        ctr_lat, ctr_lon = bbox_center(polys)
    except Exception:
        ctr_lat, ctr_lon = polys[0][0]
    try:
        # derive bbox again to compute area
        lat_min = min(lat for poly in polys for lat, _ in poly)
        lat_max = max(lat for poly in polys for lat, _ in poly)
        lon_min = min(lon for poly in polys for _, lon in poly)
        lon_max = max(lon for poly in polys for _, lon in poly)
        area_sqmi = bbox_area_sqmi(lat_min, lon_min, lat_max, lon_max)
    except Exception:
        area_sqmi = 0.25

    area_id = _generate_area_id()
    p = area_paths(area_id)
    os.makedirs(p["base"], exist_ok=True)
    os.makedirs(p["tiles"], exist_ok=True)
    os.makedirs(p["logs"], exist_ok=True)

    # Persist original KML
    try:
        with open(os.path.join(p["base"], "source.kml"), "wb") as out_kml:
            out_kml.write(data)
    except Exception:
        pass

    # Normalize polygons to JSON-serializable list of [lat, lon]
    geometry = [[ [float(lat), float(lon)] for (lat, lon) in poly ] for poly in polys]

    area_obj = {
        "id": area_id,
        "name": name,
        "center": {"lat": ctr_lat, "lon": ctr_lon},
        "area_sqmi": area_sqmi,
        "zoom": zoom,
        "geometry": geometry,
        "created_at": time.time(),
        "status": {"fetch": "idle", "scan": "idle"},
        "paths": {
            "tiles": p["tiles"],
            "all_results": p["all_results"],
            "dumpsters": p["dumpsters"],
            "reviewed": p["reviewed"],
        },
    }
    _safe_write_json(p["area_json"], area_obj)

    idx = load_areas_index()
    idx.setdefault("areas", []).append({
        "id": area_id,
        "name": name,
        "center": {"lat": ctr_lat, "lon": ctr_lon},
        "area_sqmi": area_sqmi,
        "zoom": zoom,
        "created_at": area_obj["created_at"],
    })
    save_areas_index(idx)
    return jsonify({"area_id": area_id, "area": area_obj})


@app.route("/areas/<area_id>", methods=["GET"])
def get_area(area_id: str):
    p = area_paths(area_id)
    area_obj = _safe_read_json(p["area_json"], None)
    if not area_obj:
        return jsonify({"error": "area not found"}), 404
    return jsonify(area_obj)


@app.route("/areas/<area_id>", methods=["PATCH"])
def rename_area(area_id: str):
    payload = request.get_json(force=True) or {}
    new_name = (payload.get("name") or "").strip()
    if not new_name:
        return jsonify({"error": "name required"}), 400
    p = area_paths(area_id)
    area_obj = _safe_read_json(p["area_json"], None)
    if not area_obj:
        return jsonify({"error": "area not found"}), 404
    area_obj["name"] = new_name
    _safe_write_json(p["area_json"], area_obj)
    # update index
    idx = load_areas_index()
    for a in idx.get("areas", []):
        if a.get("id") == area_id:
            a["name"] = new_name
            break
    save_areas_index(idx)
    return jsonify({"status": "ok", "area": area_obj})


@app.route("/areas/<area_id>", methods=["DELETE"])
def delete_area(area_id: str):
    p = area_paths(area_id)
    if not os.path.isdir(p["base"]):
        return jsonify({"error": "area not found"}), 404
    try:
        shutil.rmtree(p["base"])  # delete AOI directory
    except Exception as e:
        return jsonify({"error": f"failed to delete: {e}"}), 500
    # update index
    idx = load_areas_index()
    idx["areas"] = [a for a in idx.get("areas", []) if a.get("id") != area_id]
    save_areas_index(idx)
    return jsonify({"status": "deleted"})


AREA_JOBS = {}
AREA_LOCK = threading.Lock()


@app.route("/areas/<area_id>/fetch/start", methods=["POST"])
def area_fetch_start(area_id: str):
    p = area_paths(area_id)
    area_obj = _safe_read_json(p["area_json"], None)
    if not area_obj:
        return jsonify({"error": "area not found"}), 404

    center = area_obj.get("center") or {}
    zoom = area_obj.get("zoom")

    geometry = area_obj.get("geometry")
    if geometry:
        # Write geometry to a sidecar file for the fetcher
        geo_path = os.path.join(p["base"], "geometry.json")
        try:
            with open(geo_path, "w", encoding="utf-8") as gf:
                # Accept either list of polygons or wrapper dict
                if isinstance(geometry, dict):
                    json.dump(geometry, gf)
                else:
                    json.dump({"polygons": geometry}, gf)
        except Exception as e:
            return jsonify({"error": f"failed to write geometry: {e}"}), 500

        cmd = [
            sys.executable, "grab_imagery.py",
            "--geometry_json", geo_path,
            "--save_tiles_dir", p["tiles"],
            "--no_mosaic",
        ]
        # If center available, pass for auto-zoom probe
        if center.get("lat") is not None and center.get("lon") is not None:
            cmd += ["--lat", str(center.get("lat")), "--lon", str(center.get("lon"))]
        if zoom is not None and str(zoom).strip() != "":
            cmd += ["--zoom", str(int(zoom))]
        else:
            cmd += ["--auto_zoom"]
    else:
        # Legacy bbox mode
        area_sqmi = area_obj.get("area_sqmi") or 0.25
        cmd = [
            sys.executable, "grab_imagery.py",
            "--lat", str(center.get("lat")),
            "--lon", str(center.get("lon")),
            "--area_sqmi", str(area_sqmi),
            "--tiff", p["tiff"],
            "--png", p["png"],
            "--save_tiles_dir", p["tiles"],
        ]
        if zoom is not None and str(zoom).strip() != "":
            cmd += ["--zoom", str(int(zoom))]
        else:
            cmd += ["--auto_zoom"]

    os.makedirs(p["logs"], exist_ok=True)

    with AREA_LOCK:
        jobs = AREA_JOBS.setdefault(area_id, {"fetch": None, "scan": None})
        if proc_is_running(jobs.get("fetch")):
            return jsonify({"status": "error", "message": "fetch already running"}), 409
        out_fp = open(p["fetch_stdout"], "a", encoding="utf-8")
        err_fp = open(p["fetch_stderr"], "a", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=out_fp, stderr=err_fp)
        jobs["fetch"] = proc

    return jsonify({"status": "started", "pid": proc.pid, "cmd": cmd})


@app.route("/areas/<area_id>/fetch/status")
def area_fetch_status(area_id: str):
    p = area_paths(area_id)
    with AREA_LOCK:
        proc = (AREA_JOBS.get(area_id) or {}).get("fetch")
        running = proc_is_running(proc)
    zooms = get_available_zooms_for_area(area_id)
    fetched = _count_tiles_under(p["tiles"]) if os.path.isdir(p["tiles"]) else 0
    target = _parse_fetch_target_total(p["fetch_stdout"]) or None
    
    # If we can't parse the target from logs, try to estimate it
    if target is None:
        try:
            area_data = get_area(area_id)
            if area_data and area_data.get("area_sqmi") and area_data.get("zoom"):
                target = _estimate_fetch_target(area_data["area_sqmi"], area_data["zoom"])
        except Exception:
            pass
    
    percent = None
    if target and target > 0:
        percent = max(0, min(100, int(100 * fetched / target)))
    return jsonify({
        "running": running,
        "pid": getattr(proc, "pid", None),
        "zooms": zooms,
        "progress": {"fetched": fetched, "target": target, "percent": percent},
        "stdout": tail_lines(p["fetch_stdout"], 200),
        "stderr": tail_lines(p["fetch_stderr"], 200),
    })


@app.route("/areas/<area_id>/scan/start", methods=["POST"])
def area_scan_start(area_id: str):
    payload = request.get_json(force=True) or {}
    p = area_paths(area_id)
    if not os.path.isdir(p["tiles"]):
        return jsonify({"error": "tiles not found for area"}), 404

    detection_type = payload.get("detection_type") or "dumpsters"
    try:
        config = get_detection_type(detection_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    model = payload.get("model") or "google/gemini-2.5-pro"
    rpm = float(payload.get("rpm") or 60)
    min_conf = float(payload.get("min_confidence") or config.confidence_threshold)
    context_radius = int(payload.get("context_radius") or 0)
    coarse_factor = int(payload.get("coarse_factor") or 0)
    coarse_downscale = int(payload.get("coarse_downscale") or 256)
    coarse_threshold = float(payload.get("coarse_threshold") or config.coarse_threshold)
    limit = payload.get("limit")
    limit = int(limit) if (limit is not None and str(limit).strip() != "") else None
    concurrency = int(payload.get("concurrency") or 1)

    # Use detection-type-specific output file
    out_file = os.path.join(p["base"], get_output_filename(detection_type))
    events_log = os.path.join(p["logs"], "events.jsonl")

    # Optionally create an aoi_run in DB and capture run_id
    run_id = None
    db_status_msg = "DB: disabled (module unavailable)" if DbClient is None else None
    if DbClient is not None:
        try:
            area_obj = _safe_read_json(p["area_json"], None)
            geo = area_obj.get("geometry") if area_obj else None
            db = DbClient()
            if isinstance(geo, list) and geo and isinstance(geo[0], list):
                run_id = db.create_aoi_run(
                    mode=("kml" if geo else "point_radius"),
                    params={
                        "detection_type": detection_type,
                        "model": model,
                        "rpm": rpm,
                        "context_radius": context_radius,
                        "coarse_factor": coarse_factor,
                        "coarse_threshold": coarse_threshold,
                        "area_id": area_id,
                    },
                    polygons_latlon=geo,
                )
            else:
                center = area_obj.get("center") or {}
                center_lat = float(center.get("lat"))
                center_lon = float(center.get("lon"))
                area_sqmi_val = float(area_obj.get("area_sqmi") or 0.25)
                # Build a square around the center approximating the requested area
                miles_per_deg_lat = 69.0
                miles_per_deg_lon = 69.0 * max(0.000001, math.cos(math.radians(center_lat)))
                side_mi = max(0.0, math.sqrt(max(area_sqmi_val, 0.0)))
                half_lat = (side_mi / 2.0) / miles_per_deg_lat
                half_lon = (side_mi / 2.0) / miles_per_deg_lon
                lat_min = center_lat - half_lat
                lat_max = center_lat + half_lat
                lon_min = center_lon - half_lon
                lon_max = center_lon + half_lon
                square = [[lat_min, lon_min], [lat_min, lon_max], [lat_max, lon_max], [lat_max, lon_min], [lat_min, lon_min]]
                run_id = db.create_aoi_run(
                    mode="point_radius",
                    params={
                        "detection_type": detection_type,
                        "model": model,
                        "rpm": rpm,
                        "context_radius": context_radius,
                        "coarse_factor": coarse_factor,
                        "coarse_threshold": coarse_threshold,
                        "area_id": area_id,
                    },
                    polygons_latlon=[square],
                )
            try:
                db.close()
            except Exception:
                pass
            db_status_msg = f"DB: enabled (run_id={run_id})"
        except Exception as e:
            run_id = None
            db_status_msg = f"DB: connection failed: {e}"
            try:
                print(f"[DB] area_scan_start aoi_run failed: {e}", file=sys.stderr)
            except Exception:
                pass

    cmd = [
        sys.executable, "scan_objects.py",
        "--tiles_dir", p["tiles"],
        "--out", out_file,
        "--log_all", p["all_results"],
        "--detection_type", detection_type,
        "--rpm", str(rpm),
        "--model", model,
        "--min_confidence", str(min_conf),
        "--resume",
    ]
    # For construction: use boxes; if run_id exists, also emit to DB
    if detection_type == "construction":
        cmd += ["--construction_boxes"]
        if run_id:
            cmd += ["--emit_boxes_to_db", "--run_id", str(run_id)]
    if concurrency and concurrency > 1:
        cmd += ["--concurrency", str(concurrency)]
    if context_radius and context_radius > 0:
        cmd += ["--context_radius", str(context_radius)]
    if coarse_factor and coarse_factor > 1:
        cmd += ["--coarse_factor", str(coarse_factor),
                "--coarse_downscale", str(coarse_downscale),
                "--coarse_threshold", str(coarse_threshold),
                "--coarse_log", p["coarse"]]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    # events log for UI visibility
    cmd += ["--events_log", events_log]

    # Events log for this area scan
    try:
        events_log = os.path.join(p["logs"], "events.jsonl")
        cmd += ["--events_log", events_log]
    except Exception:
        pass

    os.makedirs(p["logs"], exist_ok=True)

    with AREA_LOCK:
        jobs = AREA_JOBS.setdefault(area_id, {"fetch": None, "scan": None})
        if proc_is_running(jobs.get("scan")):
            return jsonify({"status": "error", "message": "scan already running"}), 409
        out_fp = open(p["scan_stdout"], "a", encoding="utf-8")
        err_fp = open(p["scan_stderr"], "a", encoding="utf-8")
        # Prepend a launch banner to stderr for visibility
        try:
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            banner = [
                f"=== Scan Launch {ts} ===",
                f"area_id: {area_id}",
                f"tiles_dir: {p['tiles']}",
                f"detection_type: {detection_type}",
                f"model: {model}",
                f"rpm: {rpm}",
                f"concurrency: {concurrency}",
                f"min_confidence: {min_conf}",
                f"context_radius: {context_radius}",
                f"coarse_factor: {coarse_factor}",
                f"coarse_downscale: {coarse_downscale}",
                f"coarse_threshold: {coarse_threshold}",
                f"run_id: {run_id}",
                db_status_msg or "DB: n/a",
                f"cmd: {' '.join(map(str, cmd))}",
                "===\n",
            ]
            err_fp.write("\n".join(banner))
            err_fp.flush()
        except Exception:
            pass
        proc = subprocess.Popen(cmd, stdout=out_fp, stderr=err_fp)
        jobs["scan"] = proc

    # Mirror into global SCAN_STATE so Scan tab + SSE can display this run
    with SCAN_LOCK:
        SCAN_STATE["proc"] = proc
        SCAN_STATE["started_at"] = time.time()
        SCAN_STATE["args"] = {
            "area_id": area_id,
            "tiles_dir": p["tiles"],
            "detection_type": detection_type,
            "model": model,
            "rpm": rpm,
            "concurrency": concurrency,
            "context_radius": context_radius,
            "coarse_factor": coarse_factor,
        }
        SCAN_STATE["files"] = {
            "all_results": p["all_results"],
            "out": out_file,
            "coarse": p["coarse"],
            "stdout": p["scan_stdout"],
            "stderr": p["scan_stderr"],
            "events": events_log,
        }
        SCAN_STATE["baseline"] = {
            "all_results": safe_count_lines(p["all_results"]) if os.path.exists(p["all_results"]) else 0,
            "out": safe_count_lines(out_file) if os.path.exists(out_file) else 0,
            "coarse": safe_count_lines(p["coarse"]) if os.path.exists(p["coarse"]) else 0,
        }

    return jsonify({"status": "started", "pid": proc.pid, "cmd": cmd, "events_log": events_log})


@app.route("/areas/<area_id>/scan/status")
def area_scan_status(area_id: str):
    p = area_paths(area_id)
    def _count(path: str) -> int:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    with AREA_LOCK:
        proc = (AREA_JOBS.get(area_id) or {}).get("scan")
        running = proc_is_running(proc)
    total_tiles = _count_tiles_under(p["tiles"]) if os.path.isdir(p["tiles"]) else None
    processed = _count(p["all_results"]) if os.path.exists(p["all_results"]) else 0
    percent = None
    if total_tiles and total_tiles > 0:
        percent = max(0, min(100, int(100 * processed / total_tiles)))
    return jsonify({
        "running": running,
        "pid": getattr(proc, "pid", None),
        "counts": {
            "all_results": _count(p["all_results"]),
            "out": _count(p["dumpsters"]),
            "coarse": _count(p["coarse"]),
        },
        "progress": {"processed": processed, "total": total_tiles, "percent": percent},
        "stdout": tail_lines(p["scan_stdout"], 200),
        "stderr": tail_lines(p["scan_stderr"], 200),
    })


@app.route("/areas/<area_id>/mosaic/zooms")
def area_mosaic_zooms(area_id: str):
    zooms = get_available_zooms_for_area(area_id)
    return jsonify(zooms)


@app.route("/areas/<area_id>/mosaic/tiles/<int:z>")
def area_mosaic_tiles(area_id: str, z: int):
    data = list_tiles_at_zoom_for_area(area_id, z)
    return jsonify(data)


@app.route("/areas/<area_id>/image/<path:image_path>")
def area_get_image(area_id: str, image_path: str):
    p = area_paths(area_id)
    return send_from_directory(p["tiles"], image_path)


@app.route("/areas/<area_id>/detections")
def area_get_detections(area_id: str):
    p = area_paths(area_id)
    results = []
    if os.path.exists(p["all_results"]):
        with open(p["all_results"], "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("positive"):
                    results.append(obj)
    reviewed = set()
    if os.path.exists(p["reviewed"]):
        with open(p["reviewed"], "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    robj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if robj.get("path"):
                    reviewed.add(robj["path"])
    filtered = [r for r in results if r.get("path") not in reviewed]
    return jsonify(filtered)


@app.route("/areas/<area_id>/review", methods=["POST"])
def area_review_detection(area_id: str):
    p = area_paths(area_id)
    data = request.get_json(force=True) or {}
    image_path = data.get("path")
    is_approved = data.get("approved")
    if not image_path:
        return jsonify({"error": "Missing path"}), 400
    with open(p["reviewed"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"path": image_path, "approved": bool(is_approved)}) + "\n")
    return jsonify({"status": "ok"})


@app.route("/areas/<area_id>/review/toggle", methods=["POST"])
def area_toggle_review_status(area_id: str):
    p = area_paths(area_id)
    data = request.get_json(force=True) or {}
    image_path = data.get("path")
    if not image_path:
        return jsonify({"error": "Missing path"}), 400
    review_map = {}
    if os.path.exists(p["reviewed"]):
        with open(p["reviewed"], "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("path"):
                    review_map[obj["path"]] = bool(obj.get("approved"))
    if "approved" in data:
        new_approved = bool(data["approved"])
    else:
        cur = review_map.get(image_path)
        new_approved = True if cur is None else (not cur)
    with open(p["reviewed"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"path": image_path, "approved": new_approved}) + "\n")
    return jsonify({"status": "ok", "path": image_path, "approved": new_approved, "reviewed": True})
@app.route("/scan/start", methods=["POST"])
def scan_start():
    payload = request.get_json(force=True) or {}
    area_id = payload.get("area_id") or None
    tiles_dir = os.path.abspath(payload.get("tiles_dir") or TILES_DIR)
    detection_type = payload.get("detection_type") or "dumpsters"
    
    try:
        config = get_detection_type(detection_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    model = payload.get("model") or "google/gemini-2.5-pro"
    rpm = float(payload.get("rpm") or 60)
    min_conf = float(payload.get("min_confidence") or config.confidence_threshold)
    context_radius = int(payload.get("context_radius") or 0)
    coarse_factor = int(payload.get("coarse_factor") or 0)
    coarse_downscale = int(payload.get("coarse_downscale") or 256)
    coarse_threshold = float(payload.get("coarse_threshold") or config.coarse_threshold)
    limit = payload.get("limit")
    limit = int(limit) if (limit is not None and str(limit).strip() != "") else None
    concurrency = int(payload.get("concurrency") or 1)

    # Decide output files: area-scoped if area_id provided, else global
    if area_id:
        p = area_paths(area_id)
        os.makedirs(p["base"], exist_ok=True)
        os.makedirs(p["logs"], exist_ok=True)
        all_results = p["all_results"]
        out_file = os.path.join(p["base"], get_output_filename(detection_type))
        coarse_log = p["coarse"] if (coarse_factor and coarse_factor > 1) else None
        stdout_log = p["scan_stdout"]
        stderr_log = p["scan_stderr"]
        # Optional: create aoi_run in DB using area geometry (or synthesize from center+area)
        run_id = None
        db_status_msg = "DB: disabled (module unavailable)" if DbClient is None else None
        if DbClient is not None:
            try:
                area_obj = _safe_read_json(p["area_json"], None)
                geo = area_obj.get("geometry") if area_obj else None
                db = DbClient()
                if isinstance(geo, list) and geo and isinstance(geo[0], list):
                    run_id = db.create_aoi_run(
                        mode=("kml" if geo else "point_radius"),
                        params={
                            "detection_type": detection_type,
                            "model": model,
                            "rpm": rpm,
                            "context_radius": context_radius,
                            "coarse_factor": coarse_factor,
                            "coarse_threshold": coarse_threshold,
                            "area_id": area_id,
                        },
                        polygons_latlon=geo,
                    )
                else:
                    center = area_obj.get("center") or {}
                    center_lat = float(center.get("lat"))
                    center_lon = float(center.get("lon"))
                    area_sqmi_val = float(area_obj.get("area_sqmi") or 0.25)
                    # Build a square around the center approximating the requested area
                    miles_per_deg_lat = 69.0
                    miles_per_deg_lon = 69.0 * max(0.000001, math.cos(math.radians(center_lat)))
                    side_mi = max(0.0, math.sqrt(max(area_sqmi_val, 0.0)))
                    half_lat = (side_mi / 2.0) / miles_per_deg_lat
                    half_lon = (side_mi / 2.0) / miles_per_deg_lon
                    lat_min = center_lat - half_lat
                    lat_max = center_lat + half_lat
                    lon_min = center_lon - half_lon
                    lon_max = center_lon + half_lon
                    square = [[lat_min, lon_min], [lat_min, lon_max], [lat_max, lon_max], [lat_max, lon_min], [lat_min, lon_min]]
                    run_id = db.create_aoi_run(
                        mode="point_radius",
                        params={
                            "detection_type": detection_type,
                            "model": model,
                            "rpm": rpm,
                            "context_radius": context_radius,
                            "coarse_factor": coarse_factor,
                            "coarse_threshold": coarse_threshold,
                            "area_id": area_id,
                        },
                        polygons_latlon=[square],
                    )
                try:
                    db.close()
                except Exception:
                    pass
            except Exception as e:
                run_id = None
                db_status_msg = f"DB: connection failed: {e}"
                try:
                    print(f"[DB] aoi_run create failed: {e}", file=sys.stderr)
                except Exception:
                    pass
            else:
                db_status_msg = f"DB: enabled (run_id={run_id})"
    else:
        # Output files (fixed names in cwd)
        all_results = ALL_RESULTS_FILE
        out_file = get_output_filename(detection_type)
        coarse_log = "coarse.jsonl" if coarse_factor and coarse_factor > 1 else None
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        stdout_log = f"scan_{ts}.out.log"
        stderr_log = f"scan_{ts}.err.log"

    cmd = [
        sys.executable,
        "scan_objects.py",
        "--tiles_dir", tiles_dir,
        "--out", out_file,
        "--log_all", all_results,
        "--detection_type", detection_type,
        "--rpm", str(rpm),
        "--model", model,
        "--min_confidence", str(min_conf),
        "--resume",
    ]
    # For construction: use boxes; if run_id exists, also emit to DB
    if detection_type == "construction":
        cmd += ["--construction_boxes"]
        if area_id and run_id:
            cmd += ["--emit_boxes_to_db", "--run_id", str(run_id)]
    if concurrency and concurrency > 1:
        cmd += ["--concurrency", str(concurrency)]
    if context_radius and context_radius > 0:
        cmd += ["--context_radius", str(context_radius)]
    if coarse_factor and coarse_factor > 1:
        cmd += ["--coarse_factor", str(coarse_factor),
                "--coarse_downscale", str(coarse_downscale),
                "--coarse_threshold", str(coarse_threshold)]
        if coarse_log:
            cmd += ["--coarse_log", coarse_log]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    # Events log path (area-scoped if area_id provided)
    if area_id:
        events_log = os.path.join(area_paths(area_id)["logs"], "events.jsonl")
    else:
        events_log = os.path.abspath("scan_events.jsonl")
    cmd += ["--events_log", events_log]

    # Prepare stdout/stderr logs (area-scoped if provided)

    with SCAN_LOCK:
        if proc_is_running(SCAN_STATE.get("proc")):
            return jsonify({"status": "error", "message": "Scan already running"}), 409
        SCAN_STATE["files"]["all_results"] = all_results
        SCAN_STATE["files"]["out"] = out_file
        SCAN_STATE["files"]["coarse"] = coarse_log
        SCAN_STATE["files"]["stdout"] = stdout_log
        SCAN_STATE["files"]["stderr"] = stderr_log
        SCAN_STATE["files"]["events"] = events_log
        SCAN_STATE["baseline"]["all_results"] = safe_count_lines(all_results)
        SCAN_STATE["baseline"]["out"] = safe_count_lines(out_file)
        SCAN_STATE["baseline"]["coarse"] = safe_count_lines(coarse_log) if coarse_log else 0
        SCAN_STATE["started_at"] = time.time()
        # Persist args including area_id so status/progress reflects this run
        SCAN_STATE["args"] = {**payload, "area_id": area_id, "tiles_dir": tiles_dir}

        stdout_fp = open(stdout_log, "a", encoding="utf-8")
        stderr_fp = open(stderr_log, "a", encoding="utf-8")
        # Prepend a launch banner to stderr for visibility
        try:
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            banner = [
                f"=== Scan Launch {ts} ===",
                f"area_id: {area_id}",
                f"tiles_dir: {tiles_dir}",
                f"detection_type: {detection_type}",
                f"model: {model}",
                f"rpm: {rpm}",
                f"concurrency: {concurrency}",
                f"min_confidence: {min_conf}",
                f"context_radius: {context_radius}",
                f"coarse_factor: {coarse_factor}",
                f"coarse_downscale: {coarse_downscale}",
                f"coarse_threshold: {coarse_threshold}",
                f"run_id: {run_id}",
                (db_status_msg if area_id else "DB: n/a (global run)") if 'db_status_msg' in locals() else "DB: n/a",
                f"cmd: {' '.join(map(str, cmd))}",
                "===\n",
            ]
            stderr_fp.write("\n".join(banner))
            stderr_fp.flush()
        except Exception:
            pass
        proc = subprocess.Popen(cmd, stdout=stdout_fp, stderr=stderr_fp)
        SCAN_STATE["proc"] = proc

    return jsonify({"status": "started", "pid": proc.pid, "cmd": cmd})


@app.route("/scan/stop", methods=["POST"])
def scan_stop():
    with SCAN_LOCK:
        proc = SCAN_STATE.get("proc")
        if proc and proc_is_running(proc):
            try:
                proc.terminate()
                # Give it a moment to terminate gracefully
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    proc.kill()
                    proc.wait()
                return jsonify({"status": "stopped", "pid": proc.pid})
            except Exception as e:
                return jsonify({"error": f"Failed to stop scan: {e}"}), 500
        else:
            return jsonify({"error": "No scan running"}), 404


@app.route("/scan/estimate", methods=["POST"])
def scan_estimate():
    """Calculate estimated API requests for scan settings."""
    payload = request.get_json(force=True) or {}
    tiles_dir = os.path.abspath(payload.get("tiles_dir") or TILES_DIR)
    coarse_factor = int(payload.get("coarse_factor") or 0)
    limit = payload.get("limit")
    limit = int(limit) if (limit is not None and str(limit).strip() != "") else None
    
    if not os.path.isdir(tiles_dir):
        return jsonify({"error": "Tiles directory not found"}), 404
    
    total_tiles = _count_tiles_under(tiles_dir)
    
    if coarse_factor and coarse_factor > 1:
        # With coarse filtering: only process top-left tiles of each coarse_factor x coarse_factor block
        # This is a rough estimate - the actual count depends on tile coordinates
        coarse_tiles = total_tiles // (coarse_factor * coarse_factor)
        # Assume some percentage of coarse tiles are positive and need refinement
        # Conservative estimate: 20% of coarse tiles are positive
        estimated_refinement_rate = 0.2
        refined_tiles = int(coarse_tiles * estimated_refinement_rate * coarse_factor * coarse_factor)
        min_requests = coarse_tiles + refined_tiles
        max_requests = coarse_tiles + total_tiles  # Worst case: all coarse are positive
    else:
        # No coarse filtering: one request per tile
        min_requests = total_tiles
        max_requests = total_tiles
    
    if limit is not None:
        min_requests = min(min_requests, limit)
        max_requests = min(max_requests, limit)
    
    return jsonify({
        "total_tiles": total_tiles,
        "min_requests": min_requests,
        "max_requests": max_requests,
        "coarse_factor": coarse_factor,
        "limit": limit,
    })


@app.route("/scan/status")
def scan_status():
    with SCAN_LOCK:
        proc = SCAN_STATE.get("proc")
        running = proc_is_running(proc)
        files = SCAN_STATE.get("files", {})
        baseline = SCAN_STATE.get("baseline", {})
        args = SCAN_STATE.get("args", {})
        
        all_lines = safe_count_lines(files.get("all_results"))
        out_lines = safe_count_lines(files.get("out"))
        coarse_lines = safe_count_lines(files.get("coarse")) if files.get("coarse") else 0
        
        # Calculate progress (accounting for coarse filtering)
        tiles_dir = (args or {}).get("tiles_dir", "tiles")
        total_tiles = _count_tiles_under(tiles_dir) if os.path.isdir(tiles_dir) else None
        
        # For coarse filtering, progress includes both coarse and refinement stages
        coarse_factor = (args or {}).get("coarse_factor", 0)
        if coarse_factor and coarse_factor > 1:
            # Two-stage process: coarse + refinement
            coarse_processed = max(0, coarse_lines - baseline.get("coarse", 0))
            refinement_processed = max(0, all_lines - baseline.get("all_results", 0))
            
            # Total progress is coarse progress + refinement progress
            processed = coarse_processed + refinement_processed
            
            # Expected: coarse tiles + estimated refinement tiles (assume 20% positive rate)
            if total_tiles and total_tiles > 0:
                expected_coarse = total_tiles // (coarse_factor * coarse_factor)
                estimated_refinement = int(expected_coarse * 0.2 * coarse_factor * coarse_factor)
                expected_total = expected_coarse + estimated_refinement
            else:
                expected_total = None
        else:
            # Single-stage process: direct tile processing
            processed = max(0, all_lines - baseline.get("all_results", 0))
            expected_total = total_tiles
        
        percent = None
        if expected_total and expected_total > 0:
            percent = max(0, min(100, int(100 * processed / expected_total)))
        if (not running) and expected_total and expected_total > 0:
            percent = 100
        
        progress = {
            "running": running,
            "pid": getattr(proc, "pid", None),
            "started_at": SCAN_STATE.get("started_at"),
            "args": args,
            "counts": {
                "all_results": max(0, all_lines - baseline.get("all_results", 0)),
                "out": max(0, out_lines - baseline.get("out", 0)),
                "coarse": max(0, coarse_lines - baseline.get("coarse", 0)),
            },
            "progress": {"processed": processed, "total": expected_total, "percent": percent},
        }
    return jsonify(progress)


@app.route("/scan/logs")
def scan_logs():
    max_lines = int(request.args.get("n", 200))
    with SCAN_LOCK:
        files = SCAN_STATE.get("files", {})
        stdout_log = files.get("stdout")
        stderr_log = files.get("stderr")
    return jsonify({
        "stdout": tail_lines(stdout_log, max_lines) if stdout_log else "",
        "stderr": tail_lines(stderr_log, max_lines) if stderr_log else "",
    })


@app.route("/db/status")
def db_status():
    """Quick DB diagnostics: connection ok, DSN (redacted), basic table presence.

    Returns JSON with fields: connected, dsn, error, tables_present.
    """
    out = {"connected": False, "dsn": None, "error": None, "tables_present": {}}
    if DbClient is None:
        out["error"] = "db module unavailable"
        return jsonify(out), 200
    # Build DSN via config util to show what we're using (redacted password)
    try:
        from db.config import build_dsn
        dsn = build_dsn()
        # redact password in dsn
        out["dsn"] = dsn.replace("password=" + (os.getenv("DB_PASS") or ""), "password=***")
    except Exception as e:
        out["dsn"] = None
    try:
        db = DbClient()
        out["connected"] = True
        with db.conn.cursor() as cur:
            # Simple checks for expected tables
            for t in ("aoi_runs", "detections", "imagery_tiles", "sites"):
                try:
                    cur.execute("SELECT to_regclass(%s);", (t,))
                    present = bool(cur.fetchone()[0])
                except Exception:
                    present = False
                out["tables_present"][t] = present
        try:
            db.close()
        except Exception:
            pass
    except Exception as e:
        out["error"] = str(e)
        try:
            print(f"[DB] status check failed: {e}", file=sys.stderr)
        except Exception:
            pass
    return jsonify(out)


# ---- LLM events visibility ----
def _read_jsonl(path: str) -> list:
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return out


@app.route("/scan/events/list")
def scan_events_list():
    try:
        offset = int(request.args.get("offset", 0))
    except Exception:
        offset = 0
    try:
        limit = int(request.args.get("limit", 500))
        limit = max(1, min(limit, 2000))
    except Exception:
        limit = 500
    with SCAN_LOCK:
        events_path = (SCAN_STATE.get("files") or {}).get("events")
    if not events_path or not os.path.exists(events_path):
        return jsonify({"items": [], "next": offset, "total": 0})
    items = []
    total = 0
    try:
        with open(events_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                total = i + 1
                if i < offset:
                    continue
                if len(items) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return jsonify({"items": items, "next": offset + len(items), "total": total})


@app.route("/scan/events/tail")
def scan_events_tail():
    try:
        n = int(request.args.get("n", 200))
        n = max(1, min(n, 1000))
    except Exception:
        n = 200
    with SCAN_LOCK:
        events_path = (SCAN_STATE.get("files") or {}).get("events")
    if not events_path or not os.path.exists(events_path):
        return jsonify({"items": [], "total": 0})
    lines = []
    try:
        with open(events_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        lines = []
    items = []
    total = len(lines)
    for raw in lines[-n:]:
        try:
            items.append(json.loads(raw))
        except Exception:
            continue
    return jsonify({"items": items, "total": total})


@app.route("/scan/events/clear", methods=["POST"])
def scan_events_clear():
    with SCAN_LOCK:
        events_path = (SCAN_STATE.get("files") or {}).get("events")
    if not events_path:
        return jsonify({"status": "ok", "cleared": False})
    try:
        os.makedirs(os.path.dirname(os.path.abspath(events_path)) or ".", exist_ok=True)
        with open(events_path, "w", encoding="utf-8") as f:
            f.write("")
        return jsonify({"status": "ok", "cleared": True})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/scan/events/stream")
def scan_events_stream():
    """Server-Sent Events (SSE) stream of LLM request events for the active scan.

    Query params:
      - tail: int (optional) number of most recent events to preload (default: 200)
    Respects Last-Event-ID header to avoid re-sending events on reconnect (expects seq values).
    """
    try:
        tail_n = int(request.args.get("tail", 200))
        tail_n = max(0, min(tail_n, 2000))
    except Exception:
        tail_n = 200

    with SCAN_LOCK:
        events_path = (SCAN_STATE.get("files") or {}).get("events")

    def sse_format(ev: dict) -> str:
        # Send id for reconnection, and a single-line JSON in data
        ev_id = ev.get("seq")
        payload = json.dumps(ev, ensure_ascii=False)
        parts = []
        if isinstance(ev_id, int):
            parts.append(f"id: {ev_id}")
        parts.append(f"data: {payload}")
        return "\n".join(parts) + "\n\n"

    @stream_with_context
    def generate():
        # If no events path yet, keep heartbeating until available or client disconnects
        while not events_path or not os.path.exists(events_path):
            try:
                yield ": waiting for events...\n\n"
                time.sleep(1.0)
            except GeneratorExit:
                return
            except Exception:
                time.sleep(1.0)
            # Re-check in case scan started after initial check
            with SCAN_LOCK:
                _p = (SCAN_STATE.get("files") or {}).get("events")
            if _p and os.path.exists(_p):
                break

        with SCAN_LOCK:
            path = (SCAN_STATE.get("files") or {}).get("events")
        if not path:
            # nothing to stream; keep connection alive lightly
            while True:
                yield ": no events path\n\n"
                time.sleep(2.0)

        # Preload tail
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            lines = []

        last_event_id = request.headers.get("Last-Event-ID")
        start_idx = 0
        if last_event_id:
            try:
                last_id_int = int(last_event_id)
                # find first line with seq > last_id_int
                for i, ln in enumerate(lines):
                    try:
                        ev = json.loads(ln)
                        if int(ev.get("seq") or 0) > last_id_int:
                            start_idx = i
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        else:
            # use tail
            if tail_n > 0 and len(lines) > tail_n:
                start_idx = len(lines) - tail_n

        for ln in lines[start_idx:]:
            try:
                ev = json.loads(ln)
                yield sse_format(ev)
            except Exception:
                continue

        # Follow the file for new lines
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(0, os.SEEK_END)
                while True:
                    where = f.tell()
                    line = f.readline()
                    if not line:
                        time.sleep(0.5)
                        f.seek(where)
                        # send heartbeat every few cycles to keep proxies happy
                        yield ": hb\n\n"
                        continue
                    try:
                        ev = json.loads(line)
                        yield sse_format(ev)
                    except Exception:
                        continue
        except GeneratorExit:
            return
        except Exception:
            # end stream on unexpected errors
            return

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Allow XHR streaming and intermediaries
        "X-Accel-Buffering": "no",
    }
    return Response(generate(), headers=headers)


@app.route("/geocode", methods=["POST"])
def geocode_lookup():
    """Reverse-geocode a lat/lon (or z/x/y) and return an address.

    JSON body accepts: { lat, lon } OR { z, x, y }, optional { area_id, provider }.
    Provider: arcgis (default), nominatim, google, mapbox.
    Caches results in geocode_cache.json (global) or runs/<area>/geocode_cache.json.
    """
    payload = request.get_json(force=True) or {}
    area_id = payload.get("area_id") or None
    provider = (payload.get("provider") or "arcgis").lower()

    lat = payload.get("lat")
    lon = payload.get("lon")
    z = payload.get("z")
    x = payload.get("x")
    y = payload.get("y")
    px = payload.get("px")
    py = payload.get("py")

    if (lat is None or lon is None):
        try:
            zi = int(z) if z is not None else None
            xi = int(x) if x is not None else None
            yi = int(y) if y is not None else None
        except Exception:
            zi = xi = yi = None
        if None not in (zi, xi, yi):
            try:
                # Clamp within valid range
                n = 2 ** zi
                xi = max(0, min(xi, n - 1))
                yi = max(0, min(yi, n - 1))
            except Exception:
                pass
            try:
                if px is not None and py is not None:
                    lat, lon = _pixel_to_latlon(zi, xi, yi, int(px), int(py))
                else:
                    lat, lon = _tile_center_latlon(zi, xi, yi)
            except Exception:
                lat, lon = None, None
    if lat is None or lon is None:
        return jsonify({"error": "lat/lon or z/x/y required"}), 400

    cache_path = _geocode_cache_path(area_id)
    # Cache key: if px/py used or lat/lon explicitly given, prefer precise lat/lon key
    precise = (px is not None and py is not None) or (payload.get("lat") is not None and payload.get("lon") is not None)
    if precise:
        key = f"{round(float(lat), 6)},{round(float(lon), 6)}"
    elif None not in (z, x, y):
        key = f"{int(z)}/{int(x)}/{int(y)}"
    else:
        key = f"{round(float(lat), 6)},{round(float(lon), 6)}"

    # Load cache
    with GEOCODE_LOCK:
        cache = _safe_read_json(cache_path, {})
        if key in cache and isinstance(cache[key], str) and cache[key].strip():
            return jsonify({"address": cache[key], "cached": True})

    # Dispatch provider
    try:
        if provider == "arcgis":
            addr = _geocode_provider_arcgis(float(lat), float(lon))
        elif provider == "nominatim":
            addr = _geocode_provider_nominatim(float(lat), float(lon))
        elif provider == "google":
            api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
            if not api_key:
                return jsonify({"error": "GOOGLE_MAPS_API_KEY not set"}), 400
            addr = _geocode_provider_google(float(lat), float(lon), api_key)
        elif provider == "mapbox":
            token = os.environ.get("MAPBOX_ACCESS_TOKEN")
            if not token:
                return jsonify({"error": "MAPBOX_ACCESS_TOKEN not set"}), 400
            addr = _geocode_provider_mapbox(float(lat), float(lon), token)
        else:
            return jsonify({"error": f"unknown provider: {provider}"}), 400
    except Exception as e:
        return jsonify({"error": f"geocode failed: {e}"}), 502

    if addr:
        with GEOCODE_LOCK:
            cache = _safe_read_json(cache_path, {})
            cache[key] = addr
            _safe_write_json(cache_path, cache)
        return jsonify({"address": addr, "cached": False})
    else:
        return jsonify({"address": None, "cached": False})


@app.route("/scan/clear", methods=["POST"])
def scan_clear_global():
    """Clear global scan artifacts across all detection types.

    Removes:
    - all_results.jsonl (resume log)
    - coarse.jsonl (coarse stage log)
    - positives-only files for all detection types (e.g., dumpsters.jsonl, construction_sites.jsonl)
    """
    cleared = []
    errors = []
    # Start with shared logs
    files_set = {ALL_RESULTS_FILE, "coarse.jsonl"}
    # Add positives-only outputs for all detection types
    try:
        for t in get_available_detection_types():
            files_set.add(get_output_filename(t))
    except Exception:
        # Fallback: ensure at least dumpsters.jsonl is included
        files_set.add("dumpsters.jsonl")
    for f in sorted(files_set):
        try:
            if os.path.exists(f):
                os.remove(f)
                cleared.append(f)
        except Exception as e:
            errors.append({"file": f, "error": str(e)})
    return jsonify({"status": "ok", "cleared": cleared, "errors": errors})


@app.route("/areas/<area_id>/scan/clear", methods=["POST"])
def scan_clear_area(area_id: str):
    """Clear scan artifacts for an area across all detection types.

    Removes area-scoped:
    - all_results.jsonl, coarse.jsonl, reviewed_results.jsonl
    - positives-only files for all detection types under the area base dir
    """
    p = area_paths(area_id)
    if not os.path.isdir(p["base"]):
        return jsonify({"error": "area not found"}), 404
    cleared = []
    errors = []
    files_set = {p["all_results"], p["coarse"], p["reviewed"], p.get("dumpsters")}
    # Add detection-type-specific positives files
    try:
        for t in get_available_detection_types():
            files_set.add(os.path.join(p["base"], get_output_filename(t)))
    except Exception:
        # Fallback to dumpsters file path already included
        pass
    for f in sorted({f for f in files_set if f}):
        try:
            if os.path.exists(f):
                os.remove(f)
                cleared.append(f)
        except Exception as e:
            errors.append({"file": f, "error": str(e)})
    return jsonify({"status": "ok", "area_id": area_id, "cleared": cleared, "errors": errors})


@app.route("/image/<path:image_path>")
def get_image(image_path):
    return send_from_directory(TILES_DIR, image_path)


@app.route("/review", methods=["POST"])
def review_detection():
    data = request.get_json()
    image_path = data.get("path")
    is_approved = data.get("approved")

    with open(REVIEWED_RESULTS_FILE, "a") as f:
        f.write(
            json.dumps({"path": image_path, "approved": is_approved}) + "\n"
        )

    return jsonify({"status": "ok"})


@app.route("/review/toggle", methods=["POST"])
def toggle_review_status():
    """Toggle or set the review status for a tile."""
    data = request.get_json()
    image_path = data.get("path")
    
    if not image_path:
        return jsonify({"error": "Missing path"}), 400
    
    # Load current review status
    review_map = load_review_status_map()
    current_status = review_map.get(image_path, {})
    
    # Determine new status
    if "approved" in data:
        # Set specific status
        new_approved = bool(data["approved"])
    else:
        # Toggle current status
        if current_status.get("reviewed"):
            # If already reviewed, toggle approval
            new_approved = not current_status.get("approved", False)
        else:
            # If not reviewed, default to approved
            new_approved = True
    
    # Write the new status
    with open(REVIEWED_RESULTS_FILE, "a") as f:
        f.write(
            json.dumps({"path": image_path, "approved": new_approved}) + "\n"
        )
    
    return jsonify({
        "status": "ok", 
        "path": image_path,
        "approved": new_approved,
        "reviewed": True
    })


# ---- DB-backed API (detections and sites) ----
def _db_client_or_error():
    if DbClient is None:
        return None, (jsonify({"error": "DB module not available"}), 500)
    try:
        return DbClient(), None
    except Exception as e:
        return None, (jsonify({"error": f"DB connection failed: {e}"}), 500)


@app.route("/api/detections")
def api_detections():
    db, err = _db_client_or_error()
    if err:
        return err
    try:
        bbox = request.args.get("bbox")  # west,south,east,north (lon/lat)
        run_id = request.args.get("run_id")
        zcta = request.args.get("zcta")
        date_from = request.args.get("date_from")  # YYYY-MM-DD
        date_to = request.args.get("date_to")
        limit = int(request.args.get("limit") or 500)
        limit = max(1, min(limit, 5000))

        where = []
        params = []
        if run_id:
            where.append("d.run_id = %s")
            params.append(run_id)
        if date_from:
            where.append("d.created_at >= %s")
            params.append(date_from)
        if date_to:
            where.append("d.created_at <= %s")
            params.append(date_to)
        if bbox:
            try:
                west, south, east, north = [float(x) for x in bbox.split(',')]
                where.append("ST_Intersects(d.bbox_3857, ST_Transform(ST_MakeEnvelope(%s,%s,%s,%s,4326),3857))")
                params.extend([west, south, east, north])
            except Exception:
                pass
        if zcta:
            # Filter via sites zcta by joining through site_detections
            where.append("sd.site_id IN (SELECT id FROM sites WHERE zcta = %s)")
            params.append(zcta)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
            SELECT d.id, d.run_id, d.confidence, d.label, d.edge_touch,
                   ST_AsGeoJSON(ST_Transform(d.bbox_3857,4326)) AS geom,
                   ST_Y(d.centroid_4326) AS lat, ST_X(d.centroid_4326) AS lon,
                   d.created_at, it.z, it.x, it.y
            FROM detections d
            LEFT JOIN imagery_tiles it ON it.id = d.tile_id
            LEFT JOIN site_detections sd ON sd.detection_id = d.id
            {where_sql}
            ORDER BY d.created_at DESC
            LIMIT %s
        """
        params.append(limit)
        with db.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({
                "id": str(r[0]), "run_id": r[1], "confidence": float(r[2]) if r[2] is not None else None,
                "label": r[3], "edge_touch": bool(r[4]), "geom": json.loads(r[5]) if r[5] else None,
                "lat": r[6], "lon": r[7], "created_at": r[8].isoformat() if hasattr(r[8], 'isoformat') else r[8],
                "z": r[9], "x": r[10], "y": r[11],
            })
        return jsonify(out)
    finally:
        try:
            db.close()
        except Exception:
            pass


@app.route("/api/sites")
def api_sites():
    db, err = _db_client_or_error()
    if err:
        return err
    try:
        bbox = request.args.get("bbox")
        zcta = request.args.get("zcta")
        date_from = request.args.get("date_from")
        date_to = request.args.get("date_to")
        limit = int(request.args.get("limit") or 500)
        limit = max(1, min(limit, 5000))

        where = []
        params = []
        if zcta:
            where.append("zcta = %s")
            params.append(zcta)
        if date_from:
            where.append("last_seen >= %s")
            params.append(date_from)
        if date_to:
            where.append("first_seen <= %s")
            params.append(date_to)
        if bbox:
            try:
                west, south, east, north = [float(x) for x in bbox.split(',')]
                where.append("ST_Intersects(aabb_3857, ST_Transform(ST_MakeEnvelope(%s,%s,%s,%s,4326),3857))")
                params.extend([west, south, east, north])
            except Exception:
                pass
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
            SELECT id, zcta, detections_count, conf, first_seen, last_seen,
                   ST_AsGeoJSON(ST_Transform(aabb_3857,4326)) AS geom,
                   ST_Y(centroid_4326) AS lat, ST_X(centroid_4326) AS lon
            FROM sites
            {where_sql}
            ORDER BY last_seen DESC NULLS LAST
            LIMIT %s
        """
        params.append(limit)
        with db.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out = []
        for r in rows:
            out.append({
                "id": str(r[0]), "zcta": r[1], "detections_count": r[2], "conf": float(r[3]) if r[3] is not None else None,
                "first_seen": r[4].isoformat() if r[4] else None, "last_seen": r[5].isoformat() if r[5] else None,
                "geom": json.loads(r[6]) if r[6] else None, "lat": r[7], "lon": r[8],
            })
        return jsonify(out)
    finally:
        try:
            db.close()
        except Exception:
            pass


# -------- Annotations (global legacy) --------
@app.route("/annotations", methods=["GET"])
def annotations_get_global():
    image_path = request.args.get("path")
    if not image_path:
        return jsonify({"error": "path required"}), 400
    latest = _load_latest_annotation_for_path(ANNOTATIONS_FILE, image_path)
    return jsonify(latest)


@app.route("/annotations", methods=["POST"])
def annotations_post_global():
    payload = request.get_json(force=True) or {}
    image_path = payload.get("path")
    if not image_path:
        return jsonify({"error": "path required"}), 400
    # Minimal validation; ensure boxes is a list
    boxes = payload.get("boxes")
    if boxes is None or not isinstance(boxes, list):
        return jsonify({"error": "boxes list required"}), 400
    # Append with timestamp if missing
    payload.setdefault("created_at", time.time())
    _append_jsonl(ANNOTATIONS_FILE, payload)
    return jsonify({"status": "ok"})


# -------- Annotations (AOI-scoped) --------
@app.route("/areas/<area_id>/annotations", methods=["GET"])
def annotations_get_area(area_id: str):
    p = area_paths(area_id)
    image_path = request.args.get("path")
    if not image_path:
        return jsonify({"error": "path required"}), 400
    latest = _load_latest_annotation_for_path(p["annotations"], image_path)
    return jsonify(latest)


@app.route("/areas/<area_id>/annotations", methods=["POST"])
def annotations_post_area(area_id: str):
    p = area_paths(area_id)
    payload = request.get_json(force=True) or {}
    image_path = payload.get("path")
    if not image_path:
        return jsonify({"error": "path required"}), 400
    boxes = payload.get("boxes")
    if boxes is None or not isinstance(boxes, list):
        return jsonify({"error": "boxes list required"}), 400
    payload.setdefault("created_at", time.time())
    _append_jsonl(p["annotations"], payload)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
