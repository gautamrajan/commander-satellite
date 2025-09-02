import json
import os
import sys
import time
import threading
import subprocess
import shutil
from datetime import datetime
from flask import Flask, jsonify, render_template, request, send_from_directory

# Import detection type configuration
from detection_types import get_available_detection_types, get_detection_type, get_output_filename

app = Flask(__name__)

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
    with open(ALL_RESULTS_FILE, "r") as f:
        for line in f:
            results.append(json.loads(line))
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
    },
}
SCAN_LOCK = threading.Lock()


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
				"image_url": f"/image/{rel_path.replace('\\', '/')}",
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
				"image_url": f"/areas/{area_id}/image/{rel_path.replace('\\', '/')}",
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

    center = area_obj.get("center")
    area_sqmi = area_obj.get("area_sqmi") or 0.25
    zoom = area_obj.get("zoom")

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

    os.makedirs(p["logs"], exist_ok=True)

    with AREA_LOCK:
        jobs = AREA_JOBS.setdefault(area_id, {"fetch": None, "scan": None})
        if proc_is_running(jobs.get("scan")):
            return jsonify({"status": "error", "message": "scan already running"}), 409
        out_fp = open(p["scan_stdout"], "a", encoding="utf-8")
        err_fp = open(p["scan_stderr"], "a", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=out_fp, stderr=err_fp)
        jobs["scan"] = proc

    return jsonify({"status": "started", "pid": proc.pid, "cmd": cmd})


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

    # Prepare stdout/stderr logs (area-scoped if provided)

    with SCAN_LOCK:
        if proc_is_running(SCAN_STATE.get("proc")):
            return jsonify({"status": "error", "message": "Scan already running"}), 409
        SCAN_STATE["files"]["all_results"] = all_results
        SCAN_STATE["files"]["out"] = out_file
        SCAN_STATE["files"]["coarse"] = coarse_log
        SCAN_STATE["files"]["stdout"] = stdout_log
        SCAN_STATE["files"]["stderr"] = stderr_log
        SCAN_STATE["baseline"]["all_results"] = safe_count_lines(all_results)
        SCAN_STATE["baseline"]["out"] = safe_count_lines(out_file)
        SCAN_STATE["baseline"]["coarse"] = safe_count_lines(coarse_log) if coarse_log else 0
        SCAN_STATE["started_at"] = time.time()
        # Persist args including area_id so status/progress reflects this run
        SCAN_STATE["args"] = {**payload, "area_id": area_id, "tiles_dir": tiles_dir}

        stdout_fp = open(stdout_log, "a", encoding="utf-8")
        stderr_fp = open(stderr_log, "a", encoding="utf-8")
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
