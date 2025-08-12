import json
import os
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

TILES_DIR = os.path.abspath("tiles")
ALL_RESULTS_FILE = "all_results.jsonl"
REVIEWED_RESULTS_FILE = "reviewed_results.jsonl"


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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detections")
def get_detections():
    results = load_results()
    reviewed_paths = load_reviewed_paths()
    
    positive_results = [
        r for r in results if r.get("positive") and r["path"] not in reviewed_paths
    ]
    
    return jsonify(positive_results)


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


if __name__ == "__main__":
    app.run(debug=True)
