## Annotations: Bounding Boxes, Rotation, and Review Workflow

This document explains the annotation feature added to the review app: how data is stored, the HTTP endpoints, and how to use the UI to draw, move, rotate, and resize oriented bounding boxes with labels. It is compatible with both area-scoped (AOI) runs and the legacy global mode.

### Summary

- New storage: `annotations.jsonl` written append-only.
  - AOI: `runs/<AREA_ID>/annotations.jsonl`
  - Legacy: `annotations.jsonl` at repo root
- New endpoints:
  - `GET  /areas/<area_id>/annotations?path=z/x/y.jpg`
  - `POST /areas/<area_id>/annotations`
  - `GET  /annotations?path=z/x/y.jpg` (legacy)
  - `POST /annotations` (legacy)
- UI: Annotate in Review and Mosaic tabs. Draw boxes, move by drag, rotate with top handle or angle input, resize via corner handles, label per box, save.

---

### Data model (JSONL)

Each line is a JSON object containing the latest snapshot of annotations for a single tile path. The latest line for a given `path` is the effective state. Lines are appended only; no in-place edits.

Required fields written by the UI:

```json
{
  "path": "22/721523/418914.jpg",
  "z": 22,
  "x": 721523,
  "y": 418914,
  "context_radius": 1,
  "canvas_size": { "width": 768, "height": 768 },
  "boxes": [
    {
      "cx": 510.2,
      "cy": 292.5,
      "w": 140.0,
      "h": 60.0,
      "angle_deg": 32.0,
      "ncx": 0.664,
      "ncy": 0.381,
      "nw": 0.182,
      "nh": 0.078,
      "x": 450.9,
      "y": 263.1,
      "w_aabb": 146.8,
      "h_aabb": 79.0,
      "nx": 0.587,
      "ny": 0.343,
      "nw_aabb": 0.191,
      "nh_aabb": 0.103,
      "poly": [
        [0.620, 0.315],
        [0.806, 0.352],
        [0.708, 0.447],
        [0.522, 0.409]
      ],
      "label": "dumpster"
    }
  ],
  "created_at": 1734542047.123
}
```

Notes:

- Oriented box representation:
  - Center-based with angle: `cx, cy, w, h, angle_deg`.
  - Normalized center and size: `ncx, ncy, nw, nh` relative to `canvas_size`.
  - Normalized polygon (`poly`) lists four corners clockwise in normalized canvas space.
- AABB compatibility:
  - Axis-aligned bounding box is included for downstream consumers: `x, y, w_aabb, h_aabb` and normalized `nx, ny, nw_aabb, nh_aabb`.
- Backward-compatibility:
  - If legacy lines have `x,y,w,h` or `nx,ny,nw,nh`, the UI maps them to an oriented box with `angle_deg = 0` when loading.

Storage guidance:

- Treat `annotations.jsonl` as an append-only log. The latest occurrence of a `path` is the authoritative snapshot.
- Large outputs (including `*.jsonl`) should not be committed to git.

---

### HTTP API

Preferred (AOI-scoped):

```http
GET /areas/<area_id>/annotations?path=z/x/y.jpg
```

Response:

```json
{ "path": "z/x/y.jpg", "boxes": [ ... ] }
```

```http
POST /areas/<area_id>/annotations
Content-Type: application/json

{
  "path": "z/x/y.jpg",
  "z": 22, "x": 721523, "y": 418914,
  "context_radius": 1,
  "canvas_size": {"width": 768, "height": 768},
  "boxes": [ { ... see schema above ... } ]
}
```

Legacy global (same payload/shape):

```http
GET /annotations?path=z/x/y.jpg
POST /annotations
```

Behavior:

- `GET` returns the latest line for the specified `path`, or `{ path, boxes: [] }` if none.
- `POST` appends the provided object as one JSONL line. The server injects `created_at` if missing.

---

### UI/UX – How to annotate

Where to open:

- Review tab: `Annotate` button under Approve/Reject.
- Mosaic tab: select a tile; click `Annotate` in the sidebar.

Canvas & context:

- Context radius selector: 0 (center only), 1 (3x3), 2 (5x5). The background stitches neighboring tiles; the central tile is outlined in yellow.
- Grid lines indicate tile boundaries.

Drawing and editing:

- Draw new box: click-drag on empty canvas.
- Move: drag inside a selected box.
- Rotate: drag the cyan circular handle above the selected box; or set the angle in the boxes list.
- Resize: drag any of the four corner square handles (opposite corner stays fixed; angle preserved).
- Label: edit in the list (each box has its own label).
- Save: persists to `annotations.jsonl` (AOI or global, depending on where you started). Latest line wins.
- Close: dismisses without saving.

Interaction priorities (so clicks do what you expect):

1) Rotate-handle hit
2) Corner resize handle hit
3) Box body (move)
4) Else draw new box

Implementation details for stability:

- Background tiles are drawn to an offscreen canvas and then blitted to avoid flicker and ensure boxes/labels are always painted over the imagery.
- `.jpg` first, automatic `.png` fallback if needed.
- Event guards prevent duplicate box creation; we only respond to left-click.

---

### Backend changes

- `review_app.py`
  - `area_paths()` now includes an `annotations` path.
  - New routes: `GET/POST /areas/<area_id>/annotations` and global `GET/POST /annotations`.
  - Utility: `_append_jsonl()` and `_load_latest_annotation_for_path()`.
  - Mosaic tile JSON includes `z` per tile (used by the UI for correct context image URLs).

---

### Export considerations

- Use `poly` for precise geometry (normalized polygon). For YOLO/COCO exporters, prefer the oriented representation (`cx, cy, w, h, angle_deg`) and/or polygon; fall back to AABB if needed.
- Because the canvas may include context tiles, annotations are defined in stitched-canvas pixel space; normalized values are included to remain robust to different canvas sizes.

---

### Future enhancements (optional)

- Canvas rotation control (global view rotate) in addition to per-box rotation.
- Keyboard shortcuts (e.g., Q/E ±5°, Shift for fine-tuning).
- Multi-select and batch operations.
- Server-side index to accelerate `GET` of latest annotations for very large logs.

---

### Quick API examples

```bash
# Fetch latest for a tile in an AOI
curl "http://localhost:5000/areas/<AREA_ID>/annotations?path=22/721523/418914.jpg"

# Save annotations for a tile in an AOI
curl -X POST "http://localhost:5000/areas/<AREA_ID>/annotations" \
  -H 'Content-Type: application/json' \
  -d '{
    "path":"22/721523/418914.jpg",
    "z":22,"x":721523,"y":418914,
    "context_radius":1,
    "canvas_size":{"width":768,"height":768},
    "boxes":[{"cx":510.2,"cy":292.5,"w":140,"h":60,"angle_deg":32,"label":"dumpster"}]
  }'
```


