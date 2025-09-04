# grab_imagery.py
import math, time, io, argparse, sys, os, json
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import numpy as np
from PIL import Image
from typing import List, Tuple

# Local utils
try:
    from kml_utils import (
        parse_kml_polygons,
        bbox_of_polygons,
        any_polygon_contains,
    )
except Exception:
    # Optional: geometry mode may import lazily
    parse_kml_polygons = None
    bbox_of_polygons = None
    any_polygon_contains = None

# Friendlier host for public basemap pulls
IMAGERY_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# ---------- tile math ----------
def latlon_to_tile(lat, lon, z):
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def tile_to_latlon(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y/n))))
    return lat, lon

def lonlat_to_merc(lon, lat):
    R = 6378137.0
    x = math.radians(lon) * R
    y = math.log(math.tan(math.pi/4 + math.radians(lat)/2)) * R
    return x, y

def pixel_size_m(lat_deg, z):
    R = 6378137.0
    return math.cos(math.radians(lat_deg)) * 2 * math.pi * R / (256 * (2**z))

# ---------- bbox helpers ----------
def square_bbox_from_area(lat, lon, area_sqmi):
    L = math.sqrt(area_sqmi)  # miles (side length)
    h = L / 2.0
    miles_per_deg_lat = 69.0
    miles_per_deg_lon = 69.0 * max(0.000001, math.cos(math.radians(lat)))
    dlat = h / miles_per_deg_lat
    dlon = h / miles_per_deg_lon
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon

# ---------- HTTP ----------
def fetch_tile_bytes(z, x, y, timeout=15):
    url = IMAGERY_URL.format(z=z, y=y, x=x)
    req = Request(url, headers={
        "User-Agent": "tile-grabber/1.0",
        "Referer": "https://www.arcgis.com"
    })
    with urlopen(req, timeout=timeout) as r:
        return r.read()

# ---------- placeholder detection ----------
def looks_like_placeholder(image_rgb: Image.Image) -> bool:
    """Heuristic for Esri 'no data' tiles:
       - mostly flat gray (#C9C9C9~#D0D0D0), maybe tiny white text/dashes
       Decide 'placeholder' if >95% pixels within 12 gray value of 200 and near-neutral.
    """
    arr = np.asarray(image_rgb, dtype=np.uint8)
    # how close to gray?
    rg = np.abs(arr[:,:,0].astype(int) - arr[:,:,1].astype(int))
    gb = np.abs(arr[:,:,1].astype(int) - arr[:,:,2].astype(int))
    gray_like = (rg < 6) & (gb < 6)

    # close to the typical background gray ~200
    mean_gray = arr.mean(axis=2)
    near_200 = np.abs(mean_gray - 200) < 12

    mask = gray_like & near_200
    frac = mask.mean()
    return frac > 0.95

def tile_has_real_imagery(z, x, y) -> bool:
    try:
        data = fetch_tile_bytes(z, x, y)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # Reject uniform or near-uniform gray placeholders
        if looks_like_placeholder(img):
            return False
        # also reject truly uniform tiles
        extrema = img.getextrema()
        if all(lo == hi for (lo, hi) in extrema):
            return False
        return True
    except (HTTPError, URLError, OSError):
        return False

# ---------- zoom selection ----------
def probe_highest_zoom_with_imagery(lat, lon, max_zoom=23, min_zoom=10):
    for z in range(max_zoom, min_zoom - 1, -1):
        x, y = latlon_to_tile(lat, lon, z)
        if tile_has_real_imagery(z, x, y):
            return z
    return min_zoom

def cap_zoom_by_edge(lat_center, meters_edge, z, max_edge_px):
    while z > 0:
        ps = pixel_size_m(lat_center, z)
        if meters_edge / ps <= max_edge_px:
            return z
        z -= 1
    return z

# ---------- writer ----------
def build_mosaic(lat_min, lon_min, lat_max, lon_max, z, out_tiff, out_png=None,
                 delay=0.03, max_tiles=None, save_tiles_dir=None):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    from PIL import Image as PILImage

    # tile ranges
    x_min, y_max = latlon_to_tile(lat_min, lon_min, z)  # SW
    x_max, y_min = latlon_to_tile(lat_max, lon_max, z)  # NE

    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    width = cols * 256
    height = rows * 256
    total_tiles = cols * rows
    if max_tiles and total_tiles > max_tiles:
        raise SystemExit(f"Refusing to fetch {total_tiles} tiles (> max {max_tiles}). Reduce area or zoom.")

    # georeference
    tl_lat, tl_lon = tile_to_latlon(x_min, y_min, z)
    br_lat, br_lon = tile_to_latlon(x_max + 1, y_max + 1, z)
    x0_m, y1_m = lonlat_to_merc(tl_lon, tl_lat)   # top-left
    x1_m, y0_m = lonlat_to_merc(br_lon, br_lat)   # bottom-right
    resx = (x1_m - x0_m) / width
    resy = (y1_m - y0_m) / height
    transform = from_origin(x0_m, y1_m, resx, resy)

    profile = dict(
        driver="GTiff",
        height=height, width=width, count=3,
        dtype="uint8",
        crs=CRS.from_epsg(3857),
        transform=transform,
        compress="deflate",
        tiled=True, blockxsize=256, blockysize=256
    )

    print(f"Tiles: {cols} x {rows} = {total_tiles} | size: {width} x {height}px | z={z}")
    k = 0
    with rasterio.open(out_tiff, "w", **profile) as dst:
        for j, y in enumerate(range(y_min, y_max + 1)):
            for i, x in enumerate(range(x_min, x_max + 1)):
                try:
                    data = fetch_tile_bytes(z, x, y)
                    img = PILImage.open(io.BytesIO(data)).convert("RGB")
                    if looks_like_placeholder(img):
                        # fall back one level for just this tile if possible
                        substituted = False
                        if z > 0:
                            try:
                                x2, y2 = x//2, y//2
                                data2 = fetch_tile_bytes(z-1, x2, y2)
                                img2 = PILImage.open(io.BytesIO(data2)).convert("RGB").resize((256,256), PILImage.BILINEAR)
                                img = img2
                                substituted = True
                            except Exception:
                                pass
                        if not substituted:
                            img = PILImage.new("RGB", (256,256), (255,255,255))
                except Exception:
                    img = PILImage.new("RGB", (256,256), (255,255,255))

                # optionally persist tile to disk in z/x/y.jpg structure
                if save_tiles_dir:
                    try:
                        tile_dir = os.path.join(save_tiles_dir, str(z), str(x))
                        os.makedirs(tile_dir, exist_ok=True)
                        tile_path = os.path.join(tile_dir, f"{y}.jpg")
                        img.save(tile_path, format="JPEG")
                    except Exception:
                        # do not fail mosaic writing if tile save fails
                        pass

                arr = np.asarray(img, dtype=np.uint8)
                window = rasterio.windows.Window(i*256, j*256, 256, 256)
                for b in range(3):
                    dst.write(arr[:,:,b], b+1, window=window)

                k += 1
                if k % 100 == 0:
                    print(f"{k}/{total_tiles} tiles")
                time.sleep(delay)

    print(f"Saved GeoTIFF: {out_tiff}")

    if out_png:
        # create downsampled preview (~4k longest edge)
        with rasterio.open(out_tiff) as src:
            longest = max(src.width, src.height)
            scale = longest / 4000 if longest > 4000 else 1.0
            out_w = int(src.width / scale); out_h = int(src.height / scale)
            data = src.read(out_shape=(3, out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
            from PIL import Image as PILImage
            PILImage.fromarray(data.transpose(1,2,0)).save(out_png, quality=92)
        print(f"Saved preview PNG: {out_png}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Fetch Esri World_Imagery tiles around a point or a KML/geometry polygon.")
    # Point + bbox mode (legacy)
    ap.add_argument("--lat", type=float, required=False)
    ap.add_argument("--lon", type=float, required=False)
    ap.add_argument("--area_sqmi", type=float, required=False, help="Square miles around the point (square bbox).")
    # Geometry/KML mode
    ap.add_argument("--geometry_json", type=str, default=None, help="Path to JSON containing polygon list.")
    ap.add_argument("--kml", type=str, default=None, help="Path to KML file defining polygon(s).")
    ap.add_argument("--auto_zoom", action="store_true", help="Probe highest available zoom at the point.")
    ap.add_argument("--max_zoom", type=int, default=23)
    ap.add_argument("--min_zoom", type=int, default=10)
    ap.add_argument("--zoom", type=int, default=None, help="Override zoom (skips probing).")
    ap.add_argument("--max_edge_px", type=int, default=12000, help="Cap longest image edge; auto lowers zoom if needed.")
    ap.add_argument("--max_tiles", type=int, default=20000, help="Safety cap on number of tiles (default ~20000).")
    ap.add_argument("--tiff", type=str, default=None, help="Output GeoTIFF path (required unless --no_mosaic).")
    ap.add_argument("--png", type=str, default=None, help="Optional preview PNG path (ignored if --no_mosaic).")
    ap.add_argument("--no_mosaic", action="store_true", help="Skip GeoTIFF/PNG outputs; only save z/x/y tiles.")
    ap.add_argument("--delay", type=float, default=0.03, help="Delay between tile requests.")
    ap.add_argument("--save_tiles_dir", type=str, default=None, help="Optional directory to save tiles as z/x/y.jpg.")
    args = ap.parse_args()

    def tile_center_latlon(x: int, y: int, z: int) -> Tuple[float, float]:
        lat0, lon0 = tile_to_latlon(x, y, z)
        lat1, lon1 = tile_to_latlon(x + 1, y + 1, z)
        return ((lat0 + lat1) / 2.0, (lon0 + lon1) / 2.0)

    # Determine mode
    geometry_mode = bool(args.geometry_json or args.kml)

    if not geometry_mode:
        # Validate required inputs for bbox mode
        if args.lat is None or args.lon is None or args.area_sqmi is None:
            raise SystemExit("--lat, --lon, and --area_sqmi are required when not using --geometry_json/--kml")
        if args.no_mosaic and not args.save_tiles_dir:
            raise SystemExit("--no_mosaic requires --save_tiles_dir to persist tiles")

        lat_min, lon_min, lat_max, lon_max = square_bbox_from_area(args.lat, args.lon, args.area_sqmi)
        print(f"BBox: {lat_min:.6f},{lon_min:.6f} to {lat_max:.6f},{lon_max:.6f}")

        # choose zoom
        if args.zoom is not None:
            z = args.zoom
            print(f"Using fixed zoom: {z}")
        else:
            z_probe = probe_highest_zoom_with_imagery(args.lat, args.lon, args.max_zoom, args.min_zoom) if args.auto_zoom else 19
            # keep within edge cap
            R = 6371000.0
            meters_edge = math.radians(lat_max - lat_min) * R  # N-S edge
            z = cap_zoom_by_edge(args.lat, meters_edge, z_probe, args.max_edge_px)
            if z < z_probe:
                print(f"Zoom {z_probe} -> {z} to keep edge ≤ {args.max_edge_px}px")
            else:
                print(f"Using zoom: {z}")

        if args.no_mosaic:
            # Approximate grid, then fetch all tiles in bbox (no filtering)
            x_min, y_max = latlon_to_tile(lat_min, lon_min, z)
            x_max, y_min = latlon_to_tile(lat_max, lon_max, z)
            cols = x_max - x_min + 1
            rows = y_max - y_min + 1
            total_tiles = cols * rows
            if args.max_tiles and total_tiles > args.max_tiles:
                raise SystemExit(f"Refusing to fetch {total_tiles} tiles (> max {args.max_tiles}). Reduce area or zoom.")
            print(f"Fetching {total_tiles} tiles")
            k = 0
            for j, y in enumerate(range(y_min, y_max + 1)):
                for i, x in enumerate(range(x_min, x_max + 1)):
                    try:
                        data = fetch_tile_bytes(z, x, y)
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        if looks_like_placeholder(img):
                            # try parent
                            substituted = False
                            if z > 0:
                                try:
                                    x2, y2 = x//2, y//2
                                    data2 = fetch_tile_bytes(z-1, x2, y2)
                                    img2 = Image.open(io.BytesIO(data2)).convert("RGB").resize((256,256), Image.BILINEAR)
                                    img = img2
                                    substituted = True
                                except Exception:
                                    pass
                            if not substituted:
                                img = Image.new("RGB", (256,256), (255,255,255))
                    except Exception:
                        img = Image.new("RGB", (256,256), (255,255,255))

                    if args.save_tiles_dir:
                        try:
                            tile_dir = os.path.join(args.save_tiles_dir, str(z), str(x))
                            os.makedirs(tile_dir, exist_ok=True)
                            tile_path = os.path.join(tile_dir, f"{y}.jpg")
                            img.save(tile_path, format="JPEG")
                        except Exception:
                            pass
                    k += 1
                    if k % 100 == 0:
                        print(f"{k}/{total_tiles} tiles")
                    time.sleep(args.delay)
            return
        else:
            # Build mosaic + optionally save tiles
            if not args.tiff:
                raise SystemExit("--tiff is required unless --no_mosaic is set")
            build_mosaic(lat_min, lon_min, lat_max, lon_max, z,
                         out_tiff=args.tiff, out_png=args.png,
                         delay=args.delay, max_tiles=args.max_tiles,
                         save_tiles_dir=args.save_tiles_dir)
            return

    # Geometry mode
    if geometry_mode:
        if args.no_mosaic and not args.save_tiles_dir:
            raise SystemExit("--no_mosaic requires --save_tiles_dir to persist tiles")
        polys: List[List[Tuple[float, float]]] = []
        if args.geometry_json:
            with open(args.geometry_json, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            if isinstance(obj, dict) and 'polygons' in obj and isinstance(obj['polygons'], list):
                polys = obj['polygons']
            elif isinstance(obj, list):
                polys = obj
            else:
                raise SystemExit("geometry_json must be a list of polygons or {'polygons': [...]} format")
        elif args.kml:
            if parse_kml_polygons is None:
                raise SystemExit("KML parsing utilities not available")
            polys = parse_kml_polygons(args.kml)

        if not polys:
            raise SystemExit("No polygons found for geometry mode")

        # Determine bbox and zoom
        lat_min, lon_min, lat_max, lon_max = bbox_of_polygons(polys)
        print(f"BBox: {lat_min:.6f},{lon_min:.6f} to {lat_max:.6f},{lon_max:.6f}")
        if args.zoom is not None:
            z = args.zoom
            print(f"Using fixed zoom: {z}")
        else:
            # Use provided lat/lon for probing if available, else bbox center
            probe_lat = args.lat if args.lat is not None else (lat_min + lat_max) / 2.0
            probe_lon = args.lon if args.lon is not None else (lon_min + lon_max) / 2.0
            z_probe = probe_highest_zoom_with_imagery(probe_lat, probe_lon, args.max_zoom, args.min_zoom) if args.auto_zoom else 19
            # Cap zoom only matters for mosaic, but keep consistent behavior
            R = 6371000.0
            meters_edge = math.radians(lat_max - lat_min) * R
            z = cap_zoom_by_edge(probe_lat, meters_edge, z_probe, args.max_edge_px)
            if z < z_probe:
                print(f"Zoom {z_probe} -> {z} to keep edge ≤ {args.max_edge_px}px")
            else:
                print(f"Using zoom: {z}")

        # Compute tile candidates within bbox
        x_min, y_max = latlon_to_tile(lat_min, lon_min, z)  # SW
        x_max, y_min = latlon_to_tile(lat_max, lon_max, z)  # NE
        # Preselect tiles whose centers lie inside any polygon
        selected: List[Tuple[int, int]] = []
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                c_lat, c_lon = tile_center_latlon(x, y, z)
                if any_polygon_contains(c_lat, c_lon, polys):
                    selected.append((x, y))

        total_tiles = len(selected)
        if args.max_tiles and total_tiles > args.max_tiles:
            raise SystemExit(f"Refusing to fetch {total_tiles} tiles (> max {args.max_tiles}). Reduce area or zoom.")
        print(f"Fetching {total_tiles} tiles")

        # Optionally build mosaic: treat bbox mosaic as in legacy, but many users only need tiles
        if not args.no_mosaic and args.tiff:
            # Build mosaic of bbox to preserve existing behavior, while still saving only selected tiles
            # For simplicity and to avoid masking, we keep the bbox mosaic identical to legacy.
            build_mosaic(lat_min, lon_min, lat_max, lon_max, z,
                         out_tiff=args.tiff, out_png=args.png,
                         delay=0.0, max_tiles=None,  # we already enforce caps
                         save_tiles_dir=None)
            # Then fetch just selected tiles to save_tiles_dir if requested
        
        # Fetch and save selected tiles
        k = 0
        for (x, y) in selected:
            try:
                data = fetch_tile_bytes(z, x, y)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                if looks_like_placeholder(img):
                    substituted = False
                    if z > 0:
                        try:
                            x2, y2 = x//2, y//2
                            data2 = fetch_tile_bytes(z-1, x2, y2)
                            img2 = Image.open(io.BytesIO(data2)).convert("RGB").resize((256,256), Image.BILINEAR)
                            img = img2
                            substituted = True
                        except Exception:
                            pass
                    if not substituted:
                        img = Image.new("RGB", (256,256), (255,255,255))
            except Exception:
                img = Image.new("RGB", (256,256), (255,255,255))

            if args.save_tiles_dir:
                try:
                    tile_dir = os.path.join(args.save_tiles_dir, str(z), str(x))
                    os.makedirs(tile_dir, exist_ok=True)
                    tile_path = os.path.join(tile_dir, f"{y}.jpg")
                    img.save(tile_path, format="JPEG")
                except Exception:
                    pass
            k += 1
            if k % 100 == 0:
                print(f"{k}/{total_tiles} tiles")
            time.sleep(args.delay)

        return

if __name__ == "__main__":
    main()
