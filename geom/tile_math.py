import math
from typing import Tuple, List

R_M = 6378137.0  # Web Mercator sphere radius (meters)
TILE_SIZE = 256


def _map_size(z: int) -> int:
    return TILE_SIZE * (1 << z)


def pixel_to_lonlat(z: int, x: int, y: int, px: float, py: float) -> Tuple[float, float]:
    """Convert a pixel (px,py) within tile z/x/y (0..255) to lon/lat (EPSG:4326)."""
    n = 1 << z
    map_size = _map_size(z)
    gx = x * TILE_SIZE + px
    gy = y * TILE_SIZE + py
    lon = gx / map_size * 360.0 - 180.0
    # inverse Mercator
    y_norm = 1 - 2 * (gy / map_size)
    lat_rad = math.atan(math.sinh(math.pi * y_norm))
    lat = math.degrees(lat_rad)
    return lon, lat


def lonlat_to_merc(lon: float, lat: float) -> Tuple[float, float]:
    x = math.radians(lon) * R_M
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R_M
    return x, y


def bbox_px_to_merc_polygon(
    z: int,
    x: int,
    y: int,
    box_px: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    """Convert a tile-local pixel box [x0,y0,x1,y1] into a 3857 polygon ring (closed)."""
    x0, y0, x1, y1 = box_px
    # four corners: TL, TR, BR, BL
    corners_px = [
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
    ]
    ring: List[Tuple[float, float]] = []
    for (px, py) in corners_px:
        lon, lat = pixel_to_lonlat(z, x, y, px, py)
        xm, ym = lonlat_to_merc(lon, lat)
        ring.append((xm, ym))
    # close ring
    ring.append(ring[0])
    return ring


def ring_to_wkt(ring_xy: List[Tuple[float, float]]) -> str:
    """Return POLYGON WKT string for a single ring in 3857."""
    coords = ", ".join(f"{x} {y}" for x, y in ring_xy)
    return f"POLYGON(({coords}))"


def bbox_center_lonlat(
    z: int, x: int, y: int, box_px: Tuple[float, float, float, float]
) -> Tuple[float, float]:
    x0, y0, x1, y1 = box_px
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    lon, lat = pixel_to_lonlat(z, x, y, cx, cy)
    return lon, lat

