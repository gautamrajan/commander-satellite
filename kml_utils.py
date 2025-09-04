import xml.etree.ElementTree as ET
from typing import List, Tuple

LatLon = Tuple[float, float]
Polygon = List[LatLon]


def _tag_name(tag: str) -> str:
    """Return the local tag name without namespace."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def _parse_coordinates(text: str) -> List[LatLon]:
    """Parse a KML <coordinates> text block into a list of (lat, lon) tuples.

    KML stores coordinates as lon,lat[,alt], separated by whitespace.
    Returns a list of (lat, lon) in that order for easier downstream usage.
    """
    pts: List[LatLon] = []
    if not text:
        return pts
    # Split by any whitespace and commas within tokens
    for token in text.strip().replace('\n', ' ').split():
        if not token:
            continue
        parts = token.split(',')
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue
        pts.append((lat, lon))
    return pts


def parse_kml_polygons(path: str) -> List[Polygon]:
    """Extract outer polygons from a KML file.

    - Supports Placemarks with Polygon and MultiGeometry/Polygon children.
    - Only outerBoundaryIs rings are considered (holes are ignored in v1).
    - Returns a list of polygons, each a list of (lat, lon) vertices.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    polygons: List[Polygon] = []

    def walk_for_polygons(elem: ET.Element) -> None:
        tn = _tag_name(elem.tag)
        if tn == 'Polygon':
            # Look for outerBoundaryIs/LinearRing/coordinates
            outer = None
            for child in elem.iter():
                if _tag_name(child.tag) == 'outerBoundaryIs':
                    outer = child
                    break
            if outer is not None:
                coords_text = None
                for child in outer.iter():
                    if _tag_name(child.tag) == 'coordinates':
                        coords_text = child.text
                        break
                pts = _parse_coordinates(coords_text or '')
                if len(pts) >= 3:
                    polygons.append(pts)
            else:
                # Fallback: any coordinates under this polygon
                coords_text = None
                for child in elem.iter():
                    if _tag_name(child.tag) == 'coordinates':
                        coords_text = child.text
                        break
                pts = _parse_coordinates(coords_text or '')
                if len(pts) >= 3:
                    polygons.append(pts)
            return
        # Recurse
        for c in list(elem):
            walk_for_polygons(c)

    walk_for_polygons(root)
    return polygons


def bbox_of_polygons(polys: List[Polygon]) -> Tuple[float, float, float, float]:
    """Return (lat_min, lon_min, lat_max, lon_max) across all polygons."""
    lat_min = float('inf'); lon_min = float('inf')
    lat_max = float('-inf'); lon_max = float('-inf')
    for poly in polys:
        for lat, lon in poly:
            if lat < lat_min: lat_min = lat
            if lat > lat_max: lat_max = lat
            if lon < lon_min: lon_min = lon
            if lon > lon_max: lon_max = lon
    if lat_min == float('inf'):
        raise ValueError('No coordinates in polygons')
    return lat_min, lon_min, lat_max, lon_max


def bbox_center(polys: List[Polygon]) -> LatLon:
    lat_min, lon_min, lat_max, lon_max = bbox_of_polygons(polys)
    return ( (lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0 )


def bbox_area_sqmi(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> float:
    """Approximate area in square miles for a lat/lon-aligned bounding box."""
    import math
    miles_per_deg_lat = 69.0
    # approximate longitudinal miles at mid-latitude
    mid_lat = (lat_min + lat_max) / 2.0
    miles_per_deg_lon = 69.0 * max(0.000001, math.cos(math.radians(mid_lat)))
    height_mi = abs(lat_max - lat_min) * miles_per_deg_lat
    width_mi = abs(lon_max - lon_min) * miles_per_deg_lon
    return max(0.0, height_mi * width_mi)


def point_in_polygon(lat: float, lon: float, poly: Polygon) -> bool:
    """Ray casting point-in-polygon on (lat, lon) vertices.

    Treat lon as X and lat as Y. Polygon is assumed to be closed or open; algorithm handles both.
    """
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        yi, xi = poly[i][0], poly[i][1]
        yj, xj = poly[j][0], poly[j][1]
        # Check edge (xi,yi)-(xj,yj)
        intersect = ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def any_polygon_contains(lat: float, lon: float, polys: List[Polygon]) -> bool:
    for poly in polys:
        if point_in_polygon(lat, lon, poly):
            return True
    return False

