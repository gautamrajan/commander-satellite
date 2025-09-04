import os
import sys
import time
import json
import math
import argparse
from typing import Optional, Dict, Any, Tuple
from urllib.request import urlopen, Request
from urllib.parse import urlencode


def reverse_geocode_arcgis(lat: float, lon: float, timeout: float = 10.0) -> Optional[str]:
    base = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/reverseGeocode"
    qs = urlencode({
        "f": "pjson",
        "location": f"{lon},{lat}",
        "outSR": 4326,
        "langCode": "en",
    })
    url = f"{base}?{qs}"
    req = Request(url, headers={"User-Agent": "sat-data-fetcher/1.0"})
    with urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="ignore"))
    addr = None
    try:
        addr = data.get("address", {}).get("Match_addr")
    except Exception:
        addr = None
    return addr


def reverse_geocode_nominatim(lat: float, lon: float, timeout: float = 10.0) -> Optional[str]:
    base = "https://nominatim.openstreetmap.org/reverse"
    qs = urlencode({
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "zoom": 18,
        "addressdetails": 1,
    })
    url = f"{base}?{qs}"
    # Provide a descriptive UA per usage policy
    req = Request(url, headers={"User-Agent": "sat-data-fetcher/1.0 (contact: example@example.com)"})
    with urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="ignore"))
    return data.get("display_name")


def reverse_geocode_google(lat: float, lon: float, api_key: str, timeout: float = 10.0) -> Optional[str]:
    base = "https://maps.googleapis.com/maps/api/geocode/json"
    qs = urlencode({
        "latlng": f"{lat},{lon}",
        "key": api_key,
    })
    url = f"{base}?{qs}"
    req = Request(url, headers={"User-Agent": "sat-data-fetcher/1.0"})
    with urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="ignore"))
    if data.get("status") == "OK" and data.get("results"):
        return data["results"][0].get("formatted_address")
    return None


def reverse_geocode_mapbox(lat: float, lon: float, token: str, timeout: float = 10.0) -> Optional[str]:
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json?language=en&access_token={token}"
    req = Request(url, headers={"User-Agent": "sat-data-fetcher/1.0"})
    with urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="ignore"))
    feats = data.get("features") or []
    if not feats:
        return None
    return feats[0].get("place_name")


def main():
    ap = argparse.ArgumentParser(description="Reverse geocode positive detections JSONL to addresses.")
    ap.add_argument("--in", dest="in_path", type=str, required=True, help="Input positives JSONL (e.g., dumpsters.jsonl)")
    ap.add_argument("--out", dest="out_path", type=str, required=True, help="Output JSONL with addresses")
    ap.add_argument("--provider", type=str, default="arcgis", choices=["arcgis", "nominatim", "google", "mapbox"], help="Geocoder provider")
    ap.add_argument("--rpm", type=float, default=1.0, help="Requests per minute cap (respect provider limits)")
    ap.add_argument("--cache", type=str, default="geocode_cache.json", help="Path to cache JSON (dict keyed by 'z/x/y')")
    args = ap.parse_args()

    # Load cache
    cache: Dict[str, Any] = {}
    if os.path.exists(args.cache):
        try:
            with open(args.cache, "r", encoding="utf-8") as cf:
                cache = json.load(cf)
        except Exception:
            cache = {}

    # Provider setup
    provider = args.provider
    google_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    mapbox_token = os.environ.get("MAPBOX_ACCESS_TOKEN")

    # Rate limit
    min_interval = 60.0 / args.rpm if args.rpm and args.rpm > 0 else 0.0
    last_call = 0.0

    def geocode(lat: float, lon: float) -> Optional[str]:
        nonlocal last_call
        now = time.time()
        if min_interval > 0 and (now - last_call) < min_interval:
            time.sleep(min_interval - (now - last_call))
        try:
            if provider == "arcgis":
                addr = reverse_geocode_arcgis(lat, lon)
            elif provider == "nominatim":
                addr = reverse_geocode_nominatim(lat, lon)
            elif provider == "google":
                if not google_key:
                    raise SystemExit("GOOGLE_MAPS_API_KEY not set in environment")
                addr = reverse_geocode_google(lat, lon, google_key)
            elif provider == "mapbox":
                if not mapbox_token:
                    raise SystemExit("MAPBOX_ACCESS_TOKEN not set in environment")
                addr = reverse_geocode_mapbox(lat, lon, mapbox_token)
            else:
                addr = None
        finally:
            last_call = time.time()
        return addr

    # Process input
    count_in = 0
    count_out = 0
    with open(args.in_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            count_in += 1
            z, x, y = obj.get("z"), obj.get("x"), obj.get("y")
            lat, lon = obj.get("lat"), obj.get("lon")
            if (lat is None or lon is None) and None not in (z, x, y):
                # compute center for back-compat if lat/lon missing
                n = 2.0 ** z
                # top-left and bottom-right, then average
                lon0 = x / n * 360.0 - 180.0
                lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y/n))))
                lon1 = (x + 1) / n * 360.0 - 180.0
                lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*(y+1)/n))))
                lat = (lat0 + lat1) / 2.0
                lon = (lon0 + lon1) / 2.0
            key = f"{z}/{x}/{y}"
            addr = cache.get(key)
            if not addr and lat is not None and lon is not None:
                try:
                    addr = geocode(float(lat), float(lon))
                except Exception:
                    addr = None
                if addr:
                    cache[key] = addr
            obj_out = dict(obj)
            if lat is not None and lon is not None:
                obj_out["lat"] = float(lat)
                obj_out["lon"] = float(lon)
            if addr:
                obj_out["address"] = addr
            fout.write(json.dumps(obj_out) + "\n")
            count_out += 1

    # Save cache
    try:
        with open(args.cache, "w", encoding="utf-8") as cf:
            json.dump(cache, cf)
    except Exception:
        pass

    print(f"Processed {count_in} → wrote {count_out}. Cache size: {len(cache)}")


if __name__ == "__main__":
    main()

