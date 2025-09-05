from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import json

from .config import build_dsn


def _connect(dsn: str):
    try:
        import psycopg  # type: ignore

        return psycopg.connect(dsn)
    except Exception:
        import psycopg2  # type: ignore

        return psycopg2.connect(dsn)


class DbClient:
    def __init__(self, dsn: Optional[str] = None) -> None:
        self.dsn = dsn or os.getenv("POSTGRES_DSN") or build_dsn()
        self.conn = _connect(self.dsn)
        try:
            # enable autocommit for convenience
            self.conn.autocommit = True  # type: ignore[attr-defined]
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def upsert_imagery_tile(
        self,
        *,
        z: int,
        x: int,
        y: int,
        path: Optional[str],
        center_lon: Optional[float],
        center_lat: Optional[float],
        bbox_wkt_3857: Optional[str],
    ) -> int:
        sql = (
            "INSERT INTO imagery_tiles (z,x,y,path,center_4326,geom_3857)\n"
            "VALUES (%s,%s,%s,%s,\n"
            "        CASE WHEN %s::double precision IS NULL OR %s::double precision IS NULL THEN NULL ELSE ST_SetSRID(ST_MakePoint(%s::double precision,%s::double precision),4326) END,\n"
            "        CASE WHEN %s::text IS NULL THEN NULL ELSE ST_SetSRID(ST_GeomFromText(%s::text),3857) END)\n"
            "ON CONFLICT (z,x,y) DO UPDATE SET path = COALESCE(EXCLUDED.path, imagery_tiles.path)\n"
            "RETURNING id;"
        )
        params = (
            z,
            x,
            y,
            path,
            center_lon,
            center_lat,
            center_lon,
            center_lat,
            bbox_wkt_3857,
            bbox_wkt_3857,
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return int(row[0])

    @staticmethod
    def _multipolygon_wkt_from_latlon(polys: list[list[list[float]]]) -> str:
        """Build MULTIPOLYGON WKT from [[ [lat,lon], ... ], ...].

        WKT uses X Y order (lon lat). Ensures ring closure.
        """
        mp_parts: list[str] = []
        for poly in polys:
            if not poly:
                continue
            ring = [(float(pt[1]), float(pt[0])) for pt in poly]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            ring_txt = ", ".join(f"{lon} {lat}" for lon, lat in ring)
            mp_parts.append(f"(({ring_txt}))")
        if not mp_parts:
            raise ValueError("No polygons for MULTIPOLYGON")
        return f"MULTIPOLYGON({', '.join(mp_parts)})"

    def create_aoi_run(
        self,
        *,
        mode: str,
        params: Optional[dict],
        polygons_latlon: list[list[list[float]]],
    ) -> str:
        """Insert a new row into aoi_runs with given polygons (lat/lon) projected to 3857."""
        wkt4326 = self._multipolygon_wkt_from_latlon(polygons_latlon)
        sql = (
            "INSERT INTO aoi_runs (mode, params, geom_3857)\n"
            "VALUES (%s, %s::jsonb, ST_Transform(ST_SetSRID(ST_GeomFromText(%s),4326),3857))\n"
            "RETURNING id;"
        )
        params_json = json.dumps(params or {})
        with self.conn.cursor() as cur:
            cur.execute(sql, (mode, params_json, wkt4326))
            return str(cur.fetchone()[0])

    def insert_detection(
        self,
        *,
        run_id: Optional[str],
        tile_id: Optional[int],
        model: Optional[str],
        model_ver: Optional[str],
        confidence: Optional[float],
        label: Optional[str],
        bbox_px: Sequence[int],
        bbox_wkt_3857: str,
        centroid_lon: float,
        centroid_lat: float,
        edge_touch: bool,
        raw: Optional[Dict[str, Any]],
    ) -> str:
        sql = (
            "INSERT INTO detections (run_id, tile_id, model, model_ver, confidence, label, bbox_px, bbox_3857, centroid_4326, edge_touch, raw)\n"
            "VALUES (%s,%s,%s,%s,%s,%s,%s, ST_SetSRID(ST_GeomFromText(%s),3857), ST_SetSRID(ST_MakePoint(%s,%s),4326), %s, %s::jsonb)\n"
            "RETURNING id;"
        )
        params = (
            run_id,
            tile_id,
            model,
            model_ver,
            confidence,
            label,
            list(map(int, bbox_px)),
            bbox_wkt_3857,
            float(centroid_lon),
            float(centroid_lat),
            bool(edge_touch),
            json.dumps(raw) if raw is not None else None,
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return str(row[0])

    def promote_detection_to_site(
        self,
        *,
        detection_id: str,
        bbox_wkt_3857: str,
        conf: Optional[float] = None,
        iou_threshold: float = 0.3,
        area_threshold_m2: float = 600.0,
        assign_zcta: bool = True,
    ) -> str:
        """Link detection to an existing site if IoU>=threshold; else create a new site.

        Returns the site id.
        """
        # Find best overlapping site by IoU
        sql_select = (
            "WITH geom AS (SELECT ST_SetSRID(ST_GeomFromText(%s),3857) g)\n"
            "SELECT s.id,\n"
            "       COALESCE(\n"
            "         NULLIF(ST_Area(ST_Intersection(s.aabb_3857, g.g)),0) / NULLIF(ST_Area(ST_Union(s.aabb_3857, g.g)),0),\n"
            "         0\n"
            "       ) AS iou\n"
            "FROM sites s, geom g\n"
            "WHERE ST_Intersects(s.aabb_3857, g.g)\n"
            "ORDER BY iou DESC\n"
            "LIMIT 1;"
        )
        with self.conn.cursor() as cur:
            cur.execute(sql_select, (bbox_wkt_3857,))
            row = cur.fetchone()
            if row and float(row[1] or 0.0) >= iou_threshold:
                site_id = row[0]
                # Update existing site geometry and stats
                cur.execute(
                    (
                        "WITH geom AS (SELECT ST_SetSRID(ST_GeomFromText(%s),3857) g)\n"
                        "UPDATE sites SET\n"
                        "  aabb_3857 = ST_Envelope(ST_Union(aabb_3857, (SELECT g FROM geom))),\n"
                        "  last_seen = CURRENT_DATE,\n"
                        "  detections_count = detections_count + 1,\n"
                        "  conf = GREATEST(COALESCE(conf,0), COALESCE(%s,0))\n"
                        "WHERE id = %s\n"
                    ),
                    (bbox_wkt_3857, conf, site_id),
                )
            else:
                # Optionally gate new site creation by area threshold
                cur.execute(
                    (
                        "WITH geom AS (SELECT ST_SetSRID(ST_GeomFromText(%s),3857) g) SELECT ST_Area((SELECT g FROM geom));"
                    ),
                    (bbox_wkt_3857,),
                )
                area_m2 = float(cur.fetchone()[0] or 0.0)
                if area_m2 < float(area_threshold_m2):
                    site_id = None
                else:
                    cur.execute(
                        (
                            "WITH geom AS (SELECT ST_SetSRID(ST_GeomFromText(%s),3857) g)\n"
                            "INSERT INTO sites (aabb_3857, centroid_4326, first_seen, last_seen, status, conf, detections_count)\n"
                            "VALUES ( (SELECT g FROM geom), ST_Transform(ST_Centroid((SELECT g FROM geom)),4326), CURRENT_DATE, CURRENT_DATE, 'active', %s, 1)\n"
                            "RETURNING id;"
                        ),
                        (bbox_wkt_3857, conf),
                    )
                    site_id = cur.fetchone()[0]
                    if assign_zcta and site_id:
                        try:
                            # Assign ZCTA via centroid containment if table exists
                            cur.execute(
                                (
                                    "WITH s AS (SELECT id, ST_Centroid(aabb_3857) c FROM sites WHERE id=%s)\n"
                                    "UPDATE sites SET zcta = z.zcta5ce10 FROM zcta z, s\n"
                                    "WHERE sites.id = s.id AND ST_Contains(z.geom_3857, s.c);"
                                ),
                                (site_id,),
                            )
                        except Exception:
                            pass

            # Link detection if we have a site id
            if site_id:
                cur.execute(
                    "INSERT INTO site_detections (site_id, detection_id) VALUES (%s,%s) ON CONFLICT DO NOTHING;",
                    (site_id, detection_id),
                )
                return str(site_id)
            return ""
