-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- for gen_random_uuid()

-- AOI runs (normalized geometry in 3857)
CREATE TABLE IF NOT EXISTS aoi_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  mode text NOT NULL, -- 'zcta' | 'point_radius' | 'kml'
  params jsonb,
  geom_3857 geometry(MultiPolygon, 3857) NOT NULL,
  status text NOT NULL DEFAULT 'queued',
  stats jsonb
);

-- Fetched imagery tiles
CREATE TABLE IF NOT EXISTS imagery_tiles (
  id bigserial PRIMARY KEY,
  z int NOT NULL,
  x int NOT NULL,
  y int NOT NULL,
  path text,
  center_4326 geometry(Point, 4326),
  geom_3857 geometry(Polygon, 3857),
  UNIQUE (z, x, y)
);
CREATE INDEX IF NOT EXISTS imagery_tiles_geom_gix ON imagery_tiles USING GIST (geom_3857);

-- Model detections (construction only in this branch)
CREATE TABLE IF NOT EXISTS detections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid REFERENCES aoi_runs(id) ON DELETE SET NULL,
  tile_id bigint REFERENCES imagery_tiles(id) ON DELETE SET NULL,
  model text,
  model_ver text,
  confidence numeric(4,3),
  label text,
  bbox_px int[], -- [x0,y0,x1,y1] in pixel coords
  bbox_3857 geometry(Polygon, 3857) NOT NULL,
  centroid_4326 geometry(Point, 4326) NOT NULL,
  edge_touch boolean DEFAULT false,
  raw jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS detections_bbox_gix ON detections USING GIST (bbox_3857);

-- Persistent sites
CREATE TABLE IF NOT EXISTS sites (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  zcta text,
  aabb_3857 geometry(Polygon, 3857) NOT NULL,
  centroid_4326 geometry(Point, 4326) NOT NULL,
  first_seen date,
  last_seen date,
  status text,
  conf numeric(4,3),
  detections_count int NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS sites_geom_gix ON sites USING GIST (aabb_3857);
CREATE INDEX IF NOT EXISTS sites_zcta_idx ON sites (zcta);

-- Linking table
CREATE TABLE IF NOT EXISTS site_detections (
  site_id uuid REFERENCES sites(id) ON DELETE CASCADE,
  detection_id uuid REFERENCES detections(id) ON DELETE CASCADE,
  PRIMARY KEY (site_id, detection_id)
);

-- Optional ZCTA reference table (load externally; included here only as a placeholder)
-- CREATE TABLE IF NOT EXISTS zcta (
--   zcta5ce10 text PRIMARY KEY,
--   geom geometry(MultiPolygon, 4326),
--   geom_3857 geometry(MultiPolygon, 3857)
-- );
-- CREATE INDEX IF NOT EXISTS zcta_geom3857_gix ON zcta USING GIST (geom_3857);

