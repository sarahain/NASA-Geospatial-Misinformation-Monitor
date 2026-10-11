-- PostGIS schema for the NASA Geospatial Misinformation Monitor.
-- This is OPTIONAL. The Week 2 Python prototype runs entirely on CSV files.
-- These tables describe how the same data would be stored for spatial
-- indexing and querying once a PostgreSQL/PostGIS backend is introduced.
--
-- Setup (requires PostgreSQL with the PostGIS extension available):
--   createdb misinfo_monitor
--   psql -d misinfo_monitor -f database/schema.sql

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS claims (
    claim_id           INTEGER PRIMARY KEY,
    claim_text         TEXT NOT NULL,
    claim_date         DATE,
    category           TEXT,
    extracted_location TEXT,
    latitude           DOUBLE PRECISION,
    longitude          DOUBLE PRECISION,
    geom               GEOGRAPHY(POINT, 4326),
    source             TEXT DEFAULT 'synthetic',
    created_at         TIMESTAMPTZ DEFAULT now()
);

-- Normalized / resolved locations kept separately so the baseline spaCy
-- extraction and the resolved location can be compared.
CREATE TABLE IF NOT EXISTS locations (
    location_id         SERIAL PRIMARY KEY,
    claim_id            INTEGER REFERENCES claims(claim_id),
    raw_location        TEXT,
    normalized_location TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    geom                GEOGRAPHY(POINT, 4326),
    location_confidence DOUBLE PRECISION,
    candidate_count     INTEGER
);

CREATE TABLE IF NOT EXISTS satellite_detections (
    detection_id SERIAL PRIMARY KEY,
    satellite    TEXT,
    instrument   TEXT,
    acq_date     DATE,
    acq_time     TEXT,
    latitude     DOUBLE PRECISION,
    longitude    DOUBLE PRECISION,
    frp          DOUBLE PRECISION,
    confidence   TEXT,
    geom         GEOGRAPHY(POINT, 4326)
);

CREATE TABLE IF NOT EXISTS evidence_matches (
    match_id                 SERIAL PRIMARY KEY,
    claim_id                 INTEGER REFERENCES claims(claim_id),
    detection_id             INTEGER REFERENCES satellite_detections(detection_id),
    distance_km              DOUBLE PRECISION,
    temporal_difference_hours DOUBLE PRECISION,
    evidence_score           DOUBLE PRECISION,
    verdict                  TEXT
);

-- GiST spatial indexes on the geography columns.
CREATE INDEX IF NOT EXISTS idx_claims_geom ON claims USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_locations_geom ON locations USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_detections_geom ON satellite_detections USING GIST (geom);

CREATE INDEX IF NOT EXISTS idx_detections_acq_date ON satellite_detections (acq_date);
CREATE INDEX IF NOT EXISTS idx_claims_category ON claims (category);
