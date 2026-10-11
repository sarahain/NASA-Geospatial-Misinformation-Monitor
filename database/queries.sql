-- Example spatial queries for the optional PostGIS backend.
-- These are reference queries; the Week 2 Python prototype does not require them.

-- KEY DEMONSTRATION QUERY
-- Find satellite fire detections within 25 km of a given claim location.
-- ST_DWithin on geography uses meters, so 25 km = 25000.
SELECT
    d.detection_id,
    d.satellite,
    d.instrument,
    d.acq_date,
    ST_Distance(c.geom, d.geom) / 1000.0 AS distance_km
FROM claims c
JOIN satellite_detections d
  ON ST_DWithin(c.geom, d.geom, 25000)
WHERE c.claim_id = :claim_id
ORDER BY distance_km ASC;

-- Nearest detection to each claim, with its distance in km.
SELECT DISTINCT ON (c.claim_id)
    c.claim_id,
    c.claim_text,
    d.detection_id,
    ST_Distance(c.geom, d.geom) / 1000.0 AS nearest_detection_km
FROM claims c
JOIN satellite_detections d
  ON ST_DWithin(c.geom, d.geom, 50000)
ORDER BY c.claim_id, distance_km;

-- Count detections within 25 km of each claim (spatial match rate input).
SELECT
    c.claim_id,
    c.category,
    COUNT(d.detection_id) AS detections_within_25km
FROM claims c
LEFT JOIN satellite_detections d
  ON ST_DWithin(c.geom, d.geom, 25000)
GROUP BY c.claim_id, c.category
ORDER BY c.claim_id;

-- ST_Intersects example: detections falling inside a region polygon.
-- (region_boundaries is a hypothetical table of administrative polygons.)
-- SELECT d.detection_id, r.name
-- FROM satellite_detections d
-- JOIN region_boundaries r
--   ON ST_Intersects(r.geom::geometry, d.geom::geometry);

-- Threshold sensitivity: how many claims have at least one detection within
-- increasing distance thresholds.
SELECT
    threshold_km,
    COUNT(DISTINCT c.claim_id) AS claims_with_match
FROM claims c
CROSS JOIN (VALUES (5), (10), (25), (50)) AS t(threshold_km)
JOIN satellite_detections d
  ON ST_DWithin(c.geom, d.geom, t.threshold_km * 1000)
GROUP BY threshold_km
ORDER BY threshold_km;
