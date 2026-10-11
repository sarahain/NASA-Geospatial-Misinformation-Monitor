# Wildfire Case Study: Lahaina, Maui (2023)

A reproducible walkthrough of the Week 2 pipeline on a single documented
wildfire event. The claim below is **synthetic** (written for this project to
resemble a plausible social-media post); it is not a real sourced tweet.

## 1. Event name
2023 Hawaii / Maui wildfires (Lahaina fire).

## 2. Event date
August 8, 2023 (fire began; destruction of Lahaina occurred Aug 8–9, 2023).

## 3. Location
Lahaina, Maui County, Hawaii, USA.

## 4. Sample claim (synthetic)
> "Large fire reported near Lahaina"
(claim_id 4 in `data/wildfire_claims.csv`, dated 2023-08-08)

## 5. Extracted location
spaCy `en_core_web_sm` extracts the GPE entity **`Lahaina`** from the claim
text (baseline NER, no correction rules applied).

## 6. Coordinates
Nominatim geocodes `Lahaina` to approximately:
- Latitude: 20.8739
- Longitude: -156.6777
- Resolved name: *Lahaina, Maui County, Hawaii, United States*

A 0.5° bounding box is built around this point for the FIRMS query.

## 7. NASA observations
Evidence source: NASA FIRMS active fire detections (VIIRS NOAA-20).

**Important limitation:** the FIRMS *area* API used in this prototype serves
near-real-time (NRT) products, which cover only roughly the last two months.
The event date (2023-08-08) is well outside that window, so a live NRT query
for this historical date returns **zero detections**. Retrieving the actual
August 2023 detections requires the FIRMS archive download, which is future
work. No detection values are invented here.

## 8. Spatial matching result
With no detections returned for the historical date, there is nothing to match
spatially. Had detections been available, the pipeline would compute the
Haversine distance from the claim coordinate to each detection and count those
within the configurable threshold (default 25 km).

## 9. Final evidence verdict
`INSUFFICIENT_EVIDENCE` — under the current NRT-only prototype, no satellite
evidence is available for this date. This is **not** a judgement that the claim
is false; the 2023 Lahaina fire is a well-documented real event. The verdict
reflects a data-access limitation of the prototype, not the truth of the claim.

## 10. Limitations
- NRT-only retrieval; historical events need the FIRMS archive.
- Geocoding returns a single representative point, not the fire perimeter.
- Baseline NER / geocoding can mis-resolve ambiguous place names.
- Absence of a detection never establishes that a claim is false.
- A single sensor (VIIRS NOAA-20) is queried in the default pipeline; MODIS and
  NOAA-21 are compared separately in `src/sensor_comparison.py`.
