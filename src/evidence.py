"""Structured evidence records and a transparent prototype confidence score.

The evidence score is a weighted sum of four normalized sub-scores in [0, 1]:

    spatial_score    = max(0, 1 - nearest_detection_km / MAX_SPATIAL_KM)
    temporal_score   = max(0, 1 - temporal_difference_hours / MAX_TEMPORAL_HOURS)
    detection_score  = min(1, detections_within_threshold / DETECTION_SATURATION)
    location_score   = location_confidence (from location_resolution)

    evidence_score   = 0.40*spatial + 0.25*temporal + 0.15*detection + 0.20*location

This is a prototype evidence CONFIDENCE score, not a measure of scientific
certainty. A low score means weak supporting evidence, not proof of falsehood.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

import firms
import location_resolution as lr
import ner
import verify

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLAIMS_PATH = DATA_DIR / "wildfire_claims.csv"
OUTPUT_PATH = DATA_DIR / "evidence_records.csv"

EVIDENCE_SOURCE = "NASA FIRMS VIIRS"
MAX_SPATIAL_KM = 50.0
MAX_TEMPORAL_HOURS = 48.0
DETECTION_SATURATION = 5
AMBIGUITY_THRESHOLD = 0.4
WEIGHTS = {"spatial": 0.40, "temporal": 0.25, "detection": 0.15, "location": 0.20}


def _temporal_diff_hours(claim_date, acq_date, acq_time):
    try:
        claim_dt = datetime.strptime(str(claim_date), "%Y-%m-%d")
        acq_time = str(acq_time).zfill(4)
        det_dt = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M")
        return round(abs((det_dt - claim_dt).total_seconds()) / 3600.0, 1)
    except (ValueError, TypeError):
        return None


def _nearest(lat, lon, detections):
    nearest = None
    for det in detections:
        try:
            d = verify.haversine(lat, lon, float(det["latitude"]), float(det["longitude"]))
        except (ValueError, KeyError):
            continue
        if nearest is None or d < nearest[0]:
            nearest = (d, det)
    return nearest


def _score(spatial_km, temporal_hours, detections_within, location_confidence):
    spatial = max(0.0, 1 - spatial_km / MAX_SPATIAL_KM) if spatial_km is not None else 0.0
    temporal = max(0.0, 1 - temporal_hours / MAX_TEMPORAL_HOURS) if temporal_hours is not None else 0.0
    detection = min(1.0, detections_within / DETECTION_SATURATION)
    location = location_confidence
    total = (
        WEIGHTS["spatial"] * spatial
        + WEIGHTS["temporal"] * temporal
        + WEIGHTS["detection"] * detection
        + WEIGHTS["location"] * location
    )
    return round(min(1.0, total), 3)


def build_evidence(claim, resolution, detections, threshold_km=verify.DISTANCE_THRESHOLD_KM):
    lat = resolution["latitude"]
    lon = resolution["longitude"]
    record = {
        "claim_id": claim["claim_id"],
        "claim_text": claim["text"],
        "claim_date": claim["date"],
        "normalized_location": resolution["normalized_location"],
        "latitude": lat,
        "longitude": lon,
        "evidence_source": EVIDENCE_SOURCE,
        "satellite": None,
        "instrument": None,
        "observation_date": None,
        "nearest_detection_km": None,
        "temporal_difference_hours": None,
        "number_of_detections": 0,
        "evidence_score": 0.0,
        "verdict": "INSUFFICIENT_EVIDENCE",
    }

    if lat is None or not resolution["raw_location"]:
        return record

    if resolution["location_confidence"] < AMBIGUITY_THRESHOLD:
        record["verdict"] = "LOCATION_AMBIGUOUS"
        record["evidence_score"] = _score(None, None, 0, resolution["location_confidence"])
        return record

    if detections is None:
        record["verdict"] = "INSUFFICIENT_EVIDENCE"
        return record

    nearest = _nearest(lat, lon, detections)
    within = []
    for det in detections:
        try:
            if verify.haversine(lat, lon, float(det["latitude"]), float(det["longitude"])) <= threshold_km:
                within.append(det)
        except (ValueError, KeyError):
            continue

    if nearest is not None:
        dist, det = nearest
        record["nearest_detection_km"] = round(dist, 2)
        record["satellite"] = det.get("satellite")
        record["instrument"] = det.get("instrument")
        record["observation_date"] = det.get("acq_date")
        record["temporal_difference_hours"] = _temporal_diff_hours(
            claim["date"], det.get("acq_date"), det.get("acq_time")
        )

    record["number_of_detections"] = len(within)
    record["evidence_score"] = _score(
        record["nearest_detection_km"],
        record["temporal_difference_hours"],
        len(within),
        resolution["location_confidence"],
    )
    record["verdict"] = "SUPPORTED" if within else "INSUFFICIENT_EVIDENCE"
    return record


def build_records(claims_path=CLAIMS_PATH, output_path=OUTPUT_PATH):
    nlp = ner.load_model()
    claims = pd.read_csv(claims_path)
    firms_enabled = bool(firms.get_map_key())

    rows = []
    for _, claim in claims.iterrows():
        resolution = lr.resolve_location(claim["text"], nlp, use_context=True)
        if firms_enabled and resolution["latitude"] is not None:
            detections = firms.fetch_detections(resolution["latitude"], resolution["longitude"], claim["date"])
        else:
            detections = None
        rows.append(build_evidence(claim, resolution, detections))

    records = pd.DataFrame(rows)
    records.to_csv(output_path, index=False)
    return records


if __name__ == "__main__":
    print(build_records().to_string(index=False))
