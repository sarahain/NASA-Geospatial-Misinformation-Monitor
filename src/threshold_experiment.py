import datetime

import pandas as pd

import firms
import verify

THRESHOLDS_KM = [5, 10, 25, 50]
CALIFORNIA_BBOX = "-124,36,-118,42"


def threshold_sweep(lat, lon, date, thresholds=THRESHOLDS_KM):
    detections = firms.fetch_detections(lat, lon, date)
    return _sweep(lat, lon, detections, thresholds)


def _sweep(lat, lon, detections, thresholds):
    distances = []
    if detections is not None:
        for d in detections:
            try:
                distances.append(verify.haversine(lat, lon, float(d["latitude"]), float(d["longitude"])))
            except (ValueError, KeyError):
                continue

    rows = []
    for t in thresholds:
        matches = None if detections is None else sum(1 for d in distances if d <= t)
        rows.append({"threshold_km": t, "matches_within_threshold": matches})
    return pd.DataFrame(rows)


def densest_center(detections):
    coords = []
    for d in detections:
        try:
            coords.append((float(d["latitude"]), float(d["longitude"])))
        except (ValueError, KeyError):
            continue
    best, best_count = None, -1
    for lat, lon in coords:
        count = sum(1 for la, lo in coords if verify.haversine(lat, lon, la, lo) <= 50)
        if count > best_count:
            best, best_count = (lat, lon), count
    return best


def recent_region_sweep(bbox=CALIFORNIA_BBOX, date=None, thresholds=THRESHOLDS_KM):
    date = date or datetime.date.today().isoformat()
    detections = firms.fetch_area(bbox, date)
    if not detections:
        return None, None, date
    center = densest_center(detections)
    if center is None:
        return None, None, date
    return center, _sweep(center[0], center[1], detections, thresholds), date


if __name__ == "__main__":
    center, table, date = recent_region_sweep()
    print(f"Recent California FIRMS data ({date}), densest center: {center}")
    print(table.to_string(index=False) if table is not None else "no detections")
