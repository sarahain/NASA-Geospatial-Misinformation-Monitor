import csv
import io
import os

import requests

API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SOURCE = "VIIRS_NOAA20_NRT"
BBOX_HALF_WIDTH = 0.25
DAY_RANGE = 1

FIELDS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "frp",
]


def get_map_key():
    return os.environ.get("FIRMS_MAP_KEY")


def build_bbox(lat, lon):
    west = lon - BBOX_HALF_WIDTH
    south = lat - BBOX_HALF_WIDTH
    east = lon + BBOX_HALF_WIDTH
    north = lat + BBOX_HALF_WIDTH
    return f"{west},{south},{east},{north}"


def fetch_area(bbox, date, map_key=None, source=SOURCE, day_range=DAY_RANGE):
    map_key = map_key or get_map_key()
    if not map_key:
        return None

    url = f"{API_BASE}/{map_key}/{source}/{bbox}/{day_range}/{date}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    reader = csv.DictReader(io.StringIO(response.text))
    detections = []
    for row in reader:
        detections.append({field: row.get(field, "") for field in FIELDS})
    return detections


def fetch_detections(lat, lon, date, map_key=None, source=SOURCE):
    return fetch_area(build_bbox(lat, lon), date, map_key=map_key, source=source)


if __name__ == "__main__":
    if not get_map_key():
        print("FIRMS_MAP_KEY not set; skipping FIRMS request.")
    else:
        print(fetch_detections(34.0259, -118.7798, "2025-01-07"))
