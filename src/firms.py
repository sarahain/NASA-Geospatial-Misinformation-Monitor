import csv
import io
import os
from datetime import date as _date, datetime, timezone

import requests

API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# FIRMS serves recent fires as Near Real-Time (NRT, ~last 2 months only) and
# older fires as Standard Processing (SP, the multi-year archive). A tweet from
# up to ~2 years ago therefore needs the SP ("archive") source -- querying NRT
# for an old date returns NOTHING even if a fire really happened. We pick the
# source automatically from the query date.
SOURCE = "VIIRS_NOAA20_NRT"          # default / recent (back-compat name)
SOURCE_NRT = "VIIRS_NOAA20_NRT"      # recent (~last 2 months)
SOURCE_ARCHIVE = "VIIRS_SNPP_SP"     # standard-processing archive (2012-01-20 -> present)
# Dates older than this many days are served by the SP archive, not NRT.
NRT_MAX_AGE_DAYS = 60

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


def _parse_date(date):
    """Return a ``date`` from an ISO string / datetime / date, or None."""
    if date is None:
        return None
    if isinstance(date, datetime):
        return date.date()
    if isinstance(date, _date):
        return date
    s = str(date).strip()
    if not s:
        return None
    # Accept "YYYY-MM-DD" or full ISO timestamps like "2023-07-01T12:00:00Z".
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def choose_source(date, today=None):
    """Pick the FIRMS source for a query date.

    Recent dates (< NRT_MAX_AGE_DAYS old) use NRT; older dates use the SP
    archive so historical tweets (e.g. 1-2 years back) return real detections.
    Unparseable/absent dates default to the archive (safer for old data).
    """
    d = _parse_date(date)
    if d is None:
        return SOURCE_ARCHIVE
    today = today or datetime.now(timezone.utc).date()
    age_days = (today - d).days
    return SOURCE_NRT if age_days < NRT_MAX_AGE_DAYS else SOURCE_ARCHIVE


# Every FIRMS active-fire product the Area API serves, with the sensor facts a
# researcher needs to judge "which data is best" for a given claim.
PRODUCTS = {
    "VIIRS_SNPP_NRT":   {"satellite": "Suomi NPP",  "instrument": "VIIRS",  "resolution_m": 375,  "revisit": "~2/day (polar)",   "kind": "NRT", "coverage": "global"},
    "VIIRS_SNPP_SP":    {"satellite": "Suomi NPP",  "instrument": "VIIRS",  "resolution_m": 375,  "revisit": "~2/day (polar)",   "kind": "archive", "coverage": "global"},
    "VIIRS_NOAA20_NRT": {"satellite": "NOAA-20",    "instrument": "VIIRS",  "resolution_m": 375,  "revisit": "~2/day (polar)",   "kind": "NRT", "coverage": "global"},
    "VIIRS_NOAA20_SP":  {"satellite": "NOAA-20",    "instrument": "VIIRS",  "resolution_m": 375,  "revisit": "~2/day (polar)",   "kind": "archive", "coverage": "global"},
    "VIIRS_NOAA21_NRT": {"satellite": "NOAA-21",    "instrument": "VIIRS",  "resolution_m": 375,  "revisit": "~2/day (polar)",   "kind": "NRT", "coverage": "global"},
    "VIIRS_NOAA21_SP":  {"satellite": "NOAA-21",    "instrument": "VIIRS",  "resolution_m": 375,  "revisit": "~2/day (polar)",   "kind": "archive", "coverage": "global"},
    "MODIS_NRT":        {"satellite": "Terra+Aqua", "instrument": "MODIS",  "resolution_m": 1000, "revisit": "~4/day (polar)",   "kind": "NRT", "coverage": "global"},
    "MODIS_SP":         {"satellite": "Terra+Aqua", "instrument": "MODIS",  "resolution_m": 1000, "revisit": "~4/day (polar)",   "kind": "archive", "coverage": "global"},
    "LANDSAT_NRT":      {"satellite": "Landsat 8/9", "instrument": "OLI/TIRS", "resolution_m": 30, "revisit": "8 days",          "kind": "NRT", "coverage": "US + Canada"},
    "GOES_NRT":         {"satellite": "GOES-16/18", "instrument": "ABI",    "resolution_m": 2000, "revisit": "every 10 min (geo)", "kind": "NRT", "coverage": "Americas"},
}
LIVE_PRODUCTS = [p for p, m in PRODUCTS.items() if m["kind"] == "NRT"]

# Sensor families. Each has an NRT product (recent) and an SP archive product.
SENSORS = {
    "VIIRS": {"nrt": "VIIRS_NOAA20_NRT", "sp": "VIIRS_SNPP_SP", "resolution_m": 375},
    "MODIS": {"nrt": "MODIS_NRT", "sp": "MODIS_SP", "resolution_m": 1000},
}


def choose_sensor_source(date, sensor, today=None, availability=None):
    """Like ``choose_source`` but for a named sensor family ('VIIRS' / 'MODIS').

    When a FIRMS ``availability`` table is given, the product whose date range
    actually contains ``date`` wins (NRT preferred); otherwise the 60-day rule.
    """
    d = _parse_date(date)
    if availability and d is not None:
        ds = d.isoformat()
        for kind in ("nrt", "sp"):
            src = SENSORS[sensor][kind]
            lo, hi = availability.get(src, ("", ""))
            if lo and hi and lo <= ds <= hi:
                return src
    kind = "nrt" if choose_source(date, today=today) == SOURCE_NRT else "sp"
    return SENSORS[sensor][kind]


def fetch_area_status(bbox, date, map_key=None, source=None, day_range=DAY_RANGE):
    """Fetch detections and report *why* when nothing comes back.

    Returns ``(detections, status, message)`` where status is one of
    ``ok`` / ``no_key`` / ``api_error`` / ``invalid_response``. ``detections``
    is a list (possibly empty) only when status is ``ok``; otherwise None.
    """
    map_key = map_key or get_map_key()
    if not map_key:
        return None, "no_key", "FIRMS_MAP_KEY not set"

    source = source or choose_source(date)
    url = f"{API_BASE}/{map_key}/{source}/{bbox}/{day_range}/{date}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        return None, "api_error", f"{type(exc).__name__}: {exc}"[:200]

    text = response.text
    first_line = text.splitlines()[0] if text.strip() else ""
    # FIRMS returns HTTP 200 with a plain-text message for bad keys / bad params.
    if "latitude" not in first_line:
        return None, "invalid_response", first_line[:200] or "empty body"

    reader = csv.DictReader(io.StringIO(text))
    # Keep every column FIRMS returns (products differ: Landsat has path/row and
    # no FRP, GOES/VIIRS add brightness + daynight) but guarantee the core FIELDS.
    detections = [{**{field: "" for field in FIELDS}, **row} for row in reader]
    if source and not any(d.get("instrument") for d in detections[:1]):
        inst = PRODUCTS.get(source, {}).get("instrument", "")
        for d in detections:
            d["instrument"] = d["instrument"] or inst
    return detections, "ok", ""


def fetch_area(bbox, date, map_key=None, source=None, day_range=DAY_RANGE):
    detections, _status, _msg = fetch_area_status(bbox, date, map_key=map_key,
                                                  source=source, day_range=day_range)
    return detections


def fetch_detections(lat, lon, date, map_key=None, source=None, day_range=DAY_RANGE):
    """Fetch FIRMS detections near (lat, lon) for ``date``.

    ``source`` defaults to an automatic NRT-vs-archive choice based on the date.
    """
    return fetch_area(build_bbox(lat, lon), date, map_key=map_key, source=source,
                      day_range=day_range)


def fetch_detections_status(lat, lon, date, map_key=None, source=None, day_range=DAY_RANGE):
    return fetch_area_status(build_bbox(lat, lon), date, map_key=map_key, source=source,
                             day_range=day_range)


def data_availability(map_key=None):
    """Return {source: (min_date, max_date)} from the FIRMS availability endpoint,
    or None if it cannot be fetched."""
    map_key = map_key or get_map_key()
    if not map_key:
        return None
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{map_key}/ALL"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None
    out = {}
    for row in csv.DictReader(io.StringIO(response.text)):
        if row.get("data_id"):
            out[row["data_id"]] = (row.get("min_date", ""), row.get("max_date", ""))
    return out or None


if __name__ == "__main__":
    if not get_map_key():
        print("FIRMS_MAP_KEY not set; skipping FIRMS request.")
    else:
        print(fetch_detections(34.0259, -118.7798, "2025-01-07"))
