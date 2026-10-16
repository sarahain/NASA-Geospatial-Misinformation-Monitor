import json
import os
import time
from pathlib import Path

import requests
from geopy.geocoders import Nominatim

USER_AGENT = "nasa-globe-misinformation-monitor-week2"
REQUEST_DELAY = 1.0

# Optional Google Geocoding fallback (Week 3). The key is NEVER hardcoded; it is
# read from the environment. The project runs normally when it is absent.
GOOGLE_API_KEY_ENV = "GOOGLE_GEOCODING_API_KEY"
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Persistent on-disk cache so repeated runs (and the notebook) do not re-hit
# Nominatim for places already resolved. This both respects Nominatim's usage
# policy and makes the evaluation fast and reproducible.
_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "geocode_cache.json"


def _load_cache():
    if _CACHE_PATH.exists():
        try:
            data = json.loads(_CACHE_PATH.read_text())
            return (
                data.get("candidates", {}),
                data.get("single", {}),
                data.get("google", {}),
            )
        except (ValueError, OSError):
            pass
    return {}, {}, {}


def _save_cache():
    try:
        _CACHE_PATH.write_text(
            json.dumps(
                {"candidates": _candidate_cache, "single": _cache, "google": _google_cache}
            )
        )
    except OSError:
        pass


_geolocator = Nominatim(user_agent=USER_AGENT)
_candidate_cache, _cache, _google_cache = _load_cache()


def geocode_candidates(location, limit=5):
    if not location:
        return []

    if location in _candidate_cache:
        return _candidate_cache[location]

    try:
        results = _geolocator.geocode(location, exactly_one=False, limit=limit, addressdetails=True)
        time.sleep(REQUEST_DELAY)
    except Exception:
        results = None

    candidates = []
    for r in results or []:
        candidates.append(
            {
                "latitude": r.latitude,
                "longitude": r.longitude,
                "display_name": r.address,
                "importance": float(r.raw.get("importance", 0.0)),
                "type": r.raw.get("type", ""),
            }
        )
    _candidate_cache[location] = candidates
    _save_cache()
    return candidates


def google_api_key():
    return os.environ.get(GOOGLE_API_KEY_ENV)


def google_available():
    return bool(google_api_key())


def google_geocode_candidates(location, limit=5):
    """Google Geocoding fallback. Returns [] if no key is configured.

    Output is normalized to the same candidate shape as ``geocode_candidates``
    so the resolver can treat Nominatim and Google results identically.
    """
    if not location:
        return []

    key = google_api_key()
    if not key:
        return []

    if location in _google_cache:
        return _google_cache[location]

    try:
        response = requests.get(
            GOOGLE_GEOCODE_URL,
            params={"address": location, "key": key},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        payload = {}

    candidates = []
    for r in (payload.get("results") or [])[:limit]:
        geometry = r.get("geometry", {}).get("location", {})
        if "lat" not in geometry or "lng" not in geometry:
            continue
        # Google has no 'importance' field; approximate from result precision so
        # the shared ranker still works (rooftop/locality > approximate region).
        location_type = r.get("geometry", {}).get("location_type", "")
        importance = {
            "ROOFTOP": 0.8,
            "RANGE_INTERPOLATED": 0.7,
            "GEOMETRIC_CENTER": 0.6,
            "APPROXIMATE": 0.5,
        }.get(location_type, 0.5)
        candidates.append(
            {
                "latitude": geometry["lat"],
                "longitude": geometry["lng"],
                "display_name": r.get("formatted_address", ""),
                "importance": importance,
                "type": (r.get("types") or [""])[0],
            }
        )
    _google_cache[location] = candidates
    _save_cache()
    return candidates


def geocode(location):
    if not location:
        return None

    if location in _cache:
        return _cache[location]

    try:
        result = _geolocator.geocode(location)
        time.sleep(REQUEST_DELAY)
    except Exception:
        result = None

    if result is None:
        _cache[location] = None
        _save_cache()
        return None

    coords = {
        "latitude": result.latitude,
        "longitude": result.longitude,
        "display_name": result.address,
    }
    _cache[location] = coords
    _save_cache()
    return coords


if __name__ == "__main__":
    for place in ["Malibu, California", "Lahaina", "Northern California"]:
        print(place, "->", geocode(place))
