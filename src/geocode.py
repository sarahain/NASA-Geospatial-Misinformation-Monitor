import time

from geopy.geocoders import Nominatim

USER_AGENT = "nasa-globe-misinformation-monitor-week2"
REQUEST_DELAY = 1.0

_geolocator = Nominatim(user_agent=USER_AGENT)
_cache = {}
_candidate_cache = {}


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
        return None

    coords = {
        "latitude": result.latitude,
        "longitude": result.longitude,
        "display_name": result.address,
    }
    _cache[location] = coords
    return coords


if __name__ == "__main__":
    for place in ["Malibu, California", "Lahaina", "Northern California"]:
        print(place, "->", geocode(place))
