import time

from geopy.geocoders import Nominatim

USER_AGENT = "nasa-globe-misinformation-monitor-week2"
REQUEST_DELAY = 1.0

_geolocator = Nominatim(user_agent=USER_AGENT)
_cache = {}


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
