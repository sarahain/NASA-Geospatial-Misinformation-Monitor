from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_KM = 6371.0
DISTANCE_THRESHOLD_KM = 25.0
EVIDENCE_SOURCE = "NASA FIRMS VIIRS"


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def verify(lat, lon, detections, threshold_km=DISTANCE_THRESHOLD_KM):
    if detections is None:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "nearest_detection_km": None,
            "number_of_detections": 0,
            "evidence_source": EVIDENCE_SOURCE,
        }

    distances = []
    for det in detections:
        try:
            d = haversine(lat, lon, float(det["latitude"]), float(det["longitude"]))
            distances.append(d)
        except (ValueError, KeyError):
            continue

    if not distances:
        return {
            "verdict": "NO_MATCH",
            "nearest_detection_km": None,
            "number_of_detections": 0,
            "evidence_source": EVIDENCE_SOURCE,
        }

    nearest = min(distances)
    within = [d for d in distances if d <= threshold_km]
    verdict = "SUPPORTED" if within else "NO_MATCH"

    return {
        "verdict": verdict,
        "nearest_detection_km": round(nearest, 2),
        "number_of_detections": len(within),
        "evidence_source": EVIDENCE_SOURCE,
    }
