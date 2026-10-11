from pathlib import Path

import pandas as pd

import firms
import verify

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "sensor_comparison.csv"

SENSORS = [
    {"sensor": "MODIS", "source": "MODIS_NRT", "spatial_resolution": "1 km"},
    {"sensor": "VIIRS NOAA-20", "source": "VIIRS_NOAA20_NRT", "spatial_resolution": "375 m"},
    {"sensor": "VIIRS NOAA-21", "source": "VIIRS_NOAA21_NRT", "spatial_resolution": "375 m"},
]


def _summarize(lat, lon, detections):
    distances, frps = [], []
    for d in detections:
        try:
            distances.append(verify.haversine(lat, lon, float(d["latitude"]), float(d["longitude"])))
        except (ValueError, KeyError):
            pass
        try:
            frps.append(float(d["frp"]))
        except (ValueError, KeyError):
            pass
    nearest = round(min(distances), 2) if distances else None
    avg_frp = round(sum(frps) / len(frps), 2) if frps else None
    return nearest, avg_frp


def compare_sensors(lat, lon, date, output_path=OUTPUT_PATH):
    rows = []
    for s in SENSORS:
        detections = firms.fetch_detections(lat, lon, date, source=s["source"])
        if detections is None:
            nearest = avg_frp = None
            found = None
        else:
            found = len(detections)
            nearest, avg_frp = _summarize(lat, lon, detections)
        rows.append(
            {
                "sensor": s["sensor"],
                "spatial_resolution": s["spatial_resolution"],
                "detections_found": found,
                "nearest_detection_km": nearest,
                "average_frp": avg_frp,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    import geocode

    coords = geocode.geocode("Lahaina")
    print(compare_sensors(coords["latitude"], coords["longitude"], "2023-08-08").to_string(index=False))
