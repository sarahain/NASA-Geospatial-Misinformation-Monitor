"""Week 3 Deliverable 12: connect improved geographic extraction back to NASA.

This does NOT rebuild the FIRMS pipeline. It takes a handful of successfully
resolved wildfire claims from the Week 3 evaluation set and runs them through
the EXISTING NASA FIRMS verification code (``firms`` + ``verify``), proving that
the improved extraction / resolution feeds the larger research project.

Honesty notes
-------------
* The Week 3 claims are synthetic and carry no real timestamp, so there is no
  genuine claim date to match against. FIRMS NRT only covers roughly the last
  two months, so we query the *current* NRT window and report exactly what the
  satellites show. No satellite match is fabricated.
* Without a ``FIRMS_MAP_KEY`` the verdict is INSUFFICIENT_EVIDENCE and the
  detection fields are left empty -- again, nothing is invented.
"""
from datetime import date
from pathlib import Path

import pandas as pd

import firms
import ner_methods
import location_resolution as lr
import verify
import week3_build_dataset as w3data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_SET_PATH = w3data.HYDRATED_PATH  # needs real tweet text (rebuilt locally)
OUTPUT_PATH = DATA_DIR / "week3_end_to_end_examples.csv"

NUM_EXAMPLES = 5


def _nearest(lat, lon, detections):
    nearest = None
    for det in detections or []:
        try:
            d = verify.haversine(lat, lon, float(det["latitude"]), float(det["longitude"]))
        except (ValueError, KeyError):
            continue
        if nearest is None or d < nearest[0]:
            nearest = (d, det)
    return nearest


def run(test_set_path=TEST_SET_PATH, output_path=OUTPUT_PATH, num_examples=NUM_EXAMPLES):
    if test_set_path == w3data.HYDRATED_PATH and not w3data.HYDRATED_PATH.exists():
        w3data.build()  # hydrate real tweet text locally from the HumAID corpus
    claims = pd.read_csv(test_set_path).fillna({"expected_location": ""})
    wildfire = claims[claims["category"] == "wildfire"]

    extractor = ner_methods.get_extractor("spacy")
    map_key = firms.get_map_key()
    firms_enabled = bool(map_key)
    query_date = date.today().isoformat()

    rows = []
    for _, claim in wildfire.iterrows():
        if len(rows) >= num_examples:
            break
        res = lr.resolve_claim(claim["text"], extractor, use_context=True)
        if res["ambiguity_status"] != "resolved" or res["latitude"] is None:
            continue  # only pass genuinely resolved wildfire claims to NASA

        lat, lon = res["latitude"], res["longitude"]
        row = {
            "claim": claim["text"],
            "extracted_location": res["primary_location"],
            "normalized_location": res["normalized_location"],
            "coordinates": f"{round(lat, 4)}, {round(lon, 4)}",
            "NASA_source": f"NASA FIRMS {firms.SOURCE}",
            "nearest_detection": "",
            "distance_km": "",
            "temporal_difference": "",
            "verdict": "INSUFFICIENT_EVIDENCE",
        }

        if firms_enabled:
            detections = firms.fetch_detections(lat, lon, query_date)
            verdict = verify.verify(lat, lon, detections)
            row["verdict"] = verdict["verdict"]
            nearest = _nearest(lat, lon, detections)
            if nearest is not None:
                dist, det = nearest
                row["nearest_detection"] = (
                    f"{det.get('acq_date', '')} @ ({det.get('latitude')}, {det.get('longitude')})"
                )
                row["distance_km"] = round(dist, 2)
                try:
                    det_date = date.fromisoformat(det.get("acq_date"))
                    row["temporal_difference"] = f"{abs((date.today() - det_date).days)} days (vs query date)"
                except (ValueError, TypeError):
                    row["temporal_difference"] = ""
        else:
            row["NASA_source"] += " (FIRMS_MAP_KEY not set)"

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result, firms_enabled, query_date


if __name__ == "__main__":
    df, enabled, qdate = run()
    print(f"FIRMS enabled: {enabled} | NRT query date: {qdate}")
    print(f"Resolved wildfire examples passed to NASA FIRMS: {len(df)}\n")
    print(df.to_string(index=False))
