from pathlib import Path

import pandas as pd

import firms
import geocode
import ner
import verify

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLAIMS_PATH = DATA_DIR / "wildfire_claims.csv"
RESULTS_PATH = DATA_DIR / "verification_results.csv"


def main():
    claims = pd.read_csv(CLAIMS_PATH)
    ner_results = ner.run_ner()
    map_key = firms.get_map_key()
    firms_enabled = bool(map_key)

    if not firms_enabled:
        print("FIRMS_MAP_KEY not set. NER and geocoding will run; "
              "FIRMS verification will be skipped.\n")

    date_by_id = dict(zip(claims["claim_id"], claims["date"]))

    rows = []
    for _, row in ner_results.iterrows():
        claim_id = row["claim_id"]
        location = row["extracted_location"]
        coords = geocode.geocode(location) if location else None

        if coords:
            lat, lon = coords["latitude"], coords["longitude"]
            display_name = coords["display_name"]
        else:
            lat = lon = display_name = None

        if firms_enabled and coords:
            detections = firms.fetch_detections(lat, lon, date_by_id[claim_id])
        else:
            detections = None

        result = verify.verify(lat, lon, detections) if coords else {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "nearest_detection_km": None,
            "number_of_detections": 0,
            "evidence_source": verify.EVIDENCE_SOURCE,
        }

        rows.append(
            {
                "claim_id": claim_id,
                "text": row["text"],
                "expected_location": row["expected_location"],
                "extracted_location": location,
                "ner_match": row["ner_match"],
                "latitude": lat,
                "longitude": lon,
                "display_name": display_name,
                "verdict": result["verdict"],
                "nearest_detection_km": result["nearest_detection_km"],
                "number_of_detections": result["number_of_detections"],
                "evidence_source": result["evidence_source"],
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    print_summary(results, firms_enabled)
    return results


def print_summary(results, firms_enabled):
    detected = (results["extracted_location"] != "").sum()
    ner_ok = results["ner_match"].isin(["correct", "partial"]).sum()
    geocoded = results["latitude"].notna().sum()
    checked = results[results["verdict"] != "INSUFFICIENT_EVIDENCE"].shape[0] if firms_enabled else 0

    print("=== Verification Summary ===")
    print(f"Claims processed: {len(results)}")
    print(f"Locations detected: {detected}")
    print(f"NER correct/partial: {ner_ok}")
    print(f"Successfully geocoded: {geocoded}")
    print(f"Claims checked against FIRMS: {checked}")
    print(f"Supported: {(results['verdict'] == 'SUPPORTED').sum()}")
    print(f"No match: {(results['verdict'] == 'NO_MATCH').sum()}")
    print(f"Insufficient evidence: {(results['verdict'] == 'INSUFFICIENT_EVIDENCE').sum()}")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
