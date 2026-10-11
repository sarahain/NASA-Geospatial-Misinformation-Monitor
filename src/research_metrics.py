from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VERIFICATION_PATH = DATA_DIR / "verification_results.csv"
EVIDENCE_PATH = DATA_DIR / "evidence_records.csv"
OUTPUT_PATH = DATA_DIR / "research_metrics.csv"


def _rate(n, total):
    return round(100 * n / total, 1) if total else 0.0


def compute_metrics(verification_path=VERIFICATION_PATH, output_path=OUTPUT_PATH):
    if not verification_path.exists():
        raise SystemExit(
            f"{verification_path} not found. Run `python src/main.py` first to generate it."
        )

    df = pd.read_csv(verification_path).fillna({"extracted_location": ""})
    total = len(df)

    metrics = {
        "number_of_claims": total,
        "ner_extraction_rate": _rate((df["extracted_location"] != "").sum(), total),
        "ner_exact_match_rate": _rate((df["ner_match"] == "correct").sum(), total),
        "ner_partial_match_rate": _rate((df["ner_match"] == "partial").sum(), total),
        "geocoding_success_rate": _rate(df["latitude"].notna().sum(), total),
        "satellite_evidence_match_rate": _rate((df["verdict"] == "SUPPORTED").sum(), total),
        "average_nearest_detection_km": round(df["nearest_detection_km"].dropna().mean(), 2)
        if df["nearest_detection_km"].notna().any()
        else None,
    }

    if EVIDENCE_PATH.exists():
        ev = pd.read_csv(EVIDENCE_PATH)
        metrics["location_ambiguity_rate"] = _rate((ev["verdict"] == "LOCATION_AMBIGUOUS").sum(), len(ev))
    else:
        metrics["location_ambiguity_rate"] = None

    summary = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    summary.to_csv(output_path, index=False)
    return metrics


def print_summary(metrics):
    print("=== Research Metrics (Week 2) ===")
    print(f"Claims analysed              : {metrics['number_of_claims']}")
    print(f"NER extraction rate          : {metrics['ner_extraction_rate']}%")
    print(f"NER exact-match rate         : {metrics['ner_exact_match_rate']}%")
    print(f"NER partial-match rate       : {metrics['ner_partial_match_rate']}%")
    print(f"Geocoding success rate       : {metrics['geocoding_success_rate']}%")
    print(f"Location ambiguity rate      : {metrics['location_ambiguity_rate']}%")
    print(f"Satellite evidence match rate: {metrics['satellite_evidence_match_rate']}%")
    print(f"Average nearest detection    : {metrics['average_nearest_detection_km']} km")


if __name__ == "__main__":
    print_summary(compute_metrics())
