"""Run the existing Week 3 geographic pipeline over collected X posts (Week 5).

Takes the RELEVANT X posts produced by ``src/x_ingestion.py`` and runs them
through the **existing** extraction + resolution pipeline (``ner_methods`` +
``location_resolution``), writing a SEPARATE processed result -- the raw post
text is never modified, and raw vs processed files are kept distinct.

Outputs
-------
    data/processed/week5_x_ner_results.csv   (git-ignored; has text)
        post_id, created_at, text, detected_locations, primary_location,
        normalized_location, latitude, longitude, location_confidence,
        ambiguity_status

    data/week5_x_nasa_ready_summary.csv      (tracked; NO text)
        total_unique_posts, posts_with_detected_location, successfully_geocoded,
        posts_with_usable_timestamp, potentially_nasa_ready

"NASA-ready" means a post *could* be used for a NASA satellite comparison -- it is
environmental/disaster content (already filtered as RELEVANT), has a resolvable
geographic location, and a usable timestamp. It does **not** mean any satellite
verification has happened.

Geocoding uses the network; this script only runs it when you execute it after a
collection. If the input file is absent it exits cleanly without doing anything.
"""
import csv
from datetime import datetime
from pathlib import Path

import x_ingestion

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_PATH = x_ingestion.RELEVANT_PATH
NER_RESULTS_PATH = PROCESSED_DIR / "week5_x_ner_results.csv"
NASA_READY_SUMMARY_PATH = DATA_DIR / "week5_x_nasa_ready_summary.csv"

NER_FIELDS = [
    "post_id", "created_at", "text", "detected_locations", "primary_location",
    "normalized_location", "latitude", "longitude", "location_confidence",
    "ambiguity_status",
]
NASA_SUMMARY_FIELDS = [
    "total_unique_posts", "posts_with_detected_location", "successfully_geocoded",
    "posts_with_usable_timestamp", "potentially_nasa_ready",
]


def _usable_timestamp(created_at):
    s = str(created_at or "").strip()
    if not s:
        return False
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def run(input_path=INPUT_PATH, ner_results_path=NER_RESULTS_PATH,
        nasa_summary_path=NASA_READY_SUMMARY_PATH):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"[week5_x_ner] no collected posts at {input_path}. "
              "Run src/x_ingestion.py (search mode) first; nothing to do.")
        return None

    import location_resolution as lr
    import ner_methods

    extractor = ner_methods.get_extractor("spacy")

    with open(input_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    results = []
    for row in rows:
        text = row.get("text", "")
        res = lr.resolve_claim(text, extractor, use_context=True)
        results.append({
            "post_id": row.get("post_id", ""),
            "created_at": row.get("created_at", ""),
            "text": text,
            "detected_locations": "; ".join(res.get("all_locations", [])),
            "primary_location": res.get("primary_location", ""),
            "normalized_location": res.get("normalized_location", ""),
            "latitude": res.get("latitude"),
            "longitude": res.get("longitude"),
            "location_confidence": res.get("location_confidence"),
            "ambiguity_status": res.get("ambiguity_status", ""),
        })

    ner_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ner_results_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=NER_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    summary = build_nasa_ready_summary(results)
    _write_summary(summary, nasa_summary_path)
    return results, summary


def build_nasa_ready_summary(results):
    """Aggregate NASA-readiness metrics (no text) from NER results."""
    total = len(results)
    with_location = sum(1 for r in results if r["primary_location"])
    geocoded = sum(1 for r in results if r["latitude"] not in (None, "", "None"))
    with_ts = sum(1 for r in results if _usable_timestamp(r["created_at"]))
    nasa_ready = sum(
        1 for r in results
        if r["latitude"] not in (None, "", "None") and _usable_timestamp(r["created_at"])
    )
    return {
        "total_unique_posts": total,
        "posts_with_detected_location": with_location,
        "successfully_geocoded": geocoded,
        "posts_with_usable_timestamp": with_ts,
        "potentially_nasa_ready": nasa_ready,
    }


def _write_summary(summary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=NASA_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary)
    return path


if __name__ == "__main__":
    out = run()
    if out:
        _, summary = out
        print("=== Week 5 X NER + NASA-ready summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"\nResults : {NER_RESULTS_PATH} (git-ignored; has text)")
        print(f"Summary : {NASA_READY_SUMMARY_PATH} (tracked; no text)")
        print("\nNote: 'potentially_nasa_ready' does NOT mean satellite verification has happened.")
