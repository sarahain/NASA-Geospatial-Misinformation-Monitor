"""Connect LLM-annotated X posts to NASA FIRMS satellite evidence.

Final stage of the pipeline:

    x_ingestion.py  ->  llm_annotate.py  ->  [THIS]  ->  verification CSV

For each annotated post it:
    1. takes the LLM's ``llm_location`` and ``created_at`` date,
    2. geocodes the location to lat/lon (src/geocode.py, cached Nominatim),
    3. picks the right FIRMS source for that date (NRT vs SP archive) and fetches
       active-fire detections near that point on that day (src/firms.py),
    4. decides a verdict with the existing haversine check (src/verify.py).

IMPORTANT: NASA FIRMS detects **fire / thermal anomalies only**. It can support
or contradict *wildfire* claims. It CANNOT verify floods, tornados, hurricanes,
typhoons, mudslides, or earthquakes -- those are marked NOT_APPLICABLE_NON_FIRE
so a missing fire detection is never misread as "misinformation".

No X API credits are spent here. FIRMS needs ``FIRMS_MAP_KEY``; without it the
geocode step still runs and fire verdicts become INSUFFICIENT_EVIDENCE.
"""
import argparse
import csv
from pathlib import Path

import firms
import geocode
import verify

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_INPUT = PROCESSED_DIR / "x_llm_annotated.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "x_tweet_verification.csv"

# Only these hazard types are checkable against FIRMS (fire/thermal) evidence.
FIRE_VERIFIABLE_TYPES = {"wildfire"}

OUTPUT_FIELDS = [
    "post_id", "text", "created_at",
    "llm_hazard_type", "llm_location", "llm_claim", "llm_stance",
    "llm_misinfo_risk", "llm_confidence",
    "latitude", "longitude", "display_name",
    "firms_source", "verdict", "nearest_detection_km",
    "number_of_detections", "evidence_source",
]


def _date_only(created_at):
    return str(created_at or "")[:10]


def verify_row(row, firms_enabled=True, geocode_fn=None, detections_fn=None):
    """Verify one annotated post. Returns an output record dict.

    ``geocode_fn(location)`` and ``detections_fn(lat, lon, date, source)`` can be
    injected for testing; they default to the real geocode / FIRMS calls.
    """
    geocode_fn = geocode_fn or geocode.geocode
    detections_fn = detections_fn or (lambda lat, lon, d, source: firms.fetch_detections(
        lat, lon, d, source=source))

    hazard = (row.get("llm_hazard_type") or "").lower()
    location = row.get("llm_location") or ""
    date = _date_only(row.get("created_at"))

    base = {k: row.get(k, "") for k in (
        "post_id", "text", "created_at", "llm_hazard_type", "llm_location",
        "llm_claim", "llm_stance", "llm_misinfo_risk", "llm_confidence")}
    base.update({
        "latitude": "", "longitude": "", "display_name": "",
        "firms_source": "", "verdict": "", "nearest_detection_km": "",
        "number_of_detections": 0, "evidence_source": verify.EVIDENCE_SOURCE,
    })

    coords = geocode_fn(location) if location else None
    if coords:
        base["latitude"] = coords["latitude"]
        base["longitude"] = coords["longitude"]
        base["display_name"] = coords.get("display_name", "")

    # FIRMS can only speak to fire hazards.
    if hazard not in FIRE_VERIFIABLE_TYPES:
        base["verdict"] = "NOT_APPLICABLE_NON_FIRE"
        return base

    if not coords:
        base["verdict"] = "INSUFFICIENT_EVIDENCE"
        return base

    source = firms.choose_source(date)
    base["firms_source"] = source
    if not firms_enabled:
        base["verdict"] = "INSUFFICIENT_EVIDENCE"
        return base

    detections = detections_fn(coords["latitude"], coords["longitude"], date, source)
    result = verify.verify(coords["latitude"], coords["longitude"], detections)
    base["verdict"] = result["verdict"]
    base["nearest_detection_km"] = result["nearest_detection_km"]
    base["number_of_detections"] = result["number_of_detections"]
    base["evidence_source"] = result["evidence_source"]
    return base


def _read_rows(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[verify_tweets] input not found: {path}. Run llm_annotate.py first.")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, limit=None,
        firms_enabled=None):
    rows = _read_rows(input_path)
    if limit is not None:
        rows = rows[:limit]
    if firms_enabled is None:
        firms_enabled = bool(firms.get_map_key())

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            rec = verify_row(r, firms_enabled=firms_enabled)
            writer.writerow(rec)
            results.append(rec)
    return results


def summarize(results):
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return counts


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Verify LLM-annotated X posts against NASA FIRMS.")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="Annotated CSV from llm_annotate.py.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output verification CSV.")
    p.add_argument("--limit", type=int, default=None, help="Verify at most N posts.")
    p.add_argument("--dry-run", action="store_true", help="Show config only; verify nothing.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    firms_enabled = bool(firms.get_map_key())

    if args.dry_run:
        print("=== verify_tweets DRY RUN (no verification performed) ===")
        print(f"input         : {args.input}")
        print(f"output        : {args.output}")
        print(f"FIRMS_MAP_KEY : {'detected' if firms_enabled else 'NOT set -> fire verdicts = INSUFFICIENT_EVIDENCE'}")
        print(f"fire-verifiable hazard types: {sorted(FIRE_VERIFIABLE_TYPES)}")
        return 0

    if not firms_enabled:
        print("[verify_tweets] FIRMS_MAP_KEY not set; fire verdicts will be INSUFFICIENT_EVIDENCE.")
    results = run(input_path=args.input, output_path=args.output, limit=args.limit,
                  firms_enabled=firms_enabled)
    print("\n--- verify_tweets result ---")
    print(f"posts verified : {len(results)}")
    for verdict, n in sorted(summarize(results).items()):
        print(f"  {verdict:24s}: {n}")
    print(f"output         : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
