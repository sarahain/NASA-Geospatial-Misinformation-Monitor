"""Build a reproducible environmental/disaster subset summary from HumAID.

This does **not** modify the full processed HumAID dataset
(``data/processed/humaid_claims.csv``), which stays local and git-ignored. It
reads that local file, keeps only **environmental disaster** events (wildfire,
flood, hurricane, cyclone -- "weather / climate-driven" hazards relevant to this
project's taxonomy and to NASA satellite evidence), runs the preserved spaCy
extractor to count how many posts contain a detected geographic entity, and
writes only **aggregate per-event statistics** -- never any real tweet text --
to a tracked CSV.

Filtering criterion
-------------------
An event is kept if its ``event_type`` (derived from the event folder name) is in
``ENVIRONMENTAL_EVENT_TYPES``. Geophysical events present in HumAID
(earthquakes) are **excluded**: they are not in this project's environmental
hazard taxonomy and are not verifiable with the fire/thermal satellite evidence
the pipeline uses.

Output (tracked, aggregate only)
--------------------------------
    data/week4_environmental_subset_summary.csv
      columns: event, event_type, number_of_posts,
               posts_with_detected_location, location_extraction_rate

A "detected location" means spaCy tagged at least one GPE / LOC / FAC entity in
the post. This is an *extraction* signal only -- it is NOT a confirmed
geolocation, and HumAID posts are not geotagged.
"""
from pathlib import Path

import pandas as pd

import humaid_loader
import ner

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HUMAID_CORPUS_PATH = humaid_loader.OUTPUT_PATH  # local, git-ignored full dataset
OUTPUT_PATH = DATA_DIR / "week4_environmental_subset_summary.csv"

# Keyword -> event_type, matched against the HumAID event folder name.
EVENT_TYPE_KEYWORDS = [
    ("wildfire", "wildfire"),
    ("flood", "flood"),
    ("hurricane", "hurricane"),
    ("cyclone", "cyclone"),
    ("typhoon", "typhoon"),
    ("earthquake", "earthquake"),
]

# Only these event types are considered environmental/disaster for the subset.
ENVIRONMENTAL_EVENT_TYPES = {"wildfire", "flood", "hurricane", "cyclone", "typhoon"}

OUTPUT_FIELDS = [
    "event",
    "event_type",
    "number_of_posts",
    "posts_with_detected_location",
    "location_extraction_rate",
]


def event_to_type(event):
    low = str(event).lower()
    for keyword, hazard in EVENT_TYPE_KEYWORDS:
        if keyword in low:
            return hazard
    return "other"


def _load_corpus():
    if not HUMAID_CORPUS_PATH.exists():
        raise SystemExit(
            f"Processed HumAID dataset not found at {HUMAID_CORPUS_PATH}.\n"
            "Build it locally first:  python src/humaid_loader.py\n"
            "(It stays git-ignored; only this aggregate summary is tracked.)"
        )
    return pd.read_csv(HUMAID_CORPUS_PATH)


def build(output_path=OUTPUT_PATH):
    df = _load_corpus()
    df = df.copy()
    df["event_type"] = df["event"].map(event_to_type)

    subset = df[df["event_type"].isin(ENVIRONMENTAL_EVENT_TYPES)].copy()

    nlp = ner.load_model()
    # nlp.pipe is far faster than per-text calls over tens of thousands of posts.
    texts = subset["original_text"].astype(str).tolist()
    detected = []
    for doc in nlp.pipe(texts, batch_size=256):
        detected.append(any(ent.label_ in ner.LOCATION_LABELS for ent in doc.ents))
    subset["has_location"] = detected

    rows = []
    for (event, event_type), group in subset.groupby(["event", "event_type"], sort=True):
        n = len(group)
        with_loc = int(group["has_location"].sum())
        rows.append(
            {
                "event": event,
                "event_type": event_type,
                "number_of_posts": n,
                "posts_with_detected_location": with_loc,
                "location_extraction_rate": round(100 * with_loc / n, 1) if n else 0.0,
            }
        )

    result = pd.DataFrame(rows, columns=OUTPUT_FIELDS)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    summary = build()
    print("=== Week 4 environmental/disaster subset (HumAID) ===")
    print(f"Environmental event types kept: {sorted(ENVIRONMENTAL_EVENT_TYPES)}")
    print(summary.to_string(index=False))
    total_posts = int(summary["number_of_posts"].sum())
    total_loc = int(summary["posts_with_detected_location"].sum())
    print(f"\nTotals: {len(summary)} events, {total_posts} posts, "
          f"{total_loc} with a detected location "
          f"({round(100 * total_loc / total_posts, 1) if total_posts else 0.0}%).")
    print(f"Written to {OUTPUT_PATH} (aggregate only; no tweet text).")
