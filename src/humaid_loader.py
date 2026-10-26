"""Load real disaster post text from the HumAID crisis dataset.

Week 4 upgrade: the project now uses **real** disaster-related social-media text
rather than relying primarily on synthetic claims. This loader reads the actual
tweet text from a locally downloaded copy of the HumAID corpus
(Alam, Ofli & Imran, 2021 -- https://crisisnlp.qcri.org/humaid_dataset) and
writes a processed dataset for experimentation.

Data category
-------------
Every record produced here is tagged::

    data_origin  = REAL_DATASET      # not LIVE_API, not SYNTHETIC
    dataset_name = HumAID

The real post text, original humanitarian label and event are preserved; the
text is only *technically* normalized (a leading ``RT @user:`` and URLs are
stripped and whitespace collapsed) -- it is never rewritten or paraphrased.

Licensing / research integrity
------------------------------
HumAID's terms (see ``data/raw/events_set1/Licensing.txt``) require the dataset
contents to be kept **confidential**; only tweet IDs may be shared/redistributed.
Therefore:

* The processed output ``data/processed/humaid_claims.csv`` **contains real
  tweet text and is NOT committed to Git** (it is listed in ``.gitignore``).
* The raw HumAID download under ``data/raw/`` is likewise never committed.
* To reproduce this file, download HumAID yourself (instructions below) and run
  this loader locally.

Usage
-----
One-time: download the HumAID "Event wise dataset (set1)" from
https://crisisnlp.qcri.org/humaid_dataset and extract it so that
``data/raw/events_set1/`` exists, then::

    python src/humaid_loader.py

The loader prints a summary and writes ``data/processed/humaid_claims.csv``.
"""
import argparse
import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "events_set1"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = PROCESSED_DIR / "humaid_claims.csv"

DATA_ORIGIN = "REAL_DATASET"
DATASET_NAME = "HumAID"
DATASET_CITATION = (
    "HumAID (Alam, Ofli & Imran, 2021); https://crisisnlp.qcri.org/humaid_dataset"
)

OUTPUT_FIELDS = [
    "claim_id",
    "original_text",
    "event",
    "humanitarian_label",
    "data_origin",
    "dataset_name",
]

DOWNLOAD_HINT = (
    f"HumAID corpus not found at {RAW_DIR}.\n"
    "Download the 'Event wise dataset (set1)' from "
    "https://crisisnlp.qcri.org/humaid_dataset and extract it so that\n"
    f"  {RAW_DIR}\n"
    "exists (it should contain one sub-folder per event, each with *.tsv files),\n"
    "then re-run this loader. The raw data and the processed output are kept out\n"
    "of Git because HumAID's terms only permit redistributing tweet IDs."
)


def normalize_text(text):
    """Minimal, *technical-only* tweet normalization.

    Strips a leading ``RT @user:`` retweet marker and bare URLs, and collapses
    runs of whitespace. The actual claim wording is left untouched -- no
    lowercasing, spell-correction, de-hashtagging, or emoji removal, so the
    noisy real-world language is preserved for analysis.
    """
    text = re.sub(r"^RT @\w+:\s*", "", str(text))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _event_tsv_files(raw_dir):
    """Yield (event_name, tsv_path) for every HumAID split under ``raw_dir``.

    The event name is the dataset sub-folder (e.g. ``greece_wildfires_2018``);
    the HumAID layout is ``events_set1/<event>/<event>_{train,dev,test}.tsv``.
    """
    for tsv in sorted(raw_dir.rglob("*.tsv")):
        event = tsv.parent.name
        yield event, tsv


def load_records(raw_dir=RAW_DIR, events=None):
    """Read HumAID TSVs into processed records.

    Parameters
    ----------
    raw_dir : Path
        The extracted ``events_set1`` directory.
    events : iterable[str] | None
        Optional allow-list of event folder names. ``None`` loads every event
        found locally.

    Returns a list of dicts with the Week 4 ``REAL_DATASET`` schema. Duplicate
    tweet IDs (a tweet may appear in more than one split) are de-duplicated.
    """
    if not raw_dir.exists():
        raise SystemExit(DOWNLOAD_HINT)

    allow = set(events) if events else None
    seen_ids = set()
    records = []
    next_id = 1
    for event, tsv in _event_tsv_files(raw_dir):
        if allow is not None and event not in allow:
            continue
        with open(tsv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                tweet_id = (row.get("tweet_id") or "").strip()
                raw_text = row.get("tweet_text", "")
                label = (row.get("class_label") or "").strip()
                if not raw_text:
                    continue
                if tweet_id and tweet_id in seen_ids:
                    continue
                if tweet_id:
                    seen_ids.add(tweet_id)
                text = normalize_text(raw_text)
                if not text:
                    continue
                records.append(
                    {
                        "claim_id": f"humaid_{next_id}",
                        "original_text": text,
                        "event": event,
                        "humanitarian_label": label,
                        "data_origin": DATA_ORIGIN,
                        "dataset_name": DATASET_NAME,
                    }
                )
                next_id += 1
    return records


def write_processed(records, output_path=OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def build(raw_dir=RAW_DIR, output_path=OUTPUT_PATH, events=None):
    records = load_records(raw_dir=raw_dir, events=events)
    write_processed(records, output_path=output_path)
    return records


def _summary(records):
    by_event, by_label = {}, {}
    for r in records:
        by_event[r["event"]] = by_event.get(r["event"], 0) + 1
        by_label[r["humanitarian_label"]] = by_label.get(r["humanitarian_label"], 0) + 1
    return by_event, by_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the processed HumAID real-text dataset.")
    parser.add_argument(
        "--events",
        nargs="*",
        default=None,
        help="Optional event folder names to include (default: all events found).",
    )
    args = parser.parse_args()

    records = build(events=args.events)
    by_event, by_label = _summary(records)
    print(f"Loaded {len(records)} real HumAID posts ({DATA_ORIGIN}, dataset={DATASET_NAME})")
    print(f"  source citation : {DATASET_CITATION}")
    print(f"  written to      : {OUTPUT_PATH} (git-ignored; contains real text)")
    print(f"  events ({len(by_event)}):")
    for event, n in sorted(by_event.items()):
        print(f"    {event}: {n}")
    print(f"  humanitarian labels ({len(by_label)}):")
    for label, n in sorted(by_label.items(), key=lambda kv: -kv[1]):
        print(f"    {label}: {n}")
