"""Load real multimodal disaster posts from the CrisisMMD dataset.

Week 4 upgrade: alongside HumAID text-only posts, the project can use CrisisMMD
(Alam, Ofli & Imran, 2018 -- https://crisisnlp.qcri.org/crisismmd), whose posts
pair real tweet **text** with an associated **image** and several human
annotations. This loader reads the CrisisMMD annotation TSVs from a locally
downloaded copy and writes a processed dataset for experimentation, preserving:

    * the real post text (technically normalized only),
    * the event,
    * the informative label      (informative / not_informative),
    * the humanitarian label     (e.g. infrastructure_and_utility_damage),
    * the damage severity         (e.g. severe_damage; from the image task),
    * the image reference          (the relative image path), if available.

Data category
-------------
Every record is tagged::

    data_origin  = REAL_DATASET      # not LIVE_API, not SYNTHETIC
    dataset_name = CrisisMMD

Licensing / research integrity
------------------------------
CrisisMMD is released under CC-BY-NC-SA 4.0 and the underlying posts are also
subject to the social-media platform's terms. To stay on the safe side of both
the dataset license and platform redistribution restrictions, the processed
output ``data/processed/crisismmd_claims.csv`` **contains real post text and is
NOT committed to Git** (it is listed in ``.gitignore``), and the raw download is
never committed. Reproduce it locally by downloading CrisisMMD yourself.

The post text is only *technically* normalized (strip a leading ``RT @user:``
and URLs, collapse whitespace); it is never rewritten. Annotation labels are the
dataset's own -- none are fabricated. The image file itself is referenced by
path, not copied into the processed CSV.

Usage
-----
One-time: download "CrisisMMD v2.0" from https://crisisnlp.qcri.org/crisismmd
and extract it under ``data/raw/`` so that an ``annotations/`` folder of per-event
``*.tsv`` files exists (the loader searches for it), then::

    python src/crisismmd_loader.py

If the dataset is not present the loader prints clear download instructions and
exits without error-ing out the rest of the pipeline.
"""
import argparse
import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_ROOT = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = PROCESSED_DIR / "crisismmd_claims.csv"

DATA_ORIGIN = "REAL_DATASET"
DATASET_NAME = "CrisisMMD"
DATASET_CITATION = (
    "CrisisMMD (Alam, Ofli & Imran, 2018); https://crisisnlp.qcri.org/crisismmd"
)

OUTPUT_FIELDS = [
    "claim_id",
    "post_text",
    "event",
    "event_type",
    "informative_label",
    "humanitarian_label",
    "damage_severity",
    "image_reference",
    "data_origin",
    "dataset_name",
]

# Column-name candidates, tolerant to the original annotation layout and to
# exported variants. First present column wins.
COLUMN_ALIASES = {
    "tweet_id": ["tweet_id"],
    "post_text": ["tweet_text", "text"],
    "event": ["event_name", "event"],
    "informative_label": ["text_info", "informative"],
    "humanitarian_label": ["text_human", "humanitarian", "label"],
    "damage_severity": ["image_damage", "damage", "damage_severity"],
    "image_reference": ["image", "image_path"],
}

# Map a CrisisMMD event name to this project's coarse hazard taxonomy.
EVENT_TYPE_KEYWORDS = [
    ("wildfire", "wildfire"),
    ("flood", "flood"),
    ("earthquake", "earthquake"),
    ("hurricane", "hurricane"),
    ("cyclone", "cyclone"),
    ("typhoon", "typhoon"),
]

DOWNLOAD_HINT = (
    "CrisisMMD annotation files not found under {root}.\n"
    "Download 'CrisisMMD v2.0' from https://crisisnlp.qcri.org/crisismmd and\n"
    "extract it under data/raw/ so that a folder of per-event annotation TSVs\n"
    "(an 'annotations/' directory, e.g. annotations/california_wildfires.tsv)\n"
    "exists, then re-run this loader.\n\n"
    "CrisisMMD is CC-BY-NC-SA 4.0 and the posts are subject to platform terms,\n"
    "so the raw download and the processed text output are kept OUT of Git."
)


def normalize_text(text):
    """Minimal, technical-only normalization (mirrors humaid_loader)."""
    text = re.sub(r"^RT @\w+:\s*", "", str(text))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def event_to_type(event):
    low = str(event).lower()
    for keyword, hazard in EVENT_TYPE_KEYWORDS:
        if keyword in low:
            return hazard
    return "other"


def find_annotation_files(raw_root=RAW_ROOT):
    """Locate CrisisMMD per-event annotation TSVs under ``raw_root``.

    Prefers files inside an ``annotations`` directory (the original layout).
    Falls back to any ``*.tsv`` within a CrisisMMD-looking folder. Returns a
    sorted list of Paths (possibly empty).
    """
    if not raw_root.exists():
        return []
    annotation_dirs = [p for p in raw_root.rglob("annotations") if p.is_dir()]
    files = []
    for d in annotation_dirs:
        files.extend(d.glob("*.tsv"))
    if not files:
        # Fallback: any tsv under a path mentioning crisismmd.
        files = [
            p
            for p in raw_root.rglob("*.tsv")
            if "crisismmd" in str(p).lower()
        ]
    return sorted(set(files))


def _resolve_columns(fieldnames):
    """Map our logical fields to the actual header names present in a file."""
    present = {name: name for name in (fieldnames or [])}
    resolved = {}
    for logical, candidates in COLUMN_ALIASES.items():
        for cand in candidates:
            if cand in present:
                resolved[logical] = cand
                break
    return resolved


def _event_from_path(path):
    """Derive the event name from a CrisisMMD annotation filename."""
    return path.stem.replace("_final", "").strip()


def load_records(raw_root=RAW_ROOT):
    """Read CrisisMMD annotation TSVs into processed records.

    Returns ``(records, files)``. ``files`` is the list of annotation files that
    were read, so callers can tell "no dataset present" from "present but empty".
    """
    files = find_annotation_files(raw_root)
    records = []
    next_id = 1
    seen = set()
    for tsv in files:
        with open(tsv, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            cols = _resolve_columns(reader.fieldnames)
            if "post_text" not in cols:
                continue  # not a recognizable CrisisMMD annotation file
            file_event = _event_from_path(tsv)
            for row in reader:
                raw_text = row.get(cols["post_text"], "")
                text = normalize_text(raw_text)
                if not text:
                    continue
                tweet_id = (row.get(cols.get("tweet_id", ""), "") or "").strip()
                image_ref = (row.get(cols.get("image_reference", ""), "") or "").strip()
                # De-duplicate on (tweet_id, image) so multi-image tweets are kept.
                key = (tweet_id, image_ref) if tweet_id else (text, image_ref)
                if key in seen:
                    continue
                seen.add(key)
                event = (row.get(cols.get("event", ""), "") or "").strip() or file_event
                records.append(
                    {
                        "claim_id": f"crisismmd_{next_id}",
                        "post_text": text,
                        "event": event,
                        "event_type": event_to_type(event),
                        "informative_label": (row.get(cols.get("informative_label", ""), "") or "").strip(),
                        "humanitarian_label": (row.get(cols.get("humanitarian_label", ""), "") or "").strip(),
                        "damage_severity": (row.get(cols.get("damage_severity", ""), "") or "").strip(),
                        "image_reference": image_ref,
                        "data_origin": DATA_ORIGIN,
                        "dataset_name": DATASET_NAME,
                    }
                )
                next_id += 1
    return records, files


def write_processed(records, output_path=OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def build(raw_root=RAW_ROOT, output_path=OUTPUT_PATH):
    """Build the processed CrisisMMD dataset if the raw data is present.

    Returns the list of records (empty if the dataset is not available). Prints
    download instructions instead of raising when nothing is found, so this step
    degrades gracefully within the larger pipeline.
    """
    records, files = load_records(raw_root=raw_root)
    if not files:
        print(DOWNLOAD_HINT.format(root=raw_root))
        return []
    write_processed(records, output_path=output_path)
    return records


def _summary(records):
    by_event = {}
    for r in records:
        by_event[r["event"]] = by_event.get(r["event"], 0) + 1
    return by_event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the processed CrisisMMD real-text dataset.")
    parser.add_argument(
        "--raw-root",
        default=str(RAW_ROOT),
        help="Directory to search for the extracted CrisisMMD dataset (default: data/raw).",
    )
    args = parser.parse_args()

    records = build(raw_root=Path(args.raw_root))
    if records:
        by_event = _summary(records)
        print(f"Loaded {len(records)} real CrisisMMD posts ({DATA_ORIGIN}, dataset={DATASET_NAME})")
        print(f"  source citation : {DATASET_CITATION}")
        print(f"  written to      : {OUTPUT_PATH} (git-ignored; contains real text)")
        print(f"  events ({len(by_event)}):")
        for event, n in sorted(by_event.items()):
            print(f"    {event}: {n}")
