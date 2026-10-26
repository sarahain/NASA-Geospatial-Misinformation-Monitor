"""Load REAL tweets from an exported "advanced search scraper" CSV.

This ingests a CSV of **real** tweets you already exported (e.g. from a Twitter
advanced-search scraper) into the project's processed-data layout, so real posts
can be used in place of synthetic claims.

Provenance / honesty
--------------------
These rows are tagged::

    data_origin  = SCRAPED_IMPORT     # NOT LIVE_API, NOT REAL_DATASET
    dataset_name = twitter_advanced_search_scrape

``SCRAPED_IMPORT`` is a **distinct, honest** origin: the project's official-API
path is ``src/x_ingestion.py`` (``data_origin = LIVE_API``), and research
datasets are ``REAL_DATASET``. Data obtained by scraping web pages is neither, so
it is never relabelled as API data. Scraping may conflict with the platform's
Terms of Service; that is the data provider's (your) responsibility, and this
loader only *reads a file you supply* -- it does not scrape anything itself.

Privacy / licensing
-------------------
The processed output contains real post text and public author handles, so it is
**git-ignored** (under ``data/processed/``) and never committed. The raw CSV is
kept under ``data/raw/`` (also git-ignored).

Topic note
----------
The tweets are used exactly as supplied; this loader does not invent or alter
content. If the export was collected with an off-topic query (e.g. ``depressed``)
it will contain no disaster locations -- use ``--disaster-only`` to keep only
posts whose text mentions an environmental-hazard keyword, or re-export with
disaster queries (``California wildfire``, ``evacuation flood`` ...).

Usage
-----
    python src/scraped_tweets_loader.py                 # default CSV under data/raw/scraped/
    python src/scraped_tweets_loader.py --csv path.csv  # any scraper CSV
    python src/scraped_tweets_loader.py --disaster-only  # keep only hazard-related posts
"""
import argparse
import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "raw" / "scraped" / "twitter_advanced_search_scrape.csv"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = PROCESSED_DIR / "scraped_tweets.csv"

DATA_ORIGIN = "SCRAPED_IMPORT"
DATASET_NAME = "twitter_advanced_search_scrape"

# Logical field -> candidate source column names (first present wins). Tolerant
# to column-order / naming differences across scraper exports.
COLUMN_ALIASES = {
    "post_id": ["Post_ID", "post_id", "id"],
    "text": ["Tweet_Content", "text", "tweet_text", "content"],
    "created_at": ["UTC_Time", "created_at", "date", "time"],
    "author_handle": ["Author_Handle", "author_handle", "username", "handle"],
    "query_used": ["Query_Str", "query", "query_used"],
    "language": ["Language", "language", "lang"],
    "image_reference": ["Tweet_Image_URL", "image", "image_url"],
    "source_reference": ["Tweet_URL", "Post_URL", "url", "link"],
}

OUTPUT_FIELDS = [
    "claim_id",
    "text",
    "created_at",
    "author_handle",
    "query_used",
    "language",
    "image_reference",
    "source_reference",
    "data_origin",
    "dataset_name",
]

# Environmental-hazard keywords for the optional topical filter.
DISASTER_KEYWORDS = [
    "wildfire", "wild fire", "bushfire", "brush fire", "forest fire",
    "fire", "smoke", "evacuat", "flood", "flooding", "hurricane",
    "cyclone", "typhoon", "storm surge", "drought", "mudslide",
    "landslide", "earthquake",
]
_DISASTER_RE = re.compile("|".join(re.escape(k) for k in DISASTER_KEYWORDS), re.IGNORECASE)


def normalize_text(text):
    """Technical-only normalization: strip RT marker + URLs, collapse whitespace."""
    text = re.sub(r"^RT @\w+:\s*", "", str(text))
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _resolve_columns(fieldnames):
    present = set(fieldnames or [])
    resolved = {}
    for logical, candidates in COLUMN_ALIASES.items():
        for cand in candidates:
            if cand in present:
                resolved[logical] = cand
                break
    return resolved


def is_disaster_related(text):
    return bool(_DISASTER_RE.search(text or ""))


def load_records(csv_path=DEFAULT_CSV, disaster_only=False):
    """Read a scraper CSV into processed records (real tweet text preserved)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise SystemExit(
            f"Scraped tweet CSV not found at {csv_path}.\n"
            "Place your exported CSV there, or pass --csv <path>."
        )

    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = _resolve_columns(reader.fieldnames)
        if "text" not in cols:
            raise SystemExit(
                f"No recognizable tweet-text column in {csv_path} "
                f"(looked for {COLUMN_ALIASES['text']})."
            )

        records = []
        seen = set()
        next_id = 1
        for row in reader:
            text = normalize_text(row.get(cols["text"], ""))
            if not text:
                continue
            post_id = (row.get(cols.get("post_id", ""), "") or "").strip()
            key = post_id or text
            if key in seen:
                continue
            seen.add(key)
            if disaster_only and not is_disaster_related(text):
                continue
            records.append(
                {
                    "claim_id": f"scraped_{next_id}",
                    "text": text,
                    "created_at": (row.get(cols.get("created_at", ""), "") or "").strip(),
                    "author_handle": (row.get(cols.get("author_handle", ""), "") or "").strip(),
                    "query_used": (row.get(cols.get("query_used", ""), "") or "").strip(),
                    "language": (row.get(cols.get("language", ""), "") or "").strip(),
                    "image_reference": (row.get(cols.get("image_reference", ""), "") or "").strip(),
                    "source_reference": (row.get(cols.get("source_reference", ""), "") or "").strip(),
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


def build(csv_path=DEFAULT_CSV, output_path=OUTPUT_PATH, disaster_only=False):
    records = load_records(csv_path=csv_path, disaster_only=disaster_only)
    write_processed(records, output_path=output_path)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load real tweets from a scraper CSV export.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to the scraper CSV export.")
    parser.add_argument(
        "--disaster-only",
        action="store_true",
        help="Keep only posts whose text mentions an environmental-hazard keyword.",
    )
    args = parser.parse_args()

    all_records = load_records(csv_path=args.csv, disaster_only=False)
    disaster_records = [r for r in all_records if is_disaster_related(r["text"])]
    kept = build(csv_path=args.csv, output_path=OUTPUT_PATH, disaster_only=args.disaster_only)

    print(f"Loaded {len(all_records)} real tweets ({DATA_ORIGIN}, dataset={DATASET_NAME})")
    print(f"  source CSV : {args.csv}")
    print(f"  written to : {OUTPUT_PATH} (git-ignored; real text + handles)")
    print(f"  disaster-related by keyword: {len(disaster_records)} / {len(all_records)}")
    print(f"  rows written ({'disaster-only' if args.disaster_only else 'all'}): {len(kept)}")
    if len(disaster_records) == 0:
        print(
            "\n  NOTE: none of these tweets mention an environmental-hazard keyword, so"
            "\n  there are no disaster locations for the geospatial pipeline to verify."
            "\n  Re-export the scrape with disaster queries (e.g. 'California wildfire')."
        )
