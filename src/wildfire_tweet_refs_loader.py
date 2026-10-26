"""Load a curated list of REAL wildfire tweet *references* (URLs) from a spreadsheet.

Input is a small, hand-curated spreadsheet (CSV or XLSX) whose ``tweet`` column
contains **real tweet URLs** (e.g. the Jan-2025 California / Pacific Palisades /
Eaton wildfire posts). This loader extracts the verifiable part -- the tweet id,
author handle, and source URL -- into the project's processed layout.

What this is / isn't
--------------------
* The tweets are **real** (`data_origin = REAL_DATASET`), referenced by URL. This
  is a small **reference list**, not an established benchmark.
* The spreadsheet's label columns are **not reliable**: ``positive_or_negative``
  is a dump of raw model/console output (e.g. "Device set to use cpu",
  "model.safetensors: 100%...", QA prompts) and ``misinformation_likelihood`` is
  empty. The raw label cell is preserved verbatim in ``label_raw`` and flagged in
  ``label_reliability = unreliable_llm_output`` -- it is **not** treated as
  ground truth. No labels are fabricated or cleaned into fake ground truth.
* The post **text is not included**. Tweet text is obtained only by hydrating the
  ids through the **official X API** (``src/x_ingestion.py``) -- never by
  scraping. This mirrors the HumAID "ids are shareable, text is hydrated" model.

Privacy / git
-------------
Output ``data/processed/wildfire_tweet_refs.csv`` is under the git-ignored
``data/processed/`` directory; the raw spreadsheet under ``data/raw/`` is ignored
too.

Usage
-----
    python src/wildfire_tweet_refs_loader.py                     # default under data/raw/scraped/
    python src/wildfire_tweet_refs_loader.py --input path.xlsx   # any CSV/XLSX
"""
import argparse
import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "raw" / "scraped" / "wildfire_tweet_refs.xlsx"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = PROCESSED_DIR / "wildfire_tweet_refs.csv"

DATA_ORIGIN = "REAL_DATASET"
DATASET_NAME = "curated_wildfire_tweet_refs"

TWEET_URL_RE = re.compile(r"(?:x|twitter)\.com/([^/\s]+)/status/(\d+)", re.IGNORECASE)

# Candidate source column names.
TWEET_COL_CANDIDATES = ["tweet", "Tweet", "url", "URL", "Tweet_URL", "Post_URL", "link"]
LABEL_COL_CANDIDATES = ["positive_or_negative", "label", "sentiment"]
MISINFO_COL_CANDIDATES = ["misinformation_likelihood", "misinfo", "misinformation"]

OUTPUT_FIELDS = [
    "claim_id",
    "tweet_id",
    "author_handle",
    "source_url",
    "label_raw",
    "misinformation_likelihood",
    "label_reliability",
    "needs_api_hydration",
    "data_origin",
    "dataset_name",
]


def _read_rows(input_path):
    """Read a CSV or XLSX into a list of dict rows (string values)."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise SystemExit(
            f"Input spreadsheet not found at {input_path}.\n"
            "Place your curated tweet-URL file there, or pass --input <path>."
        )
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"reading {input_path.suffix} needs pandas+openpyxl: {exc}")
        df = pd.read_excel(input_path).astype(object)
        return [
            {k: ("" if pd.isna(v) else str(v)) for k, v in rec.items()}
            for rec in df.to_dict(orient="records")
        ], list(df.columns)
    with open(input_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def _pick(colnames, candidates):
    for c in candidates:
        if c in colnames:
            return c
    return None


def _clean_cell(value):
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def load_records(input_path=DEFAULT_INPUT):
    """Extract unique real tweet references from the spreadsheet.

    De-duplicates on ``tweet_id`` (the same URL may appear many times with
    different raw label dumps); the first occurrence's label cell is kept.
    """
    rows, colnames = _read_rows(input_path)
    tweet_col = _pick(colnames, TWEET_COL_CANDIDATES)
    if not tweet_col:
        raise SystemExit(
            f"No tweet-URL column found in {input_path} (looked for {TWEET_COL_CANDIDATES})."
        )
    label_col = _pick(colnames, LABEL_COL_CANDIDATES)
    misinfo_col = _pick(colnames, MISINFO_COL_CANDIDATES)

    records = []
    seen = set()
    next_id = 1
    total_refs = 0
    for row in rows:
        m = TWEET_URL_RE.search(str(row.get(tweet_col, "")))
        if not m:
            continue
        total_refs += 1
        handle, tweet_id = m.group(1), m.group(2)
        if tweet_id in seen:
            continue
        seen.add(tweet_id)
        label_raw = _clean_cell(row.get(label_col, "")) if label_col else ""
        misinfo = _clean_cell(row.get(misinfo_col, "")) if misinfo_col else ""
        records.append(
            {
                "claim_id": f"wfref_{next_id}",
                "tweet_id": tweet_id,
                "author_handle": handle,
                "source_url": f"https://x.com/{handle}/status/{tweet_id}",
                "label_raw": label_raw,
                "misinformation_likelihood": misinfo,
                # The label column is raw LLM/console output, not trustworthy.
                "label_reliability": "unreliable_llm_output" if label_raw else "none",
                "needs_api_hydration": "true",  # text comes only from the official API
                "data_origin": DATA_ORIGIN,
                "dataset_name": DATASET_NAME,
            }
        )
        next_id += 1
    return records, total_refs


def write_processed(records, output_path=OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def build(input_path=DEFAULT_INPUT, output_path=OUTPUT_PATH):
    records, total_refs = load_records(input_path=input_path)
    write_processed(records, output_path=output_path)
    return records, total_refs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load real wildfire tweet references from a CSV/XLSX.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to the curated CSV/XLSX.")
    args = parser.parse_args()

    records, total_refs = build(input_path=args.input)
    print(f"Loaded {len(records)} unique real tweet references "
          f"(from {total_refs} URL rows) [{DATA_ORIGIN}, dataset={DATASET_NAME}]")
    print(f"  input      : {args.input}")
    print(f"  written to : {OUTPUT_PATH} (git-ignored)")
    print("  NOTE: labels are raw LLM/console output (flagged unreliable); "
          "misinformation_likelihood is empty.")
    print("  NOTE: tweet TEXT is not included -- hydrate ids via the official X API "
          "(src/x_ingestion.py), never by scraping.")
