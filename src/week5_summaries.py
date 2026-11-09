"""Generate Week 5 collection + relevance aggregate summaries (tracked, no text).

Reads the raw candidate posts collected by ``src/x_ingestion.py`` and the
collection checkpoint, and produces:

    data/week5_x_collection_summary.csv   (tracked; NO post text)
    data/week5_relevance_metrics.csv      (tracked; NO post text)
    data/week5_relevance_review_sample.csv (git-ignored; HAS post text; <= 50 rows)

All relevance figures are recomputed transparently with ``relevance_filter`` from
the stored candidate text (fully offline, no network, no API credits). If no
collection has happened yet, this refuses to write fake results.
"""
import csv
import random
from collections import Counter
from pathlib import Path

import relevance_filter
import x_ingestion

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

RAW_CANDIDATES_PATH = x_ingestion.RAW_CANDIDATES_PATH
CHECKPOINT_PATH = x_ingestion.CHECKPOINT_PATH
COLLECTION_SUMMARY_PATH = x_ingestion.COLLECTION_SUMMARY_PATH
RELEVANCE_METRICS_PATH = x_ingestion.RELEVANCE_METRICS_PATH
REVIEW_SAMPLE_PATH = x_ingestion.REVIEW_SAMPLE_PATH

REVIEW_SAMPLE_SIZE = 50
REVIEW_SEED = 42

COLLECTION_SUMMARY_FIELDS = [
    "requested_target", "unique_posts_collected", "duplicate_posts_removed",
    "queries_used", "collection_start", "collection_end",
    "posts_with_location_entities", "posts_with_place_metadata",
    "posts_with_exact_coordinates", "language_distribution",
]
RELEVANCE_METRICS_FIELDS = [
    "raw_candidates", "relevant_posts", "possibly_relevant", "irrelevant_posts",
    "relevance_retention_rate", "posts_with_location", "posts_with_event_name",
    "posts_with_emergency_terms", "posts_with_damage_terms",
    "figurative_posts_rejected", "duplicate_posts_removed",
]
REVIEW_SAMPLE_FIELDS = [
    "post_id", "text", "relevance_label", "relevance_score", "query_used",
    "detected_location", "reason_kept_or_rejected",
]


def _load_candidates(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _classify_all(candidates):
    return [(row, relevance_filter.classify(row.get("text", ""))) for row in candidates]


def build_relevance_metrics(classified, duplicate_posts_removed=0):
    raw = len(classified)
    labels = Counter(r["relevance_label"] for _, r in classified)
    relevant = labels.get("RELEVANT", 0)
    retention = round(relevant / raw, 3) if raw else 0.0
    return {
        "raw_candidates": raw,
        "relevant_posts": relevant,
        "possibly_relevant": labels.get("POSSIBLY_RELEVANT", 0),
        "irrelevant_posts": labels.get("IRRELEVANT", 0),
        "relevance_retention_rate": retention,
        "posts_with_location": sum(1 for _, r in classified if r["has_location"]),
        "posts_with_event_name": sum(1 for _, r in classified if r["has_event_name"]),
        "posts_with_emergency_terms": sum(1 for _, r in classified if r["has_emergency_terms"]),
        "posts_with_damage_terms": sum(1 for _, r in classified if r["has_damage_terms"]),
        "figurative_posts_rejected": sum(
            1 for _, r in classified
            if r["is_figurative"] and r["relevance_label"] == "IRRELEVANT"
        ),
        "duplicate_posts_removed": duplicate_posts_removed,
    }


def build_collection_summary(candidates, classified, checkpoint):
    unique = len(candidates)
    retrieved = checkpoint.get("candidates_retrieved", unique) or unique
    duplicate_removed = max(0, retrieved - unique)
    langs = Counter((c.get("lang", "") or "unknown") for c in candidates)
    lang_dist = ";".join(f"{k}:{v}" for k, v in sorted(langs.items(), key=lambda kv: -kv[1]))
    return {
        "requested_target": checkpoint.get("target", ""),
        "unique_posts_collected": unique,
        "duplicate_posts_removed": duplicate_removed,
        "queries_used": len(checkpoint.get("queries_used", []) or []),
        "collection_start": checkpoint.get("collection_start", ""),
        "collection_end": checkpoint.get("last_updated", ""),
        "posts_with_location_entities": sum(
            1 for _, r in classified if r["has_location"] or r["has_event_name"]
        ),
        "posts_with_place_metadata": sum(1 for c in candidates if c.get("geo_place_id")),
        "posts_with_exact_coordinates": sum(1 for c in candidates if c.get("geo_coordinates")),
        "language_distribution": lang_dist,
    }, duplicate_removed


def build_review_sample(classified, size=REVIEW_SAMPLE_SIZE, seed=REVIEW_SEED):
    """Sample up to ``size`` candidates (stratified-ish across labels) for review."""
    rng = random.Random(seed)
    buckets = {"RELEVANT": [], "POSSIBLY_RELEVANT": [], "IRRELEVANT": []}
    for row, res in classified:
        buckets.get(res["relevance_label"], buckets["IRRELEVANT"]).append((row, res))
    picked = []
    per_bucket = max(1, size // 3)
    for label, items in buckets.items():
        rng.shuffle(items)
        picked.extend(items[:per_bucket])
    rng.shuffle(picked)
    picked = picked[:size]
    out = []
    for row, res in picked:
        out.append({
            "post_id": row.get("post_id", ""),
            "text": row.get("text", ""),
            "relevance_label": res["relevance_label"],
            "relevance_score": res["relevance_score"],
            "query_used": row.get("query_used", ""),
            "detected_location": relevance_filter.extract_location_hint(row.get("text", "")),
            "reason_kept_or_rejected": res["reason"],
        })
    return out


def _write(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def run(raw_path=RAW_CANDIDATES_PATH, checkpoint_path=CHECKPOINT_PATH):
    candidates = _load_candidates(raw_path)
    if not candidates:
        print(f"[week5_summaries] no candidates at {raw_path}. "
              "Run a collection first; not writing fake results.")
        return None

    checkpoint = x_ingestion.load_checkpoint(checkpoint_path)
    classified = _classify_all(candidates)

    collection_summary, dup_removed = build_collection_summary(candidates, classified, checkpoint)
    relevance_metrics = build_relevance_metrics(classified, duplicate_posts_removed=dup_removed)
    review_sample = build_review_sample(classified)

    _write(COLLECTION_SUMMARY_PATH, COLLECTION_SUMMARY_FIELDS, [collection_summary])
    _write(RELEVANCE_METRICS_PATH, RELEVANCE_METRICS_FIELDS, [relevance_metrics])
    _write(REVIEW_SAMPLE_PATH, REVIEW_SAMPLE_FIELDS, review_sample)

    return {
        "collection_summary": collection_summary,
        "relevance_metrics": relevance_metrics,
        "review_sample_rows": len(review_sample),
    }


def acceptance_check(relevance_metrics, min_retention=0.25):
    """Quality gate: recommend STOP if relevance retention is very low."""
    retention = relevance_metrics["relevance_retention_rate"]
    if relevance_metrics["raw_candidates"] == 0:
        return "NO_DATA", "No candidates collected yet."
    if retention < min_retention:
        return "STOP", (
            f"relevance retention {retention:.0%} is low (<{min_retention:.0%}). "
            "Improve queries before spending more paid API reads."
        )
    return "OK", f"relevance retention {retention:.0%} looks acceptable."


if __name__ == "__main__":
    out = run()
    if out:
        print("=== Week 5 collection summary ===")
        for k, v in out["collection_summary"].items():
            print(f"  {k}: {v}")
        print("\n=== Week 5 relevance metrics ===")
        for k, v in out["relevance_metrics"].items():
            print(f"  {k}: {v}")
        status, msg = acceptance_check(out["relevance_metrics"])
        print(f"\nAcceptance check: {status} -- {msg}")
        print(f"\nReview sample rows (git-ignored, has text): {out['review_sample_rows']}")
