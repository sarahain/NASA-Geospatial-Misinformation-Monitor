# Week 5 — Official X API Collection Pipeline

A safe, reproducible, budget-aware pipeline to collect ~2,000 **real, relevant**
disaster/environment X posts and feed them into the existing NER → geocoding →
NASA FIRMS pipeline. **No paid collection runs until you provide a token and
explicitly authorize it** (`--confirm-paid-run`).

## 1. Why the official X API (and not scraping)

The collector uses only the **official X API v2** endpoints:
- recent search (`/2/tweets/search/recent`) for current posts, and
- tweet lookup (`/2/tweets?ids=...`) for rehydrating known IDs.

Scraping x.com is **not** used, and no unofficial scraping package is used. Web
scraping violates X's Terms of Service and produces data of uncertain provenance;
the official API gives documented fields, stable IDs, and ToS-compliant access.

## 2. How credentials are protected

- Read **only** from the `X_BEARER_TOKEN` environment variable (or `.env`, which is
  git-ignored). `.env.example` is a token-free template.
- The token value is **never printed** — the dry run reports only
  `X_BEARER_TOKEN detected: YES/NO`.
- The token is **never logged**: `logs/x_collection_errors.log` records timestamp,
  endpoint, query, HTTP status, and error type only (no headers, no credentials).
- `.env`, `.env.*`, `secrets/`, `credentials/`, and `logs/` are git-ignored.

## 3. Safe-by-default guardrails (so you don't waste budget)

- `--target` defaults to **10** and never defaults higher.
- Any `--target` greater than 10 **requires** `--confirm-paid-run`; otherwise the
  script prints a refusal and exits without any API request.
- `--dry-run` validates configuration and lists the queries that *would* run, but
  makes **no** API call.
- Before a real run it prints the requested target, currently saved unique posts,
  and remaining target, plus:
  `Check current X Developer billing before running paid collection.`
  (No dollar estimate — X pricing changes.)
- `--candidate-limit` and `--max-pages` cap how much is read.

## 4. How duplicate posts are removed

Posts are de-duplicated by `post_id`. If the same post is returned by several
queries, **one** record is kept and every originating query is recorded in
`matched_queries` (pipe-separated). The `--target` counts **unique** posts, not
raw API rows.

## 5. How checkpointing works

Progress is saved to `data/processed/x_collection_checkpoint.json` after **every
page/batch**: seen post IDs, relevant IDs, per-query pagination tokens, counts, and
start time. If collection stops (rate limit, network error, API error, Ctrl-C),
re-running resumes from the checkpoint without re-collecting already-seen IDs. The
raw candidate and relevant CSVs are flushed to disk on each page too.

## 6. Why not all posts contain geographic metadata

Most X posts are **not** geotagged. The API returns `geo.place_id` / coordinates
only when the author attached a place, which is rare. We store `geo_place_id` and
`geo_coordinates` **only when present**, never inferring them. A mentioned place
name (in the text) is handled later by NER/geocoding and is **not** the same as a
confirmed GPS location.

## 7. How real text stays local and git-ignored

Collected text lands in git-ignored files under `data/processed/`
(`x_raw_candidates.csv`, `x_relevant_posts.csv`, `x_rehydrated_posts.csv`,
checkpoint) and `data/week5_relevance_review_sample.csv`. Only **text-free
aggregate** CSVs are tracked (`week5_x_collection_summary.csv`,
`week5_relevance_metrics.csv`, `week5_x_nasa_ready_summary.csv`).

## 8. How posts feed into NER, geocoding, and NASA evidence

After collection:
1. `src/week5_x_ner.py` runs the existing Week 3 spaCy extraction +
   `location_resolution` (geocoding) over the RELEVANT posts →
   `data/processed/week5_x_ner_results.csv` (detected/primary/normalized location,
   lat/long, confidence, ambiguity). Raw text is never modified; raw and processed
   files are kept separate.
2. It writes `data/week5_x_nasa_ready_summary.csv` — posts that are relevant, have a
   resolvable location, and a usable timestamp are counted as *potentially*
   NASA-ready (an eligibility count, **not** a claim that satellite verification
   happened).
3. Those records can then flow into the existing NASA FIRMS/MODIS/VIIRS comparison.

## 9. Relevance filtering (Stage A + Stage B)

The collector optimizes for **research relevance**, not raw post count — a smaller
useful set beats a large noisy one. Keyword presence alone is never treated as
relevance.

- **Stage A — query design** (`config/x_queries.txt`): multiple narrow boolean
  queries (hazard term + impact/evacuation/smoke context), not one broad "fire"
  query. Optional `lang:en`, `-is:retweet`, and short exclusion phrases
  (`config/x_exclude_terms.txt`) where the API supports them.
- **Stage B — local relevance filter** (`src/relevance_filter.py`): every post is
  classified locally (no API cost) as `RELEVANT` / `POSSIBLY_RELEVANT` /
  `IRRELEVANT` with a transparent `relevance_score` built from hazard, location,
  impact, emergency, smoke, and a figurative penalty. Figurative uses ("this song
  is fire", "the team is on fire") are rejected. Only `RELEVANT` posts enter the
  final target set; `--candidate-limit` bounds how many raw candidates are read.

`src/week5_summaries.py` then writes the relevance metrics, the collection
summary, and a ≤50-row manual-review sample, and runs an acceptance check that
recommends **STOP** (improve queries) if relevance retention is very low — so paid
reads are not wasted on poor queries.

## 10. HumAID rehydration vs collecting current posts

- **Rehydration** (`--mode rehydrate`): starts from historical HumAID tweet **IDs**
  and retrieves their current text by official id-lookup, merging HumAID
  event/label metadata. Output is `data_origin = REAL_DATASET_HYDRATED`. Deleted/
  private IDs are recorded with status `DELETED_OR_PRIVATE` / `UNAVAILABLE` and
  get **no** text. Useful to reconstruct a labelled historical set, but subject to
  tweet availability years later.
- **Search** (`--mode search`): collects **current** public posts matching the
  disaster queries. Output is `data_origin = LIVE_API`. Useful for fresh events,
  but has no pre-existing humanitarian labels.

## Commands (run only when you choose to)

```bash
# 0. One-time: provide your token (never commit it)
cp .env.example .env    # then edit .env to add X_BEARER_TOKEN, or:
export X_BEARER_TOKEN=...

# 1. Dry run — no API request, token value never printed
python src/x_ingestion.py --mode search --target 2000 --dry-run

# 2. Safe 10-post validation (smallest paid test)
python src/x_ingestion.py --mode search --target 10 --confirm-paid-run

# 3. 100-post run
python src/x_ingestion.py --mode search --target 100 --confirm-paid-run

# 4. Full 2,000 unique RELEVANT posts (reads more candidates as needed)
python src/x_ingestion.py --mode search --target 2000 --candidate-limit 4000 --confirm-paid-run

# 5. (Alternative) HumAID rehydration
python src/x_ingestion.py --mode rehydrate --input data/week3_location_test.csv --target 2000 --confirm-paid-run

# 6. After collection: aggregates + NER/geocoding
python src/week5_summaries.py
python src/week5_x_ner.py
```
