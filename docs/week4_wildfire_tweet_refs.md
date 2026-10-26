# Week 4 — Curated Wildfire Tweet References + Official-API "Local Tweets"

This covers two related, **compliant** additions for using real, current wildfire
tweets — and explains why a web scraper is deliberately **not** provided.

## 1. Curated wildfire tweet references (`wildfire_tweet_refs_loader.py`)

Input: a hand-curated spreadsheet (CSV/XLSX) whose `tweet` column holds **real
tweet URLs** — here, Jan-2025 California / Pacific Palisades / Eaton wildfire
posts (accounts like `ESA_EO`, `SkyNews`, `sentdefender`).

Output: `data/processed/wildfire_tweet_refs.csv` (git-ignored) with
`claim_id, tweet_id, author_handle, source_url, label_raw,
misinformation_likelihood, label_reliability, needs_api_hydration, data_origin,
dataset_name`.

Honesty notes:
- `data_origin = REAL_DATASET`, `dataset_name = curated_wildfire_tweet_refs`. It's
  a small **reference list** (16 unique tweets in the provided file), **not** an
  established benchmark.
- The spreadsheet's label columns are **not reliable**: `positive_or_negative` is
  raw model/console output (`"Device set to use cpu"`, `model.safetensors`
  progress bars, QA prompts), and `misinformation_likelihood` is empty. The raw
  cell is preserved as `label_raw` and flagged `label_reliability =
  unreliable_llm_output`. **No clean labels are invented.**
- The **tweet text is not stored**. It is obtained only by hydrating ids through
  the official X API (below) — never by scraping.

Run:
```bash
python src/wildfire_tweet_refs_loader.py --input data/raw/scraped/wildfire_tweet_refs.xlsx
```

## 2. Local, updated tweets via the official API (not a scraper)

You asked for a scraper to find "other local updated tweets." A scraper that pulls
tweets from X's web pages **violates X's Terms of Service** and contradicts this
project's own rule (*"use the official X API only; do not scrape X webpages"*), so
one is intentionally **not** included.

The compliant equivalent is built into `src/x_ingestion.py`:

- **Find local, recent wildfire tweets** — `--location`:
  ```bash
  export X_BEARER_TOKEN=...        # your own token; never hardcoded/committed
  python src/x_ingestion.py --location "Los Angeles" --max-results 50
  ```
  This builds an official recent-search query combining wildfire terms with the
  place, via `build_local_query()`. (The `place:` / `point_radius:` geo operators
  require elevated/Academic API access on some tiers; on basic access the
  free-text location term still works. We never scrape to bypass access limits.)

- **Hydrate the curated references' text** — `hydrate_tweet_ids()` fetches text for
  known tweet ids through the official id-lookup endpoint, so the
  `wildfire_tweet_refs.csv` ids can become usable text legitimately.

Both paths require your own `X_BEARER_TOKEN`; without it the module skips
gracefully. API content is written under the git-ignored `data/processed/`.

## Why references + API instead of scraped text?

- **Compliance:** official API only; no ToS-violating scraping in the repo.
- **Provenance:** real posts are either research datasets (`REAL_DATASET`), live
  API pulls (`LIVE_API`), a user-supplied scraped CSV (`SCRAPED_IMPORT`), or
  synthetic (`SYNTHETIC`) — never mislabelled.
- **Integrity:** unreliable LLM-derived labels are flagged, not trusted; empty
  fields stay empty; tweet text is hydrated from the source of record.
