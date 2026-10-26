# Week 4 — Importing Real Tweets from a Scraper CSV

`src/scraped_tweets_loader.py` ingests a CSV of **real** tweets you exported
yourself (e.g. a Twitter advanced-search scraper) into the project's processed
layout, so real posts can be used instead of synthetic claims.

## Provenance (honest labelling)

| Origin | Meaning | Module |
|--------|---------|--------|
| `REAL_DATASET` | established research datasets | HumAID, CrisisMMD loaders |
| `LIVE_API` | official X API v2 | `x_ingestion.py` |
| **`SCRAPED_IMPORT`** | **a user-supplied scraped CSV** | **`scraped_tweets_loader.py`** |
| `SYNTHETIC` | hand-written test claims | project synthetic set |

Scraped data is tagged `data_origin = SCRAPED_IMPORT` — it is **never** relabelled
as `LIVE_API`, because it did not come from the official API. Web scraping may
conflict with the platform's Terms of Service; the loader only *reads a file you
provide* and does not scrape anything itself. Deciding whether the export was
obtained permissibly is the data provider's responsibility.

## Privacy / what is committed

The processed output `data/processed/scraped_tweets.csv` contains real post text
and public author handles, so it is **git-ignored**. The raw CSV under
`data/raw/scraped/` is git-ignored too. Nothing from this import is committed.

## Columns produced

`claim_id, text, created_at, author_handle, query_used, language,
image_reference, source_reference, data_origin, dataset_name`

Text gets technical-only normalization (strip a leading `RT @user:` and URLs,
collapse whitespace); content is never rewritten. Rows are de-duplicated on
`Post_ID`. The loader tolerates column-name variations across scraper exports.

## Topic check (important)

The loader reports how many posts mention an environmental-hazard keyword
(wildfire, flood, hurricane, evacuate, smoke, …) and can keep only those with
`--disaster-only`.

The sample CSV provided (`Twitter Advanced Search Scraper 2 (1).csv`) was
collected with the query `(depressed)` — **1,631 real tweets, but only 15 mention
any hazard keyword, and those are almost all metaphorical** ("your sex is on
fire", "that song is fire"). It therefore has **no disaster locations** for the
geographic-extraction → geocoding → NASA FIRMS pipeline to verify. It validates
the loader format but is not a usable disaster corpus.

To get real tweets this project can actually use end-to-end, re-export the scrape
with **disaster queries**, e.g.:

```
California wildfire
evacuation wildfire
Maui fire
Paradise wildfire
flood evacuation
```

Then rebuild:

```bash
python src/scraped_tweets_loader.py --csv data/raw/scraped/<your_export>.csv --disaster-only
```

## Note on "real vs synthetic"

The project already contains **43,409 real disaster tweets** from HumAID
(`data_origin = REAL_DATASET`). The synthetic set is retained only as a
controlled baseline for the Week 4 real-vs-synthetic comparison (its headline
result — real text scores far lower than synthetic — depends on keeping both).
Real data is the primary evaluation source; synthetic is not presented as tweets.
