# Week 4 Mentor Summary

## Main Goal
Move the project from relying primarily on synthetic claims to using **real
disaster-related social-media text** for evaluation, under three clearly-separated
data categories (`REAL_DATASET`, `LIVE_API`, `SYNTHETIC`). The research aim is to
measure whether geographic entity extraction behaves differently on realistic,
noisy crisis posts than on clean synthetic claims, without fabricating any text
or labels.

## What Was Implemented
- **HumAID real-text loader** (`src/humaid_loader.py`) → `data/processed/humaid_claims.csv`
  (`data_origin = REAL_DATASET`, `dataset_name = HumAID`). Loaded **43,409 real
  posts** across **13 events**; every post has an `original_text`, `event`, and
  original `humanitarian_label`. Text is only technically normalized (strip
  `RT @user:` + URLs), never rewritten.
- **CrisisMMD loader** (`src/crisismmd_loader.py`) — parser is implemented and
  unit-tested, but the CrisisMMD raw data is **not present locally**, so no
  CrisisMMD records have been produced yet. Not integrated into any downstream
  stage.
- **Optional X API ingestion** (`src/x_ingestion.py`, `data_origin = LIVE_API`) —
  official X API v2 only, credentials from `X_BEARER_TOKEN` env var. **No token
  was used; no live data has been retrieved.**
- **Real-vs-synthetic experiment** (`src/week4_real_vs_synthetic.py`) →
  `data/week4_real_vs_synthetic.csv` (aggregate metrics only).
- **Error analysis** (`docs/week4_real_post_error_analysis.md`).
- **Tests** (`tests/test_week4.py`), all offline.

## Key Results
From `data/week4_real_vs_synthetic.csv` (same spaCy extractor on both sets):

| Data category | Posts | Extraction rate | Exact / manual match | Ambiguous | Missing |
|---------------|------:|----------------:|---------------------:|----------:|--------:|
| `SYNTHETIC` (manual labels) | 100 | 80.0% | 85.0% | 4.0% | 9.0% |
| `REAL_DATASET` — HumAID (manual labels) | 100 | 64.0% | 52.0% | 4.0% | 29.0% |
| `REAL_DATASET` — HumAID (unlabelled sample, n=300) | 300 | 47.7% | NA | 5.0% | NA |

- HumAID processed corpus: **43,409 posts**, **13 events**, humanitarian labels
  present on all rows; **no geographic/geotag field** (location ground truth does
  not exist in this corpus).
- The unlabelled-corpus row reports **extraction statistics only**; match and
  missing rates are `NA` because no location ground truth exists (no labels
  invented).
- **Test suite: 33 passed** (full repo), no network required.

## Most Important Finding
The **same** extractor drops sharply from synthetic to real crisis text:
- Exact / manual match: **85% → 52%** (a 33-point gap).
- Extraction rate: **80% → 64%**.
- Missing-location rate: **9% → 29%** (roughly 3×).

Controlled synthetic claims substantially **overstate** how well geographic
extraction works. Evaluating only on synthetic data would have produced an
over-optimistic picture of the pipeline's weakest step.

## Why This Matters
Synthetic claims are written as one clean sentence that names a place in plain
text ("Wildfire spreading near Malibu tonight"). Real crisis posts rarely do
that: the place is inside a hashtag (`#yeg`, `#FortMcMurray`), an abbreviation
(`Ft.`, `B.C.`, `Wash.`), a nickname or misspelling (`Elliott City`), buried
among several locations, or simply absent because the post is about a
*consequence* (a fundraiser, a vigil) rather than the disaster's location. For a
geospatial misinformation verifier this is the crux: if the location step fails
on realistic language, every downstream satellite check is starved of input or
fed the wrong place. Measuring that gap honestly is a prerequisite for trusting
any verdict the system produces.

## Limitations
- **HumAID location labels are incomplete** — the only manual location ground
  truth is our 100-post annotated subset; the full 43,409-post corpus has no
  location labels.
- **Many posts contain no explicit geographic entity** — a large share of the
  29% missing rate is posts that are not *about* a place (relief efforts,
  sympathy, policy).
- **Real posts are noisy** — hashtags, abbreviations, vague references ("here",
  "the area"), reposts/aggregated news, emojis/symbols, and missing context all
  degrade extraction.
- **CrisisMMD is not yet locally available** — only a loader exists; no records
  produced, nothing integrated.
- **Live X ingestion depends on API credentials** — optional and inactive; no
  token was used, so no live data exists.
- **Real post text is not committed to GitHub** — HumAID permits sharing tweet
  IDs only; CrisisMMD/X content is likewise kept out of Git. `data/processed/`
  is git-ignored; only aggregate metrics (`data/week4_real_vs_synthetic.csv`) are
  tracked.
- HumAID posts are **not geotagged** in this corpus; an extracted location word
  is treated as claim content to be checked, never as a confirmed geolocation.

## Current Architecture
```
real/synthetic crisis post        [implemented: synthetic set + HumAID real text]
  -> crisis classification          [planned: categories are dataset-provided/manual, no classifier]
  -> geographic entity extraction   [implemented: spaCy baseline; transformer = experimental/optional]
  -> location resolution            [implemented: alias map + candidate ranking + ambiguity flags]
  -> geocoding                      [implemented: Nominatim primary; Google fallback optional]
  -> NASA evidence retrieval        [implemented: NASA FIRMS near-real-time]
  -> spatial/temporal matching      [implemented: Haversine distance + date window]
  -> multimodal evidence fusion     [planned: CrisisMMD images only referenced by path, not fused]
```
- **Implemented:** extraction → resolution → geocoding → FIRMS retrieval →
  spatial/temporal matching, plus the transparent evidence score.
- **Experimental:** transformer NER comparison (optional, from Week 3);
  live X ingestion code path (untested against the live API here).
- **Planned:** crisis classification as a model, CrisisMMD integration,
  multimodal fusion.

## 5-Minute Demo Order
1. `data/week4_real_vs_synthetic.csv` — the headline synthetic-vs-real table.
2. `docs/week4_real_post_error_analysis.md` — the "why" with trimmed real
   examples (hashtags, abbreviations, consequence-only posts).
3. `src/humaid_loader.py` module docstring — show `REAL_DATASET` / HumAID schema
   and licensing stance; optionally run `python src/humaid_loader.py` to show
   "43,409 posts / 13 events" live.
4. `src/week4_real_vs_synthetic.py` — show it uses one spaCy extractor on both
   sets and emits `NA` where labels don't exist (no fabricated numbers).
5. `.gitignore` — show `data/processed/` is ignored while the aggregate metrics
   CSV is tracked (licensing in practice).
6. Run `python -m pytest -q` — **33 passed**.
7. (Optional) `src/crisismmd_loader.py` + `src/x_ingestion.py` docstrings — show
   they exist as clean, optional entry points that degrade gracefully.

## Questions For My Mentor
1. HumAID has no location ground truth beyond our 100-post annotated subset — is
   it worth hand-annotating a larger stratified real sample, or should effort go
   toward a weak-supervision / silver-label approach for location?
2. Given 29% of real posts have no extractable location (consequence posts), should
   a **"locatable vs not"** gate run before extraction, and how should that
   interact with the misinformation verdict?
3. For the synthetic-vs-real gap: is a short domain adaptation / hashtag-and-
   abbreviation normalization layer an acceptable scope, or should we keep the
   extractor fixed and report the gap as the scientific result?
4. CrisisMMD adds images and damage-severity labels — is multimodal fusion in
   scope for this project, or should the image signal stay as future work while
   we focus on text + FIRMS?
5. NASA FIRMS is near-real-time (~2 months); most HumAID events are historical
   (2016–2019) and return no match. Should we prioritize FIRMS archive access, or
   pivot evaluation to recent events we can actually verify against satellites?
6. What evidence threshold / reporting standard would you want before any post is
   labeled beyond `INSUFFICIENT_EVIDENCE`, given absence of a satellite detection
   is not proof of falsity?
