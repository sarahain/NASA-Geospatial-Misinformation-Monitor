# NASA GLOBE-Based Geospatial Misinformation Monitor

An undergraduate research prototype that investigates whether location-based
environmental claims can be checked against real NASA Earth observations,
rather than being judged by text alone.

## Project Overview

Most misinformation classifiers rely heavily on language. They learn to spot
suspicious wording, tone, or source patterns, but they rarely check a claim
against the physical world. This project explores a different approach:
grounding location-tagged environmental claims in satellite observations.

The Week 2 prototype focuses on wildfire claims. A claim such as *"Wildfire
spreading near Malibu tonight"* names a place and an event. The pipeline
extracts the place, converts it to coordinates, and asks NASA FIRMS whether
satellites actually detected active fires near that location around that date.

## Problem

Language-only misinformation detection can be fooled by well-written false
claims and can misfire on poorly written true ones. Environmental claims are a
useful testbed because they often make checkable physical assertions ("there is
a fire here, now") that can be compared against satellite observations. The open
question is how far that grounding can actually be pushed with off-the-shelf NLP
and public NASA data.

## Research Significance

If even a coarse spatial-temporal check against satellite data can add useful
evidence to a text claim, it suggests a path toward verification systems that
are harder to fool with fluent language and that produce transparent,
inspectable evidence instead of an opaque score. It also surfaces the limits:
where geocoding fails, where satellites simply do not observe, and where absence
of evidence must not be mistaken for disproof.

## Research Questions

- **RQ1:** How accurately can a general-purpose NER model extract geographic
  entities from location-based environmental claims?
- **RQ2:** Does geocoding and contextual location resolution improve geographic
  entity accuracy over baseline spaCy NER?
- **RQ3:** Can satellite fire detections provide useful external evidence for
  verifying location-tagged wildfire claims?
- **RQ4:** How sensitive are verification results to spatial and temporal
  matching thresholds?
- **RQ5:** What limitations arise when satellite observations are used as
  evidence for misinformation adjudication?

### Week 3 Research Sub-Questions (geographic extraction)

- **W3-RQ1:** How accurately does baseline spaCy identify geographic entities in
  short environmental claims?
- **W3-RQ2:** Does a specialized or alternative NER method outperform
  general-purpose spaCy?
- **W3-RQ3:** How much does geographic resolution improve the final location
  accuracy after NER?
- **W3-RQ4:** Which types of geographic expressions cause the most errors?
- **W3-RQ5:** How does performance change between easy, medium, and difficult
  claims?

## Current Pipeline

```
Claim
  -> spaCy NER (baseline geographic entity)
  -> Location resolution (geocoder candidates + contextual ranking)
  -> Reverse geocoding / coordinates (Nominatim)
  -> NASA FIRMS (VIIRS NOAA-20 by default; MODIS / NOAA-21 compared separately)
  -> Spatial-temporal comparison (Haversine distance + date window)
  -> Transparent evidence score
  -> Verdict (SUPPORTED / INSUFFICIENT_EVIDENCE / LOCATION_AMBIGUOUS)
```

A verdict other than `SUPPORTED` does **not** mean a claim is fake. The absence
of a satellite detection is not proof of misinformation; it only means the
current evidence source did not support the claim. `CONTRADICTED` is defined in
the verdict set but is intentionally not emitted by the current conservative
logic.

## Week 2 Progress

- Synthetic wildfire claim corpus (~30 claims).
- Baseline spaCy NER with a simple quality evaluation.
- Second-stage location resolution kept **separate** from the baseline so the
  two can be compared.
- Nominatim geocoding with caching and rate limiting.
- NASA FIRMS integration (VIIRS NOAA-20) plus a MODIS vs VIIRS comparison.
- Transparent evidence score and structured evidence records.
- Threshold sensitivity experiment.
- Research-metrics summary, an architecture document, and one wildfire case
  study.
- Optional PostGIS schema and example spatial queries (not yet wired into the
  live pipeline).

## Week 3 — Geographic Entity Extraction Evaluation

Week 3 isolates and stress-tests the weakest-evaluated step of the pipeline:
**turning claim text into a geographic location.** It extends the project
cleanly without rebuilding any Week 1/2 component.

What was done:

- **100 real tweets were evaluated** from the **HumAID** crisis dataset
  (Alam, Ofli & Imran, 2021 — <https://crisisnlp.qcri.org/humaid_dataset>),
  across four environmental-hazard events: Canada 2016 and Greece 2018
  **wildfires** (52 claims) and Sri Lanka 2017 and Maryland 2018 **floods**
  (48 claims). The claims are real and unmodified (only a leading `RT @user:`
  and URLs stripped); they naturally include hashtag-embedded places, abbreviations,
  nicknames, regions, multiple-location claims, and claims with no location.
- **Research integrity / licensing:** HumAID's terms permit sharing **tweet IDs
  only**, not tweet text. The committed `data/week3_location_test.csv` therefore
  holds tweet IDs + our annotations (no text); the real text is rehydrated
  locally by `src/week3_build_dataset.py` into a git-ignored file. The
  `expected_location` labels are **our** ground truth; `difficulty` is **computed
  objectively** from claim features (not hand-tuned). The previous labelled
  synthetic set is preserved at `data/week3_location_test_synthetic.csv`.
- **spaCy was preserved as the baseline** (Method A, `en_core_web_sm`,
  unchanged). Its outputs are recorded as-is.
- **A second extraction method was evaluated** (Method B): the pretrained
  transformer NER `dslim/bert-base-NER`, behind a shared interface
  (`src/ner_methods.py`) that returns the same output format. This is a **model
  comparison**, not fine-tuning; Method B is optional and degrades gracefully.
- **Geographic resolution was applied** (`src/location_resolution.py`): a small
  alias map → geocoder candidates → contextual ranking → coordinates →
  `location_confidence` + `ambiguity_status` (`resolved` / `ambiguous` /
  `unresolved`). Multiple-location claims keep all detected locations, pick a
  `primary_location`, and record *why*.
- **Nominatim is the primary geocoder; Google is an optional fallback**
  (`GOOGLE_GEOCODING_API_KEY`, never hardcoded), used only when Nominatim is
  unresolved/ambiguous and only if strictly more confident. Requests are cached
  on disk (`data/geocode_cache.json`) to respect rate limits.
- **Ambiguous locations are tracked, not hidden** — regions and ambiguous names
  are flagged rather than silently resolved to a confident wrong point.
- **Error analysis was performed** (`docs/week3_error_analysis.md`, ~13 worked
  examples) and performance is broken down by difficulty.

### Week 3 Results (100 real HumAID tweets)

Computed from the models as-is (see `src/week3_evaluation.py`); nothing was
hand-corrected. 93 of the 100 claims contain a location; 7 contain none (for
those, correctly extracting nothing is scored as an exact true-negative).

| Method | Extraction | Exact | Partial | Incorrect | Missing | Geocoding success |
|--------|-----------:|------:|--------:|----------:|--------:|------------------:|
| A — spaCy baseline (NER only) | 64% | 52% | 12% | 7% | 29% | 63% |
| B — transformer NER (NER only) | 82% | 58% | 16% | 15% | 11% | 78% |
| C — spaCy + location resolution | 64% | 37% | 12% | 22% | 29% | 63% |

Methods A and B are scored on the raw NER span against the raw expected label;
Method C outputs a *normalized* location and is therefore scored against both
the raw and the `expected_normalized_location` label (a match against either
counts). A/B/C exact rates are over all 100 claims (including the 7 correct
no-location true-negatives); the located-only spaCy exact rate is ~48%.

Performance by difficulty (spaCy + resolution; difficulty is computed from claim
features — hashtag-only / abbreviation / region / multiple / no-location → hard):

| Difficulty | Claims | Exact match | Geocoding success |
|------------|-------:|------------:|------------------:|
| easy | 45 | 55.6% | 80.0% |
| medium | 8 | 0.0% | 87.5% |
| hard | 47 | 25.5% | 42.6% |

(The 8 `medium` claims are "City, State" cases where the resolver returns the
city and drops the state, so they score `partial`, not `exact`.)

Key takeaways:

- **W3-RQ1/RQ2:** The transformer NER **extracts more** than the spaCy baseline
  (82% vs 64% extraction; 11% vs 29% `missing`; 58% vs 52% exact), recovering
  hashtag-embedded places, abbreviations and nicknames the small spaCy model
  drops. **But on noisy real tweets it is also less precise** (15% vs 7%
  `incorrect`) — so it is not unconditionally better, a nuance the real data
  exposes that the synthetic set did not.
- **W3-RQ3:** Resolution is the hardest step on real data: only 31% of located
  claims geocode to a clearly correct place, 37% are unresolved, and some
  `incorrect` scores are artifacts of localized display names (`Ελλάς`, `Norge`,
  `Éire`). An honest, sobering result.
- **W3-RQ4:** Dominant error sources are hashtag-only locations and NER recall
  gaps, wrong-entity selection (donor vs event site), and the geocoder
  mis-resolving nicknames/regions (e.g. *Fort. Muck.* → a site in India).
- **W3-RQ5:** Accuracy is highest on easy claims (55.6% exact, 80% geocoding) and
  degrades sharply on hard claims (25.5% / 43%).

Five resolved wildfire claims were passed through the **existing** NASA FIRMS
pipeline (`src/week3_end_to_end.py` → `data/week3_end_to_end_examples.csv`) to
confirm the improved extraction feeds the larger project. These are 2016/2018
events, which fall outside the ~2-month FIRMS NRT window, so the current window
is queried and all return `NO_MATCH` — the honest outcome; no satellite match is
fabricated.

### Running Week 3

```bash
# One-time: download HumAID "Event wise dataset (set1)" from
# https://crisisnlp.qcri.org/humaid_dataset and extract so data/raw/events_set1/ exists.
python src/week3_build_dataset.py # -> rehydrates real tweet text locally (ids + annotations are committed)
python src/week3_evaluation.py    # -> week3_spacy_baseline / geocoding / comparison / difficulty / metrics_summary CSVs
python src/week3_end_to_end.py    # -> week3_end_to_end_examples.csv (uses FIRMS_MAP_KEY if set)
jupyter notebook notebooks/week3_location_evaluation.ipynb
```

## Dataset

`data/wildfire_claims.csv` contains ~30 **synthetic / demo** wildfire-related
claims written for this project. They are **not** real social media posts and
are not assertions about real events. They mix easy geographic cases (e.g.
"Malibu, California") with intentionally ambiguous ones (e.g. "near the border",
"in the mountains") to stress-test the NER and resolution steps.

Columns: `claim_id`, `text`, `date`, `expected_location`, `category`.

## Methods

- **NER:** spaCy `en_core_web_sm`, entity labels GPE / LOC / FAC. Evaluation
  labels each extraction `correct` / `partial` / `incorrect` / `missing` by
  normalized string comparison against an expected location.
- **Location resolution** (`src/location_resolution.py`): spaCy candidate +
  multiple geocoder candidates + light contextual ranking (boosts candidates
  whose region tokens also appear in the claim). Returns a confidence and
  candidate count; low confidence is treated as `LOCATION_AMBIGUOUS` downstream.
- **Evidence score** (`src/evidence.py`): a transparent weighted sum of a
  spatial score, temporal score, detection-strength score, and location
  confidence, in `[0, 1]`. It is a prototype *confidence* score, not a measure
  of scientific certainty. The exact formula is documented in the module.
- **Verification** (`src/verify.py`): Haversine distance with a configurable
  threshold (default 25 km).

## NASA Data Sources

- **NASA FIRMS** active fire / thermal anomaly detections (near-real-time Area
  API). VIIRS NOAA-20 is the default; VIIRS NOAA-21 and MODIS are compared in
  `src/sensor_comparison.py`. VIIRS senses at ~375 m, MODIS at ~1 km.
- NRT products cover roughly the last two months; historical events require the
  FIRMS archive (future work).

## Experiments

1. **NER quality** (`src/evaluate.py` -> `data/ner_evaluation.csv`): baseline
   extraction accuracy by outcome category.
2. **Location resolution A/B/C** (`src/location_experiment.py` ->
   `data/location_evaluation.csv`): baseline spaCy vs spaCy+Nominatim vs
   contextual resolver.
3. **MODIS vs VIIRS** (`src/sensor_comparison.py`): detections, nearest
   detection, FRP, and date coverage per sensor for one case.
4. **Threshold sensitivity** (`src/threshold_experiment.py`): match counts at
   5 / 10 / 25 / 50 km.
5. **Research metrics** (`src/research_metrics.py` ->
   `data/research_metrics.csv`): aggregate rates across the corpus.

## Preliminary Results

These are early, small-sample numbers from the synthetic corpus and will change
as the dataset grows. They are reported as-is, including a negative result.

- Baseline NER: ~53% exact, ~23% partial, ~20% missing, ~3% incorrect;
  extraction rate ~80%.
- Location resolution **did not** beat the baseline on exact string match
  (~43% vs ~53%): geocoding sometimes resolves ambiguous names confidently to
  the wrong place, and the contextual resolver did not improve on plain
  geocoding for this set. This is kept as an honest negative result.
- About 10% of claims were flagged `LOCATION_AMBIGUOUS` by the resolver.
- Satellite match rate on the synthetic claims is 0%, because their dates fall
  outside the FIRMS NRT window. On recent real California fire data, the
  threshold experiment shows match counts growing with the radius (e.g. a
  handful within 5 km rising to more within 50 km), illustrating that threshold
  choice is a research design decision rather than a fixed constant.

## Limitations

- The claim corpus is synthetic and small.
- NRT-only FIRMS retrieval; historical dates return nothing without the archive.
- Geocoding yields a single representative point, not a fire perimeter, and can
  mis-resolve ambiguous names.
- Absence of a detection never establishes that a claim is false.
- The evidence score is a transparent heuristic, not a validated model.
- PostGIS and LLM-based classification are not yet implemented.

## Future Work

- Load data into PostGIS and run the spatial queries from the live pipeline.
- NASA GLOBE citizen-science observations as an additional evidence layer.
- Larger, real and properly sourced claim corpora.
- Improved geographic entity resolution for ambiguous / informal place names.
- LLM-assisted contextual disambiguation and classification.
- FIRMS archive retrieval for historical events.
- Verification pipelines for the other taxonomy categories (flood, drought, air
  quality, deforestation).

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### NASA FIRMS API Key

FIRMS verification needs a free MAP_KEY from
<https://firms.modaps.eosdis.nasa.gov/api/>. The key is **not** stored in this
repo. Provide it through an environment variable:

```bash
export FIRMS_MAP_KEY=your_key_here
```

If `FIRMS_MAP_KEY` is not set, the pipeline still runs the NER and geocoding
steps and reports FIRMS verification as skipped.

## Running

```bash
python src/main.py                 # NER -> geocode -> FIRMS -> verify
python src/evaluate.py             # NER quality evaluation
python src/location_experiment.py  # baseline vs resolved location
python src/sensor_comparison.py    # MODIS vs VIIRS
python src/evidence.py             # structured evidence records + score
python src/research_metrics.py     # aggregate research metrics
```

The optional PostGIS schema and queries live in `database/`.

## Repository Structure

```
NASA-Geospatial-Misinformation-Monitor/
    README.md
    requirements.txt
    config/claim_taxonomy.json
    database/            schema.sql, queries.sql   (optional PostGIS)
    docs/architecture.md
    case_studies/wildfire_case_study.md
    data/                wildfire_claims.csv, week3_location_test.csv (inputs)
                         + generated CSVs (incl. week3_*.csv)
    src/                 ner, ner_methods, geocode, firms, verify,
                         location_resolution, location_experiment, evidence,
                         sensor_comparison, threshold_experiment,
                         research_metrics, main,
                         week3_evaluation, week3_end_to_end
    docs/architecture.md, week3_error_analysis.md
    notebooks/week2_demo.ipynb, week3_location_evaluation.ipynb
```

## Background and Influences

- **LIAR-PLUS** and **HealthVer** influenced the evidence-verification framing,
  where a claim is paired with supporting evidence rather than classified from
  text alone.
- **NASA FIRMS** provides MODIS/VIIRS active fire observations used as the
  physical evidence source.

## References

- LIAR-PLUS: <https://github.com/Tariq60/LIAR-PLUS>
- HealthVer: <https://github.com/sarrouti/HealthVer>
- NASA FIRMS: <https://firms.modaps.eosdis.nasa.gov/>
- MODIS data: <https://modis.gsfc.nasa.gov/data/>
- NASA GLOBE observations: <https://observer.globe.gov/get-data>
