# Week 4 — Environmental / Disaster Subset (HumAID)

A reproducible, focused subset of the HumAID real-text corpus for environmental
disaster experiments. This document records **how** the subset is defined so the
summary can be regenerated exactly.

## What this is

- A **tracked, aggregate-only** summary:
  `data/week4_environmental_subset_summary.csv`.
- Built by `src/week4_environmental_subset.py` from the **local, git-ignored**
  full processed dataset `data/processed/humaid_claims.csv`
  (produced by `src/humaid_loader.py`).
- The full processed HumAID dataset is **not modified** by this step, and **no
  real tweet text is copied** into the tracked summary — only counts and rates.

## Filtering criteria

1. **Start from the full processed HumAID corpus** (43,409 posts, 13 events).
2. **Derive an `event_type`** from each event's folder name by keyword:
   `wildfire → wildfire`, `flood → flood`, `hurricane → hurricane`,
   `cyclone → cyclone`, `typhoon → typhoon`, `earthquake → earthquake`.
3. **Keep only environmental (weather / climate-driven) hazards** —
   `event_type ∈ {wildfire, flood, hurricane, cyclone, typhoon}`.
4. **Exclude geophysical events** — HumAID's earthquakes
   (`ecuador_earthquake_2016`, `italy_earthquake_aug_2016`,
   `kaikoura_earthquake_2016`, `puebla_mexico_earthquake_2017`) are left out:
   they are not part of this project's environmental hazard taxonomy (wildfire,
   flood, drought, air quality, deforestation) and are not verifiable with the
   fire/thermal satellite evidence (NASA FIRMS) the pipeline uses.

This leaves **9 events** (4 earthquake events removed).

## Columns

| Column | Meaning |
|--------|---------|
| `event` | HumAID event folder name (e.g. `greece_wildfires_2018`) |
| `event_type` | derived hazard type (wildfire / flood / hurricane / cyclone) |
| `number_of_posts` | posts in that event (from the local full corpus) |
| `posts_with_detected_location` | posts where spaCy tagged ≥1 GPE/LOC/FAC entity |
| `location_extraction_rate` | `posts_with_detected_location / number_of_posts` (%) |

**Important:** a "detected location" is an **extraction** signal only — it means a
place was *mentioned* and tagged by the NER model. It is **not** a confirmed
geolocation, and HumAID posts are **not** geotagged. No location ground-truth
labels exist for this corpus, so this subset reports extraction statistics only
(no exact-match or missing rates are claimed here).

## Results (actual)

9 environmental events · 36,435 posts · 16,528 with a detected location
(**45.4%** overall extraction rate).

| event | event_type | number_of_posts | posts_with_detected_location | location_extraction_rate |
|-------|------------|----------------:|-----------------------------:|-------------------------:|
| canada_wildfires_2016 | wildfire | 2242 | 1072 | 47.8 |
| cyclone_idai_2019 | cyclone | 3933 | 1614 | 41.0 |
| greece_wildfires_2018 | wildfire | 1515 | 867 | 57.2 |
| hurricane_harvey_2017 | hurricane | 9112 | 3255 | 35.7 |
| hurricane_irma_2017 | hurricane | 9399 | 4163 | 44.3 |
| hurricane_maria_2017 | hurricane | 7278 | 3359 | 46.2 |
| hurricane_matthew_2016 | hurricane | 1654 | 1386 | 83.8 |
| maryland_floods_2018 | flood | 742 | 620 | 83.6 |
| srilanka_floods_2017 | flood | 560 | 192 | 34.3 |

Extraction rate varies widely by event (34% – 84%), consistent with the Week 4
finding that place-name visibility in real crisis text is highly inconsistent.
These are extraction counts on the full per-event corpus and are distinct from
the labelled 100-post exact-match evaluation in `week4_real_vs_synthetic.csv`.

## Reproduce

```bash
python src/humaid_loader.py              # (re)build the local git-ignored full corpus
python src/week4_environmental_subset.py # -> data/week4_environmental_subset_summary.csv
```
