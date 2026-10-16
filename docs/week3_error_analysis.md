# Week 3 — Error Analysis: Geographic Entity Extraction & Resolution

**Research question:** *How reliably can geographic locations be extracted and
resolved from short environmental misinformation / social-media claims?*

This document records the **recurring failure types** observed when running the
100-claim Week 3 evaluation set (`data/week3_location_test.csv`) through:

- **Method A** — spaCy baseline NER (`en_core_web_sm`), preserved from Weeks 1–2
- **Method B** — transformer NER (`dslim/bert-base-NER`), a *model comparison*
- **spaCy + location resolution** — NER → alias normalization → Nominatim
  (optional Google fallback) → candidate ranking → coordinates + ambiguity

All examples below are taken directly from the generated result CSVs. **No
evaluation output was edited to hide failures.** The claims are synthetic and
labelled `source_type = synthetic`; they are not presented as real tweets.

> Note on scoring: for the 11 claims that contain **no** location, correctly
> extracting nothing is scored `exact` (a true-negative). Headline exact-match
> rates therefore include those 11 claims; the "located-only" rates in the
> metrics summary exclude them.

---

## Recurring failure types

1. **Abbreviations** (`NJ`, `NYC`, `LA`) — missed by spaCy's small model.
2. **Ambiguous city names** — a place name that exists in several
   states/countries; the geocoder may pick the wrong one.
3. **Regions instead of cities** (`Northern California`, `Midwest`,
   `Pacific Northwest`, `Southwest`) — not cleanly geocodable to a single point.
4. **Multiple locations in one claim** — which is the *event* location?
5. **Locations spaCy fails to recognize** (`Lahaina`, `Amazon`,
   `Washington State`).
6. **Non-location words recognized / resolved as places** — e.g. the brand
   token "Amazon" resolving to an Amazon warehouse.
7. **Location missing from the claim** — correctly expected to yield nothing.
8. **Geocoder selecting the wrong city/state** — right name, wrong place.

---

## Worked examples

| # | Claim | Expected | Extracted (method) | Resolved location | Failure type | Possible improvement |
|---|-------|----------|--------------------|-------------------|--------------|----------------------|
| 1 | *Heavy flooding reported in NJ after record rainfall* | NJ | spaCy: *(none)* · transformer: **NJ** | New Jersey, United States | (1) abbreviation missed by spaCy | Use the transformer for extraction; keep the alias map (`NJ → New Jersey`). |
| 2 | *Entire community of Lahaina destroyed by wildfire* | Lahaina | spaCy: *(none)* · transformer: **Lahaina** | — | (5) spaCy fails on less-common town names | Transformer recovers these; consider `en_core_web_trf` for the spaCy branch. |
| 3 | *Trees being cleared across Washington State forests* | Washington State | spaCy: *(none)* · transformer: **Washington** | — | (5) spaCy misses "Washington State" | Transformer extracts it; handle "State" suffix in normalization. |
| 4 | *Massive deforestation happening in the Amazon rainforest* | Amazon | spaCy: *(none)* · transformer: **Amazon** | Amazon Fulfillment Center, Lakeville, Minnesota | (5) + (6) spaCy misses it; geocoder resolves "Amazon" to a warehouse | Add a biome/region gazetteer for environmental terms; down-rank commercial POIs. |
| 5 | *Fire crews responding to flames outside Paradise CA* | Paradise | spaCy: *(none)* · transformer: **Paradise** | — | (1)+(5) the "CA" token fused to "Paradise CA" confuses spaCy | Normalize trailing state abbreviations before NER. |
| 6 | *The whole town of Paradise is burning again* | Paradise | spaCy/transformer: **Paradise** | Paradise, **Clark County, Nevada** | (2)+(8) wrong Paradise — resolved to Nevada, not California | Use claim context (fire/wildfire → Paradise, CA) or require state disambiguation. |
| 7 | *Wildfires are tearing across Northern California again* | Northern California | spaCy/transformer: **Northern California** | *California Pets Shop… Bahrain* | (3)+(8) region resolved to an unrelated POI abroad | Detect region phrases and map to a bounding box / centroid instead of a point. Currently flagged `ambiguous` (confidence 0.0). |
| 8 | *Drought is devastating crops in the Midwest* | Midwest | spaCy/transformer: **Midwest** | Midwest, **Natrona County, Wyoming** | (3)+(8) region collides with a small town literally named "Midwest" | Region phrase list; prefer region polygons over same-named towns. |
| 9 | *Extreme drought gripping the entire Southwest* | Southwest | spaCy/transformer: **Southwest** | Southwest, **Cameroon** | (3)+(8) US region resolved to a region of Cameroon | Constrain the geocoder with a country/viewbox hint from claim context. |
| 10 | *Illegal logging clearing forests across Brazil* | Brazil | spaCy/transformer: **Brazil** | **Brasil** (Portuguese) | scoring artifact, not a real failure — correct country, localized name | Compare against geocoder country codes, not display strings. |
| 11 | *Deforestation spreading across the state of Rondonia* | Rondonia, Brazil | transformer: **Rondonia** | **Rondônia**, Região Norte, Brasil | scoring artifact — correct place, diacritics differ | Normalize accents / use place IDs for matching. |
| 12 | *Smoke from the wildfire near Malibu is reaching Los Angeles* | Malibu; Los Angeles | both: **Malibu**, Los Angeles | Malibu, Los Angeles County, CA | (4) multiple locations — primary chosen as **Malibu** via `after_preposition:near` | Working as intended; record all candidates + the selection reason. |
| 13 | *Wildfire smoke from Canada is drifting into New York City* | Canada; New York City | both: **Canada**, New York City | *(flagged ambiguous)* | (4) two locations, no locating preposition → `first_of_multiple_ambiguous` | Event-type reasoning (smoke *source* vs *impact*) to pick the primary. |
| 14 | *Massive wildfire burning in the hills* | *(none)* | *(none)* | — | (7) no location present — correct true-negative | Desired behaviour; the system does not invent a location. |
| 15 | *Flames spreading fast across the canyon north of the city* | *(none)* | *(none)* | — | (7) vague, no geographic entity — correct true-negative | Desired behaviour; such claims are unverifiable geographically. |

---

## Cross-cutting observations

- **spaCy's main weakness is recall on abbreviations and less-common /
  non-US place names** (examples 1–5): 9 of 100 claims are `missing` for the
  baseline, and the transformer recovers **all 8** that contain a real location.
- **The resolution step's main weakness is regions and ambiguous names**
  (examples 6–9): exact-match on the *resolved* string drops versus the raw NER
  string because the geocoder confidently returns a point for a region, often
  the wrong one. The system **flags these as `ambiguous`/low-confidence rather
  than hiding the uncertainty**, which is the intended behaviour.
- **Some "incorrect" geocodes are scoring artifacts** (examples 10–11):
  localized names (`Brasil`) and diacritics (`Rondônia`) are actually correct.
  String comparison understates true geocoding quality; place-ID / country-code
  matching would fix this.
- **No false-positive locations from the transformer** were observed on this set
  (transformer `incorrect_rate = 0`), i.e. failure type (6) came from the
  *geocoder*, not the NER model, except where a valid token (`Amazon`) maps to a
  commercial POI.

See `data/week3_model_comparison.csv`, `data/week3_geocoding_results.csv`, and
`data/week3_difficulty_analysis.csv` for the full numbers behind these examples.
