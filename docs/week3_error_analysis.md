# Week 3 — Error Analysis: Geographic Entity Extraction & Resolution

**Research question:** *How reliably can geographic locations be extracted and
resolved from short environmental misinformation / social-media claims?*

This document records the **recurring failure types** observed when running the
100-claim Week 3 evaluation set through the extraction and resolution pipeline.

## Data

The evaluation set is **100 real tweets** from the **HumAID** crisis dataset
(Alam, Ofli & Imran, 2021 — <https://crisisnlp.qcri.org/humaid_dataset>),
drawn from four environmental-hazard events:

| Category | Events | Geography |
|----------|--------|-----------|
| wildfire (52) | `canada_wildfires_2016`, `greece_wildfires_2018` | Fort McMurray / Alberta / B.C.; Athens / Attica / Mati |
| flood (48) | `srilanka_floods_2017`, `maryland_floods_2018` | Sri Lanka (Galle, Matara…); Ellicott City / Maryland |

The claim **text is real and unmodified** (only a leading `RT @user:` and URLs
were stripped). Per HumAID's license, the committed dataset
(`data/week3_location_test.csv`) stores **tweet IDs + our annotations only**;
the real text is rehydrated locally by `src/week3_build_dataset.py`. The
`expected_location` labels are **our** ground truth (assigned by reading each
tweet); `difficulty` is **computed objectively** from claim features
(`classify_difficulty`), not hand-tuned. 7 of 100 claims contain no extractable
location (true-negatives). **No evaluation output was edited to hide failures.**

## Headline numbers (real data)

| Method | Extraction | Exact | Partial | Incorrect | Missing | Geocoding |
|--------|-----------:|------:|--------:|----------:|--------:|----------:|
| A — spaCy baseline | 64% | 52% | 12% | 7% | 29% | 63% |
| B — transformer NER (`dslim/bert-base-NER`) | 82% | 58% | 16% | 15% | 11% | 78% |
| C — spaCy + resolution | 64% | 37% | 12% | 22% | 29% | 63% |

Geocoding over the 93 located claims: 31 correct, 11 partial, 21 incorrect,
37 unresolved (ambiguity rate 4%, unresolved 37%).

---

## Recurring failure types

1. **Hashtag-only locations** (`#yeg`, `#FortMcMurray`, `#Athens`, `#SriLanka`)
   — the place exists only inside a hashtag; spaCy's small model does not tag it.
2. **Informal nicknames / abbreviations** (`Fort. Muck.` for Fort McMurray,
   `B.C.`, `Ft. Macmurray`) — missed by NER or mis-resolved by the geocoder.
3. **Wrong entity chosen among several** — the first entity is a donor/other
   country, not the event location (e.g. *Norway … in #SriLanka* → `Norway`).
4. **Regions instead of cities** (`South-west`, `Northern …`) — not a single
   geocodable point; often resolved abroad.
5. **Multiple locations in one claim** (`#Mati #Athens #Ireland`) — which is the
   event site?
6. **Partial-span extraction** — spaCy returns `Fort` or `Fort St.` instead of
   the full place, which then geocodes to the wrong location.
7. **Geocoder picks the wrong place for a real name** (`Fort. Muck.` →
   *Bihar School of Yoga, India*).
8. **Localized / non-Latin display names** (`Greece` → `Ελλάς`, `Norway` →
   `Norge`, `Ireland` → `Éire`) — correct place, but string comparison scores it
   `incorrect` (a scoring artifact, not a real error).
9. **Location missing from the claim** — correctly expected to yield nothing.

---

## Worked examples (from the generated result CSVs)

| # | cid | Claim (real tweet, trimmed) | Expected | Extracted (spaCy) | Resolved | Failure type | Possible improvement |
|---|-----|------|----------|-------------------|----------|--------------|----------------------|
| 1 | 12 | *…If you're in the **#yeg** area, please consider helping. #ymmfire* | #yeg (Edmonton) | *(none)* | — | (1) hashtag-only | Expand hashtags / add a hashtag-aware tokenizer before NER. |
| 2 | 27 | *Vigil held in **#Athens** for #wildfire victims #GreeceFires* | #Athens | *(none)* | — | (1) hashtag-only | Same; many crisis tweets geo-tag only via hashtags. |
| 3 | 1 | *#NorthernAlbertaWildfires … destroyed **Ft. Macmurray*** | Ft. Macmurray | *(none)* | — | (2) abbreviation/misspelling | Normalize `Ft.`→`Fort`; fuzzy-match rare place spellings. |
| 4 | 2 | *…friends who live(d) in **Fort. Muck.*** | Fort McMurray | `Fort` (partial) | *Bihar School of Yoga, India* | (2)+(6)+(7) nickname → wrong country | Don't geocode 1-token fragments confidently; use claim context. |
| 5 | 53 | ***Norway** contributes … flood response in **#SriLanka*** | Sri Lanka | `Norway` (incorrect) | Norge | (3) donor chosen over event site | Prefer the hazard location (hashtag/`#SriLanka`) over donor entities. |
| 6 | 56 | *Despite rains in **South-west**, … 840,000 in #SriLanka hit by drought* | Sri Lanka | `South-west` (incorrect) | *…Soomaaliya (Somalia)* | (4) region resolved abroad | Treat region words as non-point; constrain by country context. |
| 7 | 32 | *Irish man confirmed dead in Greek wildfires **#Mati #Athens #Ireland*** | Mati; Athens | `Ireland` (incorrect) | Éire / Ireland | (5) multiple locations, picked the nationality | Rank by event context; down-weight nationality mentions. |
| 8 | 5 | *Evacuation order issued for fire burning near **Fort St. John**.* | Fort St. John | `Fort St.` (partial) | *Fort Street, Britannia Village* | (6) partial span → wrong place | Extend entity span; geocode the full phrase. |
| 9 | 82 | *Flash Flood Warning … **#Baltimore** & **#Howard** counties in Maryland* | Howard/Baltimore County | `Maryland` (incorrect vs county) | Maryland, US | (1)+(5) county-level missed | Parse "#X county" patterns; accept county-level matches. |
| 10 | 31 | *…fire-stricken **Greeks** … says Greek PM Alexis Tsipras* | Greece | `Greece` (exact) | **Ελλάς** | (8) localized name scored incorrect | Match by country code / place-id, not display string. |
| 11 | 87 | *Maryland's governor declared a state of emergency … **Elliott City*** | Elliott City | `Maryland` | Maryland, US | (2)+(3) misspelled city, state chosen | Fuzzy-match `Elliott`→`Ellicott`; prefer city over state. |
| 12 | 9 | ***B.C.** wildfire crosses into **Alberta** …* | British Columbia; Alberta | `B.C.` | British Columbia, Canada | (2)+(5) resolved correctly, scored incorrect at extraction only | Alias `B.C.`→British Columbia; credit multi-location matches. |
| 13 | 34 | *Lucy Neatby is hosting a Fund Raiser for the **Alberta** Wildfire victims.* | Alberta | *(none)* | — | spaCy misses a plain province name | Transformer recovers `Alberta`; consider `en_core_web_trf`. |

---

## Cross-cutting observations

- **spaCy's main weakness on real tweets is recall** (`missing = 29%`): it drops
  hashtag-embedded places, abbreviations, nicknames, and even some plain names
  (`Alberta`). The transformer cuts `missing` to 11% and raises extraction from
  64% → 82%.
- **But the transformer trades recall for precision on noisy text**
  (`incorrect = 15%` vs spaCy's 7%): on real tweets it tags more non-event or
  spurious spans, so "better" is not unconditional — exactly the kind of nuance a
  real dataset exposes that the earlier synthetic set did not.
- **Resolution is the hardest step on real data.** Only 31% of located claims
  geocode to a clearly correct place; 37% are unresolved (nothing clean to
  geocode) and 21% are `incorrect` — though several of those are **scoring
  artifacts** from localized display names (`Ελλάς`, `Norge`, `Éire`), so true
  geocoding quality is somewhat higher than 31%.
- **Difficulty tracks the model.** Exact-match (spaCy + resolution) is 55.6% on
  easy claims but only 25.5% on hard claims; geocoding success falls from 80% to
  43%. (The 8 `medium` "City, State" claims score 0% *exact* because the resolver
  returns the city and drops the state — they count as `partial`, not `exact`.)

See `data/week3_model_comparison.csv`, `data/week3_geocoding_results.csv`, and
`data/week3_difficulty_analysis.csv` for the full numbers behind these examples.
