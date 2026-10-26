# Week 4 — Error Analysis: Real Social-Media Posts vs Synthetic Claims

**Research question:** *Does geographic entity extraction perform differently on
realistic, noisy social-media language compared with controlled synthetic
claims?*

This document records the recurring issues observed when the **same** spaCy
geographic-entity extractor (the preserved Week 1/2 baseline) is run over two
clearly-separated data categories:

- **SYNTHETIC** — the project's 100 manually written test claims
  (`data/week3_location_test_synthetic.csv`, `data_origin = SYNTHETIC`). These
  are **not** tweets and are not assertions about real events.
- **REAL_DATASET** — 100 real disaster tweets from **HumAID**
  (Alam, Ofli & Imran, 2021 — <https://crisisnlp.qcri.org/humaid_dataset>,
  `data_origin = REAL_DATASET`, `dataset_name = HumAID`), with our manual
  location annotations as ground truth.

Numbers are reproduced by `src/week4_real_vs_synthetic.py`
(→ `data/week4_real_vs_synthetic.csv`). Real tweet **text** is kept out of Git
under HumAID's terms (only tweet IDs may be shared); the short excerpts below are
trimmed illustrative citations for error analysis, matching the Week 3 convention.

## Headline comparison

| Data category | Posts | Location extraction | Exact / manual match | Ambiguous | Missing |
|---------------|------:|--------------------:|---------------------:|----------:|--------:|
| SYNTHETIC (manual labels) | 100 | 80.0% | 85.0% | 4.0% | 9.0% |
| REAL_DATASET — HumAID (manual labels) | 100 | 64.0% | 52.0% | 4.0% | 29.0% |
| REAL_DATASET — HumAID (unlabelled corpus sample, n=300) | 300 | 47.7% | *NA* | 5.0% | *NA* |

The unlabelled-corpus row has **no ground-truth locations**, so — per the
project's research rule — only extraction statistics are reported and the match /
missing rates are `NA`. No labels were fabricated to fill them in.

**Answer to the research question: yes, markedly.** On the labelled sets the
*same* extractor scores **85% exact on synthetic text but only 52% on real
tweets**, extraction drops **80% → 64%**, and the **missing rate triples (9% →
29%)**. The controlled synthetic claims systematically overstate how well
geographic extraction works on real social-media language.

> Why the gap? The synthetic claims were written as clean, single-sentence
> statements that *name a place in plain text* ("Wildfire spreading near Malibu
> tonight"). Real crisis tweets rarely cooperate: the place is in a hashtag, an
> abbreviation, a nickname, a misspelling, buried among several locations, or
> simply absent because the post is about a *consequence* (a fundraiser, a
> vigil) rather than the disaster's location.

---

## Issues common in real posts (largely absent from synthetic claims)

Each item lists the phenomenon, a trimmed real example, and what the baseline did.

1. **Hashtag-embedded places.** The place exists only inside a `#hashtag`, which
   spaCy's small model does not tokenize as an entity.
   - *"…If you're in the **#yeg** area, please consider helping. #ymmfire"* →
     expected Edmonton (`#yeg`), extracted **nothing** (missing).
   - *"**#FortMcMurray** wildfire map interactive with evacs…"* → extracted
     nothing.
   - Synthetic claims almost never hide the location inside a hashtag.

2. **Abbreviations / initialisms with internal punctuation.** `Ft.`, `B.C.`,
   `Wash.`, `St.`
   - *"…355000 hectare … destroyed **Ft. Macmurray**"* → extracted nothing.
   - *"Video: Helicopter dumping water on wildfire in **Gold Bar, Wash.**"* →
     extracted `Gold Bar` + `Wash.` as two spans (partial).
   - The abbreviation that *does* survive (`B.C.`) is then a geocoding risk.

3. **Place nicknames.** Informal community names the gazetteer doesn't key on.
   - *"…friends who live(d) in **Fort. Muck.**"* (Fort McMurray) → mis-extracted
     and later geocoded abroad.

4. **Misspellings.** Real users misspell place names; synthetic claims don't.
   - *"…state of emergency … **Elliott City**"* (Ellicott City) → the misspelling
     pulls resolution toward the state (`Maryland`) instead of the city.

5. **Emojis and symbols inline.** Decorative characters break up the text around
   the place name.
   - *"⚡️ **Fort McMurray**-area wildfire grows…"* (recovered here, but the noise
     is a general hazard).
   - *"Floods and landslides in the **south-western** parts of #SriLanka caused:
     ➨deaths ➨displacements…"* → extracted nothing; the arrows and hashtag-only
     country defeat the small model.

6. **Missing / irregular punctuation and run-on phrasing.** Headlined, clause-
   fragment tweets ("Communities south of Fort McMurray order to leave as
   wildfire advances | CBC") lack the clean subject-verb-place structure the
   synthetic claims always have.

7. **Multiple locations in one post.** Which one is the event site?
   - *"**B.C.** wildfire crosses into **Alberta**…"* → both appear; scored against
     a multi-location label.
   - *"Irish man confirmed dead in Greek wildfires **#Mati #Athens #Ireland**"* →
     the model picked `Ireland` (the nationality), not the event site (incorrect).

8. **Vague references — "here", "the area", "downtown".** Deixis with no
   resolvable referent.
   - *"If you're in the #yeg area…"* — "the … area" is only anchored by the
     hashtag, not by a plain place name.
   - Synthetic "hard" claims include a few of these ("fire in the hills"), but
     real posts mix them *with* real place cues, which is harder.

9. **Local references / sub-city granularity.** Neighborhoods, counties, venues.
   - *"…**#Baltimore** & **#Howard** counties in Maryland"* → resolved only to the
     state, dropping the county the post actually names.

10. **Reposted / aggregated information.** Retweets and "via @outlet" news
    forwards describe an event second-hand; the salient entity is often a media
    outlet or donor, not the hazard site.
    - *"B.C. wildfire crosses into Alberta, medical teams to help evacuees -
      Langley Times **via @BlackPressMedia**"*. (We strip a leading `RT @user:`
      but embedded repost framing remains.)

11. **Posts about consequences, not the disaster itself.** Fundraisers, vigils,
    sympathy, policy — these legitimately contain **no** extractable location.
    - *"Lucy Neatby is hosting a Fund Raiser for the **Alberta** Wildfire
      victims."* → the only place (`Alberta`) is incidental and spaCy misses it.
    - *"Bring your toonies tomorrow @woodlandsrams to Rm 138 to support … fire
      loss relief efforts. #FortMacFire"* → the hazard place is hashtag-only.
    - This is a large share of the 29% `missing` rate: the post simply isn't
      *about* a location in the way a synthetic claim is engineered to be.

---

## Contrast with the synthetic claims

The synthetic set was built to stress geographic resolution, so it *does*
include ambiguous regions ("Pacific Northwest"), abbreviations ("LA", "NJ") and
no-location cases. But structurally it is far cleaner than real text:

| Dimension | Synthetic claims | Real HumAID tweets |
|-----------|------------------|--------------------|
| Form | one grammatical sentence | fragments, headlines, retweets, threads |
| Place cue | plain in-text noun phrase | hashtags, abbreviations, nicknames, misspellings |
| Noise | none | emojis, symbols (`➨`, `⚡️`), `@mentions`, URLs, HTML entities (`&amp;`) |
| Location focus | the claim is *about* the place | the post is often about a consequence |
| Multiplicity | at most 2 deliberate places | incidental donors, nationalities, outlets |

Net effect: identical model, **+33 points** of exact match on synthetic vs real
(85% → 52%) and **3×** the missing rate. Evaluating only on synthetic text would
have painted an over-optimistic picture of the extraction step.

---

## Implications for the project (and honest limits)

- **Do not report synthetic performance as if it were real-world performance.**
  The synthetic set is for controlled *pipeline testing*; the real datasets are
  for *evaluation*.
- **A location word is not a geolocation.** Extracting `#yeg`, `Alberta`, or
  `Sri Lanka` tells us a place was *mentioned*; it does not establish that the
  post originated there or that the hazard is physically there. The pipeline
  continues to treat extracted strings as claim content to be *checked*, never as
  confirmed coordinates.
- **Plausible next steps** (not yet implemented): hashtag segmentation before
  NER, a small abbreviation/nickname normalizer, fuzzy matching for misspellings,
  and down-weighting donor/nationality/outlet entities in favour of the hazard
  site. The Week 3 transformer comparison already recovers some hashtag and
  abbreviation cases at a precision cost.

Full numbers: `data/week4_real_vs_synthetic.csv`. Reproduce with
`python src/week4_real_vs_synthetic.py` (after building the HumAID datasets —
see the README).
