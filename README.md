# NASA GLOBE-Based Geospatial Misinformation Monitor

## Project Overview

Most misinformation classifiers rely heavily on language. They learn to spot
suspicious wording, tone, or source patterns, but they rarely check a claim
against the physical world. This project explores a different angle: whether
location-based environmental claims can be grounded against real Earth
observations instead of being judged by text alone.

The current prototype focuses on wildfire claims. A claim such as *"Wildfire
spreading near Malibu tonight"* mentions a place and an event. We extract the
place, turn it into coordinates, and ask NASA FIRMS whether satellites actually
detected active fires near that location around that date.

### Problem

Language-only misinformation detection can be fooled by well-written false
claims and can misfire on poorly-written true ones. Environmental claims are a
useful testbed because they often make checkable physical assertions ("there is
a fire here, now") that can be compared against satellite observations.

### Week 2 Objective

Build a baseline pipeline for extracting geographic entities from wildfire
claims and connecting them with NASA FIRMS satellite fire observations.

## Pipeline

```
Claim
  -> spaCy NER (extract geographic entity)
  -> Geocoding (OpenStreetMap / Nominatim)
  -> Coordinates
  -> NASA FIRMS (VIIRS active fire detections)
  -> Spatial / temporal comparison (Haversine distance + date window)
  -> Evidence result (SUPPORTED / NO_MATCH / INSUFFICIENT_EVIDENCE)
```

A verdict of `NO_MATCH` does **not** mean a claim is fake. The absence of a
satellite detection is not proof of misinformation; it only means the current
evidence source did not support the claim. This distinction matters for the
research framing.

## Dataset

`data/wildfire_claims.csv` contains ~30 **synthetic / demo** wildfire-related
claims created for testing the pipeline. These are **not** real social media
posts and are not assertions about real events. They mix easy geographic cases
(e.g. "Malibu, California") with intentionally ambiguous ones (e.g. "near the
border", "in the mountains") to stress-test the NER step.

Columns: `claim_id`, `text`, `date`, `expected_location`, `category`
(categories: `wildfire`, `smoke`, `evacuation`, `burned_area`).

## Repository Structure

```
NASA-Geospatial-Misinformation-Monitor/
    README.md
    requirements.txt
    .gitignore
    data/
        wildfire_claims.csv          # synthetic demo claims (input)
        ner_results.csv              # generated
        verification_results.csv     # generated
    src/
        ner.py        # spaCy NER + simple match evaluation
        geocode.py    # geopy / Nominatim geocoding with in-memory cache
        firms.py      # NASA FIRMS Area API client
        verify.py     # Haversine distance + verdict logic
        main.py       # end-to-end pipeline
    notebooks/
        week2_demo.ipynb
```

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

From the project root:

```bash
python src/main.py
```

This loads the claims, runs NER, geocodes locations, optionally queries FIRMS,
runs verification, writes `data/verification_results.csv`, and prints a summary.

## Week 2 Deliverables

- Initial wildfire claim corpus (synthetic)
- Baseline spaCy NER for geographic entity extraction
- Simple NER quality evaluation (correct / partial / incorrect / missing)
- Geocoding pipeline (Nominatim) with caching and rate limiting
- NASA FIRMS connection (VIIRS active fire detections)
- First evidence-based verification prototype (spatial + temporal comparison)

## Background and Influences

- **LIAR-PLUS** and **HealthVer** influenced the evidence-verification
  framing used here, where a claim is paired with supporting/contradicting
  evidence rather than classified from text alone.
- **NASA FIRMS** provides MODIS/VIIRS active fire observations used as the
  physical evidence source.

## Future Work

- PostGIS for spatial storage and querying
- NASA GLOBE citizen-science observations as an additional evidence layer
- Larger, real-world claim corpora
- Improved geographic entity resolution (ambiguous and informal place names)
- LLM-assisted contextual disambiguation of locations

## References

- LIAR-PLUS: <https://github.com/Tariq60/LIAR-PLUS>
- HealthVer: <https://github.com/sarrouti/HealthVer>
- NASA FIRMS: <https://firms.modaps.eosdis.nasa.gov/>
- MODIS data: <https://modis.gsfc.nasa.gov/data/>
- NASA GLOBE observations: <https://observer.globe.gov/get-data>
