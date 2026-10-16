"""Geographic location resolution (Week 2 baseline + Week 3 extensions).

Pipeline for a single claim::

    raw claim text
      -> NER location candidates (any extractor from ner_methods)
      -> select a primary location among multiple candidates (record why)
      -> normalize common abbreviations (small alias map only)
      -> query geocoder (Nominatim primary, optional Google fallback)
      -> inspect + rank candidate locations
      -> select best candidate -> latitude / longitude
      -> confidence + ambiguity_status (resolved / ambiguous / unresolved)

Design notes
------------
* No giant hardcoded place dictionary. Only a *small* alias map for common
  abbreviations (LA, NYC, NJ, ...) that short social-media text uses.
* Ambiguity is detected from the geocoder candidates (count + importance
  separation), not hardcoded per place. The system never silently pretends an
  ambiguous location is certain.
* ``resolve_location`` keeps its Week 2 signature and return keys so
  ``evidence.py`` and ``location_experiment.py`` keep working unchanged.
"""
import re

import geocode
import ner

# --- Tunables ---------------------------------------------------------------
AMBIGUITY_THRESHOLD = 0.4   # confidence below this => "ambiguous"
MULTI_AMBIGUOUS_PENALTY = 0.5  # confidence multiplier when primary pick is a guess

STOPWORDS = {
    "the", "near", "in", "at", "across", "over", "after", "from", "to", "of",
    "and", "a", "an", "is", "are", "tonight", "this", "morning", "fire",
    "wildfire", "smoke", "reported", "residents", "evacuating", "evacuated",
}

# Small alias normalization map for common short-text abbreviations only.
# Intentionally tiny -- NOT a gazetteer. Keys are matched case-insensitively.
ALIASES = {
    "la": "Los Angeles, California",
    "l.a.": "Los Angeles, California",
    "la county": "Los Angeles County, California",
    "nyc": "New York, New York",
    "ny": "New York",
    "nj": "New Jersey",
    "sf": "San Francisco, California",
    "pnw": "Pacific Northwest, United States",
}

# Prepositions that tend to introduce the *event* location in a claim.
LOCATING_PREPOSITIONS = ("near", "in", "at", "over", "across", "outside", "above")


def _tokens(text):
    return {t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOPWORDS}


# --- Abbreviation normalization --------------------------------------------
def normalize_alias(raw_location):
    """Expand a small set of common abbreviations; otherwise return unchanged."""
    if not raw_location:
        return raw_location
    return ALIASES.get(raw_location.strip().lower(), raw_location)


# --- Multiple-location handling (Week 3, Deliverable 11) --------------------
def select_primary_location(all_locations, claim_text):
    """Pick a primary location among candidates and record *why*.

    Returns ``(primary_location, reason, confident)``. ``confident`` is False
    when more than one location is present and no locating preposition
    disambiguates them -- the caller then treats the claim as LOCATION_AMBIGUOUS
    rather than blindly trusting the first entity.
    """
    if not all_locations:
        return "", "no_location_detected", False
    if len(all_locations) == 1:
        return all_locations[0], "single_location", True

    lowered = claim_text.lower()
    for prep in LOCATING_PREPOSITIONS:
        for loc in all_locations:
            if f"{prep} {loc.lower()}" in lowered:
                return loc, f"after_preposition:{prep}", True

    # Multiple locations, none clearly the event site -> keep first but flag it.
    return all_locations[0], "first_of_multiple_ambiguous", False


# --- Candidate ranking (Week 2) --------------------------------------------
def rank_candidates(candidates, claim_text, raw_location, use_context):
    raw_tokens = _tokens(raw_location)
    context_tokens = _tokens(claim_text) - raw_tokens

    ranked = []
    for c in candidates:
        score = c["importance"]
        context_hit = False
        if use_context:
            display_tokens = _tokens(c["display_name"])
            if context_tokens & display_tokens:
                score += 0.2
                context_hit = True
        ranked.append({**c, "score": score, "context_hit": context_hit})

    ranked.sort(key=lambda c: c["score"], reverse=True)
    return ranked


def _confidence(ranked):
    if not ranked:
        return 0.0
    top = ranked[0]
    base = min(top["importance"], 0.6)
    bonus = 0.2 if top["context_hit"] else 0.0
    separation = 0.0
    if len(ranked) > 1:
        separation = min(0.2, max(0.0, top["score"] - ranked[1]["score"]))
    return round(min(1.0, base + bonus + separation), 2)


def _ambiguity_status(ranked, confidence):
    if not ranked:
        return "unresolved"
    if confidence < AMBIGUITY_THRESHOLD:
        return "ambiguous"
    return "resolved"


def _geocode_and_rank(query, claim_text, use_context):
    """Geocode ``query`` with Nominatim, falling back to Google when helpful.

    Returns ``(ranked, confidence, status, geocoder_used)``. Google is tried
    only when the key exists AND Nominatim gave an unresolved/ambiguous result,
    and is kept only if it is strictly more confident. Respects Nominatim rate
    limiting and caching via ``geocode``.
    """
    candidates = geocode.geocode_candidates(query)
    geocoder_used = "Nominatim" if candidates else "None"
    ranked = rank_candidates(candidates, claim_text, query, use_context)
    confidence = _confidence(ranked)
    status = _ambiguity_status(ranked, confidence)

    if status != "resolved" and geocode.google_available():
        g_candidates = geocode.google_geocode_candidates(query)
        if g_candidates:
            g_ranked = rank_candidates(g_candidates, claim_text, query, use_context)
            g_confidence = _confidence(g_ranked)
            if g_confidence > confidence:
                ranked, confidence = g_ranked, g_confidence
                status = _ambiguity_status(ranked, confidence)
                geocoder_used = "Google"

    return ranked, confidence, status, geocoder_used


def _empty_result(method):
    return {
        "method": method,
        "all_locations": [],
        "primary_location": "",
        "primary_reason": "no_location_detected",
        "raw_location": "",
        "normalized_location": "",
        "latitude": None,
        "longitude": None,
        "candidate_count": 0,
        "location_confidence": 0.0,
        "ambiguity_status": "unresolved",
        "geocoder_used": "None",
    }


def resolve_claim(text, extractor, use_context=True):
    """Full Week 3 resolution for one claim using any ner_methods extractor."""
    extraction = extractor.extract(text)
    all_locations = extraction["all_locations"]

    primary, reason, confident = select_primary_location(all_locations, text)

    result = _empty_result(extractor.name)
    result.update(
        {
            "all_locations": all_locations,
            "primary_location": primary,
            "primary_reason": reason,
            "raw_location": primary,
        }
    )
    if not primary:
        return result

    query = normalize_alias(primary)
    ranked, confidence, status, geocoder_used = _geocode_and_rank(query, text, use_context)

    result["candidate_count"] = len(ranked)
    result["geocoder_used"] = geocoder_used
    if ranked:
        best = ranked[0]
        result["normalized_location"] = best["display_name"]
        result["latitude"] = best["latitude"]
        result["longitude"] = best["longitude"]

    # A multi-location claim with no clear primary is never reported as certain.
    if not confident and ranked:
        confidence = round(confidence * MULTI_AMBIGUOUS_PENALTY, 2)
        status = "ambiguous"

    result["location_confidence"] = confidence
    result["ambiguity_status"] = status
    return result


def resolve_location(text, nlp=None, use_context=True):
    """Backward-compatible Week 2 entry point (spaCy extractor).

    Returns the original Week 2 keys plus the new Week 3 fields, so existing
    callers (evidence.py, location_experiment.py) keep working unchanged.
    """
    import ner_methods

    nlp = nlp or ner.load_model()
    extractor = ner_methods.SpacyExtractor(nlp=nlp)
    return resolve_claim(text, extractor, use_context=use_context)


if __name__ == "__main__":
    import ner_methods

    spacy_extractor = ner_methods.SpacyExtractor()
    samples = [
        "Residents are evacuating near Paradise California",
        "Smoke from the wildfire near Malibu is reaching Los Angeles",
        "Thousands evacuated across Northern California counties",
        "Heavy flooding reported in NJ after record rainfall",
    ]
    for s in samples:
        print(s)
        print("  ", resolve_claim(s, spacy_extractor))
