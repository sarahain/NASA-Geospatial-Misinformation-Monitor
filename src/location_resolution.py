import re

import geocode
import ner

STOPWORDS = {
    "the", "near", "in", "at", "across", "over", "after", "from", "to", "of",
    "and", "a", "an", "is", "are", "tonight", "this", "morning", "fire",
    "wildfire", "smoke", "reported", "residents", "evacuating", "evacuated",
}


def _tokens(text):
    return {t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOPWORDS}


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


def resolve_location(text, nlp=None, use_context=True):
    nlp = nlp or ner.load_model()
    raw_location = ner.pick_location(ner.extract_locations(nlp(text)))

    result = {
        "raw_location": raw_location,
        "normalized_location": "",
        "latitude": None,
        "longitude": None,
        "location_confidence": 0.0,
        "candidate_count": 0,
    }
    if not raw_location:
        return result

    candidates = geocode.geocode_candidates(raw_location)
    result["candidate_count"] = len(candidates)
    if not candidates:
        return result

    ranked = rank_candidates(candidates, text, raw_location, use_context)
    best = ranked[0]
    result["normalized_location"] = best["display_name"]
    result["latitude"] = best["latitude"]
    result["longitude"] = best["longitude"]
    result["location_confidence"] = _confidence(ranked)
    return result


if __name__ == "__main__":
    nlp = ner.load_model()
    samples = [
        "Residents are evacuating near Paradise California",
        "Smoke is covering Los Angeles after the fire",
        "Thousands evacuated across Northern California counties",
        "Large fire reported near Lahaina",
    ]
    for s in samples:
        print(s)
        print("  ", resolve_location(s, nlp))
