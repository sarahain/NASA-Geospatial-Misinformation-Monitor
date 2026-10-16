"""Unified geographic-entity extraction interface for Week 3 model comparison.

Two interchangeable extractors are exposed behind a single, identical API so
the evaluation code can treat them the same way:

    Method A -- ``SpacyExtractor``       : the preserved Week 1/2 baseline
                                           (spaCy ``en_core_web_sm``).
    Method B -- ``TransformerExtractor`` : an alternative, pretrained
                                           transformer NER (``dslim/bert-base-NER``).

This is a *model comparison*, not fine-tuning: both models are used exactly as
published. Neither extractor mutates the other's output.

Every extractor returns the same structure from ``extract(text)``::

    {
        "method": <str>,               # short method id
        "all_locations": [<str>, ...], # location strings, in order of appearance
        "primary_location": <str>,     # first location, or "" (naive baseline pick)
        "available": True,             # the extractor is usable
    }

Method B is *optional*. If ``torch`` / ``transformers`` or the model weights are
unavailable, ``TransformerExtractor`` reports ``available = False`` and returns
empty extractions instead of crashing the pipeline, so the project still runs
with only the spaCy baseline. The evaluation records this rather than
fabricating transformer results.
"""
from functools import lru_cache

import ner

# spaCy entity labels treated as geographic (shared with the Week 2 baseline).
SPACY_LOCATION_LABELS = ner.LOCATION_LABELS

# Transformer (dslim/bert-base-NER CoNLL-2003 tagset) labels treated as geographic.
TRANSFORMER_LOCATION_LABELS = {"LOC"}
TRANSFORMER_MODEL = "dslim/bert-base-NER"


class SpacyExtractor:
    """Method A: the preserved spaCy baseline (``en_core_web_sm``)."""

    name = "spacy"

    def __init__(self, nlp=None):
        self._nlp = nlp or ner.load_model()
        self.available = True

    def extract(self, text):
        doc = self._nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ in SPACY_LOCATION_LABELS]
        return _package(self.name, locations)


class TransformerExtractor:
    """Method B: alternative pretrained transformer NER (optional)."""

    name = "transformer"

    def __init__(self, model_name=TRANSFORMER_MODEL):
        self.model_name = model_name
        self._pipeline = None
        self.load_error = None
        self.available = self._try_load()

    def _try_load(self):
        try:
            from transformers import pipeline
        except Exception as exc:  # transformers / torch not installed
            self.load_error = f"transformers not available: {exc}"
            return False
        try:
            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
            )
            return True
        except Exception as exc:  # model weights unavailable / offline
            self.load_error = f"could not load '{self.model_name}': {exc}"
            return False

    def extract(self, text):
        if not self.available:
            return _package(self.name, [], available=False)
        entities = self._pipeline(text)
        locations = [
            e["word"].strip()
            for e in entities
            if e.get("entity_group") in TRANSFORMER_LOCATION_LABELS and e["word"].strip()
        ]
        return _package(self.name, locations)


def _package(method, locations, available=True):
    # Preserve order but drop exact duplicates (case-insensitive).
    seen, ordered = set(), []
    for loc in locations:
        key = loc.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(loc)
    return {
        "method": method,
        "all_locations": ordered,
        "primary_location": ordered[0] if ordered else "",
        "available": available,
    }


@lru_cache(maxsize=None)
def get_extractor(method, _nlp_id=None):
    """Return a cached extractor for ``method`` ("spacy" or "transformer")."""
    if method == "spacy":
        return SpacyExtractor()
    if method == "transformer":
        return TransformerExtractor()
    raise ValueError(f"unknown extraction method: {method!r}")


if __name__ == "__main__":
    samples = [
        "Smoke from the wildfire near Malibu is reaching Los Angeles",
        "Heavy flooding reported in NJ after record rainfall",
        "Massive wildfire burning in the hills",
    ]
    for method in ("spacy", "transformer"):
        extractor = get_extractor(method)
        print(f"\n=== {method} (available={extractor.available}) ===")
        if not extractor.available:
            print("  ", getattr(extractor, "load_error", "unavailable"))
            continue
        for s in samples:
            print(f"  {s}\n    -> {extractor.extract(s)}")
