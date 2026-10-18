"""Week 3 geographic entity extraction & resolution evaluation.

Research question: *How reliably can geographic locations be extracted and
resolved from short environmental misinformation / social-media claims?*

This script runs the 100-claim evaluation set through:

    Method A : spaCy baseline NER            (preserved Week 1/2 model)
    Method B : transformer NER               (dslim/bert-base-NER, optional)
    spaCy + location resolution              (NER -> normalize -> geocode -> rank)

and writes the Week 3 deliverable CSVs:

    data/week3_spacy_baseline.csv      (D2)  per-claim spaCy baseline
    data/week3_geocoding_results.csv   (D6)  per-claim resolution + geocoding
    data/week3_model_comparison.csv    (D7)  A vs B vs spaCy+resolution
    data/week3_difficulty_analysis.csv (D10) accuracy by easy/medium/hard
    data/week3_metrics_summary.csv           headline metrics for notebook/README

All results are computed from the models as-is. Nothing is hand-corrected. The
transformer method is optional; if it is unavailable the comparison records that
honestly instead of fabricating numbers.
"""
from pathlib import Path

import pandas as pd

import location_resolution as lr
import ner
import ner_methods
import week3_build_dataset as w3data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# The evaluation needs the real tweet TEXT, which lives only in the (git-ignored)
# hydrated file. It is rebuilt locally from the HumAID corpus on demand.
TEST_SET_PATH = w3data.HYDRATED_PATH

SPACY_BASELINE_PATH = DATA_DIR / "week3_spacy_baseline.csv"
GEOCODING_PATH = DATA_DIR / "week3_geocoding_results.csv"
COMPARISON_PATH = DATA_DIR / "week3_model_comparison.csv"
DIFFICULTY_PATH = DATA_DIR / "week3_difficulty_analysis.csv"
SUMMARY_PATH = DATA_DIR / "week3_metrics_summary.csv"

EXTRACTION_OUTCOMES = ["exact", "partial", "incorrect", "missing"]


# --- Matching / scoring helpers --------------------------------------------
def _norm(text):
    return " ".join(str(text).lower().replace(",", " ").split())


def _expected_parts(expected_location):
    """Split a possibly multi-location expected value on ';'."""
    if not str(expected_location).strip():
        return []
    return [p.strip() for p in str(expected_location).split(";") if p.strip()]


def _combine_expected(expected, expected_normalized):
    """Merge the raw and normalized gold labels into one ';'-joined value.

    ``score_extraction`` matches an extraction against ANY ';'-separated part,
    so this lets the normalization-producing method (Method C) match either the
    raw label (e.g. "LA") or the normalized one (e.g. "Los Angeles, California")
    without inventing a match. Empty labels (no-location claims) stay empty.
    """
    parts = _expected_parts(expected) + _expected_parts(expected_normalized)
    # Preserve order, drop duplicates.
    seen, ordered = set(), []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(p)
    return "; ".join(ordered)


def score_extraction(expected_location, primary_location):
    """Label an extracted primary location exact / partial / incorrect / missing.

    Handles three cases:
      * expected is empty  -> the claim has NO location; extracting nothing is
        'exact' (a correct true-negative), extracting something is 'incorrect'
        (a false positive).
      * expected is a single location.
      * expected lists several locations (';'-separated); a match against ANY
        counts.
    """
    expected_parts = _expected_parts(expected_location)
    extracted = str(primary_location or "").strip()

    if not expected_parts:  # no location present in the claim
        return "exact" if not extracted else "incorrect"

    if not extracted:
        return "missing"

    best = "incorrect"
    for part in expected_parts:
        label = ner.evaluate_match(part, extracted)  # correct/partial/incorrect/missing
        if label == "correct":
            return "exact"
        if label == "partial":
            best = "partial"
    return best


def score_geocoding(expected_normalized, resolution):
    """Label resolved coordinates correct / partial / incorrect / unresolved.

    Compares the geocoder's resolved display name against the expected
    normalized location by token overlap (place tokens + region tokens). No
    ground-truth coordinates are fabricated; this is a transparent string check.
    """
    if resolution["latitude"] is None or not resolution["normalized_location"]:
        return "unresolved"

    expected = str(expected_normalized).strip()
    if not expected:
        # No expected location but something was resolved -> spurious.
        return "incorrect"

    resolved_tokens = set(_norm(resolution["normalized_location"]).split())
    parts = [p.strip() for p in expected.split(",")]
    place_tokens = set(_norm(parts[0]).split())
    region_tokens = set(_norm(" ".join(parts[1:])).split()) if len(parts) > 1 else set()

    place_hit = bool(place_tokens) and place_tokens.issubset(resolved_tokens)
    region_hit = bool(region_tokens) and bool(region_tokens & resolved_tokens)

    if place_hit and (region_hit or not region_tokens):
        return "correct"
    if place_hit or region_hit:
        return "partial"
    return "incorrect"


# --- Core evaluation --------------------------------------------------------
def evaluate(test_set_path=TEST_SET_PATH, use_transformer=True):
    """Run the full evaluation and return a dict of result DataFrames."""
    if test_set_path == w3data.HYDRATED_PATH and not w3data.HYDRATED_PATH.exists():
        w3data.build()  # hydrate real tweet text locally from the HumAID corpus
    claims = pd.read_csv(test_set_path).fillna({"expected_location": "", "expected_normalized_location": ""})

    spacy_extractor = ner_methods.get_extractor("spacy")
    transformer_extractor = ner_methods.get_extractor("transformer") if use_transformer else None
    transformer_available = bool(transformer_extractor and transformer_extractor.available)

    baseline_rows = []   # D2
    geocoding_rows = []  # D6
    per_claim = []       # internal, drives comparison + difficulty

    for _, claim in claims.iterrows():
        cid = claim["claim_id"]
        text = claim["text"]
        expected = claim["expected_location"]
        expected_norm = claim["expected_normalized_location"]

        # --- Method A: spaCy baseline (extraction only) ---
        a = spacy_extractor.extract(text)
        a_result = score_extraction(expected, a["primary_location"])
        baseline_rows.append(
            {
                "claim_id": cid,
                "text": text,
                "expected_location": expected,
                "spacy_location": a["primary_location"],
                "spacy_all_entities": "; ".join(a["all_locations"]),
                "spacy_result": a_result,
            }
        )

        # --- Method B: transformer NER (extraction only) ---
        if transformer_available:
            b = transformer_extractor.extract(text)
            b_primary = b["primary_location"]
            b_all = b["all_locations"]
            b_result = score_extraction(expected, b_primary)
        else:
            b_primary, b_all, b_result = "", [], None

        # --- spaCy + full location resolution ---
        res = lr.resolve_claim(text, spacy_extractor, use_context=True)
        # Method C outputs a *normalized* location, so it is scored against both
        # the raw and the normalized gold label (a match against either counts).
        # Otherwise correct expansions like "LA" -> "Los Angeles" would be mis-
        # scored as 'incorrect' purely because they no longer equal the raw
        # abbreviation, understating the resolution step.
        res_result = score_extraction(
            _combine_expected(expected, expected_norm),
            _leading(res["normalized_location"]) or res["primary_location"],
        )
        geo_result = score_geocoding(expected_norm, res)
        geocoding_rows.append(
            {
                "claim_id": cid,
                "expected_location": expected,
                "extracted_location": res["primary_location"],
                "normalized_location": res["normalized_location"],
                "latitude": res["latitude"],
                "longitude": res["longitude"],
                "geocoder_used": res["geocoder_used"],
                "candidate_count": res["candidate_count"],
                "ambiguity_status": res["ambiguity_status"],
                "location_confidence": res["location_confidence"],
                "geocode_result": geo_result,
            }
        )

        # --- transformer + resolution (for comparison geocoding rate only) ---
        if transformer_available:
            res_b = lr.resolve_claim(text, transformer_extractor, use_context=True)
            b_geocoded = res_b["latitude"] is not None
        else:
            b_geocoded = False

        per_claim.append(
            {
                "claim_id": cid,
                "difficulty": claim["difficulty"],
                "category": claim["category"],
                "has_expected_location": bool(_expected_parts(expected)),
                "a_primary": a["primary_location"],
                "a_produced": bool(a["primary_location"]),
                "b_produced": bool(b_primary),
                "res_produced": bool(res["primary_location"]),
                "a_result": a_result,
                "b_result": b_result,
                "res_result": res_result,
                "geo_result": geo_result,
                "a_geocoded": res["latitude"] is not None,
                "b_geocoded": b_geocoded,
                "res_geocoded": res["latitude"] is not None,
                "ambiguity_status": res["ambiguity_status"],
            }
        )

    baseline_df = pd.DataFrame(baseline_rows)
    geocoding_df = pd.DataFrame(geocoding_rows)
    per_claim_df = pd.DataFrame(per_claim)

    comparison_df = _build_comparison(per_claim_df, transformer_available)
    difficulty_df = _build_difficulty(per_claim_df)
    summary = _build_summary(baseline_df, geocoding_df, per_claim_df, transformer_available)

    return {
        "baseline": baseline_df,
        "geocoding": geocoding_df,
        "comparison": comparison_df,
        "difficulty": difficulty_df,
        "per_claim": per_claim_df,
        "summary": summary,
        "transformer_available": transformer_available,
    }


def _leading(display_name):
    return str(display_name).split(",")[0].strip() if display_name else ""


def _rates(results, total):
    counts = {o: list(results).count(o) for o in EXTRACTION_OUTCOMES}
    return {
        "exact_match_rate": _pct(counts["exact"], total),
        "partial_match_rate": _pct(counts["partial"], total),
        "incorrect_rate": _pct(counts["incorrect"], total),
        "missing_rate": _pct(counts["missing"], total),
    }


def _pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


def _build_comparison(per_claim, transformer_available):
    total = len(per_claim)
    rows = []

    # Row A: spaCy baseline
    a = per_claim["a_result"]
    rows.append(
        {
            "method": "A_spacy_baseline",
            "extraction_rate": _pct(per_claim["a_produced"].sum(), total),
            **_rates(a, total),
            "successful_geocoding_rate": _pct(per_claim["a_geocoded"].sum(), total),
        }
    )

    # Row B: transformer NER
    if transformer_available:
        b = per_claim["b_result"]
        rows.append(
            {
                "method": "B_transformer_ner",
                "extraction_rate": _pct(per_claim["b_produced"].sum(), total),
                **_rates(b, total),
                "successful_geocoding_rate": _pct(per_claim["b_geocoded"].sum(), total),
            }
        )
    else:
        rows.append(
            {
                "method": "B_transformer_ner",
                "extraction_rate": None,
                "exact_match_rate": None,
                "partial_match_rate": None,
                "incorrect_rate": None,
                "missing_rate": None,
                "successful_geocoding_rate": None,
            }
        )

    # Row C: spaCy + location resolution
    c = per_claim["res_result"]
    rows.append(
        {
            "method": "C_spacy_plus_resolution",
            "extraction_rate": _pct(per_claim["res_produced"].sum(), total),
            **_rates(c, total),
            "successful_geocoding_rate": _pct(per_claim["res_geocoded"].sum(), total),
        }
    )

    return pd.DataFrame(rows)


def _build_difficulty(per_claim):
    rows = []
    for level in ["easy", "medium", "hard"]:
        subset = per_claim[per_claim["difficulty"] == level]
        n = len(subset)
        rows.append(
            {
                "difficulty": level,
                "number_of_claims": n,
                "exact_match_rate": _pct((subset["res_result"] == "exact").sum(), n),
                "geocoding_success_rate": _pct(subset["res_geocoded"].sum(), n),
            }
        )
    return pd.DataFrame(rows)


def _build_summary(baseline_df, geocoding_df, per_claim, transformer_available):
    total = len(baseline_df)
    with_loc = int(per_claim["has_expected_location"].sum())
    geocodable = per_claim[per_claim["has_expected_location"]]
    located = per_claim[per_claim["has_expected_location"]]

    def pct(n, d):
        return _pct(n, d)

    metrics = {
        "total_claims": total,
        "claims_with_expected_location": with_loc,
        "claims_with_no_location": total - with_loc,
        # spaCy baseline (D2) -- rates over ALL 100 claims
        "spacy_extraction_rate": pct((baseline_df["spacy_location"].astype(str).str.len() > 0).sum(), total),
        "spacy_exact_match_rate": pct((baseline_df["spacy_result"] == "exact").sum(), total),
        "spacy_partial_match_rate": pct((baseline_df["spacy_result"] == "partial").sum(), total),
        "spacy_incorrect_rate": pct((baseline_df["spacy_result"] == "incorrect").sum(), total),
        "spacy_missing_rate": pct((baseline_df["spacy_result"] == "missing").sum(), total),
        # exact-match rates over ONLY the claims that contain a location
        # (excludes the true-negative 'no location' claims from the denominator)
        "spacy_exact_match_rate_located": pct((located["a_result"] == "exact").sum(), len(located)),
        "resolution_exact_match_rate_located": pct((located["res_result"] == "exact").sum(), len(located)),
        # geocoding (D6) -- measured over the geocodable subset
        "geocoding_success_rate": pct(geocodable["res_geocoded"].sum(), len(geocodable)),
        "geocoding_correct_rate": pct((geocoding_df["geocode_result"] == "correct").sum(), total),
        "geocoding_ambiguity_rate": pct((geocoding_df["ambiguity_status"] == "ambiguous").sum(), total),
        "geocoding_unresolved_rate": pct((geocoding_df["ambiguity_status"] == "unresolved").sum(), total),
        "transformer_available": transformer_available,
    }
    return pd.DataFrame(list(metrics.items()), columns=["metric", "value"])


def run(test_set_path=TEST_SET_PATH, use_transformer=True):
    results = evaluate(test_set_path, use_transformer=use_transformer)
    results["baseline"].to_csv(SPACY_BASELINE_PATH, index=False)
    results["geocoding"].to_csv(GEOCODING_PATH, index=False)
    results["comparison"].to_csv(COMPARISON_PATH, index=False)
    results["difficulty"].to_csv(DIFFICULTY_PATH, index=False)
    results["summary"].to_csv(SUMMARY_PATH, index=False)
    return results


def _print(results):
    print("=== Week 3 Evaluation ===")
    print(f"Transformer (Method B) available: {results['transformer_available']}\n")
    print("--- Metrics summary ---")
    print(results["summary"].to_string(index=False))
    print("\n--- Model comparison ---")
    print(results["comparison"].to_string(index=False))
    print("\n--- Difficulty analysis ---")
    print(results["difficulty"].to_string(index=False))


if __name__ == "__main__":
    _print(run())
