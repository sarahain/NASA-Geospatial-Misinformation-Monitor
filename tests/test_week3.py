"""Unit tests for Week 3 extraction / resolution logic (no network required)."""
import location_resolution as lr
import week3_evaluation as w3


def test_normalize_alias_expands_common_abbreviations():
    assert lr.normalize_alias("LA") == "Los Angeles, California"
    assert lr.normalize_alias("nyc") == "New York, New York"
    assert lr.normalize_alias("NJ") == "New Jersey"


def test_normalize_alias_leaves_unknown_unchanged():
    assert lr.normalize_alias("Malibu") == "Malibu"
    assert lr.normalize_alias("") == ""


def test_select_primary_single_location():
    primary, reason, confident = lr.select_primary_location(["Malibu"], "fire near Malibu")
    assert primary == "Malibu"
    assert reason == "single_location"
    assert confident is True


def test_select_primary_uses_locating_preposition():
    primary, reason, confident = lr.select_primary_location(
        ["Malibu", "Los Angeles"],
        "Smoke from the wildfire near Malibu is reaching Los Angeles",
    )
    assert primary == "Malibu"
    assert reason.startswith("after_preposition")
    assert confident is True


def test_select_primary_multiple_without_preposition_is_flagged():
    primary, reason, confident = lr.select_primary_location(
        ["Canada", "New York City"],
        "Wildfire smoke from Canada is drifting into New York City",
    )
    assert primary == "Canada"  # falls back to first, but...
    assert reason == "first_of_multiple_ambiguous"
    assert confident is False   # ...not treated as certain


def test_select_primary_no_location():
    primary, reason, confident = lr.select_primary_location([], "fire in the hills")
    assert primary == ""
    assert reason == "no_location_detected"
    assert confident is False


def test_score_extraction_exact_partial_incorrect_missing():
    assert w3.score_extraction("Los Angeles", "Los Angeles") == "exact"
    assert w3.score_extraction("Malibu, California", "Malibu") == "partial"
    assert w3.score_extraction("Canada", "Texas") == "incorrect"
    assert w3.score_extraction("Lahaina", "") == "missing"


def test_score_extraction_no_expected_location_is_true_negative():
    # Claim has no location: extracting nothing is correct (exact), a spurious
    # extraction is a false positive (incorrect).
    assert w3.score_extraction("", "") == "exact"
    assert w3.score_extraction("", "Chicago") == "incorrect"


def test_score_extraction_multiple_expected_matches_any():
    assert w3.score_extraction("Malibu; Los Angeles", "Los Angeles") == "exact"


def test_score_geocoding_labels():
    resolved = {"latitude": 34.0, "longitude": -118.0,
                "normalized_location": "Los Angeles, Los Angeles County, California, United States"}
    assert w3.score_geocoding("Los Angeles, California", resolved) == "correct"

    unresolved = {"latitude": None, "longitude": None, "normalized_location": ""}
    assert w3.score_geocoding("Los Angeles, California", unresolved) == "unresolved"
