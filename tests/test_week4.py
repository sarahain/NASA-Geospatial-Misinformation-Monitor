"""Unit tests for the Week 4 real-text loaders + experiment (no network)."""
import csv
from pathlib import Path

import crisismmd_loader as cm
import humaid_loader as hl
import week4_real_vs_synthetic as w4
import x_ingestion as xi


# --- HumAID loader ---------------------------------------------------------
def test_humaid_normalize_strips_rt_and_urls_but_keeps_content():
    raw = "RT @user: Fire near #FortMcMurray https://t.co/abc  spreading"
    assert hl.normalize_text(raw) == "Fire near #FortMcMurray spreading"


def test_humaid_normalize_preserves_hashtags_and_case():
    # No de-hashtagging / lowercasing -- noisy real text is preserved.
    assert hl.normalize_text("Evacuate NOW #yeg") == "Evacuate NOW #yeg"


def _write_tsv(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def test_humaid_load_records_schema_and_dedup(tmp_path):
    raw = tmp_path / "events_set1"
    header = ["tweet_id", "tweet_text", "class_label"]
    _write_tsv(
        raw / "greece_wildfires_2018" / "greece_wildfires_2018_train.tsv",
        [["1", "Fire in #Athens", "caution_and_advice"],
         ["2", "RT @x: smoke everywhere https://t.co/z", "other_relevant_information"]],
        header,
    )
    # Same tweet id in another split should be de-duplicated.
    _write_tsv(
        raw / "greece_wildfires_2018" / "greece_wildfires_2018_dev.tsv",
        [["1", "Fire in #Athens", "caution_and_advice"]],
        header,
    )
    records = hl.load_records(raw_dir=raw)
    assert len(records) == 2
    r = records[0]
    assert set(r) == set(hl.OUTPUT_FIELDS)
    assert r["data_origin"] == "REAL_DATASET"
    assert r["dataset_name"] == "HumAID"
    assert r["event"] == "greece_wildfires_2018"
    assert r["humanitarian_label"] == "caution_and_advice"


def test_humaid_missing_corpus_raises_with_download_hint(tmp_path):
    missing = tmp_path / "nope"
    try:
        hl.load_records(raw_dir=missing)
    except SystemExit as exc:
        assert "crisisnlp" in str(exc)
    else:
        raise AssertionError("expected SystemExit when corpus is absent")


# --- CrisisMMD loader ------------------------------------------------------
def test_crisismmd_event_to_type():
    assert cm.event_to_type("california_wildfires") == "wildfire"
    assert cm.event_to_type("srilanka_floods") == "flood"
    assert cm.event_to_type("mexico_earthquake") == "earthquake"
    assert cm.event_to_type("hurricane_maria") == "hurricane"
    assert cm.event_to_type("something_else") == "other"


def test_crisismmd_load_records_original_format(tmp_path):
    ann = tmp_path / "CrisisMMD_v2.0" / "annotations"
    header = [
        "tweet_id", "image_id", "tweet_text", "image",
        "text_info", "text_info_conf", "image_info", "image_info_conf",
        "text_human", "text_human_conf", "image_human", "image_human_conf",
        "image_damage", "image_damage_conf",
    ]
    row = [
        "918008272363368448", "918008272363368448_0",
        "#California #wildfires HORRIBLE https://t.co/x",
        "data_image/california_wildfires/x.jpg",
        "informative", "1.0", "informative", "1.0",
        "infrastructure_and_utility_damage", "1.0",
        "infrastructure_and_utility_damage", "1.0",
        "severe_damage", "1.0",
    ]
    _write_tsv(ann / "california_wildfires.tsv", [row], header)

    records, files = cm.load_records(raw_root=tmp_path)
    assert len(files) == 1 and len(records) == 1
    r = records[0]
    assert set(r) == set(cm.OUTPUT_FIELDS)
    assert r["data_origin"] == "REAL_DATASET" and r["dataset_name"] == "CrisisMMD"
    assert r["event"] == "california_wildfires" and r["event_type"] == "wildfire"
    assert r["informative_label"] == "informative"
    assert r["humanitarian_label"] == "infrastructure_and_utility_damage"
    assert r["damage_severity"] == "severe_damage"
    assert r["image_reference"].endswith(".jpg")
    assert "https://" not in r["post_text"]


def test_crisismmd_absent_is_graceful(tmp_path):
    records, files = cm.load_records(raw_root=tmp_path / "empty")
    assert records == [] and files == []
    # build() should not raise, just return [].
    assert cm.build(raw_root=tmp_path / "empty", output_path=tmp_path / "out.csv") == []


# --- X ingestion (no network) ----------------------------------------------
def test_x_token_from_env(monkeypatch):
    monkeypatch.delenv(xi.BEARER_TOKEN_ENV, raising=False)
    assert xi.get_bearer_token() is None
    monkeypatch.setenv(xi.BEARER_TOKEN_ENV, "abc123")
    assert xi.get_bearer_token() == "abc123"


def test_x_ingest_without_token_raises(monkeypatch):
    monkeypatch.delenv(xi.BEARER_TOKEN_ENV, raising=False)
    try:
        xi.ingest()
    except xi.XIngestionError as exc:
        assert xi.BEARER_TOKEN_ENV in str(exc)
    else:
        raise AssertionError("expected XIngestionError without a token")


def test_x_exclude_retweets_and_place_flatten():
    assert xi._default_exclude_retweets("fire") == "(fire) -is:retweet"
    assert xi._default_exclude_retweets("a is:retweet") == "a is:retweet"
    # Place present but no coordinates -> has_coordinates False (never assumed).
    name, country, full, has_coords = xi._flatten_place(
        {"geo": {"place_id": "p1"}},
        {"p1": {"name": "Malibu", "country": "United States", "full_name": "Malibu, CA"}},
    )
    assert (name, country, full, has_coords) == ("Malibu", "United States", "Malibu, CA", False)


# --- Week 4 experiment -----------------------------------------------------
class _FakeExtractor:
    """Deterministic extractor: returns locations keyed on the text."""

    def __init__(self, mapping):
        self._mapping = mapping

    def extract(self, text):
        locs = self._mapping.get(text, [])
        return {"all_locations": locs, "primary_location": locs[0] if locs else ""}


def test_week4_is_ambiguous_only_for_multi_without_cue():
    assert w4._is_ambiguous(["Malibu"], "fire near Malibu") is False
    assert w4._is_ambiguous(["Canada", "New York"], "smoke from Canada into New York") is True
    assert w4._is_ambiguous(
        ["Malibu", "Los Angeles"],
        "wildfire near Malibu is reaching Los Angeles",
    ) is False  # a locating preposition disambiguates -> not ambiguous


def test_week4_evaluate_group_metrics_with_labels():
    texts = ["a", "b", "c", "d"]
    extractor = _FakeExtractor({"a": ["Malibu"], "b": ["Texas"], "c": [], "d": []})
    labels = ["Malibu", "Paris", "Lahaina", ""]  # d has no location (true-negative)
    m = w4.evaluate_group(texts, labels, extractor)
    assert m["number_of_posts"] == 4
    assert m["location_extraction_rate"] == 50.0           # a, b extracted
    assert m["exact_or_manual_location_match_rate"] == 50.0  # a correct + d true-negative
    assert m["missing_location_rate"] == 25.0              # c expected Lahaina, got none


def test_week4_evaluate_group_without_labels_reports_na():
    extractor = _FakeExtractor({"x": ["Reno"], "y": []})
    m = w4.evaluate_group(["x", "y"], None, extractor)
    assert m["exact_or_manual_location_match_rate"] == "NA"
    assert m["missing_location_rate"] == "NA"
    assert m["location_extraction_rate"] == 50.0
