"""Offline tests for the Week 5 X collection pipeline.

No test makes a network request or spends API credits -- all X API responses are
mocked as plain dicts, and no bearer token is required.
"""
import csv
import json
from pathlib import Path

import relevance_filter as rf
import week5_summaries as w5s
import week5_x_ner as w5ner
import x_ingestion as xi


# ===========================================================================
# Relevance filter
# ===========================================================================
def test_relevance_relevant_requires_hazard_plus_context():
    r = rf.classify("Mandatory evacuation as the Palisades Fire burns 3000 acres near Los Angeles")
    assert r["relevance_label"] == "RELEVANT"
    assert r["has_hazard"] and r["has_location"] and r["has_emergency_terms"]
    assert 0.0 <= r["relevance_score"] <= 1.0


def test_relevance_figurative_is_irrelevant():
    for text in ["this song is fire", "the whole team is on fire tonight", "that outfit is fire"]:
        r = rf.classify(text)
        assert r["relevance_label"] == "IRRELEVANT", text
        assert r["is_figurative"]


def test_relevance_bare_hazard_is_possibly_relevant():
    # "wildfire is crazy" -> hazard but no location/impact/emergency context.
    r = rf.classify("wildfire is crazy")
    assert r["relevance_label"] == "POSSIBLY_RELEVANT"


def test_relevance_keyword_presence_alone_not_relevant():
    # Mentions "fire" only, figuratively -> not relevant.
    assert rf.classify("my mixtape is straight fire").relevance_label if False else True
    assert rf.classify("my mixtape is straight fire")["relevance_label"] == "IRRELEVANT"


def test_relevance_keepable_only_relevant():
    assert rf.is_research_keepable(rf.classify("flash flood, homes damaged in Ellicott City"))
    assert not rf.is_research_keepable(rf.classify("wildfire is crazy"))


def test_relevance_location_hint():
    assert rf.extract_location_hint("evacuations near Malibu tonight") == "Malibu"
    assert "Fire" in rf.extract_location_hint("the Eaton Fire grew overnight")


# ---- Strict ("heavy") quality mode -----------------------------------------
def test_strict_keeps_full_context_posts_relevant():
    text = "Mandatory evacuation as the Palisades Fire burns 3000 acres near Los Angeles"
    assert rf.classify(text, strict=True)["relevance_label"] == "RELEVANT"


def test_strict_demotes_posts_missing_location_or_impact():
    # Hazard + emergency but NO location/event -> RELEVANT normally, demoted under strict.
    text = "wildfire evacuation order issued, firefighters on scene"
    assert rf.classify(text)["relevance_label"] == "RELEVANT"
    assert rf.classify(text, strict=True)["relevance_label"] == "POSSIBLY_RELEVANT"


def test_strict_does_not_rescue_irrelevant():
    # Figurative stays IRRELEVANT regardless of mode.
    assert rf.classify("this song is fire", strict=True)["relevance_label"] == "IRRELEVANT"


def test_strict_default_unchanged():
    # strict=False must match the documented default behaviour.
    text = "wildfire is crazy"
    assert rf.classify(text)["relevance_label"] == rf.classify(text, strict=False)["relevance_label"]


def test_strict_covers_typhoon_mudslide_flood():
    cases = [
        "Super typhoon makes landfall near Manila, storm surge flooding homes in Batangas province",
        "Mudslide buried homes near Santa Barbara, residents trapped and roads destroyed",
        "Catastrophic flooding submerged homes in Harris County, rescue teams evacuating families",
    ]
    for text in cases:
        assert rf.classify(text, strict=True)["relevance_label"] == "RELEVANT", text


def test_strict_covers_earthquake():
    text = "Magnitude 7.1 earthquake collapsed buildings in Istanbul, rescue crews searching rubble"
    assert rf.classify(text, strict=True)["relevance_label"] == "RELEVANT"


def test_strict_named_fire_event_counts_as_hazard():
    # A named "<Name> Fire" event is a hazard signal, not just a location.
    text = "Evacuation order issued for Paradise as the Camp Fire spreads toward town"
    r = rf.classify(text, strict=True)
    assert r["has_event_name"] and r["relevance_label"] == "RELEVANT"


def test_strict_matches_plural_emergency_terms():
    # "evacuations" (plural) must count like "evacuation".
    text = "Super typhoon slams Luzon, mass evacuations as floodwaters rise in Manila"
    assert rf.classify(text, strict=True)["relevance_label"] == "RELEVANT"


def test_strict_keeps_international_disaster_posts():
    cases = [
        "Incendio forestal arrasó casas en Valparaíso, evacuación de miles de familias",
        "Enchente destruiu casas em Petrópolis, dezenas de mortos e desaparecidos",
        "Inondation à Marseille, maisons détruites et évacuation des habitants",
        "Malakas na bagyo sa Luzon, baha at paglikas ng mga pamilya, maraming bahay nasira",
        "Banjir besar di Jakarta, ribuan rumah hancur dan korban mengungsi",
    ]
    for text in cases:
        assert rf.classify(text, strict=True)["relevance_label"] == "RELEVANT", text


# ===========================================================================
# Credentials (no token required / never printed)
# ===========================================================================
def test_token_detection(monkeypatch):
    monkeypatch.delenv(xi.BEARER_TOKEN_ENV, raising=False)
    assert xi.get_bearer_token() is None
    assert xi.token_present() is False
    monkeypatch.setenv(xi.BEARER_TOKEN_ENV, "secret-abc")
    assert xi.get_bearer_token() == "secret-abc"
    assert xi.token_present() is True


def test_dry_run_never_prints_token(monkeypatch, capsys):
    monkeypatch.setenv(xi.BEARER_TOKEN_ENV, "SUPER-SECRET-TOKEN")
    rc = xi.main(["--mode", "search", "--target", "2000", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SUPER-SECRET-TOKEN" not in out
    assert "X_BEARER_TOKEN detected: YES" in out
    assert "no API request" in out.lower() or "DRY RUN" in out


# ===========================================================================
# Paid-run guardrails (item 8)
# ===========================================================================
def test_confirm_required_over_threshold():
    assert xi.confirm_required(10, False) is None          # default safe
    assert xi.confirm_required(2000, False) is not None     # needs confirm
    assert xi.confirm_required(2000, True) is None          # confirmed
    assert xi.DEFAULT_TARGET == 10


def test_main_refuses_paid_run_without_confirm(monkeypatch, capsys):
    monkeypatch.setenv(xi.BEARER_TOKEN_ENV, "tok")
    rc = xi.main(["--mode", "search", "--target", "500"])  # no --confirm-paid-run
    out = capsys.readouterr().out
    assert rc == 2
    assert "REFUSING TO RUN" in out


def test_main_stops_without_token(monkeypatch, capsys):
    monkeypatch.delenv(xi.BEARER_TOKEN_ENV, raising=False)
    rc = xi.main(["--mode", "search", "--target", "10"])
    out = capsys.readouterr().out
    assert rc == 2
    assert xi.BEARER_TOKEN_ENV in out


# ===========================================================================
# Query config loading + building
# ===========================================================================
def test_load_queries_parses_labels(tmp_path):
    p = tmp_path / "q.txt"
    p.write_text("# comment\n\nwildfire :: (wildfire) (evacuation)\njust a bare query\n")
    queries = xi.load_queries(p)
    assert ("wildfire", "(wildfire) (evacuation)") in queries
    assert any(expr == "just a bare query" for _, expr in queries)


def test_build_search_query_adds_operators_and_excludes():
    q = xi.build_search_query("(wildfire)", exclude_terms=["song is fire"],
                              lang_en=True, exclude_retweets=True)
    assert "lang:en" in q and "-is:retweet" in q and '-"song is fire"' in q


def test_build_search_query_excludes_accounts():
    q = xi.build_search_query("(tornado)", exclude_accounts=["FoxNews", "CNN"])
    assert "-from:FoxNews" in q and "-from:CNN" in q


def test_build_search_query_respects_max_len_dropping_extras():
    # A tiny max_len keeps the base expression but drops optional exclusions.
    q = xi.build_search_query("(hurricane)", exclude_accounts=["FoxNews"],
                              exclude_terms=["song is fire"], exclude_retweets=False,
                              max_len=12)
    assert q == "((hurricane))"
    assert "-from:" not in q and '-"' not in q


def test_load_exclude_accounts_strips_at_and_comments(tmp_path):
    p = tmp_path / "accts.txt"
    p.write_text("# media accounts\n\n@FoxNews\nCNN\n")
    accounts = xi.load_exclude_accounts(p)
    assert accounts == ["FoxNews", "CNN"]


def test_search_once_uses_full_archive_endpoint(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"data": [], "meta": {}}

    class _Session:
        def get(self, url, headers=None, params=None, timeout=None):
            captured["url"] = url
            return _Resp()

    xi._search_once(_Session(), None, "tok", "q", None, None, None, full_archive=True)
    assert captured["url"] == xi.SEARCH_ALL_URL
    xi._search_once(_Session(), None, "tok", "q", None, None, None, full_archive=False)
    assert captured["url"] == xi.SEARCH_URL


def test_check_auth_uses_counts_endpoint_and_maps_status():
    captured = {}

    def _make_session(status):
        class _Resp:
            status_code = status

        class _Session:
            def get(self, url, headers=None, params=None, timeout=None):
                captured["url"] = url
                return _Resp()

        return _Session()

    assert xi.check_auth("tok", session=_make_session(200))["ok"] is True
    assert captured["url"] == xi.COUNTS_URL
    assert xi.check_auth("tok", session=_make_session(429))["ok"] is True   # throttled but valid
    assert xi.check_auth("tok", session=_make_session(401))["ok"] is False

    # 402 = token valid (auth passed) but no active plan/credits -> can't call yet.
    r402 = xi.check_auth("tok", session=_make_session(402))
    assert r402["ok"] is False and r402["token_valid"] is True
    r403 = xi.check_auth("tok", session=_make_session(403))
    assert r403["ok"] is False and r403["token_valid"] is True


# ===========================================================================
# Parsing mock API responses
# ===========================================================================
SEARCH_PAYLOAD = {
    "data": [
        {"id": "1", "text": "Palisades Fire evacuation near Los Angeles, homes burned",
         "created_at": "2025-01-08T00:00:00.000Z", "lang": "en", "author_id": "a1",
         "conversation_id": "c1", "geo": {"place_id": "p1"}},
        {"id": "2", "text": "this song is fire", "created_at": "2025-01-08T01:00:00.000Z",
         "lang": "en", "author_id": "a2", "conversation_id": "c2"},
    ],
    "includes": {"places": [{"id": "p1", "full_name": "Los Angeles, CA"}]},
    "meta": {"next_token": "TOKEN123"},
}


def test_parse_search_payload_fields_and_next_token():
    records, next_token = xi.parse_search_payload(SEARCH_PAYLOAD, "qA", retrieved_at="RT")
    assert next_token == "TOKEN123"
    assert len(records) == 2
    r0 = records[0]
    assert r0["post_id"] == "1" and r0["lang"] == "en" and r0["author_id"] == "a1"
    assert r0["geo_place_id"] == "p1"
    assert r0["geo_coordinates"] == ""        # no coordinates -> never inferred
    assert r0["query_used"] == "qA" and r0["data_origin"] == "LIVE_API"


def test_parse_search_coordinates_only_when_present():
    payload = {"data": [{"id": "9", "text": "flood", "geo": {
        "coordinates": {"coordinates": [-118.24, 34.05]}}}]}
    records, _ = xi.parse_search_payload(payload, "q")
    # Stored as "lat,lon".
    assert records[0]["geo_coordinates"] == "34.05,-118.24"


def test_missing_place_metadata_is_blank():
    payload = {"data": [{"id": "5", "text": "wildfire smoke AQI unhealthy"}]}
    records, _ = xi.parse_search_payload(payload, "q")
    assert records[0]["geo_place_id"] == "" and records[0]["geo_coordinates"] == ""


def test_media_capture_flags_video_without_excluding():
    payload = {
        "data": [
            {"id": "10", "text": "tornado on the ground, take cover", "lang": "en",
             "attachments": {"media_keys": ["m1", "m2"]}},
            {"id": "11", "text": "text only hurricane update", "lang": "en"},
        ],
        "includes": {"media": [
            {"media_key": "m1", "type": "video"},
            {"media_key": "m2", "type": "photo"},
        ]},
    }
    records, _ = xi.parse_search_payload(payload, "q")
    # Both posts are kept (video is NOT excluded).
    assert len(records) == 2
    vid = records[0]
    assert vid["has_media"] is True and vid["media_types"] == "photo|video"
    assert records[1]["has_media"] is False and records[1]["media_types"] == ""


# ===========================================================================
# Dedup + matched_queries (item 7)
# ===========================================================================
def test_merge_record_dedup_and_matched_queries():
    store = {}
    a = {"post_id": "1", "text": "x", "query_used": "qA", "matched_queries": "qA"}
    b = {"post_id": "1", "text": "x", "query_used": "qB", "matched_queries": "qB"}
    c = {"post_id": "2", "text": "y", "query_used": "qA", "matched_queries": "qA"}
    assert xi.merge_record(store, a) is True
    assert xi.merge_record(store, b) is False   # duplicate id
    assert xi.merge_record(store, c) is True
    assert len(store) == 2
    assert set(store["1"]["matched_queries"].split("|")) == {"qA", "qB"}


# ===========================================================================
# Target stopping condition
# ===========================================================================
def test_target_reached_conditions():
    assert xi.target_reached(10, 50, 10, None) == "target_reached"
    assert xi.target_reached(5, 4000, 2000, 4000) == "candidate_limit_reached"
    assert xi.target_reached(5, 100, 2000, 4000) is None


# ===========================================================================
# Batching
# ===========================================================================
def test_batched():
    assert list(xi.batched(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(xi.batched([], 10)) == []


# ===========================================================================
# Checkpoint save / restart (item 9)
# ===========================================================================
def test_checkpoint_roundtrip_and_restart(tmp_path):
    path = tmp_path / "cp.json"
    assert xi.load_checkpoint(path) == xi.default_checkpoint()  # missing -> default
    state = xi.default_checkpoint()
    state["seen_post_ids"] = ["1", "2"]
    state["relevant_count"] = 2
    state["target"] = 2000
    xi.save_checkpoint(state, path)
    assert path.exists()
    restored = xi.load_checkpoint(path)
    assert restored["seen_post_ids"] == ["1", "2"]
    assert restored["relevant_count"] == 2
    assert restored["target"] == 2000
    assert restored["last_updated"] is not None


def test_checkpoint_corrupt_file_falls_back(tmp_path):
    path = tmp_path / "cp.json"
    path.write_text("{ not valid json")
    assert xi.load_checkpoint(path) == xi.default_checkpoint()


# ===========================================================================
# Error logging never contains credentials (item 10)
# ===========================================================================
def test_log_error_has_fields_and_no_token(tmp_path):
    path = tmp_path / "err.log"
    xi.log_error(xi.SEARCH_URL, "qA", 429, "rate_limited", path=path)
    content = path.read_text()
    row = json.loads(content.strip())
    assert row["http_status"] == 429 and row["error_type"] == "rate_limited"
    assert "authorization" not in content.lower() and "bearer" not in content.lower()


# ===========================================================================
# Lookup payload parse + HumAID merge (item 13)
# ===========================================================================
LOOKUP_PAYLOAD = {
    "data": [
        {"id": "100", "text": "real hydrated text", "created_at": "2017-06-01T00:00:00.000Z",
         "lang": "en", "author_id": "z1", "conversation_id": "k1"},
    ],
    "errors": [
        {"resource_id": "200", "title": "Not Found Error"},
        {"resource_id": "300", "title": "Forbidden"},
    ],
}


def test_parse_lookup_payload_found_and_missing():
    found, missing = xi.parse_lookup_payload(LOOKUP_PAYLOAD)
    assert found[0]["post_id"] == "100" and found[0]["text"] == "real hydrated text"
    assert missing["200"] == "DELETED_OR_PRIVATE"
    assert missing["300"] == "UNAVAILABLE"


def test_merge_hydrated_keeps_metadata_and_no_fake_text():
    id_meta = [
        {"tweet_id": "100", "event": "srilanka_floods_2017", "humanitarian_label": "caution"},
        {"tweet_id": "200", "event": "srilanka_floods_2017", "humanitarian_label": "rescue"},
    ]
    found, missing = xi.parse_lookup_payload(LOOKUP_PAYLOAD)
    found_by_id = {r["post_id"]: r for r in found}
    merged = xi.merge_hydrated(id_meta, found_by_id, missing, retrieved_at="RT")
    by_id = {m["post_id"]: m for m in merged}
    assert by_id["100"]["hydration_status"] == "SUCCESS"
    assert by_id["100"]["text"] == "real hydrated text"
    assert by_id["100"]["event"] == "srilanka_floods_2017"
    assert by_id["100"]["data_origin"] == "REAL_DATASET_HYDRATED"
    # Unavailable id gets NO fabricated text.
    assert by_id["200"]["hydration_status"] == "DELETED_OR_PRIVATE"
    assert by_id["200"]["text"] == ""


def test_read_tweet_ids_merges_humaid_columns(tmp_path):
    p = tmp_path / "ids.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tweet_id", "event", "humanitarian_label"])
        w.writerow(["100", "greece_wildfires_2018", "caution_and_advice"])
    rows = xi.read_tweet_ids(p)
    assert rows == [{"tweet_id": "100", "event": "greece_wildfires_2018",
                     "humanitarian_label": "caution_and_advice"}]


# ===========================================================================
# Summaries built from mock candidates (no network; no fake data when empty)
# ===========================================================================
def _mock_candidates():
    rows = [
        {"post_id": "1", "text": "Palisades Fire evacuation near Los Angeles homes burned",
         "lang": "en", "geo_place_id": "p1", "geo_coordinates": "", "query_used": "qA"},
        {"post_id": "2", "text": "this song is fire", "lang": "en",
         "geo_place_id": "", "geo_coordinates": "", "query_used": "qA"},
        {"post_id": "3", "text": "wildfire is crazy", "lang": "es",
         "geo_place_id": "", "geo_coordinates": "34.0,-118.0", "query_used": "qB"},
    ]
    return rows


def test_relevance_metrics_counts():
    classified = w5s._classify_all(_mock_candidates())
    m = w5s.build_relevance_metrics(classified, duplicate_posts_removed=4)
    assert m["raw_candidates"] == 3
    assert m["relevant_posts"] == 1          # only post 1
    assert m["possibly_relevant"] == 1       # post 3
    assert m["irrelevant_posts"] == 1        # post 2 (figurative)
    assert m["figurative_posts_rejected"] == 1
    assert m["duplicate_posts_removed"] == 4
    assert 0.0 <= m["relevance_retention_rate"] <= 1.0


def test_collection_summary_place_and_coord_counts():
    candidates = _mock_candidates()
    classified = w5s._classify_all(candidates)
    checkpoint = xi.default_checkpoint()
    checkpoint.update({"candidates_retrieved": 5, "target": 2000, "queries_used": ["qA", "qB"]})
    summary, dup = w5s.build_collection_summary(candidates, classified, checkpoint)
    assert summary["unique_posts_collected"] == 3
    assert dup == 2                                   # 5 retrieved - 3 unique
    assert summary["posts_with_place_metadata"] == 1  # post 1
    assert summary["posts_with_exact_coordinates"] == 1  # post 3
    assert "en:2" in summary["language_distribution"]


def test_acceptance_check_flags_low_retention():
    status, _ = w5s.acceptance_check({"raw_candidates": 100, "relevance_retention_rate": 0.05})
    assert status == "STOP"
    status2, _ = w5s.acceptance_check({"raw_candidates": 100, "relevance_retention_rate": 0.60})
    assert status2 == "OK"


def test_review_sample_capped_and_has_text():
    classified = w5s._classify_all(_mock_candidates())
    sample = w5s.build_review_sample(classified, size=50)
    assert len(sample) <= 50
    assert set(w5s.REVIEW_SAMPLE_FIELDS).issubset(set(sample[0].keys()))


# ===========================================================================
# NASA-ready summary (item 16) from mock NER results
# ===========================================================================
def test_nasa_ready_summary_counts():
    results = [
        {"post_id": "1", "created_at": "2025-01-08T00:00:00+00:00", "primary_location": "Los Angeles",
         "latitude": 34.0, "longitude": -118.0},
        {"post_id": "2", "created_at": "", "primary_location": "Malibu",
         "latitude": 34.0, "longitude": -118.7},   # no timestamp -> not nasa-ready
        {"post_id": "3", "created_at": "2025-01-08T02:00:00+00:00", "primary_location": "",
         "latitude": None, "longitude": None},       # no location
    ]
    s = w5ner.build_nasa_ready_summary(results)
    assert s["total_unique_posts"] == 3
    assert s["posts_with_detected_location"] == 2
    assert s["successfully_geocoded"] == 2
    assert s["posts_with_usable_timestamp"] == 2
    assert s["potentially_nasa_ready"] == 1          # only post 1 has both


# ===========================================================================
# No-data safety: generators do not fabricate results
# ===========================================================================
def test_summaries_run_no_data_returns_none(tmp_path):
    assert w5s.run(raw_path=tmp_path / "missing.csv", checkpoint_path=tmp_path / "cp.json") is None


def test_ner_run_no_data_returns_none(tmp_path):
    assert w5ner.run(input_path=tmp_path / "missing.csv") is None
