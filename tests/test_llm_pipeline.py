"""Offline tests for the LLM-annotation + satellite-verification pipeline.

No network: the LLM is faked via a ``responder`` callable, and geocode / FIRMS
are injected. No X API credits and no Ollama server are required.
"""
import csv
import json
from datetime import date

import firms
import llm_annotate as la
import verify_tweets as vt


# ---- FIRMS source selection (NRT vs SP archive) ----------------------------
def test_firms_choose_source_recent_vs_archive():
    today = date(2025, 6, 1)
    # A date a few days ago -> NRT.
    assert firms.choose_source("2025-05-20", today=today) == firms.SOURCE_NRT
    # A 2-year-old tweet -> SP archive (this is the whole point for old tweets).
    assert firms.choose_source("2023-07-01", today=today) == firms.SOURCE_ARCHIVE
    # Full ISO timestamp is accepted.
    assert firms.choose_source("2023-07-01T12:00:00Z", today=today) == firms.SOURCE_ARCHIVE
    # Unparseable/empty -> archive (safer for old data).
    assert firms.choose_source("", today=today) == firms.SOURCE_ARCHIVE
    assert firms.choose_source("not-a-date", today=today) == firms.SOURCE_ARCHIVE


# ---- LLM annotation (faked model) ------------------------------------------
def _responder(payload):
    return lambda prompt: json.dumps(payload)


def test_annotate_text_parses_and_coerces():
    ann = la.annotate_text("Palisades Fire burning near Malibu, evacuations ordered",
                           responder=_responder({
                               "is_hazard_claim": True,
                               "hazard_type": "wildfire",
                               "location": "Malibu, California",
                               "claim": "Palisades Fire burning near Malibu",
                               "stance": "asserts_event",
                               "misinfo_risk": "medium",
                               "confidence": 0.9,
                               "reason": "names a fire and a place",
                           }))
    assert ann["llm_status"] == "ok"
    assert ann["llm_is_hazard_claim"] is True
    assert ann["llm_hazard_type"] == "wildfire"
    assert ann["llm_location"] == "Malibu, California"
    assert ann["llm_confidence"] == 0.9


def test_annotate_text_coerces_invalid_enum_values():
    ann = la.annotate_text("something", responder=_responder({
        "is_hazard_claim": True,
        "hazard_type": "volcano",       # not in taxonomy -> "other"
        "stance": "???",                # invalid -> "unrelated"
        "misinfo_risk": "extreme",      # invalid -> "low"
        "confidence": "high",           # non-numeric -> 0.0
    }))
    assert ann["llm_hazard_type"] == "other"
    assert ann["llm_stance"] == "unrelated"
    assert ann["llm_misinfo_risk"] == "low"
    assert ann["llm_confidence"] == 0.0


def test_annotate_text_handles_unparseable_and_empty():
    assert la.annotate_text("x", responder=lambda p: "not json")["llm_status"] == "unparseable_json"
    assert la.annotate_text("", responder=lambda p: "{}")["llm_status"] == "empty_text"


def test_annotate_file_writes_and_resumes(tmp_path):
    inp = tmp_path / "in.csv"
    with open(inp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["post_id", "text", "relevance_label"])
        w.writeheader()
        w.writerow({"post_id": "1", "text": "wildfire near Malibu", "relevance_label": "RELEVANT"})
        w.writerow({"post_id": "2", "text": "this song is fire", "relevance_label": "IRRELEVANT"})
    out = tmp_path / "out.csv"

    resp = _responder({"is_hazard_claim": True, "hazard_type": "wildfire",
                       "location": "Malibu", "stance": "asserts_event",
                       "misinfo_risk": "low", "confidence": 0.5})
    res = la.annotate_file(input_path=inp, output_path=out, responder=resp)
    # only_relevant default -> only post 1 annotated.
    assert res["annotated"] == 1 and res["ok"] == 1
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["post_id"] for r in rows] == ["1"]

    # Re-run: post 1 already done -> nothing new annotated (resumable).
    res2 = la.annotate_file(input_path=inp, output_path=out, responder=resp)
    assert res2["annotated"] == 0


# ---- verify_tweets adapter (geocode + FIRMS injected) ----------------------
def _geo_hit(_location):
    return {"latitude": 34.03, "longitude": -118.78, "display_name": "Malibu, CA"}


def test_verify_row_wildfire_supported_when_detection_near():
    row = {"post_id": "1", "created_at": "2023-07-01T00:00:00Z",
           "llm_hazard_type": "wildfire", "llm_location": "Malibu"}
    det = lambda lat, lon, d, source: [{"latitude": "34.04", "longitude": "-118.79"}]
    rec = vt.verify_row(row, geocode_fn=_geo_hit, detections_fn=det)
    assert rec["verdict"] == "SUPPORTED"
    assert rec["firms_source"] == firms.SOURCE_ARCHIVE   # old date -> archive
    assert rec["latitude"] == 34.03


def test_verify_row_wildfire_no_match_when_detection_far():
    row = {"post_id": "1", "created_at": "2023-07-01", "llm_hazard_type": "wildfire",
           "llm_location": "Malibu"}
    det = lambda lat, lon, d, source: [{"latitude": "40.0", "longitude": "-100.0"}]
    rec = vt.verify_row(row, geocode_fn=_geo_hit, detections_fn=det)
    assert rec["verdict"] == "NO_MATCH"


def test_verify_row_non_fire_is_not_applicable():
    row = {"post_id": "2", "created_at": "2023-07-01", "llm_hazard_type": "flood",
           "llm_location": "Houston"}
    rec = vt.verify_row(row, geocode_fn=_geo_hit, detections_fn=lambda *a: [])
    assert rec["verdict"] == "NOT_APPLICABLE_NON_FIRE"


def test_verify_row_wildfire_insufficient_without_location():
    row = {"post_id": "3", "created_at": "2023-07-01", "llm_hazard_type": "wildfire",
           "llm_location": ""}
    rec = vt.verify_row(row, geocode_fn=lambda loc: None, detections_fn=lambda *a: [])
    assert rec["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_verify_row_firms_disabled_is_insufficient():
    row = {"post_id": "4", "created_at": "2023-07-01", "llm_hazard_type": "wildfire",
           "llm_location": "Malibu"}
    rec = vt.verify_row(row, firms_enabled=False, geocode_fn=_geo_hit,
                        detections_fn=lambda *a: [{"latitude": "34.04", "longitude": "-118.79"}])
    assert rec["verdict"] == "INSUFFICIENT_EVIDENCE"
