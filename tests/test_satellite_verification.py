"""Offline tests for multi-sensor satellite verification (no network)."""
import csv
import json
from datetime import date

import pytest
import requests

import firms
import satellite_verification as sv

CLAIM = {
    "post_id": "1", "text": "Fire burning near San Clemente tonight", "created_at": "2026-10-20T03:00:00.000Z",
    "llm_is_hazard_claim": "True", "llm_hazard_type": "wildfire", "llm_location": "San Clemente, California",
    "llm_claim": "fire near San Clemente", "llm_stance": "asserts_event", "llm_misinfo_risk": "low",
    "llm_confidence": "0.9", "llm_reason": "",
}
COORDS = {"latitude": 33.4270, "longitude": -117.6120, "display_name": "San Clemente, CA"}


def _geocode(location, text=""):
    return COORDS if location else None


def _det(lat, lon, acq_date, acq_time, frp="12.5", instrument="VIIRS"):
    return {"latitude": str(lat), "longitude": str(lon), "acq_date": acq_date, "acq_time": acq_time,
            "satellite": "N20", "instrument": instrument, "confidence": "n", "frp": frp}


def _fetch_with(dets_by_source):
    def fetch(lat, lon, start, source=None, day_range=None):
        return dets_by_source.get(source, []), "ok", ""
    return fetch


# ---- firms.py status handling ------------------------------------------------
class _Resp:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_fetch_area_status_ok_and_empty(monkeypatch):
    header = "latitude,longitude,acq_date,acq_time,satellite,instrument,confidence,frp\n"
    monkeypatch.setattr(firms.requests, "get", lambda url, timeout: _Resp(header + "33.4,-117.6,2026-10-20,0930,N20,VIIRS,n,10.0\n"))
    dets, status, _ = firms.fetch_area_status("b", "2026-10-19", map_key="k", source="VIIRS_NOAA20_NRT")
    assert status == "ok" and len(dets) == 1 and dets[0]["frp"] == "10.0"

    monkeypatch.setattr(firms.requests, "get", lambda url, timeout: _Resp(header))
    dets, status, _ = firms.fetch_area_status("b", "2026-10-19", map_key="k", source="VIIRS_NOAA20_NRT")
    assert status == "ok" and dets == []


def test_fetch_area_status_distinguishes_errors(monkeypatch):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    assert firms.fetch_area_status("b", "2026-10-19")[1] == "no_key"

    monkeypatch.setattr(firms.requests, "get", lambda url, timeout: _Resp("Invalid MAP_KEY."))
    dets, status, msg = firms.fetch_area_status("b", "2026-10-19", map_key="bad")
    assert dets is None and status == "invalid_response" and "Invalid" in msg

    def boom(url, timeout):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(firms.requests, "get", boom)
    assert firms.fetch_area_status("b", "2026-10-19", map_key="k")[1] == "api_error"
    # Back-compat wrapper still returns None on failure.
    assert firms.fetch_area("b", "2026-10-19", map_key="k") is None


def test_choose_sensor_source():
    today = date(2026, 11, 8)
    assert firms.choose_sensor_source("2026-11-01", "MODIS", today=today) == "MODIS_NRT"
    assert firms.choose_sensor_source("2025-10-01", "MODIS", today=today) == "MODIS_SP"
    assert firms.choose_sensor_source("2025-10-01", "VIIRS", today=today) == "VIIRS_SNPP_SP"


# ---- spatial + temporal matching --------------------------------------------
def test_annotate_detections_distance_and_time():
    claim_dt = sv.parse_claim_time(CLAIM["created_at"])
    near = _det(33.45, -117.60, "2026-10-20", "0930")       # ~2.8 km, +6.5 h
    far = _det(33.90, -117.60, "2026-10-20", "0930")        # ~52 km
    late = _det(33.45, -117.60, "2026-10-23", "0930")       # +78.5 h -> outside 48 h
    out = sv.annotate_detections(COORDS["latitude"], COORDS["longitude"], claim_dt, [near, far, late])
    assert [d["matched"] for d in out] == [True, False, False]
    assert out[0]["distance_km"] < 5 and out[0]["hours_diff"] == 6.5
    assert out[1]["distance_km"] > 25
    assert abs(out[2]["hours_diff"]) > sv.TIME_HOURS


def test_supported_when_any_sensor_matches():
    fetch = _fetch_with({"VIIRS_NOAA20_NRT": [_det(33.45, -117.60, "2026-10-20", "0930")],
                         "MODIS_NRT": []})
    rec, dets = sv.verify_row(CLAIM, geocode_fn=_geocode, fetch_fn=fetch)
    assert rec["verdict"] == "SUPPORTED"
    assert rec["matched_sensors"] == "VIIRS"
    assert rec["viirs_detections_matched"] == 1 and rec["modis_detections_matched"] == 0
    assert rec["viirs_source"] == "VIIRS_NOAA20_NRT" and rec["modis_source"] == "MODIS_NRT"
    assert 0 < rec["evidence_score"] <= 1
    assert "does not verify every detail" in rec["explanation"]
    assert len(dets["VIIRS"]) == 1 and dets["VIIRS"][0]["matched"] is True


def test_no_nearby_detection_is_not_false():
    fetch = _fetch_with({"VIIRS_NOAA20_NRT": [_det(33.90, -117.60, "2026-10-20", "0930")], "MODIS_NRT": []})
    rec, _ = sv.verify_row(CLAIM, geocode_fn=_geocode, fetch_fn=fetch)
    assert rec["verdict"] == "NO_NEARBY_DETECTION"
    assert rec["viirs_detections_total"] == 1 and rec["viirs_detections_matched"] == 0
    assert rec["evidence_score"] == 0.0
    assert "NOT proof the post is false" in rec["explanation"]
    assert "false" not in rec["verdict"].lower()


def test_api_error_when_all_sensors_fail():
    def fetch(lat, lon, start, source=None, day_range=None):
        return None, "api_error", "ConnectionError"
    rec, _ = sv.verify_row(CLAIM, geocode_fn=_geocode, fetch_fn=fetch)
    assert rec["verdict"] == "API_ERROR"
    assert rec["viirs_status"] == "api_error" and rec["modis_status"] == "api_error"


def test_partial_api_error_still_uses_working_sensor():
    def fetch(lat, lon, start, source=None, day_range=None):
        if source.startswith("MODIS"):
            return None, "api_error", "timeout"
        return [], "ok", ""
    rec, _ = sv.verify_row(CLAIM, geocode_fn=_geocode, fetch_fn=fetch)
    assert rec["verdict"] == "NO_NEARBY_DETECTION"
    assert rec["modis_status"] == "api_error" and rec["viirs_status"] == "ok"


def test_no_coverage_from_availability_table():
    # Real FIRMS shape: NRT starts after the SP archive ends -> a gap month.
    avail = {"VIIRS_NOAA20_NRT": ("2026-09-01", "2026-11-08"), "VIIRS_SNPP_SP": ("2012-01-20", "2026-07-31"),
             "MODIS_NRT": ("2026-09-01", "2026-11-08"), "MODIS_SP": ("2000-11-01", "2026-07-31")}
    called = []
    def fetch(*a, **k):
        called.append(1)
        return [], "ok", ""
    gap = dict(CLAIM, created_at="2026-08-15T00:00:00Z")
    rec, _ = sv.verify_row(gap, geocode_fn=_geocode, fetch_fn=fetch, availability=avail)
    assert rec["verdict"] == "NO_SATELLITE_COVERAGE" and not called
    # A date inside the archive picks the SP product even though it is >60 days old.
    old = dict(CLAIM, created_at="2025-10-01T00:00:00Z")
    rec, _ = sv.verify_row(old, geocode_fn=_geocode, fetch_fn=fetch, availability=avail)
    assert rec["viirs_source"] == "VIIRS_SNPP_SP" and rec["modis_source"] == "MODIS_SP"
    assert rec["verdict"] == "NO_NEARBY_DETECTION"


def test_non_fire_and_unresolved_location():
    flood = dict(CLAIM, llm_hazard_type="flood")
    rec, _ = sv.verify_row(flood, geocode_fn=_geocode, fetch_fn=_fetch_with({}))
    assert rec["verdict"] == "NOT_APPLICABLE_NON_FIRE"
    assert rec["latitude"] == COORDS["latitude"]  # still geocoded for the map

    rec, _ = sv.verify_row(dict(CLAIM, llm_location=""), geocode_fn=_geocode, fetch_fn=_fetch_with({}))
    assert rec["verdict"] == "LOCATION_UNRESOLVED"
    rec, _ = sv.verify_row(CLAIM, geocode_fn=lambda loc, text="": None, fetch_fn=_fetch_with({}))
    assert rec["verdict"] == "LOCATION_UNRESOLVED"


# ---- location resolution ------------------------------------------------------
def test_query_variants_simplify_progressively():
    v = sv.query_variants("George H. Crosby Manitou State Park, northern Minnesota")
    assert v[0] == "George H. Crosby Manitou State Park, northern Minnesota"
    assert "northern Minnesota" in v and "Minnesota" in v
    v = sv.query_variants("5875 Brenda Mines Road (ZipZone Adventure Park)")
    assert "5875 Brenda Mines Road" in v
    assert "Nerja" in sv.query_variants("Pico del Almendrón in Nerja")
    assert "San Clemente" in sv.query_variants("San Clemente and San Juan Capistrano")
    assert sv.query_variants("") == []


def test_resolve_location_flags_ambiguous_names():
    cands = {
        "Palisades": [
            {"latitude": 41.0, "longitude": -73.9, "display_name": "Palisades, Rockland County, New York", "importance": 0.36, "type": "hamlet"},
            {"latitude": 38.9, "longitude": -77.1, "display_name": "The Palisades, Washington, DC", "importance": 0.33, "type": "neighbourhood"},
        ],
        "San Clemente, California": [
            {"latitude": 33.4, "longitude": -117.6, "display_name": "San Clemente, Orange County, California", "importance": 0.62, "type": "city"},
        ],
    }
    amb = sv.resolve_location("Palisades", "Palisades Fire report", candidates_fn=lambda q: cands.get(q, []))
    assert amb["ambiguity_status"] == "ambiguous" and amb["candidate_count"] == 2
    ok = sv.resolve_location("San Clemente, California", "", candidates_fn=lambda q: cands.get(q, []))
    assert ok["ambiguity_status"] == "resolved" and ok["geocode_query"] == "San Clemente, California"
    # One obscure (low-importance) candidate is still unambiguous.
    one = sv.resolve_location("Chiricahua Wilderness", "", candidates_fn=lambda q: [
        {"latitude": 31.8, "longitude": -109.3, "display_name": "Chiricahua Wilderness, Arizona", "importance": 0.08, "type": "protected_area"}])
    assert one["ambiguity_status"] == "resolved" and one["location_confidence"] == 0.08


def test_resolve_location_uses_fallback_and_ignores_junk():
    calls = []
    def cands(q):
        calls.append(q)
        if q == "northern Minnesota":   # Nominatim returns street-level junk here
            return [{"latitude": 1, "longitude": 1, "display_name": "Minnesota Street, Manila", "importance": 0.05, "type": "road"}]
        if q == "Minnesota":
            return [{"latitude": 46.3, "longitude": -94.3, "display_name": "Minnesota, United States", "importance": 0.75, "type": "state"}]
        return []
    r = sv.resolve_location("George H. Crosby Manitou State Park, northern Minnesota", "", candidates_fn=cands)
    assert r["geocode_query"] == "Minnesota" and r["display_name"].startswith("Minnesota")
    assert calls[0].startswith("George H. Crosby")
    assert sv.resolve_location("Nowhere Land", "", candidates_fn=lambda q: []) is None


def test_explanation_carries_ambiguity_caveat():
    amb = dict(COORDS, geocode_query="Palisades", location_confidence=0.36, ambiguity_status="ambiguous")
    fetch = _fetch_with({"VIIRS_NOAA20_NRT": [_det(33.45, -117.60, "2026-10-20", "0930")], "MODIS_NRT": []})
    rec, _ = sv.verify_row(dict(CLAIM, llm_location="Palisades"), geocode_fn=lambda l, t="": amb, fetch_fn=fetch)
    assert rec["verdict"] == "SUPPORTED" and rec["ambiguity_status"] == "ambiguous"
    assert "ambiguous place name" in rec["explanation"]


def test_cached_fetch_hits_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(sv, "FIRMS_CACHE_PATH", tmp_path / "c.json")
    monkeypatch.setattr(sv, "_firms_cache", None)
    calls = []
    def fake(lat, lon, start, source=None, day_range=None):
        calls.append(1)
        return [_det(1, 1, "2026-10-20", "0000")], "ok", ""
    monkeypatch.setattr(firms, "fetch_detections_status", fake)
    sv.cached_fetch(1, 1, "2026-10-19", source="MODIS_NRT")
    dets, status, msg = sv.cached_fetch(1, 1, "2026-10-19", source="MODIS_NRT")
    assert len(calls) == 1 and msg == "cached" and len(dets) == 1
    sv.save_cache()
    assert (tmp_path / "c.json").exists()


def test_no_key_marks_api_error_without_fetching():
    rec, _ = sv.verify_row(CLAIM, geocode_fn=_geocode, firms_enabled=False)
    assert rec["verdict"] == "API_ERROR" and rec["viirs_status"] == "no_key"


def test_evidence_score_monotonic():
    assert sv.evidence_score(1.0, 1.0, 5) > sv.evidence_score(20.0, 40.0, 1) > 0
    assert sv.evidence_score(5.0, 5.0, 0) == 0.0


# ---- file round-trip ----------------------------------------------------------
def test_run_writes_csv_and_detections(tmp_path, monkeypatch):
    inp = tmp_path / "ann.csv"
    with open(inp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CLAIM))
        w.writeheader()
        w.writerow(CLAIM)
        w.writerow(dict(CLAIM, post_id="2", llm_hazard_type="flood"))
    monkeypatch.setattr(sv, "resolve_location", _geocode)
    monkeypatch.setattr(sv, "cached_fetch",
                        _fetch_with({"VIIRS_NOAA20_NRT": [_det(33.45, -117.60, "2026-10-20", "0930")]}))
    monkeypatch.setattr(sv, "save_cache", lambda: None)
    monkeypatch.setattr(firms, "data_availability", lambda: None)
    out, dets = tmp_path / "out.csv", tmp_path / "dets.json"
    results = sv.run(input_path=inp, output_path=out, detections_path=dets, firms_enabled=True)
    assert sv.summarize(results) == {"SUPPORTED": 1, "NOT_APPLICABLE_NON_FIRE": 1}
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["verdict"] == "SUPPORTED" and set(sv.OUTPUT_FIELDS) <= set(rows[0])
    store = json.loads(dets.read_text())
    assert store["1"]["VIIRS"][0]["distance_km"] < 5
    # Resume: nothing new to do, existing rows preserved.
    again = sv.run(input_path=inp, output_path=out, detections_path=dets, firms_enabled=True)
    assert len(again) == 2
