"""Streamlit integration tests: every page renders without raising.

Uses Streamlit's AppTest against a small synthetic dataset written to a temp
directory, so the tests are hermetic (no network, no real post text).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = {"views/overview.py": "Overview", "views/live.py": "Live fire map", "views/map.py": "Claims map",
         "views/claims.py": "Claim explorer", "views/verification.py": "Verification", "views/evaluation.py": "Evaluation",
         "views/cases.py": "Case studies", "views/check.py": "Check a claim", "views/data.py": "Collected data",
         "views/sources.py": "Data sources & keys", "views/dataset.py": "Annotated dataset"}


def goto(at, page):
    at.run()
    return at.switch_page(page).run()


def _fake_dataset(tmp_path):
    processed = tmp_path / "data" / "processed"
    evald = tmp_path / "data" / "evaluation"
    processed.mkdir(parents=True)
    evald.mkdir(parents=True)
    ann = pd.DataFrame([
        {"post_id": "1", "text": "Brush fire near San Clemente tonight", "created_at": "2026-10-20T03:00:00.000Z",
         "lang": "en", "geo_coordinates": "", "has_media": False, "media_types": "", "llm_is_hazard_claim": True,
         "llm_hazard_type": "wildfire", "llm_location": "San Clemente, California", "llm_claim": "fire",
         "llm_stance": "asserts_event", "llm_misinfo_risk": "low", "llm_confidence": 0.9, "llm_reason": "r",
         "llm_status": "ok"},
        {"post_id": "2", "text": "Fire in Gaviota", "created_at": "2025-10-31T16:07:30.000Z", "lang": "en",
         "geo_coordinates": "", "has_media": False, "media_types": "", "llm_is_hazard_claim": True,
         "llm_hazard_type": "wildfire", "llm_location": "Gaviota", "llm_claim": "fire", "llm_stance": "asserts_event",
         "llm_misinfo_risk": "high", "llm_confidence": 0.8, "llm_reason": "r", "llm_status": "ok"},
        {"post_id": "3", "text": "Flooding in Houston", "created_at": "2026-10-21T03:00:00.000Z", "lang": "en",
         "geo_coordinates": "", "has_media": False, "media_types": "", "llm_is_hazard_claim": True,
         "llm_hazard_type": "flood", "llm_location": "Houston", "llm_claim": "flood", "llm_stance": "asserts_event",
         "llm_misinfo_risk": "low", "llm_confidence": 0.9, "llm_reason": "r", "llm_status": "ok"},
    ])
    ann.to_csv(processed / "x_llm_annotated.csv", index=False)

    import satellite_verification as sv
    base = {f: "" for f in sv.OUTPUT_FIELDS}
    ver = pd.DataFrame([
        {**base, "post_id": "1", "text": ann.text[0], "created_at": ann.created_at[0], "llm_hazard_type": "wildfire",
         "llm_location": "San Clemente, California", "llm_claim": "fire", "llm_misinfo_risk": "low",
         "claim_date_utc": "2026-10-20T03:00:00+00:00", "latitude": 33.427, "longitude": -117.612,
         "display_name": "San Clemente, Orange County, California, United States",
         "geocode_query": "San Clemente, California", "location_confidence": 0.51, "ambiguity_status": "resolved",
         "viirs_source": "VIIRS_NOAA20_NRT", "viirs_status": "ok", "viirs_detections_total": 2,
         "viirs_detections_matched": 1, "viirs_nearest_km": 2.8, "viirs_nearest_hours": 6.5, "viirs_max_frp": 3.1,
         "viirs_observation_date": "2026-10-20", "modis_source": "MODIS_NRT", "modis_status": "ok",
         "modis_detections_total": 0, "modis_detections_matched": 0, "verdict": "SUPPORTED",
         "matched_sensors": "VIIRS", "nearest_detection_km": 2.8, "nearest_detection_hours": 6.5,
         "evidence_score": 0.7, "explanation": "supported", "evidence_source": "NASA FIRMS (VIIRS + MODIS)"},
        {**base, "post_id": "2", "text": ann.text[1], "created_at": ann.created_at[1], "llm_hazard_type": "wildfire",
         "llm_location": "Gaviota", "llm_claim": "fire", "llm_misinfo_risk": "high",
         "claim_date_utc": "2025-10-31T16:07:30+00:00", "latitude": 34.47, "longitude": -120.21,
         "display_name": "Gaviota, Santa Barbara County, California, United States",
         "geocode_query": "Gaviota", "location_confidence": 0.45, "ambiguity_status": "resolved",
         "viirs_source": "VIIRS_SNPP_SP", "viirs_status": "ok", "viirs_detections_total": 0,
         "viirs_detections_matched": 0, "modis_source": "MODIS_SP", "modis_status": "ok",
         "modis_detections_total": 0, "modis_detections_matched": 0, "verdict": "NO_NEARBY_DETECTION",
         "evidence_score": 0.0, "explanation": "none", "evidence_source": "NASA FIRMS (VIIRS + MODIS)"},
        {**base, "post_id": "3", "text": ann.text[2], "created_at": ann.created_at[2], "llm_hazard_type": "flood",
         "llm_location": "Houston", "latitude": 29.76, "longitude": -95.37, "display_name": "Houston, Texas, United States",
         "verdict": "NOT_APPLICABLE_NON_FIRE", "evidence_score": 0.0, "explanation": "n/a",
         "evidence_source": "NASA FIRMS (VIIRS + MODIS)"},
    ])
    ver.to_csv(processed / "satellite_verification.csv", index=False)
    (processed / "satellite_detections.json").write_text(json.dumps({"1": {"VIIRS": [
        {"latitude": "33.45", "longitude": "-117.60", "acq_date": "2026-10-20", "acq_time": "0930", "satellite": "N20",
         "instrument": "VIIRS", "confidence": "n", "frp": "3.1", "distance_km": 2.8, "hours_diff": 6.5, "matched": True},
        {"latitude": "33.90", "longitude": "-117.60", "acq_date": "2026-10-20", "acq_time": "0930", "satellite": "N20",
         "instrument": "VIIRS", "confidence": "n", "frp": "1.0", "distance_km": 52.0, "hours_diff": 6.5, "matched": False},
    ], "MODIS": []}}))
    pd.DataFrame({"stage": ["Annotated posts", "Wildfire"], "posts": [3, 2]}).to_csv(evald / "funnel.csv", index=False)
    pd.DataFrame([
        {"sensor": "VIIRS", "products_used": "VIIRS_NOAA20_NRT", "claims_queried_ok": 2, "claims_with_any_detection_in_box": 1,
         "claims_with_matched_detection": 1, "match_rate": 0.5, "median_nearest_km": 2.8, "median_abs_hours": 6.5,
         "median_max_frp_mw": 3.1, "viirs_only_matches": None, "modis_only_matches": None},
        {"sensor": "MODIS", "products_used": "MODIS_NRT", "claims_queried_ok": 2, "claims_with_any_detection_in_box": 0,
         "claims_with_matched_detection": 0, "match_rate": 0.0, "median_nearest_km": None, "median_abs_hours": None,
         "median_max_frp_mw": None, "viirs_only_matches": None, "modis_only_matches": None},
        {"sensor": "BOTH_OK", "products_used": "", "claims_queried_ok": 2, "claims_with_any_detection_in_box": None,
         "claims_with_matched_detection": 0, "match_rate": None, "median_nearest_km": None, "median_abs_hours": None,
         "median_max_frp_mw": None, "viirs_only_matches": 1, "modis_only_matches": 0},
    ]).to_csv(evald / "sensor_comparison.csv", index=False)
    (evald / "summary.json").write_text(json.dumps({"annotated_posts": 3, "wildfire_posts": 2, "supported": 1,
                                                    "wildfire_posts_verified": 2}))
    return tmp_path


@pytest.fixture
def app(tmp_path, monkeypatch):
    root = _fake_dataset(tmp_path)
    import dashboard.data as dd
    monkeypatch.setattr(dd, "DATA", root / "data")
    monkeypatch.setattr(dd, "PROCESSED", root / "data" / "processed")
    monkeypatch.setattr(dd, "EVAL", root / "data" / "evaluation")
    monkeypatch.setattr(dd, "spacy_model", lambda: None)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    import dashboard.live as dl
    monkeypatch.setattr(dl, "PROCESSED", root / "data" / "processed")
    monkeypatch.setattr(dl, "ENV_PATH", root / ".env")
    monkeypatch.setattr(dl.firms, "data_availability", lambda: None)
    monkeypatch.setattr(dl.firms, "fetch_area_status", lambda *a, **k: ([], "ok", ""))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    return at


@pytest.mark.parametrize("page", list(PAGES))
def test_each_page_renders_without_exception(app, page):
    goto(app, page)
    assert not app.exception, [e.message for e in app.exception]
    # Our own error boundary (dashboard.shell.render) must not have fired either.
    assert not any("Error" in e.value and ":" in e.value for e in app.error if "FIRMS_MAP_KEY" not in e.value)
    assert app.title[0].value == PAGES[page]


def test_overview_metrics_reflect_dataset(app):
    app.run()
    labels = {m.label: m.value for m in app.metric}
    assert labels["Annotated posts"] == "3"
    assert labels["Wildfire posts (FIRMS-checkable)"] == "2"
    assert labels["Satellite-supported"] == "1"


def test_live_check_requires_key(app):
    goto(app, "views/check.py")
    app.text_input[0].set_value("San Clemente, California")
    app.button[0].click().run()
    assert any("FIRMS_MAP_KEY" in e.value for e in app.error)


def test_live_fire_map_without_key_shows_instructions(app):
    goto(app, "views/live.py")
    assert any("FIRMS_MAP_KEY" in e.value for e in app.error)


def test_pages_show_empty_state_without_data(tmp_path, monkeypatch):
    import dashboard.data as dd
    import dashboard.live as dl
    (tmp_path / "data" / "processed").mkdir(parents=True)
    monkeypatch.setattr(dd, "DATA", tmp_path / "data")
    monkeypatch.setattr(dd, "PROCESSED", tmp_path / "data" / "processed")
    monkeypatch.setattr(dd, "EVAL", tmp_path / "data" / "evaluation")
    monkeypatch.setattr(dl, "PROCESSED", tmp_path / "data" / "processed")
    monkeypatch.setattr(dl, "ENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    for page in PAGES:
        goto(at, page)
        assert not at.exception, (page, [e.message for e in at.exception])


def test_annotated_dataset_merges_verdicts_and_exports_excel():
    import io
    import sys
    sys.path.insert(0, str(ROOT))
    from dashboard.data import annotated_dataset, to_excel_bytes
    ann = pd.DataFrame({"post_id": ["1", "2"], "text": ["fire", "flood"],
                        "created_at": pd.to_datetime(["2026-10-20T03:00Z", "2026-10-21T03:00Z"], utc=True),
                        "llm_hazard_type": ["wildfire", "flood"], "llm_location": ["Malibu", "Houston"]})
    ver = pd.DataFrame({"post_id": ["1"], "verdict": ["SUPPORTED"], "nearest_detection_km": [3.2]})
    df = annotated_dataset(ann, ver)
    assert dict(zip(df["post_id"], df["verdict"])) == {"1": "SUPPORTED", "2": "NOT_VERIFIED"}
    back = pd.read_excel(io.BytesIO(to_excel_bytes(df)), dtype={"post_id": str})
    assert len(back) == 2 and {"llm_hazard_type", "verdict", "nearest_detection_km"} <= set(back.columns)
