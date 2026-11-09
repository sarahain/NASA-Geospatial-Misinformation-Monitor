"""Offline tests for the live-data pages (dashboard/live.py). NASA is mocked."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import firms  # noqa: E402
from dashboard import live  # noqa: E402


def test_save_env_values_replaces_and_appends(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nX_BEARER_TOKEN=old\nFIRMS_MAP_KEY=\n")
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    live.save_env_values({"FIRMS_MAP_KEY": "abc123", "GOOGLE_GEOCODING_API_KEY": "", "OLLAMA_MODEL": "llama3.2"}, path=env)
    text = env.read_text()
    assert "FIRMS_MAP_KEY=abc123" in text and text.count("FIRMS_MAP_KEY") == 1
    assert "X_BEARER_TOKEN=old" in text and "OLLAMA_MODEL=llama3.2" in text
    assert "GOOGLE_GEOCODING_API_KEY" not in text          # blanks are not written
    assert os.environ["FIRMS_MAP_KEY"] == "abc123"         # active immediately


def test_fetch_live_merges_products_and_reports_status(monkeypatch):
    def fake(bbox, date, source=None, day_range=1, **kw):
        if source == "GOES_NRT":
            return None, "api_error", "timeout"
        if source == "LANDSAT_NRT":   # Landsat has no FRP / instrument columns
            return [{"latitude": "34.1", "longitude": "-118.2", "acq_date": "2026-11-07", "acq_time": "1830",
                     "satellite": "L9", "confidence": "H", "path": "41", "row": "36", "frp": "", "instrument": "OLI/TIRS"}], "ok", ""
        return [{"latitude": "34.0", "longitude": "-118.0", "acq_date": "2026-11-07", "acq_time": "0930",
                 "satellite": "N20", "instrument": "VIIRS", "confidence": "n", "frp": "12.5", "daynight": "D"}], "ok", ""
    monkeypatch.setattr(firms, "fetch_area_status", fake)
    df, status = live.fetch_live.__wrapped__("-125,32,-114,42", "2026-11-07", ("VIIRS_NOAA20_NRT", "LANDSAT_NRT", "GOES_NRT"), 1, "k")
    assert len(df) == 2 and set(df["product"]) == {"VIIRS_NOAA20_NRT", "LANDSAT_NRT"}
    assert status["GOES_NRT"][0] == "api_error" and status["VIIRS_NOAA20_NRT"][2] == 1
    assert df["frp"].dtype.kind == "f" and df["acq_dt"].notna().all()


def test_terrain_deck_serializes_with_fires_and_claims():
    fires = pd.DataFrame({"latitude": [34.0, 34.1], "longitude": [-118.0, -118.1], "frp": [5.0, None],
                          "product": ["VIIRS_NOAA20_NRT", "LANDSAT_NRT"], "acq_date": ["2026-11-07"] * 2,
                          "acq_time": ["0930", "1830"], "satellite": ["N20", "L9"], "confidence": ["n", "H"]})
    claims = pd.DataFrame({"latitude": [33.9], "longitude": [-117.9], "verdict": ["SUPPORTED"], "llm_location": ["X"]})
    deck = live.terrain_deck(fires, claims, (34.0, -118.0))
    layers = json.loads(deck.to_json())["layers"]
    assert [l["@@type"] for l in layers] == ["TerrainLayer", "ColumnLayer", "ScatterplotLayer"]
    # Missing FRP (Landsat) is floored to 1 so the column still renders.
    assert min(d["frp_v"] for d in layers[1]["data"]) == 1.0


def test_firms_products_catalog_is_consistent():
    assert set(firms.LIVE_PRODUCTS) == {p for p, m in firms.PRODUCTS.items() if m["kind"] == "NRT"}
    assert "GOES_NRT" in firms.LIVE_PRODUCTS and "LANDSAT_NRT" in firms.LIVE_PRODUCTS
    for fam in firms.SENSORS.values():
        assert fam["nrt"] in firms.PRODUCTS and fam["sp"] in firms.PRODUCTS


def test_fetch_area_status_keeps_product_specific_columns(monkeypatch):
    class R:
        status_code = 200
        text = ("latitude,longitude,path,row,scan,track,acq_date,acq_time,satellite,confidence,daynight\n"
                "34.1,-118.2,41,36,30,30,2026-11-07,1830,L9,H,D\n")
        def raise_for_status(self): pass
    monkeypatch.setattr(firms.requests, "get", lambda url, timeout: R())
    dets, status, _ = firms.fetch_area_status("b", "2026-11-07", map_key="k", source="LANDSAT_NRT")
    assert status == "ok" and dets[0]["path"] == "41" and dets[0]["frp"] == ""
    assert dets[0]["instrument"] == "OLI/TIRS"
