"""Offline tests for the text-free evaluation builder."""
import json

import pandas as pd

import research_evaluation as re_
import satellite_verification as sv


def _ver_row(pid, verdict, risk="low", amb="resolved", v_matched=0, m_matched=0, v_total=0, m_total=0, lat=33.4):
    base = {f: "" for f in sv.OUTPUT_FIELDS}
    base.update({
        "post_id": pid, "text": "t", "created_at": "2026-10-20T03:00:00Z", "llm_hazard_type": "wildfire",
        "llm_location": "X", "llm_misinfo_risk": risk, "latitude": lat, "longitude": -117.6,
        "display_name": "X, California, United States", "geocode_query": "X", "location_confidence": 0.5,
        "ambiguity_status": amb, "viirs_source": "VIIRS_NOAA20_NRT", "viirs_status": "ok",
        "viirs_detections_total": v_total, "viirs_detections_matched": v_matched, "modis_source": "MODIS_NRT",
        "modis_status": "ok", "modis_detections_total": m_total, "modis_detections_matched": m_matched,
        "verdict": verdict, "evidence_score": 0.5 if verdict == "SUPPORTED" else 0.0,
        "nearest_detection_km": 3.0 if verdict == "SUPPORTED" else "",
    })
    if v_matched:
        base.update({"viirs_nearest_km": 3.0, "viirs_nearest_hours": 5.0, "viirs_max_frp": 2.0})
    if m_matched:
        base.update({"modis_nearest_km": 4.0, "modis_nearest_hours": 6.0, "modis_max_frp": 20.0})
    return base


def _setup(tmp_path, monkeypatch, with_labels=True):
    processed, annot, evald = tmp_path / "processed", tmp_path / "annotations", tmp_path / "evaluation"
    processed.mkdir(), annot.mkdir()
    ann = pd.DataFrame([
        {"post_id": "1", "text": "a", "created_at": "2026-10-20T03:00:00Z", "llm_is_hazard_claim": True,
         "llm_hazard_type": "wildfire", "llm_location": "X", "llm_misinfo_risk": "low"},
        {"post_id": "2", "text": "b", "created_at": "2026-10-21T03:00:00Z", "llm_is_hazard_claim": True,
         "llm_hazard_type": "wildfire", "llm_location": "Y", "llm_misinfo_risk": "high"},
        {"post_id": "3", "text": "c", "created_at": "2026-10-22T03:00:00Z", "llm_is_hazard_claim": True,
         "llm_hazard_type": "wildfire", "llm_location": "", "llm_misinfo_risk": "low"},
        {"post_id": "4", "text": "d", "created_at": "2026-10-22T03:00:00Z", "llm_is_hazard_claim": False,
         "llm_hazard_type": "flood", "llm_location": "Z", "llm_misinfo_risk": "low"},
    ])
    ann.to_csv(processed / "x_llm_annotated.csv", index=False)
    ver = pd.DataFrame([
        _ver_row("1", "SUPPORTED", v_matched=2, v_total=3, m_matched=1, m_total=1),
        _ver_row("2", "NO_NEARBY_DETECTION", risk="high", amb="ambiguous", v_total=1),
        _ver_row("3", "LOCATION_UNRESOLVED", lat=None, amb=""),
    ])
    ver.loc[ver.post_id == "3", ["latitude", "longitude", "viirs_status", "modis_status"]] = [None, None, "", ""]
    ver.to_csv(processed / "satellite_verification.csv", index=False)
    if with_labels:
        pd.DataFrame({"post_id": ["1", "2", "3"], "human_label": ["true_event", "false_or_misleading", "unclear"],
                      "notes": ["", "", ""]}).to_csv(annot / "manual_labels.csv", index=False)
    monkeypatch.setattr(re_, "ANNOTATED_PATH", processed / "x_llm_annotated.csv")
    monkeypatch.setattr(re_, "VERIFICATION_PATH", processed / "satellite_verification.csv")
    monkeypatch.setattr(re_, "LABELS_PATH", annot / "manual_labels.csv")
    monkeypatch.setattr(re_, "LABEL_TEMPLATE_PATH", processed / "manual_labels_template.csv")
    return evald


def test_dataset_loading_normalizes_types(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    ann = re_.load_annotated()
    assert isinstance(ann["post_id"].iloc[0], str) and ann["llm_is_hazard_claim"].dtype == bool
    assert str(ann["created_at"].dt.tz) == "UTC"
    ver = re_.load_verification()
    assert ver["latitude"].isna().sum() == 1 and ver["viirs_detections_matched"].sum() == 2


def test_run_writes_all_outputs_and_metrics(tmp_path, monkeypatch):
    evald = _setup(tmp_path, monkeypatch)
    rep = re_.run(out_dir=evald)
    for name in ("funnel.csv", "verdict_counts.csv", "hazard_counts.csv", "sensor_comparison.csv",
                 "nlp_vs_satellite.csv", "location_quality.csv", "label_metrics.csv", "summary.json"):
        assert (evald / name).exists(), name
    funnel = pd.read_csv(evald / "funnel.csv").set_index("stage")["posts"]
    assert funnel["Annotated posts"] == 4
    assert funnel["Wildfire-typed posts (FIRMS-checkable)"] == 3
    assert funnel["Geocoded to coordinates"] == 2
    assert funnel["Detection within 25 km and 48 h (SUPPORTED)"] == 1
    assert rep["supported"] == 1 and rep["geocoded_ambiguous"] == 1 and rep["supported_unambiguous_location"] == 1

    sc = pd.read_csv(evald / "sensor_comparison.csv").set_index("sensor")
    assert sc.loc["VIIRS", "claims_with_matched_detection"] == 1
    assert sc.loc["BOTH_OK", "claims_with_matched_detection"] == 1  # both sensors matched post 1

    lm = pd.read_csv(evald / "label_metrics.csv").set_index("method")
    # 2 scorable labels (post 3 is 'unclear' and excluded). NLP-only: post1 low->pos (TP), post2 high->neg (TN).
    assert lm.loc["NLP-only (LLM low risk)", "n_labelled"] == 2
    assert lm.loc["NLP-only (LLM low risk)", "f1"] == 1.0
    assert lm.loc["Satellite-only (SUPPORTED)", "tp"] == 1 and lm.loc["Satellite-only (SUPPORTED)", "fp"] == 0
    assert json.loads((evald / "summary.json").read_text())["labelled_posts_scored"] == 2


def test_run_without_labels_reports_absence(tmp_path, monkeypatch):
    evald = _setup(tmp_path, monkeypatch, with_labels=False)
    rep = re_.run(out_dir=evald)
    assert not (evald / "label_metrics.csv").exists()
    assert "no human labels" in rep["label_metrics"]


def test_label_template_is_stratified(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    path, n = re_.write_label_template(re_.load_verification(), n_per_group=1)
    df = pd.read_csv(path)
    assert n == 3 and set(df["verdict"]) == {"SUPPORTED", "NO_NEARBY_DETECTION", "LOCATION_UNRESOLVED"}
    assert "human_label" in df.columns and df["human_label"].isna().all()
