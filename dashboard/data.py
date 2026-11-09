"""Cached data access for the Streamlit dashboard. All paths are local files
produced by the pipeline scripts in ``src/``; nothing is fetched at import."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
EVAL = DATA / "evaluation"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

VERDICT_ORDER = ["SUPPORTED", "NO_NEARBY_DETECTION", "NO_SATELLITE_COVERAGE", "API_ERROR",
                 "LOCATION_UNRESOLVED", "NOT_APPLICABLE_NON_FIRE"]
VERDICT_COLORS = {
    "SUPPORTED": "#2e7d32",
    "NO_NEARBY_DETECTION": "#f9a825",
    "NO_SATELLITE_COVERAGE": "#8d6e63",
    "API_ERROR": "#c62828",
    "LOCATION_UNRESOLVED": "#9e9e9e",
    "NOT_APPLICABLE_NON_FIRE": "#90a4ae",
}
VERDICT_LABELS = {
    "SUPPORTED": "Supported by satellite detection",
    "NO_NEARBY_DETECTION": "No nearby detection (insufficient evidence)",
    "NO_SATELLITE_COVERAGE": "No satellite coverage for date",
    "API_ERROR": "NASA FIRMS unavailable / not queried",
    "LOCATION_UNRESOLVED": "Location could not be geocoded",
    "NOT_APPLICABLE_NON_FIRE": "Not a fire claim (FIRMS not applicable)",
}
HAZARD_COLORS = {
    "wildfire": "#e65100", "hurricane": "#1565c0", "flood": "#0288d1", "tornado": "#6a1b9a",
    "earthquake": "#5d4037", "typhoon": "#00838f", "mudslide": "#795548",
    "severe_weather": "#546e7a", "other": "#9e9e9e", "none": "#cfd8dc",
}


def _mtime(path):
    return Path(path).stat().st_mtime if Path(path).exists() else 0


@st.cache_data(show_spinner=False)
def load_annotated(_mtime=None):
    path = PROCESSED / "x_llm_annotated.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"post_id": str})
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["llm_is_hazard_claim"] = df["llm_is_hazard_claim"].astype(str).str.lower().eq("true")
    df["llm_location"] = df["llm_location"].fillna("")
    df["llm_confidence"] = pd.to_numeric(df["llm_confidence"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_verification(_mtime=None):
    path = PROCESSED / "satellite_verification.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"post_id": str})
    numeric = [c for c in df.columns if any(c.endswith(s) for s in (
        "_km", "_hours", "_total", "_matched", "_frp", "latitude", "longitude", "evidence_score"))]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_detections(_mtime=None):
    path = PROCESSED / "satellite_detections.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError:
        return {}


def detections_frame(store, post_ids=None):
    rows = []
    for pid, sensors in store.items():
        if post_ids is not None and pid not in post_ids:
            continue
        for sensor, dets in sensors.items():
            for d in dets:
                rows.append({"post_id": pid, "sensor": sensor, **d})
    if not rows:
        return pd.DataFrame(columns=["post_id", "sensor", "latitude", "longitude", "acq_date", "acq_time",
                                     "satellite", "instrument", "confidence", "frp", "distance_km",
                                     "hours_diff", "matched"])
    df = pd.DataFrame(rows)
    for c in ("latitude", "longitude", "frp", "distance_km", "hours_diff"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_csv(name, _mtime=None):
    path = DATA / name
    return pd.read_csv(path) if path.exists() else None


def load_eval(name):
    return load_csv(f"evaluation/{name}", _mtime(EVAL / name))


def load_summary():
    path = EVAL / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_all():
    return {
        "annotated": load_annotated(_mtime(PROCESSED / "x_llm_annotated.csv")),
        "verification": load_verification(_mtime(PROCESSED / "satellite_verification.csv")),
        "detections": load_detections(_mtime(PROCESSED / "satellite_detections.json")),
    }


@st.cache_resource(show_spinner=False)
def spacy_model():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


VERIFICATION_COLUMNS = [
    "verdict", "explanation", "latitude", "longitude", "display_name", "geocode_query",
    "location_confidence", "ambiguity_status", "matched_sensors", "nearest_detection_km",
    "nearest_detection_hours", "evidence_score", "viirs_status", "viirs_detections_matched",
    "viirs_nearest_km", "modis_status", "modis_detections_matched", "modis_nearest_km",
]


def annotated_dataset(ann, ver):
    """Every annotated post with its LLM annotations and (if verified) satellite evidence."""
    if ann is None:
        return None
    df = ann.copy()
    if ver is not None:
        cols = ["post_id"] + [c for c in VERIFICATION_COLUMNS if c in ver.columns]
        df = df.merge(ver[cols], on="post_id", how="left")
    if "verdict" not in df.columns:
        df["verdict"] = None
    df["verdict"] = df["verdict"].fillna("NOT_VERIFIED")
    df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    return df.sort_values("created_at", ascending=False).reset_index(drop=True)


def to_excel_bytes(df, sheet_name="annotated_posts"):
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    return buf.getvalue()


def short(text, n=110):
    text = str(text or "").replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "..."
