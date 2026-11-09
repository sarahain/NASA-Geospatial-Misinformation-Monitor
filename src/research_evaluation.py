"""Aggregate, text-free evaluation results for the annotated X dataset.

Reads the LLM annotations and the satellite verification output and writes
tracked CSVs (no post text) to ``data/evaluation/``:

    funnel.csv              how many posts survive each pipeline stage
    verdict_counts.csv      satellite verdicts for wildfire posts
    hazard_counts.csv       LLM hazard-type distribution
    sensor_comparison.csv   VIIRS vs MODIS on the same claims
    nlp_vs_satellite.csv    LLM misinformation risk x satellite verdict
    location_quality.csv    geocoding ambiguity / fallback x satellite verdict
    label_metrics.csv       precision / recall / F1 on human-labelled posts
                            (only if data/annotations/manual_labels.csv exists)
    summary.json            headline numbers used by the dashboard

Nothing is estimated or invented: every number is a count over the real files.
If a file is missing the corresponding output is skipped and reported.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_DIR = ROOT / "data" / "evaluation"
ANNOTATED_PATH = PROCESSED_DIR / "x_llm_annotated.csv"
VERIFICATION_PATH = PROCESSED_DIR / "satellite_verification.csv"
LABELS_PATH = ROOT / "data" / "annotations" / "manual_labels.csv"
LABEL_TEMPLATE_PATH = PROCESSED_DIR / "manual_labels_template.csv"

# Human label vocabulary for manual_labels.csv (column ``human_label``).
LABEL_TRUE = "true_event"
LABEL_FALSE = "false_or_misleading"
LABEL_UNCLEAR = "unclear"

SUPPORTED = "SUPPORTED"
NO_NEARBY = "NO_NEARBY_DETECTION"


def load_annotated(path=None):
    df = pd.read_csv(path or ANNOTATED_PATH, dtype={"post_id": str})
    df["llm_is_hazard_claim"] = df["llm_is_hazard_claim"].astype(str).str.lower().eq("true")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


def load_verification(path=None):
    path = path or VERIFICATION_PATH
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, dtype={"post_id": str})
    for c in ("latitude", "longitude", "nearest_detection_km", "nearest_detection_hours", "evidence_score",
              "viirs_detections_total", "viirs_detections_matched", "modis_detections_total",
              "modis_detections_matched", "viirs_nearest_km", "modis_nearest_km",
              "viirs_nearest_hours", "modis_nearest_hours", "viirs_max_frp", "modis_max_frp"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_labels(path=None):
    path = path or LABELS_PATH
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, dtype={"post_id": str})
    df = df[df["human_label"].isin([LABEL_TRUE, LABEL_FALSE])]
    return df if len(df) else None


def funnel(ann, ver):
    fire = ann[ann["llm_hazard_type"] == "wildfire"]
    rows = [
        ("Annotated posts", len(ann)),
        ("LLM: asserts a hazard event", int(ann["llm_is_hazard_claim"].sum())),
        ("LLM: any location extracted", int(ann["llm_location"].fillna("").ne("").sum())),
        ("Wildfire-typed posts (FIRMS-checkable)", len(fire)),
        ("Wildfire posts with a location string", int(fire["llm_location"].fillna("").ne("").sum())),
    ]
    if ver is not None:
        vf = ver[ver["llm_hazard_type"] == "wildfire"]
        rows += [
            ("Wildfire posts processed by verifier", len(vf)),
            ("Geocoded to coordinates", int(vf["latitude"].notna().sum())),
            ("FIRMS queried with coverage (>=1 sensor ok)", int((vf["viirs_status"].eq("ok") | vf["modis_status"].eq("ok")).sum())),
            ("Any detection in the 0.5 deg box (+/-1 day)", int(((vf["viirs_detections_total"].fillna(0) + vf["modis_detections_total"].fillna(0)) > 0).sum())),
            ("Detection within 25 km and 48 h (SUPPORTED)", int(vf["verdict"].eq(SUPPORTED).sum())),
            ("SUPPORTED with an unambiguous geocode", int((vf["verdict"].eq(SUPPORTED) & vf["ambiguity_status"].eq("resolved")).sum())),
        ]
    return pd.DataFrame(rows, columns=["stage", "posts"])


def sensor_comparison(ver):
    vf = ver[(ver["llm_hazard_type"] == "wildfire") & ver["latitude"].notna()]
    rows = []
    for s in ("viirs", "modis"):
        ok = vf[vf[f"{s}_status"] == "ok"]
        matched = ok[ok[f"{s}_detections_matched"].fillna(0) > 0]
        rows.append({
            "sensor": s.upper(),
            "products_used": ";".join(sorted(ok[f"{s}_source"].dropna().unique())),
            "claims_queried_ok": len(ok),
            "claims_with_any_detection_in_box": int((ok[f"{s}_detections_total"].fillna(0) > 0).sum()),
            "claims_with_matched_detection": len(matched),
            "match_rate": round(len(matched) / len(ok), 3) if len(ok) else None,
            "median_nearest_km": round(matched[f"{s}_nearest_km"].median(), 2) if len(matched) else None,
            "median_abs_hours": round(matched[f"{s}_nearest_hours"].abs().median(), 1) if len(matched) else None,
            "median_max_frp_mw": round(matched[f"{s}_max_frp"].median(), 1) if len(matched) else None,
        })
    both = vf[(vf["viirs_status"] == "ok") & (vf["modis_status"] == "ok")]
    v_only = int(((both["viirs_detections_matched"] > 0) & (both["modis_detections_matched"].fillna(0) == 0)).sum())
    m_only = int(((both["modis_detections_matched"] > 0) & (both["viirs_detections_matched"].fillna(0) == 0)).sum())
    agree = int(((both["viirs_detections_matched"] > 0) & (both["modis_detections_matched"] > 0)).sum())
    rows.append({"sensor": "BOTH_OK", "products_used": "", "claims_queried_ok": len(both),
                 "claims_with_any_detection_in_box": None,
                 "claims_with_matched_detection": agree, "match_rate": None,
                 "median_nearest_km": None, "median_abs_hours": None, "median_max_frp_mw": None,
                 "viirs_only_matches": v_only, "modis_only_matches": m_only})
    return pd.DataFrame(rows)


def location_quality(ver):
    """How geocoding quality interacts with verdicts (ambiguous names can create spurious matches)."""
    vf = ver[ver["llm_hazard_type"] == "wildfire"].copy()
    vf["ambiguity_status"] = vf["ambiguity_status"].fillna("unresolved").replace("", "unresolved")
    vf["geocode_mode"] = (vf["geocode_query"].fillna("") == vf["llm_location"].fillna("")).map(
        {True: "full_string", False: "simplified_query"})
    vf.loc[vf["latitude"].isna(), "geocode_mode"] = "not_geocoded"
    ct = pd.crosstab([vf["ambiguity_status"], vf["geocode_mode"]], vf["verdict"], margins=True, margins_name="total")
    return ct.reset_index()


def nlp_vs_satellite(ver):
    vf = ver[ver["llm_hazard_type"] == "wildfire"]
    ct = pd.crosstab(vf["llm_misinfo_risk"].fillna(""), vf["verdict"], margins=True, margins_name="total")
    return ct.reset_index().rename(columns={"llm_misinfo_risk": "llm_misinfo_risk"})


def _prf(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f1, 3), "accuracy": round((tp + tn) / len(y_true), 3) if len(y_true) else None}


def label_metrics(ver, labels):
    """Compare NLP-only vs satellite-assisted predictions of 'true_event' against human labels.

    Positive class = the post describes a real event (``true_event``).
      NLP-only            : predict positive when llm_misinfo_risk == 'low'
      Satellite-only      : predict positive when verdict == SUPPORTED
      Satellite-assisted  : positive when SUPPORTED, or (NLP low risk and the satellite
                            check was not applicable / unavailable); negative when NLP
                            flags risk AND no satellite support.
    Only wildfire posts the human labelled as true/false are scored.
    """
    df = labels.merge(ver, on="post_id", how="inner")
    df = df[df["llm_hazard_type"] == "wildfire"]
    if df.empty:
        return None
    y = (df["human_label"] == LABEL_TRUE).astype(int)
    nlp = (df["llm_misinfo_risk"] == "low").astype(int)
    sat = (df["verdict"] == SUPPORTED).astype(int)
    unavailable = df["verdict"].isin(["API_ERROR", "NO_SATELLITE_COVERAGE", "LOCATION_UNRESOLVED"])
    assisted = ((sat == 1) | ((nlp == 1) & unavailable)).astype(int)
    rows = []
    for name, pred in (("NLP-only (LLM low risk)", nlp), ("Satellite-only (SUPPORTED)", sat),
                       ("Satellite-assisted", assisted)):
        rows.append({"method": name, "n_labelled": len(df), **_prf(y, pred)})
    return pd.DataFrame(rows)


def write_label_template(ver, n_per_group=20, path=None, seed=7):
    """Stratified sample of wildfire posts for manual labelling (git-ignored: has text)."""
    path = path or LABEL_TEMPLATE_PATH
    vf = ver[ver["llm_hazard_type"] == "wildfire"]
    parts = [g.sample(min(n_per_group, len(g)), random_state=seed)
             for _, g in vf.groupby("verdict") if len(g)]
    sample = pd.concat(parts) if parts else vf.head(0)
    out = sample[["post_id", "text", "created_at", "llm_location", "llm_claim", "llm_misinfo_risk",
                  "verdict", "nearest_detection_km"]].copy()
    out["human_label"] = ""
    out["notes"] = ""
    out.to_csv(path, index=False)
    return path, len(out)


def run(out_dir=EVAL_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    ann = load_annotated()
    ver = load_verification()
    labels = load_labels()

    funnel(ann, ver).to_csv(out_dir / "funnel.csv", index=False)
    ann["llm_hazard_type"].fillna("").value_counts().rename_axis("hazard_type").reset_index(name="posts") \
        .to_csv(out_dir / "hazard_counts.csv", index=False)
    report.update({
        "annotated_posts": len(ann),
        "wildfire_posts": int((ann["llm_hazard_type"] == "wildfire").sum()),
        "date_min": ann["created_at"].min().date().isoformat() if ann["created_at"].notna().any() else None,
        "date_max": ann["created_at"].max().date().isoformat() if ann["created_at"].notna().any() else None,
    })

    if ver is None:
        report["verification"] = "missing -- run src/satellite_verification.py"
    else:
        vf = ver[ver["llm_hazard_type"] == "wildfire"]
        vf["verdict"].value_counts().rename_axis("verdict").reset_index(name="posts") \
            .to_csv(out_dir / "verdict_counts.csv", index=False)
        sensor_comparison(ver).to_csv(out_dir / "sensor_comparison.csv", index=False)
        nlp_vs_satellite(ver).to_csv(out_dir / "nlp_vs_satellite.csv", index=False)
        location_quality(ver).to_csv(out_dir / "location_quality.csv", index=False)
        report.update({
            "wildfire_posts_verified": len(vf),
            "geocoded": int(vf["latitude"].notna().sum()),
            "geocoded_ambiguous": int(vf["ambiguity_status"].eq("ambiguous").sum()),
            "supported": int(vf["verdict"].eq(SUPPORTED).sum()),
            "supported_unambiguous_location": int((vf["verdict"].eq(SUPPORTED) & vf["ambiguity_status"].eq("resolved")).sum()),
            "no_nearby_detection": int(vf["verdict"].eq(NO_NEARBY).sum()),
            "verdicts": vf["verdict"].value_counts().to_dict(),
            "median_nearest_km_supported": float(vf.loc[vf["verdict"] == SUPPORTED, "nearest_detection_km"].median())
            if vf["verdict"].eq(SUPPORTED).any() else None,
        })

    if ver is not None and labels is not None:
        lm = label_metrics(ver, labels)
        if lm is not None:
            lm.to_csv(out_dir / "label_metrics.csv", index=False)
            report["labelled_posts_scored"] = int(lm["n_labelled"].iloc[0])
    else:
        report["label_metrics"] = "no human labels yet (data/annotations/manual_labels.csv)"

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build text-free evaluation CSVs.")
    p.add_argument("--make-label-template", action="store_true",
                   help="Also write a stratified manual-labelling template (git-ignored).")
    args = p.parse_args()
    rep = run()
    print(json.dumps(rep, indent=2, default=str))
    if args.make_label_template:
        ver = load_verification()
        if ver is not None:
            path, n = write_label_template(ver)
            print(f"label template: {path} ({n} rows). Fill human_label with "
                  f"{LABEL_TRUE} / {LABEL_FALSE} / {LABEL_UNCLEAR}, then copy post_id,human_label,notes "
                  f"to {LABELS_PATH}.")
