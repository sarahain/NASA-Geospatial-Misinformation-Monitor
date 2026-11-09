"""Multi-sensor NASA FIRMS verification of annotated wildfire claims.

Extends ``verify_tweets.py``: for each LLM-annotated post it

    1. geocodes ``llm_location`` (cached Nominatim),
    2. converts ``created_at`` to a UTC timestamp,
    3. queries NASA FIRMS for **both** VIIRS and MODIS, choosing NRT or the SP
       archive by date, over a +/-1 day window around the post date,
    4. computes, per detection, the great-circle distance to the claim location
       and the time difference to the post,
    5. keeps detections within ``DISTANCE_KM`` and ``TIME_HOURS`` as *matches*,
    6. emits an explainable verdict and stores the raw detections for the map.

Verdicts
--------
    SUPPORTED               one or more fire detections within the space/time window
    NO_NEARBY_DETECTION     satellites returned data for the area but no detection
                            within the window -> *insufficient evidence*, not false
    NO_SATELLITE_COVERAGE   the post date is outside the FIRMS product's date range
    API_ERROR               FIRMS could not be queried (key / network / response)
    LOCATION_UNRESOLVED     no location, or geocoding failed
    NOT_APPLICABLE_NON_FIRE FIRMS only senses fire/thermal anomalies

Absence of a detection is NEVER treated as proof that a post is false: clouds,
small fires, overpass timing, and geocoding to a city centroid all cause misses.
"""
import argparse
import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import firms
import geocode
import location_resolution as lr
import verify

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_INPUT = PROCESSED_DIR / "x_llm_annotated.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "satellite_verification.csv"
DEFAULT_DETECTIONS = PROCESSED_DIR / "satellite_detections.json"
FIRMS_CACHE_PATH = PROCESSED_DIR / "firms_cache.json"

FIRE_TYPES = {"wildfire"}
SENSORS = ("VIIRS", "MODIS")
DISTANCE_KM = verify.DISTANCE_THRESHOLD_KM   # 25 km
TIME_HOURS = 48.0
WINDOW_DAYS_BEFORE = 1
DAY_RANGE = 3                                # day-1, day, day+1

# Transparent evidence score (0..1): how strong the *supporting* evidence is.
# Not a probability that the post is true.
WEIGHTS = {"spatial": 0.5, "temporal": 0.3, "detection": 0.2}
DETECTION_SATURATION = 5

PASSTHROUGH = ["post_id", "text", "created_at", "llm_is_hazard_claim", "llm_hazard_type",
               "llm_location", "llm_claim", "llm_stance", "llm_misinfo_risk",
               "llm_confidence", "llm_reason"]
SENSOR_FIELDS = ["source", "status", "detections_total", "detections_matched",
                 "nearest_km", "nearest_hours", "max_frp", "observation_date"]
OUTPUT_FIELDS = PASSTHROUGH + [
    "claim_date_utc", "latitude", "longitude", "display_name", "geocode_query",
    "location_confidence", "ambiguity_status",
] + [f"{s.lower()}_{f}" for s in SENSORS for f in SENSOR_FIELDS] + [
    "verdict", "matched_sensors", "nearest_detection_km", "nearest_detection_hours",
    "evidence_score", "explanation", "evidence_source", "verified_at",
]


# ---- location resolution ------------------------------------------------------
_DIRECTIONAL = re.compile(r"\b(north|south|east|west|northern|southern|eastern|western|central|"
                          r"downtown|greater|rural|coastal)\b(\s+of)?", re.I)
MIN_FALLBACK_IMPORTANCE = 0.15   # ignore street-level junk when a simplified query is used


def query_variants(location):
    """Progressively simpler geocoder queries for a free-text location string."""
    loc = re.sub(r"\s+", " ", str(location or "")).strip()
    out = [loc]
    s = re.sub(r"\s*\([^)]*\)", "", loc)            # drop parentheticals
    s = re.sub(r"#\w+", "", s).strip(" ,")           # drop hashtags
    out.append(s)
    for sep in (" in ", " near ", " at ", " of "):
        if sep in s:
            out.append(s.split(sep)[-1].strip(" ,"))
    if " and " in s:
        out.append(s.split(" and ")[0].strip(" ,"))
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out.append(", ".join(parts[1:]))
        out.append(parts[-1])
    # Directional adjectives ("northern Minnesota") rarely geocode; add stripped forms.
    out += [re.sub(r"\s+", " ", _DIRECTIONAL.sub("", v)).strip(" ,") for v in list(out)]
    seen, variants = set(), []
    for v in out:
        if len(v) >= 3 and v.lower() not in seen:
            seen.add(v.lower())
            variants.append(v)
    return variants


def resolve_location(location, text="", candidates_fn=None):
    """Geocode with the Week 3 resolver: candidates -> contextual ranking ->
    confidence + ambiguity status. Falls back to simplified queries.

    Returns None or a dict with latitude, longitude, display_name,
    geocode_query, location_confidence, ambiguity_status, candidate_count.
    """
    candidates_fn = candidates_fn or geocode.geocode_candidates
    for i, q in enumerate(query_variants(location)):
        cands = candidates_fn(q) or []
        if i > 0:
            cands = [c for c in cands if c.get("importance", 0) >= MIN_FALLBACK_IMPORTANCE]
        if not cands:
            continue
        ranked = lr.rank_candidates(cands, text or "", q, use_context=True)
        confidence = lr._confidence(ranked)
        top = ranked[0]
        # A single candidate is not ambiguous even if Nominatim ranks the place
        # as obscure (low importance), e.g. a named wilderness area.
        status = lr._ambiguity_status(ranked, confidence) if len(cands) > 1 else "resolved"
        return {
            "latitude": top["latitude"], "longitude": top["longitude"],
            "display_name": top["display_name"], "geocode_query": q,
            "location_confidence": confidence,
            "ambiguity_status": status,
            "candidate_count": len(cands),
        }
    return None


# ---- FIRMS response cache (so re-runs do not re-hit NASA) ---------------------
_firms_cache = None


def _cache():
    global _firms_cache
    if _firms_cache is None:
        _firms_cache = _load_json(FIRMS_CACHE_PATH)
    return _firms_cache


def cached_fetch(lat, lon, start, source=None, day_range=DAY_RANGE):
    key = f"{source}|{firms.build_bbox(lat, lon)}|{start}|{day_range}"
    cache = _cache()
    if key in cache:
        return cache[key], "ok", "cached"
    dets, status, msg = firms.fetch_detections_status(lat, lon, start, source=source, day_range=day_range)
    if status == "ok":
        cache[key] = dets
    return dets, status, msg


def save_cache():
    if _firms_cache is not None:
        FIRMS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIRMS_CACHE_PATH.write_text(json.dumps(_firms_cache))


def parse_claim_time(created_at):
    s = str(created_at or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def detection_time(det):
    try:
        t = str(det.get("acq_time", "")).zfill(4)
        return datetime.strptime(f"{det['acq_date']} {t}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
    except (ValueError, KeyError, TypeError):
        return None


def annotate_detections(lat, lon, claim_dt, detections):
    """Attach distance_km / hours_diff / matched to each raw FIRMS row."""
    out = []
    for det in detections:
        try:
            dist = verify.haversine(lat, lon, float(det["latitude"]), float(det["longitude"]))
        except (ValueError, KeyError, TypeError):
            continue
        dt = detection_time(det)
        hours = round((dt - claim_dt).total_seconds() / 3600.0, 1) if (dt and claim_dt) else None
        matched = dist <= DISTANCE_KM and hours is not None and abs(hours) <= TIME_HOURS
        rec = dict(det)
        rec.update({"distance_km": round(dist, 2), "hours_diff": hours, "matched": matched})
        out.append(rec)
    return out


def evidence_score(nearest_km, nearest_hours, n_matched):
    if n_matched == 0:
        return 0.0
    spatial = max(0.0, 1 - nearest_km / DISTANCE_KM)
    temporal = max(0.0, 1 - abs(nearest_hours) / TIME_HOURS) if nearest_hours is not None else 0.0
    detection = min(1.0, n_matched / DETECTION_SATURATION)
    return round(WEIGHTS["spatial"] * spatial + WEIGHTS["temporal"] * temporal
                 + WEIGHTS["detection"] * detection, 3)


def _in_range(date_str, availability, source):
    """True unless the availability table says the date is outside the product."""
    if not availability or source not in availability:
        return True
    lo, hi = availability[source]
    return (not lo or date_str >= lo) and (not hi or date_str <= hi)


def query_sensor(lat, lon, claim_dt, sensor, fetch_fn=None, availability=None):
    """Query one sensor family. Returns (summary_dict, annotated_detections)."""
    fetch_fn = fetch_fn or cached_fetch
    claim_date = claim_dt.date().isoformat()
    source = firms.choose_sensor_source(claim_date, sensor, availability=availability)
    start = (claim_dt.date() - timedelta(days=WINDOW_DAYS_BEFORE)).isoformat()
    summary = {"source": source, "status": "", "detections_total": 0, "detections_matched": 0,
               "nearest_km": "", "nearest_hours": "", "max_frp": "", "observation_date": ""}

    if not _in_range(claim_date, availability, source):
        summary["status"] = "no_coverage"
        return summary, []

    detections, status, _msg = fetch_fn(lat, lon, start, source=source, day_range=DAY_RANGE)
    if status != "ok":
        summary["status"] = status
        return summary, []

    annotated = annotate_detections(lat, lon, claim_dt, detections)
    matched = [d for d in annotated if d["matched"]]
    summary.update({"status": "ok", "detections_total": len(annotated),
                    "detections_matched": len(matched)})
    if matched:
        best = min(matched, key=lambda d: d["distance_km"])
        frps = [float(d["frp"]) for d in matched if str(d.get("frp", "")).strip()]
        summary.update({
            "nearest_km": best["distance_km"],
            "nearest_hours": best["hours_diff"],
            "max_frp": round(max(frps), 1) if frps else "",
            "observation_date": best.get("acq_date", ""),
        })
    return summary, annotated


def _explain(record, sensors):
    v = record["verdict"]
    if v == "NOT_APPLICABLE_NON_FIRE":
        return (f"Hazard type '{record['llm_hazard_type'] or 'unknown'}' is not a fire; "
                "NASA FIRMS only detects thermal anomalies, so no satellite check is possible.")
    if v == "LOCATION_UNRESOLVED":
        return ("No usable location could be geocoded from the post"
                + (f" ('{record['llm_location']}')" if record["llm_location"] else "") + ".")
    caveat = ""
    if record.get("ambiguity_status") == "ambiguous":
        caveat = (f" Caveat: '{record['llm_location']}' is an ambiguous place name (resolver confidence "
                  f"{record['location_confidence']}); it was geocoded to '{record['display_name']}', which may "
                  "not be the place the post meant.")
    elif record.get("geocode_query") and record["geocode_query"] != record["llm_location"]:
        caveat = (f" Note: the full location string could not be geocoded; the simplified query "
                  f"'{record['geocode_query']}' was used, so the coordinates are coarser.")
    if v == "API_ERROR":
        return "NASA FIRMS could not be queried (" + "; ".join(
            f"{s}: {sensors[s]['status']}" for s in SENSORS) + "). Re-run later."
    if v == "NO_SATELLITE_COVERAGE":
        return "The post date is outside the FIRMS product date range for all sensors."
    if v == "SUPPORTED":
        parts = [f"{s} ({sensors[s]['source']}): {sensors[s]['detections_matched']} detection(s) "
                 f"within {DISTANCE_KM:.0f} km / {TIME_HOURS:.0f} h, nearest {sensors[s]['nearest_km']} km "
                 f"and {sensors[s]['nearest_hours']:+.1f} h from the post"
                 for s in SENSORS if sensors[s]["detections_matched"]]
        return ("Active-fire detections near the claimed location and time: " + "; ".join(parts)
                + ". This supports that a fire was burning there; it does not verify every detail of the post."
                + caveat)
    totals = ", ".join(f"{s}: {sensors[s]['detections_total']} in the 0.5 deg box"
                       for s in SENSORS if sensors[s]["status"] == "ok")
    return (f"FIRMS returned data but no detection within {DISTANCE_KM:.0f} km and {TIME_HOURS:.0f} h "
            f"of the claim ({totals}). Insufficient evidence -- NOT proof the post is false "
            "(cloud cover, small fires, overpass timing and centroid geocoding all cause misses)." + caveat)


def verify_row(row, geocode_fn=None, fetch_fn=None, availability=None, firms_enabled=True):
    """Verify one annotated post -> (record, {sensor: annotated_detections}).

    ``geocode_fn(location, text)`` defaults to :func:`resolve_location`.
    """
    geocode_fn = geocode_fn or resolve_location
    record = {k: row.get(k, "") for k in PASSTHROUGH}
    for f in OUTPUT_FIELDS:
        record.setdefault(f, "")
    record.update({"evidence_source": "NASA FIRMS (VIIRS + MODIS)",
                   "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "evidence_score": 0.0, "matched_sensors": ""})
    sensors = {s: {f: "" for f in SENSOR_FIELDS} for s in SENSORS}
    all_dets = {}

    claim_dt = parse_claim_time(row.get("created_at"))
    record["claim_date_utc"] = claim_dt.isoformat() if claim_dt else ""

    location = (row.get("llm_location") or "").strip()
    coords = geocode_fn(location, row.get("text", "")) if location else None
    if coords:
        record.update({"latitude": coords["latitude"], "longitude": coords["longitude"],
                       "display_name": coords.get("display_name", ""),
                       "geocode_query": coords.get("geocode_query", location),
                       "location_confidence": coords.get("location_confidence", ""),
                       "ambiguity_status": coords.get("ambiguity_status", "")})

    hazard = (row.get("llm_hazard_type") or "").lower()
    if hazard not in FIRE_TYPES:
        record["verdict"] = "NOT_APPLICABLE_NON_FIRE"
    elif not coords or claim_dt is None:
        record["verdict"] = "LOCATION_UNRESOLVED"
    else:
        for s in SENSORS:
            if not firms_enabled:
                sensors[s]["source"] = firms.choose_sensor_source(claim_dt.date().isoformat(), s)
                sensors[s]["status"] = "no_key"
                continue
            sensors[s], all_dets[s] = query_sensor(coords["latitude"], coords["longitude"],
                                                   claim_dt, s, fetch_fn=fetch_fn,
                                                   availability=availability)
        statuses = {sensors[s]["status"] for s in SENSORS}
        matched = [s for s in SENSORS if sensors[s]["detections_matched"]]
        if matched:
            record["verdict"] = "SUPPORTED"
            record["matched_sensors"] = "+".join(matched)
            best = min(matched, key=lambda s: sensors[s]["nearest_km"])
            record["nearest_detection_km"] = sensors[best]["nearest_km"]
            record["nearest_detection_hours"] = sensors[best]["nearest_hours"]
            record["evidence_score"] = evidence_score(
                sensors[best]["nearest_km"], sensors[best]["nearest_hours"],
                sum(sensors[s]["detections_matched"] for s in matched))
        elif "ok" in statuses:
            record["verdict"] = "NO_NEARBY_DETECTION"
        elif statuses <= {"no_coverage"}:
            record["verdict"] = "NO_SATELLITE_COVERAGE"
        else:
            record["verdict"] = "API_ERROR"

    for s in SENSORS:
        for f in SENSOR_FIELDS:
            record[f"{s.lower()}_{f}"] = sensors[s][f]
    record["explanation"] = _explain(record, sensors)
    return record, all_dets


def _read_rows(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[satellite_verification] input not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_json(path):
    path = Path(path)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    return {}


def run(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, detections_path=DEFAULT_DETECTIONS,
        limit=None, fire_only=False, resume=True, firms_enabled=None, progress=None):
    rows = _read_rows(input_path)
    if fire_only:
        rows = [r for r in rows if (r.get("llm_hazard_type") or "").lower() in FIRE_TYPES]
    if firms_enabled is None:
        firms_enabled = bool(firms.get_map_key())
    availability = firms.data_availability() if firms_enabled else None

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if resume and out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            existing = {r["post_id"]: r for r in csv.DictReader(fh)}
    det_store = _load_json(detections_path) if resume else {}

    todo = [r for r in rows if r.get("post_id") not in existing]
    if limit is not None:
        todo = todo[:limit]

    results = dict(existing)
    for i, r in enumerate(todo, 1):
        rec, dets = verify_row(r, availability=availability, firms_enabled=firms_enabled)
        results[rec["post_id"]] = rec
        if dets:
            det_store[rec["post_id"]] = dets
        if progress and (i % 25 == 0 or i == len(todo)):
            progress(i, len(todo), rec)
        if i % 25 == 0 or i == len(todo):
            _write(out, results.values())
            Path(detections_path).write_text(json.dumps(det_store))
            save_cache()
    if not todo:
        _write(out, results.values())
    return list(results.values())


def _write(path, records):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def summarize(results):
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description="Verify annotated posts against NASA FIRMS (VIIRS + MODIS).")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--limit", type=int, default=None, help="Verify at most N new posts.")
    p.add_argument("--fire-only", action="store_true", help="Only process wildfire-typed posts.")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing output and start over.")
    args = p.parse_args(argv)

    if not firms.get_map_key():
        print("[satellite_verification] FIRMS_MAP_KEY not set -> fire verdicts will be API_ERROR (no_key).")

    def _progress(i, n, rec):
        print(f"  {i}/{n}  {rec['post_id']}  {rec['verdict']}")

    results = run(input_path=args.input, output_path=args.output, limit=args.limit,
                  fire_only=args.fire_only, resume=not args.no_resume, progress=_progress)
    print("\n--- satellite_verification result ---")
    print(f"posts in output : {len(results)}")
    for verdict, n in sorted(summarize(results).items()):
        print(f"  {verdict:24s}: {n}")
    print(f"output          : {args.output}")
    return 0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    raise SystemExit(main())
