"""Official X (Twitter) API v2 collection pipeline for disaster/environment posts.

Week 5. Prepares a **safe, reproducible, budget-aware** collector so paid API
usage is not wasted. It supports two modes:

    MODE A (search)    : keyword search for current/recent public posts
    MODE B (rehydrate) : retrieve text for a file of known tweet IDs (e.g. HumAID)

Hard rules (enforced here)
--------------------------
* **Official X API v2 only** (recent-search + tweet-lookup). It does NOT scrape
  x.com and uses no unofficial scraping packages.
* Credentials come **only** from the ``X_BEARER_TOKEN`` environment variable and
  are **never printed, logged, or committed**.
* **Safe by default**: ``--target`` defaults to 10. Any target > 10 requires the
  explicit ``--confirm-paid-run`` flag, otherwise the script exits WITHOUT making
  a request.
* ``--dry-run`` validates configuration and makes **no** API request.
* Checkpointing + error logging allow safe restart without re-collecting.
* Coordinates are never inferred; place metadata is not treated as exact GPS.
* No dollar cost is estimated (X pricing changes); the script tells you to check
  your X Developer billing before a paid run.

Nothing in this module performs a network call at import time. Network access
happens only inside the collection loops, only after the paid-run guardrails pass.
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = ROOT / "config"
LOGS_DIR = ROOT / "logs"

QUERIES_FILE = CONFIG_DIR / "x_queries.txt"
EXCLUDE_FILE = CONFIG_DIR / "x_exclude_terms.txt"
EXCLUDE_ACCOUNTS_FILE = CONFIG_DIR / "x_exclude_accounts.txt"

RAW_CANDIDATES_PATH = PROCESSED_DIR / "x_raw_candidates.csv"
RELEVANT_PATH = PROCESSED_DIR / "x_relevant_posts.csv"
REHYDRATED_PATH = PROCESSED_DIR / "x_rehydrated_posts.csv"
DEFAULT_OUTPUT = RELEVANT_PATH
CHECKPOINT_PATH = PROCESSED_DIR / "x_collection_checkpoint.json"
ERROR_LOG = LOGS_DIR / "x_collection_errors.log"

REVIEW_SAMPLE_PATH = DATA_DIR / "week5_relevance_review_sample.csv"   # git-ignored (has text)
COLLECTION_SUMMARY_PATH = DATA_DIR / "week5_x_collection_summary.csv"  # tracked, no text
RELEVANCE_METRICS_PATH = DATA_DIR / "week5_relevance_metrics.csv"      # tracked, no text

# --- API constants ----------------------------------------------------------
BEARER_TOKEN_ENV = "X_BEARER_TOKEN"
SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
SEARCH_ALL_URL = "https://api.twitter.com/2/tweets/search/all"  # full-archive (needs archive access)
COUNTS_URL = "https://api.twitter.com/2/tweets/counts/recent"   # returns counts, not posts (not billed as reads)
LOOKUP_URL = "https://api.twitter.com/2/tweets"
API_MAX_PER_PAGE = 100   # recent-search page cap
API_LOOKUP_BATCH = 100   # id-lookup batch cap
QUERY_MAX_LEN_RECENT = 512     # recent-search query-length cap
QUERY_MAX_LEN_ARCHIVE = 1024   # full-archive (search/all) query-length cap

DATA_ORIGIN = "LIVE_API"                    # search mode
DATA_ORIGIN_REHYDRATE = "REAL_DATASET_HYDRATED"  # rehydrate mode
DATASET_NAME = "X_API"

# --- Safety constants --------------------------------------------------------
DEFAULT_TARGET = 10          # NEVER default higher
PAID_RUN_THRESHOLD = 10      # > this requires --confirm-paid-run
BILLING_NOTICE = "Check current X Developer billing before running paid collection."

# --- Record schema (item 6) -------------------------------------------------
BASE_FIELDS = [
    "post_id", "text", "created_at", "lang", "author_id", "conversation_id",
    "geo_place_id", "geo_coordinates", "has_media", "media_types",
    "query_used", "matched_queries",
    "retrieved_at", "data_origin", "dataset_name",
]
RELEVANCE_FIELDS = ["relevance_label", "relevance_score", "relevance_reason"]
RECORD_FIELDS = BASE_FIELDS + RELEVANCE_FIELDS
REHYDRATE_FIELDS = [
    "post_id", "text", "created_at", "lang", "author_id", "conversation_id",
    "geo_place_id", "geo_coordinates", "has_media", "media_types",
    "event", "humanitarian_label",
    "hydration_status", "retrieved_at", "data_origin", "dataset_name",
]

# Example wildfire terms for the --location local helper.
WILDFIRE_TERMS = (
    '(wildfire OR "wild fire" OR bushfire OR "forest fire" OR "brush fire" OR "fire evacuation")'
)


class XIngestionError(RuntimeError):
    """Raised for configuration / API problems the caller can report cleanly."""


# ===========================================================================
# Credentials
# ===========================================================================
def get_bearer_token():
    """Return the bearer token from the environment, or ``None`` if unset.

    The value is never printed or logged anywhere in this module.
    """
    token = os.environ.get(BEARER_TOKEN_ENV, "").strip()
    return token or None


def token_present():
    return get_bearer_token() is not None


def check_auth(token, session=None):
    """Make ONE lightweight request to verify the token authenticates.

    Uses the recent **counts** endpoint, which returns match counts rather than
    posts, so it does not consume post-read credits. Returns a dict:
    ``{"ok": bool, "status": int, "detail": str}``. Never prints the token.
    """
    requests = _require_requests()
    session = session or requests.Session()
    headers = {"Authorization": f"Bearer {token}"}
    params = {"query": "weather", "granularity": "day"}
    try:
        resp = session.get(COUNTS_URL, headers=headers, params=params, timeout=30)
    except Exception as exc:  # network error, DNS, etc.
        return {"ok": False, "status": 0, "detail": f"network error: {exc}"}
    status = resp.status_code
    detail = {
        200: "authenticated (token is connected)",
        401: "401 Unauthorized -- token is missing/invalid",
        402: "402 Payment Required -- token IS valid, but your account has no active plan/credits. "
             "Set up pay-per-use or a plan at console.x.com before any request will succeed.",
        403: "403 Forbidden -- token valid but this access tier can't use this endpoint",
        429: "429 rate limited -- token is valid; just throttled, try again shortly",
    }.get(status, f"unexpected HTTP {status}")
    # 402/403/429 all prove the credentials themselves were accepted (auth passed).
    token_valid = status in (200, 402, 403, 429)
    # "ok" = you can actually make billable calls right now (only 200/429).
    return {"ok": status in (200, 429), "status": status,
            "token_valid": token_valid, "detail": detail}


def _require_requests():
    try:
        import requests  # noqa: F401
    except Exception as exc:
        raise XIngestionError(f"the 'requests' package is required: {exc}") from exc
    import requests

    return requests


# ===========================================================================
# Config loading
# ===========================================================================
def load_queries(path=QUERIES_FILE):
    """Parse config/x_queries.txt -> list of (hazard_label, query_expr).

    Lines: ``hazard_label :: query`` or just ``query``. ``#`` comments / blanks
    are ignored. ``hazard_label`` is used only for local diversity tracking.
    """
    path = Path(path)
    if not path.exists():
        raise XIngestionError(f"queries file not found: {path}")
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "::" in line:
            label, expr = line.split("::", 1)
            queries.append((label.strip(), expr.strip()))
        else:
            queries.append((_infer_hazard(line), line))
    return queries


def load_exclude_terms(path=EXCLUDE_FILE):
    path = Path(path)
    if not path.exists():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def load_exclude_accounts(path=EXCLUDE_ACCOUNTS_FILE):
    """Parse config/x_exclude_accounts.txt -> list of bare usernames (no '@').

    Each non-comment, non-blank line is one account to exclude from results via
    the ``-from:username`` operator. A leading ``@`` is stripped if present.
    """
    path = Path(path)
    if not path.exists():
        return []
    accounts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lstrip("@")
        if line and not line.startswith("#"):
            accounts.append(line)
    return accounts


def _infer_hazard(text):
    low = text.lower()
    for key, hazard in [
        ("wildfire", "wildfire"), ("forest fire", "wildfire"), ("brush fire", "wildfire"),
        ("flood", "flood"), ("air quality", "air_quality"), ("smoke", "air_quality"),
        ("evacuation", "evacuation"), ("damage", "damage"), ("hurricane", "hurricane"),
    ]:
        if key in low:
            return hazard
    return "other"


# ===========================================================================
# Query building
# ===========================================================================
def _default_exclude_retweets(query):
    """Append ``-is:retweet`` so reposts are excluded unless already specified."""
    if "is:retweet" in query:
        return query
    return f"({query}) -is:retweet"


def build_search_query(expr, exclude_terms=None, lang_en=False, exclude_retweets=True,
                       exclude_accounts=None, max_len=512, exclude_replies=False):
    """Compose a final search query string from a base expression.

    Appends optional ``lang:en``, ``-is:retweet``, ``-is:reply``, negated
    ``-from:`` account filters, and negated exclusion phrases (only while under
    ``max_len`` to respect search query-length limits). Account exclusions are
    added before phrase exclusions because they are higher precision. Operator
    support depends on the API tier and is validated at run time.
    """
    parts = [f"({expr})"]
    if lang_en:
        parts.append("lang:en")
    if exclude_retweets:
        parts.append("-is:retweet")
    if exclude_replies:
        parts.append("-is:reply")
    query = " ".join(parts)
    # Account exclusions first (high precision, e.g. drop major-media accounts).
    for acct in (exclude_accounts or []):
        candidate = f"{query} -from:{acct}"
        if len(candidate) <= max_len:
            query = candidate
        else:
            break
    for term in (exclude_terms or []):
        candidate = f'{query} -"{term}"'
        if len(candidate) <= max_len:
            query = candidate
        else:
            break
    return query


def build_local_query(location=None, point_radius=None, terms=WILDFIRE_TERMS):
    """Build a local, updated wildfire query for the official recent-search API.

    Compliant alternative to a scraper. ``point_radius`` is ``(lon, lat, km)``.
    The ``place:`` / ``point_radius:`` geo operators require elevated API access
    on some tiers; we never scrape to work around that.
    """
    parts = [terms]
    if location:
        safe = location.replace('"', "")
        parts.append(f'("{safe}" OR place:"{safe}")')
    if point_radius:
        lon, lat, radius_km = point_radius
        parts.append(f"point_radius:[{lon} {lat} {int(radius_km)}km]")
    return " ".join(parts)


# ===========================================================================
# Response parsing (pure -> testable with mock payloads)
# ===========================================================================
def _flatten_place(tweet, places_by_id):
    """Extract place metadata WITHOUT assuming coordinates exist.

    Returns (geo_place_id, geo_coordinates, place_full_name). ``geo_coordinates``
    is "lat,lon" only when the API actually provides a coordinate; otherwise "".
    """
    geo = tweet.get("geo") or {}
    place_id = geo.get("place_id", "") or ""
    coords = ""
    c = (geo.get("coordinates") or {}).get("coordinates")
    if c and len(c) == 2:
        # X returns [longitude, latitude]; store as "lat,lon".
        coords = f"{c[1]},{c[0]}"
    place_full_name = ""
    if place_id and place_id in places_by_id:
        place_full_name = places_by_id[place_id].get("full_name", "") or ""
    return place_id, coords, place_full_name


def _flatten_media(tweet, media_by_key):
    """Return (has_media, media_types) for a tweet.

    ``media_types`` is a sorted, deduped, "|"-joined string of attached media
    types (e.g. "photo|video"). Posts with no media -> (False, ""). Video posts
    are kept, not excluded -- this only records that a video is present.
    """
    keys = (tweet.get("attachments") or {}).get("media_keys") or []
    types = sorted({media_by_key[k].get("type", "") for k in keys
                    if k in media_by_key and media_by_key[k].get("type")})
    return (bool(keys), "|".join(types))


def parse_search_tweet(tweet, places_by_id, query, retrieved_at, media_by_key=None):
    """Convert one API tweet object into a base record dict (no relevance yet)."""
    place_id, coords, _ = _flatten_place(tweet, places_by_id)
    has_media, media_types = _flatten_media(tweet, media_by_key or {})
    return {
        "post_id": tweet.get("id", ""),
        "text": tweet.get("text", ""),
        "created_at": tweet.get("created_at", ""),
        "lang": tweet.get("lang", ""),
        "author_id": tweet.get("author_id", ""),
        "conversation_id": tweet.get("conversation_id", ""),
        "geo_place_id": place_id,
        "geo_coordinates": coords,
        "has_media": has_media,
        "media_types": media_types,
        "query_used": query,
        "matched_queries": query,
        "retrieved_at": retrieved_at,
        "data_origin": DATA_ORIGIN,
        "dataset_name": DATASET_NAME,
    }


def _media_by_key(includes):
    return {m["media_key"]: m for m in includes.get("media", []) if "media_key" in m}


def parse_search_payload(payload, query, retrieved_at=None):
    """Parse a recent-search JSON payload -> (records, next_token)."""
    retrieved_at = retrieved_at or _now_iso()
    includes = payload.get("includes", {}) or {}
    places_by_id = {p["id"]: p for p in includes.get("places", []) if "id" in p}
    media_by_key = _media_by_key(includes)
    records = [
        parse_search_tweet(t, places_by_id, query, retrieved_at, media_by_key)
        for t in payload.get("data", []) or []
    ]
    next_token = (payload.get("meta", {}) or {}).get("next_token")
    return records, next_token


def parse_lookup_payload(payload, retrieved_at=None):
    """Parse a tweet-lookup JSON payload -> (found_records, missing_statuses).

    ``found_records`` are base dicts for returned tweets; ``missing_statuses`` is a
    dict ``{tweet_id: status}`` for ids in ``payload['errors']`` (deleted/private).
    """
    retrieved_at = retrieved_at or _now_iso()
    includes = payload.get("includes", {}) or {}
    places_by_id = {p["id"]: p for p in includes.get("places", []) if "id" in p}
    media_by_key = _media_by_key(includes)
    found = []
    for t in payload.get("data", []) or []:
        place_id, coords, _ = _flatten_place(t, places_by_id)
        has_media, media_types = _flatten_media(t, media_by_key)
        found.append({
            "post_id": t.get("id", ""),
            "text": t.get("text", ""),
            "created_at": t.get("created_at", ""),
            "lang": t.get("lang", ""),
            "author_id": t.get("author_id", ""),
            "conversation_id": t.get("conversation_id", ""),
            "geo_place_id": place_id,
            "geo_coordinates": coords,
            "has_media": has_media,
            "media_types": media_types,
            "retrieved_at": retrieved_at,
        })
    missing = {}
    for err in payload.get("errors", []) or []:
        tid = str(err.get("resource_id") or err.get("value") or "")
        if not tid:
            continue
        title = (err.get("title") or "").lower()
        if "not found" in title:
            missing[tid] = "DELETED_OR_PRIVATE"
        else:
            missing[tid] = "UNAVAILABLE"
    return found, missing


# ===========================================================================
# Dedup store (item 7) + relevance
# ===========================================================================
def apply_relevance(record, strict=False):
    """Attach relevance_label/score/reason to a record (local, no API)."""
    import relevance_filter

    result = relevance_filter.classify(record.get("text", ""), strict=strict)
    record["relevance_label"] = result["relevance_label"]
    record["relevance_score"] = result["relevance_score"]
    record["relevance_reason"] = result["reason"]
    return record


def merge_record(store, record):
    """Insert ``record`` into ``store`` (dict by post_id), deduping.

    If the post_id already exists, the originating query is appended to
    ``matched_queries`` (deduped) and the stored record is kept. Returns True if
    this was a new unique post, False if it was a duplicate.
    """
    pid = record.get("post_id")
    if not pid:
        return False
    if pid in store:
        existing = store[pid]
        seen = [q for q in existing.get("matched_queries", "").split("|") if q]
        q = record.get("query_used", "")
        if q and q not in seen:
            seen.append(q)
            existing["matched_queries"] = "|".join(seen)
        return False
    store[pid] = record
    return True


# ===========================================================================
# Checkpoint (item 9)
# ===========================================================================
def default_checkpoint():
    return {
        "mode": None,
        "target": None,
        "candidate_limit": None,
        "seen_post_ids": [],
        "relevant_post_ids": [],
        "candidates_retrieved": 0,
        "relevant_count": 0,
        "queries_used": [],
        "per_query_next_token": {},
        "collection_start": None,
        "last_updated": None,
    }


def load_checkpoint(path=CHECKPOINT_PATH):
    path = Path(path)
    if not path.exists():
        return default_checkpoint()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_checkpoint()
    state = default_checkpoint()
    state.update(data)
    return state


def save_checkpoint(state, path=CHECKPOINT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = _now_iso()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return path


# ===========================================================================
# Error logging (item 10) -- never logs credentials
# ===========================================================================
def log_error(endpoint, query, http_status, error_type, path=ERROR_LOG):
    """Append one structured error line. Bearer token/headers are NEVER written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _now_iso(),
        "endpoint": endpoint,
        "query": query,
        "http_status": http_status,
        "error_type": error_type,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


# ===========================================================================
# Small utilities
# ===========================================================================
def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def batched(items, size):
    """Yield successive ``size``-length chunks of ``items``."""
    items = list(items)
    for start in range(0, len(items), size):
        yield items[start:start + size]


def confirm_required(target, confirm_paid_run):
    """Return an error message if a paid run is attempted without confirmation.

    > PAID_RUN_THRESHOLD posts requires ``--confirm-paid-run``. Returns None when
    the run is allowed.
    """
    if target > PAID_RUN_THRESHOLD and not confirm_paid_run:
        return (
            f"target={target} exceeds the safe limit of {PAID_RUN_THRESHOLD}. "
            "Re-run with --confirm-paid-run to authorize a paid collection. "
            + BILLING_NOTICE
        )
    return None


def target_reached(relevant_count, candidates_retrieved, target, candidate_limit):
    """Stop condition: enough relevant posts OR candidate limit hit."""
    if relevant_count >= target:
        return "target_reached"
    if candidate_limit is not None and candidates_retrieved >= candidate_limit:
        return "candidate_limit_reached"
    return None


# ===========================================================================
# Output writers
# ===========================================================================
def write_records(records, path, fields=RECORD_FIELDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    return path


# ===========================================================================
# MODE A: search collection
# ===========================================================================
def _search_once(session, requests, token, query, next_token, start_time, end_time,
                 full_archive=False):
    """One search request. Returns (payload). Raises XIngestionError.

    ``full_archive=True`` targets the full-archive endpoint (search/all), which
    supports a ~2-year (and older) lookback via ``start_time``/``end_time``.
    ``False`` uses recent-search (last ~7 days only).
    """
    url = SEARCH_ALL_URL if full_archive else SEARCH_URL
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "query": query,
        "max_results": API_MAX_PER_PAGE,
        "tweet.fields": "created_at,lang,author_id,conversation_id,geo,attachments",
        "expansions": "geo.place_id,attachments.media_keys",
        "place.fields": "full_name,name,country,geo",
        "media.fields": "type",
    }
    if next_token:
        params["next_token"] = next_token
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    resp = session.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 401:
        log_error(url, query, 401, "unauthorized")
        raise XIngestionError("X API 401 Unauthorized -- check X_BEARER_TOKEN.")
    if resp.status_code == 403:
        log_error(url, query, 403, "forbidden")
        raise XIngestionError(
            "X API 403 Forbidden -- your access does not include this endpoint. "
            "Full-archive search (search/all) requires archive access (pay-per-use "
            "or Pro/Enterprise). Check your plan at console.x.com."
        )
    if resp.status_code == 429:
        log_error(url, query, 429, "rate_limited")
        raise XIngestionError("X API 429 rate limit. Progress is checkpointed; retry later.")
    if resp.status_code != 200:
        log_error(url, query, resp.status_code, "http_error")
        raise XIngestionError(f"X API error {resp.status_code}.")
    return resp.json()


def collect_search(queries, token, target=DEFAULT_TARGET, candidate_limit=None,
                   max_pages=None, start_time=None, end_time=None, lang_en=False,
                   exclude_terms=None, exclude_accounts=None, full_archive=False,
                   strict=False, output_path=DEFAULT_OUTPUT,
                   raw_path=RAW_CANDIDATES_PATH, checkpoint_path=CHECKPOINT_PATH,
                   exclude_replies=False):
    """Collect current posts, round-robin across queries, until target relevant.

    Writes raw candidates and relevant posts, checkpointing after each page. This
    is only invoked after the paid-run guardrail passes. Round-robin paging keeps
    any single query from dominating the dataset.
    """
    requests = _require_requests()
    session = requests.Session()

    state = load_checkpoint(checkpoint_path)
    if state["collection_start"] is None:
        state["collection_start"] = _now_iso()
    state["mode"] = "search"
    state["target"] = target
    state["candidate_limit"] = candidate_limit

    store = {}  # post_id -> record (all candidates this session)
    seen = set(state.get("seen_post_ids", []))
    relevant_ids = set(state.get("relevant_post_ids", []))
    candidates_retrieved = state.get("candidates_retrieved", 0)

    query_max_len = QUERY_MAX_LEN_ARCHIVE if full_archive else QUERY_MAX_LEN_RECENT
    built = [(label, build_search_query(expr, exclude_terms, lang_en,
                                        exclude_accounts=exclude_accounts,
                                        max_len=query_max_len,
                                        exclude_replies=exclude_replies))
             for label, expr in queries]
    active = {q: state["per_query_next_token"].get(q, None) for _, q in built}
    exhausted = set()
    pages = 0

    while True:
        stop = target_reached(len(relevant_ids), candidates_retrieved, target, candidate_limit)
        if stop:
            break
        if len(exhausted) == len(built):
            stop = "queries_exhausted"
            break
        progressed = False
        for label, query in built:
            if query in exhausted:
                continue
            if max_pages is not None and pages >= max_pages:
                exhausted.add(query)
                continue
            try:
                payload = _search_once(session, requests, token, query,
                                       active.get(query), start_time, end_time,
                                       full_archive=full_archive)
            except XIngestionError:
                # Progress so far is already checkpointed; stop cleanly.
                save_checkpoint(state, checkpoint_path)
                _flush(store, output_path, raw_path)
                raise
            pages += 1
            progressed = True
            records, next_token = parse_search_payload(payload, query)
            for rec in records:
                candidates_retrieved += 1
                if rec["post_id"] in seen:
                    merge_record(store, rec)  # update matched_queries if loaded
                    continue
                seen.add(rec["post_id"])
                apply_relevance(rec, strict=strict)
                merge_record(store, rec)
                if rec["relevance_label"] == "RELEVANT":
                    relevant_ids.add(rec["post_id"])
            # checkpoint after each page
            active[query] = next_token
            if not next_token:
                exhausted.add(query)
            state.update({
                "seen_post_ids": sorted(seen),
                "relevant_post_ids": sorted(relevant_ids),
                "candidates_retrieved": candidates_retrieved,
                "relevant_count": len(relevant_ids),
                "queries_used": [q for _, q in built],
                "per_query_next_token": active,
            })
            save_checkpoint(state, checkpoint_path)
            _flush(store, output_path, raw_path)
            if target_reached(len(relevant_ids), candidates_retrieved, target, candidate_limit):
                stop = "target_reached"
                break
        if not progressed:
            break

    _flush(store, output_path, raw_path)
    return {
        "candidates_retrieved": candidates_retrieved,
        "unique_candidates": len(store),
        "relevant": len(relevant_ids),
        "stop_reason": stop or "complete",
    }


def _flush(store, output_path, raw_path):
    """Persist all candidates and the RELEVANT subset to disk."""
    all_records = list(store.values())
    write_records(all_records, raw_path, RECORD_FIELDS)
    relevant = [r for r in all_records if r.get("relevance_label") == "RELEVANT"]
    write_records(relevant, output_path, RECORD_FIELDS)


# ===========================================================================
# MODE B: HumAID rehydration
# ===========================================================================
def read_tweet_ids(input_path, id_column=None):
    """Read tweet IDs (+ optional event/label) from a CSV for rehydration.

    Returns a list of dicts: {tweet_id, event, humanitarian_label}. Recognizes
    common column names; HumAID metadata is merged if present.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise XIngestionError(f"rehydration input not found: {input_path}")
    rows = []
    with open(input_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        id_col = id_column or _pick(cols, ["tweet_id", "post_id", "id"])
        if not id_col:
            raise XIngestionError(f"no tweet-id column in {input_path} (cols={cols})")
        event_col = _pick(cols, ["event", "event_name"])
        label_col = _pick(cols, ["humanitarian_label", "class_label", "label"])
        for row in reader:
            tid = str(row.get(id_col, "")).strip()
            if not tid:
                continue
            rows.append({
                "tweet_id": tid,
                "event": str(row.get(event_col, "")).strip() if event_col else "",
                "humanitarian_label": str(row.get(label_col, "")).strip() if label_col else "",
            })
    return rows


def _pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _lookup_once(session, requests, token, ids):
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "ids": ",".join(ids),
        "tweet.fields": "created_at,lang,author_id,conversation_id,geo,attachments",
        "expansions": "geo.place_id,attachments.media_keys",
        "place.fields": "full_name,name,country,geo",
        "media.fields": "type",
    }
    resp = session.get(LOOKUP_URL, headers=headers, params=params, timeout=30)
    if resp.status_code == 401:
        log_error(LOOKUP_URL, "id_lookup", 401, "unauthorized")
        raise XIngestionError("X API 401 Unauthorized -- check X_BEARER_TOKEN.")
    if resp.status_code == 429:
        log_error(LOOKUP_URL, "id_lookup", 429, "rate_limited")
        raise XIngestionError("X API 429 rate limit. Progress is checkpointed; retry later.")
    if resp.status_code != 200:
        log_error(LOOKUP_URL, "id_lookup", resp.status_code, "http_error")
        raise XIngestionError(f"X API error {resp.status_code}.")
    return resp.json()


def merge_hydrated(id_meta, found_by_id, missing_status, retrieved_at=None):
    """Merge looked-up text with HumAID event/label metadata (item 13).

    ``id_meta`` : list of {tweet_id, event, humanitarian_label}
    ``found_by_id`` : {tweet_id: base_record}
    ``missing_status`` : {tweet_id: status}
    Returns a list of rehydrate records with a ``hydration_status`` of SUCCESS /
    UNAVAILABLE / DELETED_OR_PRIVATE. Deleted/unavailable posts get NO text
    (never fabricated).
    """
    retrieved_at = retrieved_at or _now_iso()
    out = []
    for meta in id_meta:
        tid = meta["tweet_id"]
        base = found_by_id.get(tid)
        if base:
            rec = dict(base)
            rec["hydration_status"] = "SUCCESS"
        else:
            rec = {"post_id": tid, "text": "", "created_at": "", "lang": "",
                   "author_id": "", "conversation_id": "", "geo_place_id": "",
                   "geo_coordinates": "", "retrieved_at": retrieved_at}
            rec["hydration_status"] = missing_status.get(tid, "UNAVAILABLE")
        rec["event"] = meta.get("event", "")
        rec["humanitarian_label"] = meta.get("humanitarian_label", "")
        rec["data_origin"] = DATA_ORIGIN_REHYDRATE
        rec["dataset_name"] = "HumAID"
        out.append(rec)
    return out


def collect_rehydrate(id_meta, token, target=DEFAULT_TARGET, output_path=REHYDRATED_PATH,
                      checkpoint_path=CHECKPOINT_PATH):
    """Rehydrate up to ``target`` tweet ids via official id-lookup, checkpointing."""
    requests = _require_requests()
    session = requests.Session()
    id_meta = id_meta[:target]

    state = load_checkpoint(checkpoint_path)
    if state["collection_start"] is None:
        state["collection_start"] = _now_iso()
    state["mode"] = "rehydrate"
    state["target"] = target

    done_ids = set(state.get("seen_post_ids", []))
    records = []
    pending = [m for m in id_meta if m["tweet_id"] not in done_ids]

    for chunk in batched([m["tweet_id"] for m in pending], API_LOOKUP_BATCH):
        try:
            payload = _lookup_once(session, requests, token, chunk)
        except XIngestionError:
            save_checkpoint(state, checkpoint_path)
            if records:
                write_records(records, output_path, REHYDRATE_FIELDS)
            raise
        found, missing = parse_lookup_payload(payload)
        found_by_id = {r["post_id"]: r for r in found}
        chunk_meta = [m for m in pending if m["tweet_id"] in set(chunk)]
        records.extend(merge_hydrated(chunk_meta, found_by_id, missing))
        done_ids.update(chunk)
        state["seen_post_ids"] = sorted(done_ids)
        state["candidates_retrieved"] = len(done_ids)
        save_checkpoint(state, checkpoint_path)
        write_records(records, output_path, REHYDRATE_FIELDS)

    statuses = {}
    for r in records:
        statuses[r["hydration_status"]] = statuses.get(r["hydration_status"], 0) + 1
    return {"requested": len(id_meta), "records": len(records), "statuses": statuses}


# ===========================================================================
# Backward-compatible simple ingest (used by earlier Week 4 tests)
# ===========================================================================
def ingest(queries=None, max_results=50, **kwargs):
    """Minimal guarded entry point: requires a token, else raises.

    Retained for backward compatibility. Prefer the CLI (search/rehydrate modes).
    """
    token = get_bearer_token()
    if not token:
        raise XIngestionError(
            f"no {BEARER_TOKEN_ENV} set. Live X ingestion is optional; export your "
            "own bearer token to enable it. Credentials are never hardcoded."
        )
    raise XIngestionError(
        "ingest() is a stub; use `python src/x_ingestion.py --mode search ...` "
        "with the Week 5 guardrails (--confirm-paid-run)."
    )


# ===========================================================================
# Dry run (item 4)
# ===========================================================================
def dry_run_report(args, queries, exclude_terms, exclude_accounts=None):
    exclude_accounts = exclude_accounts or []
    endpoint = SEARCH_ALL_URL if getattr(args, "full_archive", False) else SEARCH_URL
    lines = ["=== X ingestion DRY RUN (no API request made) ==="]
    lines.append(f"mode                 : {args.mode}")
    lines.append(f"search endpoint      : {endpoint}")
    lines.append(f"full-archive (2yr)   : {'YES' if getattr(args, 'full_archive', False) else 'NO (recent, ~7 days)'}")
    lines.append(f"strict filter        : {'YES (heavy quality)' if getattr(args, 'strict', False) else 'NO (standard)'}")
    lines.append(f"exclude replies      : {'YES (-is:reply)' if getattr(args, 'no_replies', False) else 'NO'}")
    lines.append(f"requested target     : {args.target} unique "
                 f"{'relevant ' if args.mode == 'search' else ''}posts")
    lines.append(f"candidate limit      : {args.candidate_limit}")
    lines.append(f"max pages            : {args.max_pages}")
    lines.append(f"start_time           : {args.start_time}")
    lines.append(f"end_time             : {args.end_time}")
    lines.append(f"output path          : {args.output}")
    lines.append(f"raw candidates path  : {getattr(args, 'raw_output', RAW_CANDIDATES_PATH)}")
    lines.append(f"checkpoint path      : {getattr(args, 'checkpoint_file', CHECKPOINT_PATH)}")
    lines.append(f"error log            : {ERROR_LOG}")
    # NEVER print the token value -- only presence.
    lines.append(f"X_BEARER_TOKEN detected: {'YES' if token_present() else 'NO'}")
    guard = confirm_required(args.target, args.confirm_paid_run)
    lines.append(f"paid-run authorized  : {'NO -> ' + guard if guard else 'YES'}")
    lines.append(f"\n{BILLING_NOTICE}")
    if args.mode == "search":
        query_max_len = (QUERY_MAX_LEN_ARCHIVE if getattr(args, "full_archive", False)
                         else QUERY_MAX_LEN_RECENT)
        lines.append(f"\nQueries that WOULD run ({len(queries)}):")
        for label, expr in queries:
            final = build_search_query(expr, exclude_terms, lang_en=args.lang_en,
                                       exclude_accounts=exclude_accounts,
                                       max_len=query_max_len,
                                       exclude_replies=getattr(args, "no_replies", False))
            lines.append(f"  [{label}] {final}")
        if exclude_accounts:
            lines.append(f"\nExcluded accounts ({len(exclude_accounts)}): {exclude_accounts}")
        if exclude_terms:
            lines.append(f"\nExclusion terms ({len(exclude_terms)}): {exclude_terms}")
    else:
        lines.append(f"\nrehydrate input      : {args.input}")
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================
def _build_arg_parser():
    p = argparse.ArgumentParser(description="Official X API v2 disaster-post collector (Week 5).")
    p.add_argument("--mode", choices=["search", "rehydrate"], default="search",
                   help="search = current posts; rehydrate = text for known tweet IDs.")
    p.add_argument("--target", type=int, default=DEFAULT_TARGET,
                   help=f"UNIQUE posts to collect (default {DEFAULT_TARGET}; never defaults higher).")
    p.add_argument("--candidate-limit", type=int, default=None,
                   help="Max raw candidates to retrieve before stopping (search mode).")
    p.add_argument("--query-file", default=str(QUERIES_FILE), help="Path to queries file.")
    p.add_argument("--exclude-file", default=str(EXCLUDE_FILE), help="Path to exclusion-terms file.")
    p.add_argument("--exclude-accounts-file", default=str(EXCLUDE_ACCOUNTS_FILE),
                   help="Path to file of accounts to exclude via -from: (e.g. major media).")
    p.add_argument("--input", default=None, help="CSV of tweet IDs for --mode rehydrate.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    p.add_argument("--raw-output", default=str(RAW_CANDIDATES_PATH),
                   help="Raw candidates CSV path (use a separate file for a second pass).")
    p.add_argument("--checkpoint-file", default=str(CHECKPOINT_PATH),
                   help="Checkpoint JSON path. Use a SEPARATE file for an independent "
                        "second pass (e.g. the international run) so state does not mix.")
    p.add_argument("--max-pages", type=int, default=None, help="Max API pages (safety cap).")
    p.add_argument("--start-time", default=None, help="ISO-8601 UTC start (recent search).")
    p.add_argument("--end-time", default=None, help="ISO-8601 UTC end (recent search).")
    p.add_argument("--lang-en", action="store_true", help="Add lang:en to queries (if supported).")
    p.add_argument("--full-archive", action="store_true",
                   help="Use full-archive search (search/all) for a >7-day lookback (e.g. 2 years). "
                        "Requires archive access (pay-per-use or Pro/Enterprise).")
    p.add_argument("--strict", action="store_true",
                   help="Heavy quality filter: only keep posts with hazard + location/event + "
                        "impact/emergency/smoke context above a higher score. Fewer, cleaner posts.")
    p.add_argument("--no-replies", action="store_true",
                   help="Add -is:reply so reply/commentary posts are excluded at the API level "
                        "(saves budget on opinion chatter that mentions disasters).")
    p.add_argument("--location", default=None,
                   help="Add a local wildfire query near a place (official API, not scraping).")
    p.add_argument("--dry-run", action="store_true", help="Validate only; make NO API request.")
    p.add_argument("--check-auth", action="store_true",
                   help="Make ONE counts request (no post reads / no post-read credits) just to "
                        "verify the token is connected, then exit.")
    p.add_argument("--confirm-paid-run", action="store_true",
                   help=f"Required for target > {PAID_RUN_THRESHOLD}.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    queries = load_queries(args.query_file) if args.mode == "search" else []
    if args.location:
        queries = (queries or []) + [("wildfire_local", build_local_query(location=args.location))]
    exclude_terms = load_exclude_terms(args.exclude_file)
    exclude_accounts = load_exclude_accounts(args.exclude_accounts_file)

    # --- Dry run: no network, no token value printed ---
    if args.dry_run:
        print(dry_run_report(args, queries, exclude_terms, exclude_accounts))
        return 0

    # --- Connectivity check: one counts request, no post-read credits spent ---
    if args.check_auth:
        if not token_present():
            print(f"[x_ingestion] no {BEARER_TOKEN_ENV} set. Export your token, then re-run.")
            return 2
        result = check_auth(get_bearer_token())
        print(f"X_BEARER_TOKEN detected : YES")
        print(f"endpoint               : {COUNTS_URL} (counts only -- not billed as post reads)")
        print(f"http_status            : {result['status']}")
        print(f"token_valid (auth ok)  : {'YES' if result.get('token_valid') else 'NO'}")
        print(f"can_make_requests      : {'YES' if result['ok'] else 'NO'}")
        print(f"detail                 : {result['detail']}")
        return 0 if result["ok"] else 2

    # --- Paid-run guardrail (item 8) ---
    guard = confirm_required(args.target, args.confirm_paid_run)
    if guard:
        print(f"[x_ingestion] REFUSING TO RUN: {guard}")
        return 2

    if not token_present():
        print(f"[x_ingestion] no {BEARER_TOKEN_ENV} set. Export your token, then re-run. "
              "Credentials are never hardcoded or printed.")
        return 2

    token = get_bearer_token()

    # Pre-run status (item 8).
    checkpoint_path = Path(args.checkpoint_file)
    raw_path = Path(args.raw_output)
    existing = load_checkpoint(checkpoint_path)
    print(f"Requested unique post target : {args.target}")
    print(f"Current saved unique posts   : {existing.get('relevant_count', 0)}")
    print(f"Remaining target             : {max(0, args.target - existing.get('relevant_count', 0))}")
    print(BILLING_NOTICE)

    if args.mode == "search":
        result = collect_search(
            queries, token, target=args.target, candidate_limit=args.candidate_limit,
            max_pages=args.max_pages, start_time=args.start_time, end_time=args.end_time,
            lang_en=args.lang_en, exclude_terms=exclude_terms,
            exclude_accounts=exclude_accounts, full_archive=args.full_archive,
            strict=args.strict, output_path=Path(args.output),
            raw_path=raw_path, checkpoint_path=checkpoint_path,
            exclude_replies=args.no_replies,
        )
        print("\n--- search result ---")
        print(f"authentication_status : OK")
        print(f"posts_returned        : {result['candidates_retrieved']}")
        print(f"unique_posts_saved    : {result['unique_candidates']}")
        print(f"relevant_saved        : {result['relevant']}")
        print(f"stop_reason           : {result['stop_reason']}")
    else:
        if not args.input:
            print("[x_ingestion] --mode rehydrate requires --input <ids.csv>")
            return 2
        id_meta = read_tweet_ids(args.input)
        result = collect_rehydrate(id_meta, token, target=args.target, output_path=Path(args.output))
        print("\n--- rehydrate result ---")
        print(f"authentication_status : OK")
        print(f"requested             : {result['requested']}")
        print(f"records               : {result['records']}")
        print(f"statuses              : {result['statuses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
