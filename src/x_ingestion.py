"""OPTIONAL live ingestion of disaster posts from the official X (Twitter) API.

This module is **entirely optional**. The research pipeline runs without it; it
only adds a third, clearly-separated data category:

    data_origin = LIVE_API      # not REAL_DATASET, not SYNTHETIC
    dataset_name = X

Hard rules (enforced here)
--------------------------
* Uses the **official X API v2** recent-search endpoint only. It does **not**
  scrape X web pages.
* Credentials come **only** from environment variables (``X_BEARER_TOKEN``).
  Nothing is ever hardcoded, and the token is never written to disk or logged.
* The processed output ``data/processed/x_live_posts.csv`` is **git-ignored** --
  API-retrieved content and credentials are never committed.
* Coordinates are **not** assumed to exist. Location metadata is stored only when
  the API actually returns a place, and a bare place name is **not** treated as a
  confirmed geolocation.
* Only fields the API exposes publicly are stored; no private account
  information is collected.

Stored per post
---------------
    post_id, text, created_at, author_id (optional), place_name, place_country,
    place_full_name, has_coordinates, query_used, retrieved_at,
    data_origin, dataset_name

Usage
-----
    export X_BEARER_TOKEN=...        # your own API bearer token, never committed
    python src/x_ingestion.py --max-results 50 \
        --query "California wildfire" --query "evacuation wildfire"

Without credentials (or without the optional ``requests`` dependency) the module
degrades gracefully: it explains what is missing and writes nothing, rather than
crashing the pipeline.
"""
import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PATH = PROCESSED_DIR / "x_live_posts.csv"

DATA_ORIGIN = "LIVE_API"
DATASET_NAME = "X"

BEARER_TOKEN_ENV = "X_BEARER_TOKEN"
SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
API_MAX_PER_PAGE = 100  # X API v2 recent-search page cap

# Example disaster queries (wildfire-focused, matching the project scope).
DEFAULT_QUERIES = [
    "wildfire OR wildfire smoke",
    "evacuation wildfire",
    "California wildfire",
    "Maui fire",
    "Paradise wildfire",
]

OUTPUT_FIELDS = [
    "post_id",
    "text",
    "created_at",
    "author_id",
    "place_name",
    "place_country",
    "place_full_name",
    "has_coordinates",
    "query_used",
    "retrieved_at",
    "data_origin",
    "dataset_name",
]


class XIngestionError(RuntimeError):
    """Raised for configuration / API problems that the caller can report."""


def get_bearer_token():
    """Return the bearer token from the environment, or ``None`` if unset."""
    token = os.environ.get(BEARER_TOKEN_ENV, "").strip()
    return token or None


def _require_requests():
    try:
        import requests  # noqa: F401
    except Exception as exc:  # requests not installed
        raise XIngestionError(
            f"the 'requests' package is required for live X ingestion: {exc}"
        ) from exc
    import requests

    return requests


def _default_exclude_retweets(query):
    """Append ``-is:retweet`` so reposts are excluded unless already specified."""
    if "is:retweet" in query:
        return query
    return f"({query}) -is:retweet"


def _flatten_place(tweet, places_by_id):
    """Extract place metadata for a tweet without assuming coordinates exist."""
    place_name = place_country = place_full_name = ""
    has_coordinates = False
    geo = tweet.get("geo") or {}
    place_id = geo.get("place_id")
    if place_id and place_id in places_by_id:
        place = places_by_id[place_id]
        place_full_name = place.get("full_name", "") or ""
        place_name = place.get("name", "") or ""
        place_country = place.get("country", "") or ""
    # Coordinates are rare and must never be assumed; record only if present.
    coords = geo.get("coordinates") or {}
    if coords.get("coordinates"):
        has_coordinates = True
    return place_name, place_country, place_full_name, has_coordinates


def search_recent(
    query,
    bearer_token,
    max_results=50,
    start_time=None,
    end_time=None,
    include_author=True,
    exclude_retweets=True,
    session=None,
):
    """Query the official X API v2 recent-search endpoint for one query.

    Parameters
    ----------
    query : str
        A search query (e.g. ``"California wildfire"``).
    bearer_token : str
        OAuth2 bearer token (from the environment; never hardcoded).
    max_results : int
        Total posts to retrieve across pages (paginated under the hood).
    start_time, end_time : str | None
        Optional ISO-8601 UTC bounds (e.g. ``"2024-01-01T00:00:00Z"``).
    include_author : bool
        Whether to store the public ``author_id`` field.
    exclude_retweets : bool
        Whether to append ``-is:retweet`` to avoid reposted information.

    Returns a list of normalized record dicts. Raises ``XIngestionError`` on
    auth / request failure so the caller can report it cleanly.
    """
    requests = _require_requests()
    session = session or requests.Session()

    effective_query = _default_exclude_retweets(query) if exclude_retweets else query
    headers = {"Authorization": f"Bearer {bearer_token}"}
    tweet_fields = ["created_at", "geo"]
    if include_author:
        tweet_fields.append("author_id")
    params = {
        "query": effective_query,
        "max_results": min(API_MAX_PER_PAGE, max(10, max_results)),
        "tweet.fields": ",".join(tweet_fields),
        "expansions": "geo.place_id",
        "place.fields": "full_name,name,country,geo",
    }
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time

    retrieved_at = datetime.now(timezone.utc).isoformat()
    records = []
    next_token = None

    while len(records) < max_results:
        if next_token:
            params["next_token"] = next_token
        else:
            params.pop("next_token", None)

        resp = session.get(SEARCH_URL, headers=headers, params=params, timeout=30)
        if resp.status_code == 401:
            raise XIngestionError("X API returned 401 Unauthorized -- check X_BEARER_TOKEN.")
        if resp.status_code == 429:
            raise XIngestionError("X API rate limit hit (429). Try again later or lower --max-results.")
        if resp.status_code != 200:
            raise XIngestionError(f"X API error {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        data = payload.get("data", [])
        includes = payload.get("includes", {})
        places_by_id = {p["id"]: p for p in includes.get("places", [])}

        for tweet in data:
            place_name, place_country, place_full_name, has_coords = _flatten_place(
                tweet, places_by_id
            )
            records.append(
                {
                    "post_id": tweet.get("id", ""),
                    "text": tweet.get("text", ""),
                    "created_at": tweet.get("created_at", ""),
                    "author_id": tweet.get("author_id", "") if include_author else "",
                    "place_name": place_name,
                    "place_country": place_country,
                    "place_full_name": place_full_name,
                    "has_coordinates": has_coords,
                    "query_used": query,
                    "retrieved_at": retrieved_at,
                    "data_origin": DATA_ORIGIN,
                    "dataset_name": DATASET_NAME,
                }
            )
            if len(records) >= max_results:
                break

        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            break

    return records[:max_results]


def ingest(
    queries=None,
    max_results=50,
    start_time=None,
    end_time=None,
    include_author=True,
    exclude_retweets=True,
    output_path=OUTPUT_PATH,
):
    """Run live ingestion for a list of queries and write the processed CSV.

    ``max_results`` is applied **per query**. Returns the list of records.
    Raises ``XIngestionError`` if credentials / dependencies are missing so the
    caller can decide how to report it.
    """
    token = get_bearer_token()
    if not token:
        raise XIngestionError(
            f"no {BEARER_TOKEN_ENV} set. Live X ingestion is optional; export your "
            "own bearer token to enable it. Credentials are never hardcoded."
        )

    queries = queries or DEFAULT_QUERIES
    requests = _require_requests()
    session = requests.Session()

    all_records = []
    for query in queries:
        all_records.extend(
            search_recent(
                query,
                token,
                max_results=max_results,
                start_time=start_time,
                end_time=end_time,
                include_author=include_author,
                exclude_retweets=exclude_retweets,
                session=session,
            )
        )

    write_processed(all_records, output_path=output_path)
    return all_records


def write_processed(records, output_path=OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Optional live X API disaster-post ingestion.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Search query (repeatable). Defaults to a set of wildfire queries.",
    )
    parser.add_argument("--max-results", type=int, default=50, help="Posts per query (default 50).")
    parser.add_argument("--start-time", default=None, help="ISO-8601 UTC start, e.g. 2024-01-01T00:00:00Z.")
    parser.add_argument("--end-time", default=None, help="ISO-8601 UTC end, e.g. 2024-01-07T00:00:00Z.")
    parser.add_argument(
        "--no-author", action="store_true", help="Do not store the public author_id field."
    )
    parser.add_argument(
        "--include-retweets", action="store_true", help="Include reposts (default: excluded)."
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    try:
        records = ingest(
            queries=args.queries,
            max_results=args.max_results,
            start_time=args.start_time,
            end_time=args.end_time,
            include_author=not args.no_author,
            exclude_retweets=not args.include_retweets,
        )
    except XIngestionError as exc:
        print(f"[x_ingestion] skipped: {exc}")
    else:
        print(f"Ingested {len(records)} live posts ({DATA_ORIGIN}, dataset={DATASET_NAME})")
        print(f"  written to : {OUTPUT_PATH} (git-ignored; API content + not redistributed)")
        queries = args.queries or DEFAULT_QUERIES
        print(f"  queries    : {queries}")
