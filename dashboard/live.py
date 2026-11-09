"""Live NASA data, data-source management and raw-collection pages."""
import glob
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import streamlit as st
from streamlit_folium import st_folium

from dashboard import maps
from dashboard.data import PROCESSED, ROOT, VERDICT_COLORS, load_eval, short

import firms  # noqa: E402  (src/ is on sys.path via dashboard.data)

ENV_PATH = ROOT / ".env"
PLOT_TEMPLATE = "plotly_dark"

REGIONS = {
    "California": "-125,32,-114,42",
    "Western US": "-125,31,-102,49",
    "Contiguous US": "-125,24,-66,50",
    "Canada": "-141,42,-52,70",
    "Mexico + Central America": "-118,7,-77,33",
    "South America": "-82,-35,-34,13",
    "Europe + Mediterranean": "-10,30,40,60",
    "Africa": "-18,-35,52,37",
    "South + Southeast Asia": "60,-11,150,35",
    "Australia + NZ": "112,-48,180,-10",
    "World (VIIRS/MODIS only, slow)": "world",
}

# Every external service the project touches. Values are never displayed.
API_INVENTORY = [
    {"service": "NASA FIRMS Area API", "env": "FIRMS_MAP_KEY", "needs_key": True,
     "purpose": "Active-fire detections (VIIRS, MODIS, Landsat, GOES), NRT + archive",
     "url": "https://firms.modaps.eosdis.nasa.gov/api/map_key/"},
    {"service": "NASA GIBS imagery (WMTS)", "env": "", "needs_key": False,
     "purpose": "Daily true-colour satellite imagery tiles for any date", "url": "https://nasa-gibs.github.io/gibs-api-docs/"},
    {"service": "Nominatim (OpenStreetMap)", "env": "", "needs_key": False,
     "purpose": "Primary geocoder (1 req/s, cached in data/geocode_cache.json)", "url": "https://nominatim.org/release-docs/latest/api/Search/"},
    {"service": "Google Geocoding (optional)", "env": "GOOGLE_GEOCODING_API_KEY", "needs_key": True,
     "purpose": "Fallback geocoder when Nominatim is ambiguous/unresolved", "url": "https://developers.google.com/maps/documentation/geocoding"},
    {"service": "X API v2 (official)", "env": "X_BEARER_TOKEN", "needs_key": True,
     "purpose": "Post collection (paid; only via src/x_ingestion.py with --confirm-paid-run)", "url": "https://developer.x.com/"},
    {"service": "Ollama (local LLM)", "env": "OLLAMA_MODEL", "needs_key": False,
     "purpose": "Claim annotation (hazard type, location, stance, risk) -- runs locally", "url": "https://ollama.com/"},
    {"service": "Esri basemap tiles", "env": "", "needs_key": False,
     "purpose": "Dark canvas, hillshade (terrain), imagery, place labels", "url": "https://www.arcgis.com/"},
    {"service": "AWS Terrain Tiles (Mapzen terrarium)", "env": "", "needs_key": False,
     "purpose": "Elevation for the 3D terrain map", "url": "https://registry.opendata.aws/terrain-tiles/"},
]


# ---------------------------------------------------------------- .env helpers
def env_configured(name):
    return bool(os.environ.get(name, "").strip())


def save_env_values(values, path=ENV_PATH):
    """Write/replace KEY=value lines in the git-ignored .env and the live process env."""
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    for key, val in values.items():
        val = val.strip()
        if not val:
            continue
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        replaced = False
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"{key}={val}"
                replaced = True
        if not replaced:
            lines.append(f"{key}={val}")
        os.environ[key] = val
    path.write_text("\n".join(lines) + "\n")
    return path


@st.cache_data(ttl=300, show_spinner=False)
def firms_key_status(key):
    try:
        r = requests.get(f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={key}", timeout=20)
        return r.status_code, r.text[:300]
    except requests.RequestException as exc:
        return None, str(exc)


@st.cache_data(ttl=3600, show_spinner=False)
def availability_table(_key_fingerprint):
    return firms.data_availability() or {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_live(bbox, date, products, day_range, _key_fingerprint):
    """Fetch several FIRMS products for one bbox/date. Returns (DataFrame, status dict)."""
    frames, status = [], {}
    for p in products:
        dets, s, msg = firms.fetch_area_status(bbox, date, source=p, day_range=day_range)
        status[p] = (s, msg, len(dets) if dets is not None else 0)
        if dets:
            df = pd.DataFrame(dets)
            df["product"] = p
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "frp", "product", "acq_date", "acq_time",
                                     "satellite", "confidence", "daynight"]), status
    df = pd.concat(frames, ignore_index=True)
    for c in ("latitude", "longitude", "frp"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
    df["acq_dt"] = pd.to_datetime(df["acq_date"] + " " + df["acq_time"], format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    return df, status


def _bbox_tuple(bbox):
    if bbox == "world":
        return (-180.0, -90.0, 180.0, 90.0)
    w, s, e, n = (float(x) for x in bbox.split(","))
    return w, s, e, n


# ------------------------------------------------------------- 3D terrain deck
ELEVATION_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TEXTURE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TERRARIUM_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}


def terrain_deck(fires, claims, center, zoom=7, pitch=55, exaggeration=1.5, column_scale=50):
    """deck.gl scene: real elevation (mountains) + fire columns (height = FRP) + claim points."""
    layers = [pdk.Layer("TerrainLayer", elevation_decoder=TERRARIUM_DECODER, texture=TEXTURE_TILES,
                        elevation_data=ELEVATION_TILES, mesh_max_error=4.0, material=False,
                        elevation_scale=exaggeration)]
    if len(fires):
        f = fires.dropna(subset=["latitude", "longitude"]).copy()
        f["frp_v"] = f["frp"].fillna(1.0).clip(lower=1.0)
        f["color"] = f["product"].map(lambda p: _hex_rgb(maps.PRODUCT_COLORS.get(p, "#ff5252")))
        f = f[["latitude", "longitude", "frp_v", "color", "product", "acq_date", "acq_time", "satellite", "confidence"]]
        layers.append(pdk.Layer("ColumnLayer", data=f, get_position="[longitude, latitude]", get_elevation="frp_v",
                                elevation_scale=column_scale, radius=600, get_fill_color="color", pickable=True,
                                auto_highlight=True, extruded=True))
    if claims is not None and len(claims):
        c = claims.dropna(subset=["latitude", "longitude"]).copy()
        c["color"] = c["verdict"].map(lambda v: _hex_rgb(VERDICT_COLORS.get(v, "#607d8b")))
        c["label"] = c["llm_location"].astype(str)
        c = c[["latitude", "longitude", "color", "label", "verdict"]]
        layers.append(pdk.Layer("ScatterplotLayer", data=c, get_position="[longitude, latitude]", get_fill_color="color",
                                get_radius=2500, radius_min_pixels=5, pickable=True, stroked=True,
                                get_line_color=[255, 255, 255], line_width_min_pixels=1))
    view = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom, pitch=pitch, bearing=-15)
    tooltip = {"html": "<b>{product}{label}</b><br/>{satellite} {acq_date} {acq_time}<br/>FRP {frp_v} MW {verdict}",
               "style": {"background": "#111", "color": "#eee", "fontSize": "12px"}}
    return pdk.Deck(layers=layers, initial_view_state=view, map_style=None, tooltip=tooltip)


def _hex_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [210]


# ------------------------------------------------------------------- 1. live map
def live_fire_map(d):
    st.title("Live fire map")
    st.caption("Current NASA FIRMS detections from every available satellite product, with the project's verified claims overlaid.")
    key = firms.get_map_key()
    if not key:
        st.error("FIRMS_MAP_KEY is not configured. Add it on the **Data Sources & Keys** page.")
        return
    fp = key[:4]
    avail = availability_table(fp)
    today = datetime.now(timezone.utc).date()

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        region = c1.selectbox("Region", list(REGIONS))
        date = c2.date_input("Last day of window (UTC)", value=today, max_value=today)
        day_range = c3.slider("Days back", 1, 10, 1)
        view = c4.radio("View", ["2D dark map", "3D terrain"], horizontal=True)
        live_products = [p for p in firms.LIVE_PRODUCTS if p in avail and avail[p][1] >= str(date - timedelta(days=day_range))]
        default = [p for p in live_products if p in ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT")]
        if region.startswith("World"):
            live_products = [p for p in live_products if "VIIRS" in p or "MODIS" in p]
        products = st.multiselect("Satellite products", live_products, default=default or live_products[:3],
                                  help="Landsat covers US+Canada only; GOES covers the Americas (geostationary, 10-min).")
        c5, c6, c7 = st.columns([1.5, 1, 1])
        basemap = c5.selectbox("Basemap", list(maps.BASEMAPS), index=0)
        relief = c6.toggle("Terrain relief", value=True)
        show_claims = c7.toggle("Overlay verified claims", value=True)

    if not products:
        st.info("Select at least one product.")
        return
    bbox = REGIONS[region]
    start = (date - timedelta(days=day_range - 1)).isoformat()
    with st.spinner(f"Querying NASA FIRMS for {len(products)} product(s)..."):
        fires, status = fetch_live(bbox, start, tuple(products), day_range, fp)

    bad = {p: s for p, s in status.items() if s[0] != "ok"}
    if bad:
        st.warning("Some products failed: " + "; ".join(f"{p}: {s[0]} {s[1]}" for p, s in bad.items()))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fire pixels", f"{len(fires):,}")
    m2.metric("Products returning data", f"{sum(1 for s in status.values() if s[2] > 0)}/{len(products)}")
    m3.metric("Total FRP (MW)", f"{fires['frp'].sum():,.0f}" if len(fires) else "0")
    m4.metric("Latest acquisition (UTC)", fires["acq_dt"].max().strftime("%m-%d %H:%M") if len(fires) and fires["acq_dt"].notna().any() else "n/a")

    ver = d["verification"]
    claims = None
    if show_claims and ver is not None:
        w, s, e, n = _bbox_tuple(bbox)
        c = ver.dropna(subset=["latitude", "longitude"])
        claims = c[(c["longitude"].between(w, e)) & (c["latitude"].between(s, n))]
        claims = claims[(claims["created_at"].dt.date >= date - timedelta(days=day_range + 2))
                        & (claims["created_at"].dt.date <= date)]
        st.caption(f"{len(claims)} verified claim(s) from the dataset fall in this region and window.")

    w, s, e, n = _bbox_tuple(bbox)
    center = ((s + n) / 2, (w + e) / 2) if bbox != "world" else (15, 0)
    if view == "3D terrain":
        if len(fires) > 6000:
            st.info(f"Rendering the 3,000 strongest of {len(fires):,} pixels in 3D for performance.")
            fires3 = fires.nlargest(3000, "frp")
        else:
            fires3 = fires
        zoom = 7 if region in ("California",) else 5 if bbox != "world" else 2
        c1, c2 = st.columns(2)
        exag = c1.slider("Vertical exaggeration", 1.0, 4.0, 1.5, 0.5)
        col_scale = c2.slider("Fire column height scale", 10, 300, 60, 10)
        st.pydeck_chart(terrain_deck(fires3, claims, center, zoom=zoom, exaggeration=exag, column_scale=col_scale),
                        height=640)
        st.caption("3D: real elevation from AWS/Mapzen terrarium tiles draped with Esri imagery; each column is a FIRMS fire "
                   "pixel whose height is fire radiative power (MW). Drag to rotate, scroll to zoom, right-drag to tilt.")
    else:
        zoom = 6 if region == "California" else 4 if bbox != "world" else 2
        m = maps.base_map(center, zoom=zoom, basemap=basemap, relief=relief, labels=True)
        if len(fires):
            maps.add_fire_pixels(m, fires.nlargest(20000, "frp") if len(fires) > 20000 else fires)
        if claims is not None and len(claims):
            maps.add_claims(m, claims, cluster=False)
        maps.add_product_legend(m, products)
        maps.finish(m)
        st_folium(m, height=640, use_container_width=True, returned_objects=[])

    if len(fires):
        st.subheader("What each satellite saw")
        per = fires.groupby("product").agg(pixels=("latitude", "size"), total_frp_mw=("frp", "sum"),
                                           max_frp_mw=("frp", "max"), first_utc=("acq_dt", "min"),
                                           last_utc=("acq_dt", "max")).reset_index()
        per["total_frp_mw"] = per["total_frp_mw"].round(0)
        st.dataframe(per, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        hourly = fires.dropna(subset=["acq_dt"]).copy()
        hourly["hour"] = hourly["acq_dt"].dt.floor("h")
        fig = px.bar(hourly.groupby(["hour", "product"]).size().reset_index(name="pixels"), x="hour", y="pixels",
                     color="product", color_discrete_map=maps.PRODUCT_COLORS, template=PLOT_TEMPLATE, height=320,
                     title="Detections per hour (UTC) -- overpass timing by satellite")
        fig.update_layout(margin=dict(t=50, b=10), legend_title="")
        c1.plotly_chart(fig, width="stretch")
        fig = px.histogram(fires.dropna(subset=["frp"]), x="frp", color="product", nbins=40, log_y=True,
                           color_discrete_map=maps.PRODUCT_COLORS, template=PLOT_TEMPLATE, height=320,
                           title="Fire radiative power distribution (log scale)")
        fig.update_layout(margin=dict(t=50, b=10), legend_title="", xaxis_title="FRP (MW)")
        c2.plotly_chart(fig, width="stretch")
        st.download_button("Download these detections (CSV)", fires.drop(columns=["acq_dt"]).to_csv(index=False).encode(),
                           file_name=f"firms_live_{region.replace(' ', '_')}_{start}_{day_range}d.csv", mime="text/csv")
    else:
        st.info("No active-fire pixels were returned for this region and window.")


# ----------------------------------------------------------- 2. data sources
def data_sources(d):
    st.title("Data sources & keys")
    st.caption("Keys are stored only in the git-ignored .env file and never displayed.")

    st.subheader("Drop in API keys")
    with st.form("keys"):
        c1, c2 = st.columns(2)
        firms_key = c1.text_input("NASA FIRMS MAP_KEY", type="password",
                                  placeholder="configured" if env_configured("FIRMS_MAP_KEY") else "paste key (32 chars)",
                                  help="Free: https://firms.modaps.eosdis.nasa.gov/api/map_key/")
        google_key = c2.text_input("Google Geocoding key (optional)", type="password",
                                   placeholder="configured" if env_configured("GOOGLE_GEOCODING_API_KEY") else "optional")
        x_token = c1.text_input("X API bearer token (optional)", type="password",
                                placeholder="configured" if env_configured("X_BEARER_TOKEN") else "optional, paid collection only")
        ollama = c2.text_input("Ollama model (optional)", value=os.environ.get("OLLAMA_MODEL", ""), placeholder="llama3.2")
        saved = st.form_submit_button("Save to .env", type="primary")
    if saved:
        vals = {"FIRMS_MAP_KEY": firms_key, "GOOGLE_GEOCODING_API_KEY": google_key, "X_BEARER_TOKEN": x_token,
                "OLLAMA_MODEL": ollama}
        if any(v.strip() for v in vals.values()):
            save_env_values(vals)
            st.cache_data.clear()
            st.success(f"Saved to {ENV_PATH.name} (git-ignored). Values are now active in this session.")
        else:
            st.info("Nothing entered.")

    st.subheader("Connection status")
    rows = []
    for a in API_INVENTORY:
        configured = env_configured(a["env"]) if a["env"] else True
        rows.append({"service": a["service"], "purpose": a["purpose"], "key required": "yes" if a["needs_key"] else "no",
                     "configured": "yes" if configured else "NO", "env variable": a["env"] or "-", "docs": a["url"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"docs": st.column_config.LinkColumn("docs")})

    key = firms.get_map_key()
    if key:
        code, text = firms_key_status(key)
        if code == 200 and "transaction_limit" in text:
            st.success(f"NASA FIRMS key valid. Quota: {text.strip()}")
        else:
            st.error(f"NASA FIRMS key check failed (HTTP {code}): {text}")
    else:
        st.warning("NASA FIRMS key not configured -- satellite retrieval is disabled.")

    st.subheader("NASA FIRMS satellite products: coverage and which is best for what")
    avail = availability_table(key[:4]) if key else {}
    sc = load_eval("sensor_comparison.csv")
    emp = {}
    if sc is not None:
        for _, r in sc[sc["sensor"].isin(["VIIRS", "MODIS"])].iterrows():
            emp[r["sensor"]] = r
    rows = []
    for p, m in firms.PRODUCTS.items():
        lo, hi = avail.get(p, ("", ""))
        e = emp.get(m["instrument"])
        rows.append({
            "product": p, "satellite": m["satellite"], "instrument": m["instrument"],
            "pixel size (m)": m["resolution_m"], "revisit": m["revisit"], "coverage": m["coverage"],
            "type": m["kind"], "available from": lo, "available to": hi,
            "match rate on our claims": f"{e['match_rate']:.0%}" if e is not None and pd.notna(e["match_rate"]) else "n/a",
            "median nearest km": e["median_nearest_km"] if e is not None else None,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        "**How to choose (from the specs above and our measured results):**\n"
        "- **VIIRS (SNPP, NOAA-20, NOAA-21)** -- best default for *detecting* fires near a claim: 375 m pixels, three "
        "satellites giving ~6 looks/day, global, archive back to 2012. On this dataset VIIRS matched "
        f"{emp['VIIRS']['match_rate']:.0%} of geocoded wildfire claims vs MODIS {emp['MODIS']['match_rate']:.0%}.\n"
        if emp else
        "- **VIIRS** -- best default for detecting fires near a claim (375 m, three satellites, global, archive to 2012).\n")
    st.markdown(
        "- **MODIS (Terra/Aqua)** -- coarser (1 km) but the longest record (2000-) and the most-cited FRP; useful for "
        "large fires and long-term baselines.\n"
        "- **Landsat 8/9 (30 m)** -- the sharpest geolocation, but only US + Canada and an 8-day revisit: good to "
        "*confirm where* a fire was, poor for *whether* it burned on a given day.\n"
        "- **GOES-16/18 (ABI, 2 km, every 10 min)** -- best *timing* evidence in the Americas (was it burning at "
        "the hour of the post?), but coarse and misses small fires.\n"
        "- **NRT vs SP**: NRT appears within ~3 h but is reprocessed later; the Standard-Processing archive is the "
        "quality-controlled record -- the pipeline picks whichever covers the claim date.")

    st.subheader("Datasets on disk")
    files = [
        ("Annotated posts (LLM)", PROCESSED / "x_llm_annotated.csv"),
        ("Satellite verification", PROCESSED / "satellite_verification.csv"),
        ("Retrieved FIRMS detections", PROCESSED / "satellite_detections.json"),
        ("FIRMS response cache", PROCESSED / "firms_cache.json"),
        ("Collected X posts (merged)", PROCESSED / "x_relevant_merged.csv"),
        ("Geocode cache", ROOT / "data" / "geocode_cache.json"),
        ("Evaluation outputs", ROOT / "data" / "evaluation" / "summary.json"),
    ]
    rows = [{"dataset": n, "path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "exists": p.exists(),
             "size (KB)": round(p.stat().st_size / 1024) if p.exists() else None,
             "updated": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if p.exists() else ""}
            for n, p in files]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# --------------------------------------------------------- 3. collected data
@st.cache_data(show_spinner=False)
def load_collected(_mtime):
    path = PROCESSED / "x_relevant_merged.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce")
    return df


def _raw_candidate_counts():
    out = {}
    for f in glob.glob(str(PROCESSED / "x_raw_candidates*.csv")):
        name = Path(f).stem.replace("x_raw_candidates", "").strip("_") or "main"
        out[name] = len(pd.read_csv(f, usecols=["post_id"], dtype=str))  # tweets span lines; count rows not lines
    return out


def collected_data(d):
    st.title("Collected data")
    st.caption("Raw corpus from the official X API v2, before annotation.")
    path = PROCESSED / "x_relevant_merged.csv"
    df = load_collected(path.stat().st_mtime if path.exists() else 0)
    if df is None:
        st.info("No collected posts found at data/processed/x_relevant_merged.csv.")
        st.code("python src/x_ingestion.py --mode search --target 10 --confirm-paid-run", language="bash")
        return

    raw = _raw_candidate_counts()
    ann = d["annotated"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw candidates read", f"{sum(raw.values()):,}" if raw else "n/a", help="Posts returned by the API across all passes")
    c2.metric("Relevant unique posts", f"{len(df):,}")
    c3.metric("LLM-annotated", f"{len(ann):,}" if ann is not None else "0")
    c4.metric("Collection passes", df["_source"].nunique() if "_source" in df else "n/a")
    c5.metric("Geotagged by author", f"{df['geo_place_id'].notna().sum():,}", help="Place metadata attached by the poster -- rare")

    summary = ROOT / "data" / "x_merged_collection_summary.csv"
    if summary.exists():
        s = pd.read_csv(summary).iloc[0]
        st.caption(f"Date range **{s['date_range']}** - monthly: {s['monthly_coverage'].replace(';', ', ')} - "
                   f"posts with media: {s['posts_with_media']} - keyword mix: {s['hazard_mix_keyword_scan'].replace(';', ', ')}")

    left, right = st.columns(2)
    if "_source" in df:
        src = df["_source"].value_counts().rename_axis("pass").reset_index(name="posts")
        if raw:
            src["raw_candidates"] = src["pass"].map({"english_main": raw.get("main"), "english_gap": raw.get("gap"),
                                                     "english_recent": raw.get("recent"), "international": raw.get("intl")})
        fig = px.bar(src, x="pass", y=["posts"] + (["raw_candidates"] if raw else []), barmode="group",
                     title="Collection passes: raw candidates vs relevant posts kept", template=PLOT_TEMPLATE, height=330)
        fig.update_layout(margin=dict(t=50, b=10), legend_title="", xaxis_title="")
        left.plotly_chart(fig, width="stretch")
    daily = df.dropna(subset=["created_at"]).groupby(df["created_at"].dt.date).size().reset_index(name="posts")
    daily.columns = ["day", "posts"]
    fig = px.area(daily, x="day", y="posts", title="Posts per day (post creation time)", template=PLOT_TEMPLATE, height=330)
    fig.update_layout(margin=dict(t=50, b=10))
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    lang = df["lang"].fillna("?").value_counts().head(10).rename_axis("language").reset_index(name="posts")
    fig = px.bar(lang, x="language", y="posts", title="Language", template=PLOT_TEMPLATE, height=300, text="posts")
    fig.update_layout(margin=dict(t=50, b=10))
    left.plotly_chart(fig, width="stretch")
    fig = px.histogram(df.dropna(subset=["relevance_score"]), x="relevance_score", nbins=20,
                       title="Rule-based relevance score (local filter, 0-1)", template=PLOT_TEMPLATE, height=300)
    fig.update_layout(margin=dict(t=50, b=10))
    right.plotly_chart(fig, width="stretch")

    st.subheader("Queries that produced the posts")
    q = df["query_used"].fillna("").map(lambda s: s.split(" lang:")[0]).value_counts().rename_axis("query").reset_index(name="posts")
    st.dataframe(q, hide_index=True, width="stretch", height=260)

    st.subheader("Browse the raw posts")
    search = st.text_input("Search raw text", key="raw_search")
    f = df
    if search:
        f = f[f["text"].str.contains(search, case=False, na=False)]
    show = f[["post_id", "created_at", "lang", "_source", "relevance_score", "has_media", "text"]].copy() if "_source" in f \
        else f[["post_id", "created_at", "lang", "relevance_score", "text"]].copy()
    show["created_at"] = show["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    show["text"] = show["text"].map(lambda t: short(t, 200))
    st.caption(f"{len(f):,} posts")
    st.dataframe(show.sort_values("created_at", ascending=False), hide_index=True, width="stretch", height=380)
    st.download_button("Download collected posts (CSV)", f.to_csv(index=False).encode(), file_name="x_collected_posts.csv",
                       mime="text/csv")
