"""Dashboard pages. Each function renders one page from the loaded data dict."""
import json
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from dashboard import maps
from dashboard.data import (HAZARD_COLORS, VERDICT_COLORS, VERDICT_LABELS, VERDICT_ORDER,
                            annotated_dataset, detections_frame, load_csv, load_eval, load_summary, short,
                            spacy_model, to_excel_bytes)

PLOT_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 340


def _empty(msg, cmd=None):
    st.info(msg)
    if cmd:
        st.code(cmd, language="bash")


def _wildfire(ver):
    return ver[ver["llm_hazard_type"] == "wildfire"] if ver is not None else None


def _verdict_bar(counts, title):
    df = counts.rename_axis("verdict").reset_index(name="posts")
    df["verdict"] = pd.Categorical(df["verdict"], VERDICT_ORDER)
    df = df.sort_values("verdict")
    fig = px.bar(df, x="verdict", y="posts", color="verdict", color_discrete_map=VERDICT_COLORS,
                 text="posts", title=title, template=PLOT_TEMPLATE, height=CHART_HEIGHT)
    fig.update_layout(showlegend=False, xaxis_title="", margin=dict(t=50, b=10))
    return fig


def _download(df, label, name):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")


def _dataset_link(label="Open annotated dataset (spreadsheet) in a new tab"):
    st.markdown(f"<a href='/dataset' target='_blank' style='display:inline-block;padding:6px 14px;border:1px solid "
                f"#3d4756;border-radius:6px;text-decoration:none;color:#d6dae2'>{label} &#8599;</a>",
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- 1
def overview(d):
    ann, ver = d["annotated"], d["verification"]
    st.title("Overview")
    if ann is None:
        return _empty("No annotated dataset found at data/processed/x_llm_annotated.csv.",
                      "python src/llm_annotate.py")

    fire = ann[ann["llm_hazard_type"] == "wildfire"]
    vf = _wildfire(ver)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Annotated posts", f"{len(ann):,}")
    c2.metric("Assert a hazard event", f"{int(ann['llm_is_hazard_claim'].sum()):,}")
    c3.metric("Wildfire posts (FIRMS-checkable)", f"{len(fire):,}")
    c4.metric("Wildfire posts verified", f"{len(vf):,}" if vf is not None else "0",
              help="Posts run through the NASA FIRMS pipeline")
    c5.metric("Satellite-supported", f"{int(vf['verdict'].eq('SUPPORTED').sum()):,}" if vf is not None else "0",
              help="A FIRMS fire pixel within 25 km and 48 h of the claim")

    dmin, dmax = ann["created_at"].min(), ann["created_at"].max()
    st.caption(f"Posts collected via the official X API v2 between **{dmin:%Y-%m-%d}** and **{dmax:%Y-%m-%d}**; "
               "annotated with a local LLM (Ollama); geocoded with Nominatim; verified against NASA FIRMS "
               "VIIRS + MODIS (NRT and Standard-Processing archive).")

    left, right = st.columns(2)
    hz = ann["llm_hazard_type"].fillna("none").value_counts().rename_axis("hazard").reset_index(name="posts")
    fig = px.bar(hz, x="hazard", y="posts", color="hazard", color_discrete_map=HAZARD_COLORS, text="posts",
                 title="Event categories (LLM hazard type)", template=PLOT_TEMPLATE, height=CHART_HEIGHT)
    fig.update_layout(showlegend=False, xaxis_title="", margin=dict(t=50, b=10))
    left.plotly_chart(fig, width="stretch")

    tl = ann.dropna(subset=["created_at"]).copy()
    tl["week"] = tl["created_at"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    tl = tl.groupby(["week", "llm_hazard_type"]).size().reset_index(name="posts")
    fig = px.bar(tl, x="week", y="posts", color="llm_hazard_type", color_discrete_map=HAZARD_COLORS,
                 title="Posts per week by hazard type", template=PLOT_TEMPLATE, height=CHART_HEIGHT)
    fig.update_layout(legend_title="", xaxis_title="", margin=dict(t=50, b=10))
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    risk = ann["llm_misinfo_risk"].fillna("n/a").value_counts().rename_axis("risk").reset_index(name="posts")
    fig = px.pie(risk, names="risk", values="posts", hole=0.5, title="LLM misinformation-risk flag",
                 color="risk", color_discrete_map={"low": "#66bb6a", "medium": "#ffa726", "high": "#ef5350"},
                 template=PLOT_TEMPLATE, height=CHART_HEIGHT)
    fig.update_layout(margin=dict(t=50, b=10))
    left.plotly_chart(fig, width="stretch")

    if vf is not None and len(vf):
        right.plotly_chart(_verdict_bar(vf["verdict"].value_counts(), "Satellite verification verdicts (wildfire posts)"),
                           width="stretch")
    else:
        with right:
            _empty("Satellite verification has not been run yet.", "python src/satellite_verification.py --fire-only")

    st.subheader("Pipeline funnel")
    funnel = load_eval("funnel.csv")
    if funnel is not None:
        fig = go.Figure(go.Funnel(y=funnel["stage"], x=funnel["posts"], textinfo="value+percent initial",
                                  marker={"color": "#1565c0"}))
        fig.update_layout(template=PLOT_TEMPLATE, height=420, margin=dict(t=20, b=10, l=10))
        st.plotly_chart(fig, width="stretch")
    else:
        _empty("Run the evaluation script to build the funnel.", "python src/research_evaluation.py")

    st.subheader("Geographic distribution of verified claims")
    if ver is not None and ver["latitude"].notna().any():
        geo = ver.dropna(subset=["latitude", "longitude"]).copy()
        geo["country"] = geo["display_name"].fillna("").str.split(",").str[-1].str.strip()
        top = geo["country"].value_counts().head(12).rename_axis("country").reset_index(name="claims")
        c1, c2 = st.columns([1, 2])
        c1.dataframe(top, hide_index=True, width="stretch")
        fig = px.scatter_geo(geo, lat="latitude", lon="longitude", color="verdict",
                             color_discrete_map=VERDICT_COLORS, hover_name="llm_location",
                             title="Geocoded claim locations", template=PLOT_TEMPLATE, height=420)
        fig.update_geos(showcountries=True, showland=True, landcolor="#f5f5f5", fitbounds="locations")
        fig.update_layout(legend_title="", margin=dict(t=50, b=0, l=0, r=0))
        c2.plotly_chart(fig, width="stretch")
    else:
        _empty("Geocoded claim coordinates appear here after verification runs.")


# --------------------------------------------------------------------------- 2
def claim_explorer(d):
    ann, ver, store = d["annotated"], d["verification"], d["detections"]
    st.title("Claim explorer")
    if ann is None:
        return _empty("No annotated dataset found.", "python src/llm_annotate.py")

    df = ann.copy()
    if ver is not None:
        df = df.merge(ver[["post_id", "verdict", "latitude", "longitude", "display_name", "nearest_detection_km",
                           "nearest_detection_hours", "evidence_score", "explanation", "matched_sensors",
                           "geocode_query", "location_confidence", "ambiguity_status"]],
                      on="post_id", how="left")
    else:
        for c in ("verdict", "latitude", "longitude", "display_name", "explanation"):
            df[c] = None
    df["verdict"] = df["verdict"].fillna("NOT_VERIFIED")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        q = c1.text_input("Search text / location / claim", placeholder="e.g. San Clemente, evacuation, smoke")
        hz = c2.multiselect("Hazard type", sorted(df["llm_hazard_type"].fillna("none").unique()), default=["wildfire"])
        vd = c3.multiselect("Satellite verdict", sorted(df["verdict"].unique()))
        rk = c4.multiselect("LLM risk flag", sorted(df["llm_misinfo_risk"].dropna().unique()))
        c5, c6 = st.columns([1, 3])
        stance = c5.multiselect("Stance", sorted(df["llm_stance"].dropna().unique()))
        dmin, dmax = df["created_at"].min().date(), df["created_at"].max().date()
        dr = c6.slider("Post date range", dmin, dmax, (dmin, dmax))

    f = df
    if q:
        ql = q.lower()
        f = f[f["text"].str.lower().str.contains(ql, na=False) | f["llm_location"].str.lower().str.contains(ql, na=False)
              | f["llm_claim"].fillna("").str.lower().str.contains(ql, na=False)]
    if hz:
        f = f[f["llm_hazard_type"].fillna("none").isin(hz)]
    if vd:
        f = f[f["verdict"].isin(vd)]
    if rk:
        f = f[f["llm_misinfo_risk"].isin(rk)]
    if stance:
        f = f[f["llm_stance"].isin(stance)]
    f = f[(f["created_at"].dt.date >= dr[0]) & (f["created_at"].dt.date <= dr[1])]
    f = f.sort_values("created_at", ascending=False)

    st.caption(f"{len(f):,} of {len(df):,} posts match.")
    if f.empty:
        return _empty("No posts match these filters.")

    table = f[["post_id", "created_at", "llm_hazard_type", "llm_location", "llm_misinfo_risk", "verdict", "text"]].copy()
    table["created_at"] = table["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    table["text"] = table["text"].map(lambda t: short(t, 140))
    sel = st.dataframe(table, hide_index=True, width="stretch", height=320,
                       on_select="rerun", selection_mode="single-row",
                       column_config={"post_id": st.column_config.TextColumn("post id", width="small"),
                                      "text": st.column_config.TextColumn("post text", width="large")})
    rows = sel.get("selection", {}).get("rows", []) if sel else []
    row = f.iloc[rows[0]] if rows else f.iloc[0]
    if not rows:
        st.caption("Select a row to inspect a claim. Showing the first match.")
    _download(f.drop(columns=["explanation"], errors="ignore"), "Download filtered claims (CSV)", "filtered_claims.csv")

    st.divider()
    _claim_detail(row, store)


def _claim_detail(row, store):
    st.subheader("Claim detail")
    st.markdown(f"> {row['text']}")
    st.caption(f"post_id `{row['post_id']}` - posted {row['created_at']:%Y-%m-%d %H:%M} UTC - lang {row.get('lang', '')}")

    a, b, c = st.columns(3)
    with a:
        st.markdown("**LLM annotation** (local Ollama model, text only)")
        st.write({
            "hazard claim": bool(row["llm_is_hazard_claim"]), "hazard type": row["llm_hazard_type"],
            "location": row["llm_location"], "claim": row["llm_claim"], "stance": row["llm_stance"],
            "misinfo risk": row["llm_misinfo_risk"], "confidence": row["llm_confidence"],
            "reason": row["llm_reason"],
        })
    with b:
        st.markdown("**Model predictions** (independent of the LLM)")
        nlp = spacy_model()
        ents = [(e.text, e.label_) for e in nlp(str(row["text"])).ents if e.label_ in {"GPE", "LOC", "FAC"}] if nlp else []
        try:
            import relevance_filter
            rel = relevance_filter.classify(str(row["text"]))
            rel_out = {"relevance": rel["relevance_label"], "score": rel["relevance_score"], "reason": rel["reason"]}
        except Exception:
            rel_out = {"relevance": "unavailable"}
        st.write({"spaCy geographic entities": ents or "none", **rel_out,
                  "geocoded to": row.get("display_name") or "not geocoded",
                  "geocoder query": row.get("geocode_query") or "n/a",
                  "resolver confidence": row.get("location_confidence") if pd.notna(row.get("location_confidence")) else "n/a",
                  "ambiguity": row.get("ambiguity_status") or "n/a",
                  "coordinates": (f"{row['latitude']:.4f}, {row['longitude']:.4f}"
                                  if pd.notna(row.get("latitude")) else "n/a")})
    with c:
        st.markdown("**NASA satellite evidence**")
        v = row["verdict"]
        color = VERDICT_COLORS.get(v, "#607d8b")
        st.markdown(f"<div style='padding:8px 12px;border-left:5px solid {color};background:#161b26'>"
                    f"<b>{VERDICT_LABELS.get(v, v)}</b></div>", unsafe_allow_html=True)
        if v == "SUPPORTED":
            st.metric("Nearest matched detection", f"{row['nearest_detection_km']:.1f} km",
                      f"{row['nearest_detection_hours']:+.1f} h from post", delta_color="off")
            st.write(f"Sensors matched: **{row['matched_sensors']}** - evidence score {row['evidence_score']:.2f}")
        if isinstance(row.get("explanation"), str):
            st.caption(row["explanation"])

    dets = detections_frame(store, {row["post_id"]})
    if pd.notna(row.get("latitude")):
        st.markdown("**Satellite observations retrieved for this claim**")
        m = maps.base_map((row["latitude"], row["longitude"]), zoom=9, relief=False, labels=False,
                          imagery_date=row["created_at"].strftime("%Y-%m-%d"), imagery_layer="VIIRS SNPP true colour")
        maps.add_radius(m, row["latitude"], row["longitude"])
        maps.add_claims(m, pd.DataFrame([row]), cluster=False)
        if len(dets):
            maps.add_detections(m, dets)
        maps.add_legend(m, verdicts=False)
        maps.finish(m)
        st_folium(m, height=420, use_container_width=True, returned_objects=[])
        if len(dets):
            show = dets.sort_values("distance_km")[["sensor", "satellite", "acq_date", "acq_time", "distance_km",
                                                    "hours_diff", "frp", "confidence", "matched"]]
            st.dataframe(show, hide_index=True, width="stretch", height=220)
            _download(show, "Download detections (CSV)", f"detections_{row['post_id']}.csv")
        elif row["verdict"] in ("NO_NEARBY_DETECTION",):
            st.caption("FIRMS returned no active-fire pixels in the 0.5 degree search box for the +/-1 day window.")


# --------------------------------------------------------------------------- 3
def satellite_map(d):
    ver, store = d["verification"], d["detections"]
    st.title("Claims map")
    _dataset_link()
    vf = _wildfire(ver)
    if vf is None or vf.empty:
        return _empty("Run the satellite verification first.", "python src/satellite_verification.py --fire-only")
    vf = vf.dropna(subset=["latitude", "longitude"])

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        dmin, dmax = vf["created_at"].min().date(), vf["created_at"].max().date()
        dr = c1.slider("Claim date range", dmin, dmax, (dmin, dmax), key="map_dates")
        sensors = c2.multiselect("Sensor", ["VIIRS", "MODIS"], default=["VIIRS", "MODIS"])
        verdicts = c3.multiselect("Verdict", [v for v in VERDICT_ORDER if v in set(vf["verdict"])],
                                  default=[v for v in ("SUPPORTED", "NO_NEARBY_DETECTION") if v in set(vf["verdict"])])
        show_unmatched = c4.toggle("Show unmatched pixels", value=False)
        c5, c6 = st.columns([1, 2])
        img_layer = c5.selectbox("NASA GIBS imagery", ["none"] + list(maps.GIBS_LAYERS))
        img_date = c6.date_input("Imagery date", value=dmax, min_value=dmin, max_value=dmax) if img_layer != "none" else None

    f = vf[(vf["created_at"].dt.date >= dr[0]) & (vf["created_at"].dt.date <= dr[1])]
    if verdicts:
        f = f[f["verdict"].isin(verdicts)]
    dets = detections_frame(store, set(f["post_id"]))
    dets = dets[dets["sensor"].isin(sensors)]
    if not show_unmatched:
        dets = dets[dets["matched"] == True]  # noqa: E712

    m1, m2, m3 = st.columns(3)
    m1.metric("Claims shown", len(f))
    m2.metric("FIRMS pixels shown", len(dets))
    m3.metric("Claims with matched pixels", int(f["verdict"].eq("SUPPORTED").sum()))

    if f.empty:
        return _empty("No claims in this selection.")
    center = (f["latitude"].median(), f["longitude"].median())
    m = maps.base_map(center, zoom=5, imagery_date=str(img_date) if img_date else None,
                      imagery_layer=None if img_layer == "none" else img_layer)
    if len(dets):
        maps.add_detections(m, dets, show_unmatched=show_unmatched)
    maps.add_claims(m, f)
    maps.add_legend(m)
    maps.finish(m)
    st_folium(m, height=620, use_container_width=True, returned_objects=[])
    with st.expander("About this map"):
        st.markdown("Fire pixels are NASA FIRMS VIIRS (375 m) and MODIS (1 km) detections returned for each claim's "
                    "±1-day window; faded pixels were in the search box but outside the 25 km / 48 h rule. Imagery is "
                    "NASA GIBS true colour for the selected date.")


# --------------------------------------------------------------------------- 3b
def dataset(d):
    st.title("Annotated dataset")
    df = annotated_dataset(d["annotated"], d["verification"])
    if df is None:
        return _empty("No annotated dataset found at data/processed/x_llm_annotated.csv.",
                      "python src/llm_annotate.py")
    st.caption("Every collected X post with its local-LLM annotations (llm_*) and, for wildfire posts, the NASA "
               "FIRMS verdict and evidence. Click a column header to sort; use the filters to narrow rows.")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        q = c1.text_input("Search text / location / claim", key="ds_q")
        hz = c2.multiselect("Hazard type", sorted(df["llm_hazard_type"].fillna("none").unique()), key="ds_hz")
        vd = c3.multiselect("Satellite verdict", sorted(df["verdict"].unique()), key="ds_vd")
        rk = c4.multiselect("LLM risk flag", sorted(df["llm_misinfo_risk"].dropna().unique()), key="ds_rk")

    f = df
    if q:
        ql = q.lower()
        f = f[f["text"].fillna("").str.lower().str.contains(ql, regex=False)
              | f["llm_location"].str.lower().str.contains(ql, regex=False)
              | f["llm_claim"].fillna("").str.lower().str.contains(ql, regex=False)]
    if hz:
        f = f[f["llm_hazard_type"].fillna("none").isin(hz)]
    if vd:
        f = f[f["verdict"].isin(vd)]
    if rk:
        f = f[f["llm_misinfo_risk"].isin(rk)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows shown", f"{len(f):,}")
    m2.metric("Total posts", f"{len(df):,}")
    m3.metric("Wildfire posts", f"{int(f['llm_hazard_type'].eq('wildfire').sum()):,}")
    m4.metric("Satellite-supported", f"{int(f['verdict'].eq('SUPPORTED').sum()):,}")

    st.dataframe(f, hide_index=True, width="stretch", height=640,
                 column_config={"post_id": st.column_config.TextColumn("post_id", pinned=True),
                                "text": st.column_config.TextColumn("text", width="large"),
                                "llm_claim": st.column_config.TextColumn("llm_claim", width="large"),
                                "explanation": st.column_config.TextColumn("explanation", width="large")})
    a, b = st.columns(2)
    a.download_button("Download as Excel (.xlsx)", to_excel_bytes(f), file_name="annotated_posts.xlsx",
                      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with b:
        _download(f, "Download as CSV", "annotated_posts.csv")


# --------------------------------------------------------------------------- 4
def verification_results(d):
    ver, store = d["verification"], d["detections"]
    st.title("Verification")
    vf = _wildfire(ver)
    if vf is None or vf.empty:
        return _empty("Run the satellite verification first.", "python src/satellite_verification.py --fire-only")

    counts = vf["verdict"].value_counts()
    cols = st.columns(len(counts))
    for col, (v, n) in zip(cols, counts.items()):
        col.metric(VERDICT_LABELS.get(v, v), n)
    st.markdown("A verdict other than *supported* is not a misinformation label: FIRMS misses small, brief or "
                "cloud-obscured fires, and centroid geocoding can place a claim more than 25 km from the fire.")

    sup = vf[vf["verdict"] == "SUPPORTED"]
    left, right = st.columns(2)
    if len(sup):
        fig = px.histogram(sup, x="nearest_detection_km", nbins=25, title="Distance to nearest matched fire pixel (km)",
                           template=PLOT_TEMPLATE, height=CHART_HEIGHT, color_discrete_sequence=["#2e7d32"])
        fig.update_layout(margin=dict(t=50, b=10), yaxis_title="claims")
        left.plotly_chart(fig, width="stretch")
        fig = px.histogram(sup, x="nearest_detection_hours", nbins=24,
                           title="Time offset of nearest matched pixel (h; negative = before post)",
                           template=PLOT_TEMPLATE, height=CHART_HEIGHT, color_discrete_sequence=["#1565c0"])
        fig.update_layout(margin=dict(t=50, b=10), yaxis_title="claims")
        right.plotly_chart(fig, width="stretch")

    st.subheader("Sensor comparison on the same claims")
    sc = load_eval("sensor_comparison.csv")
    if sc is not None:
        st.dataframe(sc, hide_index=True, width="stretch")
        both = sc[sc["sensor"] == "BOTH_OK"]
        if len(both):
            b = both.iloc[0]
            st.caption(f"Where both sensors returned data ({int(b['claims_queried_ok'])} claims): "
                       f"{int(b['claims_with_matched_detection'])} matched on both, "
                       f"{int(b['viirs_only_matches'])} VIIRS-only, {int(b['modis_only_matches'])} MODIS-only.")

    st.subheader("Does geocoding quality affect the verdict?")
    lq = load_eval("location_quality.csv")
    if lq is not None:
        st.dataframe(lq, hide_index=True, width="stretch")
        st.caption("`ambiguous` = the place name had several comparable geocoder candidates (e.g. 'Palisades'); "
                   "`simplified_query` = the full location string failed and a shorter query was used. A SUPPORTED "
                   "verdict on an ambiguous or simplified geocode is weaker evidence and is flagged in the explanation.")

    st.subheader("Per-claim evidence")
    show_v = st.multiselect("Filter verdict", list(counts.index), default=list(counts.index), key="vr_verdict")
    t = vf[vf["verdict"].isin(show_v)].sort_values(["verdict", "evidence_score"], ascending=[True, False])
    table = t[["post_id", "created_at", "llm_location", "display_name", "ambiguity_status", "verdict", "matched_sensors",
               "nearest_detection_km", "nearest_detection_hours", "evidence_score",
               "viirs_detections_matched", "modis_detections_matched", "viirs_source", "modis_source"]].copy()
    table["created_at"] = table["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(table, hide_index=True, width="stretch", height=360)

    by_id = t.set_index("post_id")
    labels = {p: f"{p} - {short(loc, 40)}" for p, loc in zip(t["post_id"], t["llm_location"])}
    pid = st.selectbox("Explain a claim", t["post_id"].tolist(), format_func=labels.get)
    if pid:
        r = by_id.loc[pid]
        st.markdown(f"> {r['text']}")
        st.info(r["explanation"])
        dets = detections_frame(store, {pid})
        report = {
            "post_id": pid, "created_at": str(r["created_at"]), "claimed_location": r["llm_location"],
            "geocoded": {"display_name": r["display_name"], "latitude": r["latitude"], "longitude": r["longitude"],
                         "query": r["geocode_query"], "confidence": r["location_confidence"],
                         "ambiguity_status": r["ambiguity_status"]},
            "verdict": r["verdict"], "explanation": r["explanation"], "evidence_score": r["evidence_score"],
            "sensors": {s: {k: (None if pd.isna(r[f"{s}_{k}"]) else r[f"{s}_{k}"]) if not isinstance(r[f"{s}_{k}"], str) else r[f"{s}_{k}"]
                            for k in ("source", "status", "detections_total", "detections_matched", "nearest_km",
                                      "nearest_hours", "max_frp", "observation_date")} for s in ("viirs", "modis")},
            "detections": dets.to_dict(orient="records"),
            "matching_rule": "fire pixel within 25 km and 48 h of the post",
            "evidence_source": r["evidence_source"],
        }
        st.download_button("Download evidence report (JSON)", json.dumps(report, indent=2, default=str).encode(),
                           file_name=f"evidence_{pid}.json", mime="application/json")

    st.subheader("Downloads")
    c1, c2 = st.columns(2)
    with c1:
        _download(vf, "All wildfire verification results (CSV)", "satellite_verification_wildfire.csv")
    with c2:
        if st.checkbox("Prepare full FIRMS detections export (large)"):
            _download(detections_frame(store), "All retrieved FIRMS detections (CSV)", "firms_detections.csv")


# --------------------------------------------------------------------------- 5
def model_evaluation(d):
    ver = d["verification"]
    st.title("Evaluation")

    st.subheader("1. Location extraction on independently labelled real tweets (Week 3, HumAID n=100)")
    mc = load_csv("week3_model_comparison.csv")
    diff = load_csv("week3_difficulty_analysis.csv")
    if mc is not None:
        long = mc.melt(id_vars="method", var_name="metric", value_name="percent")
        fig = px.bar(long, x="metric", y="percent", color="method", barmode="group",
                     title="NER methods vs manual location labels (%)", template=PLOT_TEMPLATE, height=380)
        fig.update_layout(legend_title="", xaxis_title="", margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        c1.dataframe(mc, hide_index=True, width="stretch")
        if diff is not None:
            c2.dataframe(diff, hide_index=True, width="stretch")
    else:
        _empty("Week 3 evaluation outputs not present.", "python src/week3_evaluation.py")

    rs = load_csv("week4_real_vs_synthetic.csv")
    if rs is not None:
        st.markdown("**Real vs synthetic text** (same spaCy extractor)")
        st.dataframe(rs, hide_index=True, width="stretch")

    st.subheader("2. NLP-only judgement vs satellite evidence (this dataset)")
    vf = _wildfire(ver)
    if vf is None or vf.empty:
        _empty("Run the satellite verification first.", "python src/satellite_verification.py --fire-only")
    else:
        ct = pd.crosstab(vf["llm_misinfo_risk"].fillna("n/a"), vf["verdict"])
        fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                        title="LLM misinformation-risk flag (rows) x satellite verdict (columns)",
                        template=PLOT_TEMPLATE, height=360)
        fig.update_layout(margin=dict(t=50, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        checked = vf[vf["verdict"].isin(["SUPPORTED", "NO_NEARBY_DETECTION"])]
        if len(checked):
            rate = checked.groupby("llm_misinfo_risk")["verdict"].apply(lambda s: (s == "SUPPORTED").mean()).rename("supported_rate")
            n = checked.groupby("llm_misinfo_risk").size().rename("n")
            st.dataframe(pd.concat([n, rate.round(3)], axis=1).reset_index(), hide_index=True)
            st.caption("Reading: if satellite evidence were redundant with the text-only risk flag, the supported "
                       "rate would not differ across risk groups. Differences show where satellite data adds "
                       "independent information. No truth label is implied by either axis.")

    st.subheader("3. MODIS vs VIIRS")
    sc = load_eval("sensor_comparison.csv")
    if sc is not None:
        s2 = sc[sc["sensor"].isin(["VIIRS", "MODIS"])]
        long = s2.melt(id_vars="sensor", value_vars=["claims_queried_ok", "claims_with_any_detection_in_box",
                                                      "claims_with_matched_detection"], var_name="stage", value_name="claims")
        fig = px.bar(long, x="stage", y="claims", color="sensor", barmode="group", text="claims",
                     color_discrete_map={"VIIRS": "#d50000", "MODIS": "#ff6d00"},
                     title="Detections by sensor on identical claims / windows", template=PLOT_TEMPLATE, height=360)
        fig.update_layout(legend_title="", xaxis_title="", margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(s2, hide_index=True, width="stretch")
        st.caption("VIIRS (375 m) resolves far more small fires than MODIS (1 km); MODIS reports higher FRP per pixel. "
                   "This is consistent with the sensors' known footprints and is a reproducible result on this data.")

    st.subheader("4. Precision / recall / F1 against human labels")
    lm = load_eval("label_metrics.csv")
    if lm is not None:
        st.dataframe(lm, hide_index=True, width="stretch")
        fig = px.bar(lm.melt(id_vars="method", value_vars=["precision", "recall", "f1"]), x="method", y="value",
                     color="variable", barmode="group", template=PLOT_TEMPLATE, height=340,
                     title=f"NLP-only vs satellite-assisted (n={int(lm['n_labelled'].iloc[0])} human-labelled wildfire posts)")
        fig.update_layout(legend_title="", xaxis_title="", margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        cols = st.columns(len(lm))
        for col, (_, r) in zip(cols, lm.iterrows()):
            cm = pd.DataFrame([[r["tp"], r["fn"]], [r["fp"], r["tn"]]], index=["actual true", "actual false"],
                              columns=["pred true", "pred false"])
            f = px.imshow(cm, text_auto=True, color_continuous_scale="Greens", title=r["method"], height=280,
                          template=PLOT_TEMPLATE)
            f.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=0))
            col.plotly_chart(f, width="stretch")
    else:
        st.warning("No human-verified labels yet, so no accuracy figures are reported. To produce them:")
        st.code("python src/research_evaluation.py --make-label-template\n"
                "# label data/processed/manual_labels_template.csv (human_label = true_event / false_or_misleading / unclear)\n"
                "# copy post_id,human_label,notes to data/annotations/manual_labels.csv\n"
                "python src/research_evaluation.py", language="bash")

    st.subheader("Limitations")
    st.markdown(
        "- **Absence of a fire pixel is not evidence of falsehood**: clouds, small/brief fires, overpass timing "
        "(2-4 passes/day) and 375 m / 1 km footprints all cause misses.\n"
        "- **Geocoding is to a place centroid**, so a correct post about a fire 30 km outside a city can fall "
        "outside the 25 km radius; ambiguous names (e.g. 'Palisades', 'Washington') can resolve to the wrong place.\n"
        "- **Satellite support confirms a fire, not the post's other details** (cause, acreage, casualties, blame).\n"
        "- **Retrospective posts** discussing past fires are dated by posting time, not event time.\n"
        "- **LLM annotations are not ground truth**; the misinformation-risk flag is a text-only heuristic.\n"
        "- **Posts are English-biased and query-driven** (official X API search), not a random sample of X.")


# --------------------------------------------------------------------------- 6
def case_studies(d):
    ver, store = d["verification"], d["detections"]
    st.title("Case studies")
    st.caption("Selected automatically from the dataset: strongest supported claims in distinct regions, plus two with no nearby detection.")
    vf = _wildfire(ver)
    if vf is None or vf.empty:
        return _empty("Run the satellite verification first.", "python src/satellite_verification.py --fire-only")

    vf = vf.dropna(subset=["latitude", "longitude"]).copy()
    vf["cell"] = ((vf["latitude"] // 2).astype(int).astype(str) + "," + (vf["longitude"] // 2).astype(int).astype(str))  # 2-degree cells
    sup = vf[(vf["verdict"] == "SUPPORTED") & (vf["ambiguity_status"] == "resolved")].sort_values("evidence_score", ascending=False)
    sup = sup.drop_duplicates(subset=["cell"]).head(4)
    neg = vf[(vf["verdict"] == "NO_NEARBY_DETECTION") & (vf["ambiguity_status"] == "resolved")]
    neg = neg[~neg["cell"].isin(sup["cell"])].drop_duplicates(subset=["cell"]).head(2)
    cases = pd.concat([sup, neg])
    if cases.empty:
        return _empty("No verified wildfire claims with coordinates yet.")

    for i, (_, r) in enumerate(cases.iterrows(), 1):
        color = VERDICT_COLORS.get(r["verdict"], "#607d8b")
        with st.container(border=True):
            st.markdown(f"#### Case {i}: {r['llm_location']} "
                        f"<span style='font-size:0.7em;padding:2px 8px;border-radius:10px;background:{color};color:white'>"
                        f"{VERDICT_LABELS.get(r['verdict'], r['verdict'])}</span>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Original post**")
                st.markdown(f"> {r['text']}")
                st.markdown(f"**Posted:** {r['created_at']:%Y-%m-%d %H:%M} UTC  \n"
                            f"**LLM claim:** {r['llm_claim']}  \n"
                            f"**Geocoded:** {r['display_name']} (resolver confidence {r['location_confidence']}, {r['ambiguity_status']})  \n"
                            f"**Coordinates:** {r['latitude']:.4f}, {r['longitude']:.4f}  \n"
                            f"**FIRMS products:** {r['viirs_source']}, {r['modis_source']}")
                st.markdown("**Satellite observations**")
                for s in ("viirs", "modis"):
                    tot, mt = r[f"{s}_detections_total"], r[f"{s}_detections_matched"]
                    line = f"- **{s.upper()}**: {int(tot) if pd.notna(tot) else 0} pixel(s) in the search box, {int(mt) if pd.notna(mt) else 0} matched"
                    if pd.notna(r[f"{s}_nearest_km"]):
                        line += f"; nearest {r[f'{s}_nearest_km']:.1f} km, {r[f'{s}_nearest_hours']:+.1f} h, max FRP {r[f'{s}_max_frp']} MW"
                    st.markdown(line)
                st.markdown("**Verification explanation**")
                st.info(r["explanation"])
            with c2:
                dets = detections_frame(store, {r["post_id"]})
                m = maps.base_map((r["latitude"], r["longitude"]), zoom=9, relief=False, labels=False,
                                  imagery_date=r["created_at"].strftime("%Y-%m-%d"), imagery_layer="VIIRS SNPP true colour")
                maps.add_radius(m, r["latitude"], r["longitude"])
                maps.add_claims(m, pd.DataFrame([r]), cluster=False)
                if len(dets):
                    maps.add_detections(m, dets)
                maps.finish(m)
                st_folium(m, height=380, use_container_width=True, returned_objects=[], key=f"case_{r['post_id']}")


# --------------------------------------------------------------------------- 7
def live_check():
    st.title("Check a claim")
    st.caption("Geocodes the place and queries NASA FIRMS (VIIRS + MODIS) for that date, using the same matching rule as the batch pipeline.")
    import firms
    import satellite_verification as sv

    with st.form("live"):
        text = st.text_area("Claim text (optional)", placeholder="Brush fire spreading near San Clemente tonight")
        c1, c2 = st.columns(2)
        location = c1.text_input("Location", placeholder="San Clemente, California")
        when = c2.date_input("Event date (UTC)", value=datetime.now(timezone.utc).date())
        go_btn = st.form_submit_button("Query NASA FIRMS", type="primary")
    if not go_btn:
        return
    if not location:
        return st.error("A location is required.")
    if not firms.get_map_key():
        return st.error("FIRMS_MAP_KEY is not set. Add it to .env and restart the app.")

    row = {"post_id": "live", "text": text, "created_at": f"{when.isoformat()}T12:00:00Z",
           "llm_hazard_type": "wildfire", "llm_location": location, "llm_is_hazard_claim": True,
           "llm_claim": text, "llm_stance": "asserts_event", "llm_misinfo_risk": "", "llm_confidence": "",
           "llm_reason": ""}
    with st.spinner("Geocoding and querying NASA FIRMS (VIIRS + MODIS)..."):
        try:
            avail = firms.data_availability()
            rec, dets = sv.verify_row(row, availability=avail)
        except Exception as exc:
            return st.error(f"Verification failed: {exc}")

    color = VERDICT_COLORS.get(rec["verdict"], "#607d8b")
    st.markdown(f"<div style='padding:10px 14px;border-left:6px solid {color};background:#161b26;font-size:1.1em'>"
                f"<b>{VERDICT_LABELS.get(rec['verdict'], rec['verdict'])}</b></div>", unsafe_allow_html=True)
    st.write(rec["explanation"])
    if rec["verdict"] in ("API_ERROR",):
        st.warning(f"VIIRS status: {rec['viirs_status']}; MODIS status: {rec['modis_status']}")
    if rec["latitude"] != "":
        st.caption(f"Geocoded: {rec['display_name']} ({rec['latitude']:.4f}, {rec['longitude']:.4f}); "
                   f"products: {rec['viirs_source']}, {rec['modis_source']}")
        df = detections_frame({"live": dets})
        m = maps.base_map((rec["latitude"], rec["longitude"]), zoom=9, relief=False, imagery_date=when.isoformat(),
                          imagery_layer="VIIRS SNPP true colour")
        maps.add_radius(m, rec["latitude"], rec["longitude"])
        rec_df = pd.DataFrame([{**rec, "created_at": pd.Timestamp(rec["claim_date_utc"])}])
        maps.add_claims(m, rec_df, cluster=False)
        if len(df):
            maps.add_detections(m, df)
        maps.add_legend(m, verdicts=False)
        maps.finish(m)
        st_folium(m, height=450, use_container_width=True, returned_objects=[])
        if len(df):
            st.dataframe(df.sort_values("distance_km")[["sensor", "satellite", "acq_date", "acq_time", "distance_km",
                                                        "hours_diff", "frp", "confidence", "matched"]],
                         hide_index=True, width="stretch", height=240)
