"""Build a double-click offline package: claims map (VIIRS + MODIS), annotated dataset page, Excel file, zip.

    python src/build_offline_package.py                 # private: includes post text (do not publish)
    python src/build_offline_package.py --public        # public-safe: post IDs/links + annotations, no post text
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import folium  # noqa: E402
from dashboard import maps  # noqa: E402
from dashboard.data import (annotated_dataset, detections_frame, load_annotated,  # noqa: E402
                            load_detections, load_summary, load_verification, to_excel_bytes)

DATASET_COLUMNS = ["post_id", "post_url", "created_at", "text", "llm_hazard_type", "llm_is_hazard_claim",
                   "llm_location", "llm_claim", "llm_stance", "llm_misinfo_risk", "llm_confidence", "llm_reason",
                   "verdict", "matched_sensors", "nearest_detection_km", "nearest_detection_hours",
                   "evidence_score", "display_name", "ambiguity_status", "explanation"]

DATASET_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Annotated dataset</title><style>
body{margin:0;font:13px -apple-system,Segoe UI,sans-serif;background:#0f131a;color:#d6dae2}
header{padding:12px 16px;border-bottom:1px solid #2a3140;position:sticky;top:0;background:#0f131a;z-index:2}
h1{font-size:17px;margin:0 0 8px}input,select{background:#161b26;color:#d6dae2;border:1px solid #3d4756;border-radius:4px;padding:5px;margin-right:6px}
a{color:#4fc3f7}#wrap{overflow:auto;height:calc(100vh - 110px)}table{border-collapse:collapse;font-size:12px}
th{position:sticky;top:0;background:#1c2230;cursor:pointer;text-align:left;padding:6px;border:1px solid #2a3140;white-space:nowrap}
td{padding:5px 6px;border:1px solid #232a38;vertical-align:top;max-width:380px}tr:nth-child(even) td{background:#121722}
.SUPPORTED{color:#66bb6a;font-weight:600}.NO_NEARBY_DETECTION{color:#f9a825}.LOCATION_UNRESOLVED{color:#9e9e9e}
</style></head><body><header><h1>Annotated dataset &mdash; LLM annotations + NASA satellite verdicts</h1>
<input id="q" size="34" placeholder="Search __SEARCH__"> <select id="hz"></select> <select id="vd"></select> <select id="rk"></select>
<span id="n"></span> &nbsp; <a href="annotated_posts.xlsx">Download Excel (.xlsx)</a> &nbsp; <a href="index.html">Back to map</a></header>
<div id="wrap"><table><thead><tr id="h"></tr></thead><tbody id="b"></tbody></table></div><script>
const COLS=__COLS__, ROWS=__ROWS__;const ix=n=>COLS.indexOf(n);let sortC=ix("created_at"),asc=false;
const esc=s=>s==null?"":String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const cell=(v,c)=>c==="post_url"&&v?`<a href="${esc(v)}" target="_blank">open</a>`:esc(v);
function opts(id,col,label){const v=[...new Set(ROWS.map(r=>r[ix(col)]).filter(x=>x!=null&&x!==""))].sort();
document.getElementById(id).innerHTML=`<option value="">${label}: all</option>`+v.map(x=>`<option>${esc(x)}</option>`).join("");}
opts("hz","llm_hazard_type","Hazard");opts("vd","verdict","Verdict");opts("rk","llm_misinfo_risk","Risk");
document.getElementById("h").innerHTML=COLS.map((c,i)=>`<th onclick="srt(${i})">${c}</th>`).join("");
function srt(i){asc=sortC===i?!asc:true;sortC=i;draw();}
function draw(){const q=document.getElementById("q").value.toLowerCase(),f={hz:"llm_hazard_type",vd:"verdict",rk:"llm_misinfo_risk"};
const s=["text","llm_location","llm_claim"].map(ix).filter(i=>i>=0);
let r=ROWS.filter(x=>s.some(i=>String(x[i]??"").toLowerCase().includes(q)));
for(const id in f){const v=document.getElementById(id).value;if(v)r=r.filter(x=>String(x[ix(f[id])])===v);}
r.sort((a,b)=>{const x=a[sortC],y=b[sortC];if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*(asc?1:-1);});
document.getElementById("n").textContent=r.length.toLocaleString()+" of "+ROWS.length.toLocaleString()+" posts";
const vi=ix("verdict");document.getElementById("b").innerHTML=r.map(x=>"<tr>"+x.map((v,i)=>`<td${i===vi?` class="${v}"`:""}>${cell(v,COLS[i])}</td>`).join("")+"</tr>").join("");}
["q","hz","vd","rk"].forEach(id=>document.getElementById(id).addEventListener("input",draw));draw();
</script></body></html>"""


def quotes_post(text, value, n=60):
    """True if `value` contains any run of `n` consecutive characters copied from the post text."""
    norm = lambda s: " ".join(str(s).lower().split())  # noqa: E731
    t, v = norm(text), norm(value)
    return len(v) >= n and any(t[i:i + n] in v for i in range(len(t) - n + 1))


def scrub_quotes(df, cols=("llm_claim", "llm_reason", "explanation")):
    """Blank annotation fields that reproduce long stretches of the original post text."""
    df = df.copy()
    for c in cols:
        if c in df:
            hit = [isinstance(v, str) and quotes_post(t, v) for t, v in zip(df["text"], df[c])]
            df.loc[hit, c] = "[omitted: quotes post text]"
    return df


def build_map(ver, store, summary, out):
    vf = ver[ver["llm_hazard_type"] == "wildfire"].dropna(subset=["latitude", "longitude"]).copy()
    dets = detections_frame(store, set(vf["post_id"]))
    dets = dets[dets["matched"] == True]  # noqa: E712
    m = maps.base_map((vf["latitude"].median(), vf["longitude"].median()), zoom=4)
    for sensor, label in (("VIIRS", "VIIRS fire pixels (375 m)"), ("MODIS", "MODIS fire pixels (1 km)")):
        g = folium.FeatureGroup(name=f"{label} - matched to a claim")
        for _, d in dets[dets["sensor"] == sensor].iterrows():
            try:
                r = min(9, 2.5 + (float(d["frp"]) ** 0.5) / 3)
            except (TypeError, ValueError):
                r = 3
            folium.CircleMarker((d["latitude"], d["longitude"]), radius=r, color=maps.SENSOR_COLORS[sensor],
                                fill=True, fill_opacity=0.8, weight=0.6,
                                tooltip=f"{sensor} {d['satellite']} {d['acq_date']} - FRP {d['frp']} MW - "
                                        f"{d['distance_km']:.1f} km / {d['hours_diff']:+.0f} h from claim").add_to(g)
        g.add_to(m)
    maps.add_claims(m, vf)
    maps.add_legend(m)
    maps.finish(m)
    stats = (f"{summary.get('annotated_posts', 0):,} posts annotated &middot; "
             f"{summary.get('wildfire_posts_verified', 0):,} wildfire claims checked &middot; "
             f"{summary.get('supported', 0):,} supported by NASA satellites")
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;top:10px;left:50px;z-index:9999;background:rgba(20,20,24,.92);color:#eee;"
        "padding:8px 14px;border:1px solid #444;border-radius:6px;font:13px -apple-system,Segoe UI,sans-serif'>"
        "<b>NASA Geospatial Misinformation Monitor</b> &mdash; claims map (VIIRS + MODIS combined)<br>"
        f"<span style='color:#aaa'>{stats}</span><br>"
        "<a href='dataset.html' target='_blank' style='color:#4fc3f7'>"
        "Open annotated dataset (spreadsheet) in a new tab &#8599;</a>"
        " &nbsp;|&nbsp; <a href='annotated_posts.xlsx' style='color:#4fc3f7'>Excel file</a></div>"))
    m.save(out / "index.html")


def build_dataset_page(data, out, public):
    table = data[[c for c in DATASET_COLUMNS if c in data.columns]].copy()
    for c in ("nearest_detection_km", "nearest_detection_hours", "evidence_score"):
        if c in table:
            table[c] = table[c].round(2)
    rows = json.loads(table.to_json(orient="values", force_ascii=False))
    page = (DATASET_PAGE
            .replace("__SEARCH__", "location / claim" if public else "text / location / claim")
            .replace("__COLS__", json.dumps(list(table.columns)))
            .replace("__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")))
    (out / "dataset.html").write_text(page, encoding="utf-8")


def readme(public):
    note = ("Post text is NOT included (X terms allow sharing post IDs, not full text). Each row has a post_id and\n"
            "post_url; llm_claim is the local LLM's short summary of the claim.\n" if public else
            "Contains real social-media post text: for research use only, please do not share publicly.\n")
    return ("NASA Geospatial Misinformation Monitor - offline package\n\n"
            "1. Unzip this folder.\n"
            "2. Double-click index.html (opens in your browser): the claims map with VIIRS + MODIS fire pixels.\n"
            "3. Click 'Open annotated dataset' on the map (or double-click dataset.html) for the full table.\n"
            "4. annotated_posts.xlsx is the same dataset (all columns) for Excel.\n\n"
            "An internet connection is needed for the map background tiles; the data itself is inside the files.\n"
            + note)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public", action="store_true", help="omit post text (safe to publish)")
    ap.add_argument("--out", type=Path, help="output folder (a .zip is written next to it)")
    a = ap.parse_args()
    out = a.out or (REPO / "docs" / "mentor_package" if a.public
                    else Path.home() / "Downloads" / "NASA_Monitor_Mentor_Package")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    ann, ver, store, summary = load_annotated(0), load_verification(0), load_detections(0), load_summary()
    if a.public:
        ann, ver = scrub_quotes(ann), scrub_quotes(ver)
        ann = ann.drop(columns=["text"])
        ver["text"] = "LLM claim summary: " + ver["llm_claim"].fillna("")
    data = annotated_dataset(ann, ver.drop(columns=["text"]) if a.public else ver)
    if a.public:
        data.insert(1, "post_url", "https://x.com/i/status/" + data["post_id"].astype(str))

    build_map(ver, store, summary, out)
    build_dataset_page(data, out, a.public)
    (out / "annotated_posts.xlsx").write_bytes(to_excel_bytes(data))
    (out / "README.txt").write_text(readme(a.public))
    zip_path = shutil.make_archive(str(out), "zip", out.parent, out.name)
    for p in sorted(out.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")
    print(f"{zip_path}: {Path(zip_path).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
