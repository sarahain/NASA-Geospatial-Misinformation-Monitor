"""Folium map helpers: claim markers, real NASA FIRMS detections, and NASA GIBS
daily satellite imagery for a chosen date."""
import folium
from folium.plugins import MarkerCluster

from dashboard.data import VERDICT_COLORS, VERDICT_LABELS, short

# NASA GIBS (Global Imagery Browse Services) daily true-colour imagery, no key needed.
GIBS_LAYERS = {
    "VIIRS SNPP true colour": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    "MODIS Terra true colour": "MODIS_Terra_CorrectedReflectance_TrueColor",
    "MODIS Aqua true colour": "MODIS_Aqua_CorrectedReflectance_TrueColor",
}
GIBS_URL = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/"
            "GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg")

SENSOR_COLORS = {"VIIRS": "#ff1744", "MODIS": "#ff9100"}
PRODUCT_COLORS = {
    "VIIRS_SNPP_NRT": "#ff1744", "VIIRS_NOAA20_NRT": "#f50057", "VIIRS_NOAA21_NRT": "#d500f9",
    "MODIS_NRT": "#ff9100", "LANDSAT_NRT": "#00e5ff", "GOES_NRT": "#ffea00",
    "VIIRS_SNPP_SP": "#ff1744", "VIIRS_NOAA20_SP": "#f50057", "VIIRS_NOAA21_SP": "#d500f9", "MODIS_SP": "#ff9100",
}

# Free, key-less basemaps. Esri services are open for non-commercial research use.
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{svc}/MapServer/tile/{{z}}/{{y}}/{{x}}"
BASEMAPS = {
    "Dark (Esri)": (ESRI.format(svc="Canvas/World_Dark_Gray_Base"), "Esri, HERE, Garmin, OpenStreetMap contributors"),
    "Satellite imagery (Esri)": (ESRI.format(svc="World_Imagery"), "Esri, Maxar, Earthstar Geographics"),
    "Terrain / topographic (OpenTopoMap)": ("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", "OpenTopoMap (CC-BY-SA), SRTM"),
    "Light (OpenStreetMap)": ("OpenStreetMap", None),
}
HILLSHADE = (ESRI.format(svc="Elevation/World_Hillshade"), "Esri, USGS, NASA SRTM")
LABELS_DARK = (ESRI.format(svc="Canvas/World_Dark_Gray_Reference"), "Esri")
LABELS = (ESRI.format(svc="Reference/World_Boundaries_and_Places"), "Esri")


def base_map(center=(34.0, -118.0), zoom=5, imagery_date=None, imagery_layer=None,
             basemap="Dark (Esri)", relief=True, labels=True):
    """Dark research basemap with optional terrain hillshade (mountains) and
    place / landmark labels, plus optional NASA GIBS daily imagery."""
    tiles, attr = BASEMAPS.get(basemap, BASEMAPS["Dark (Esri)"])
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer(tiles=tiles, attr=attr or "OpenStreetMap", name=basemap, control=False).add_to(m)
    if relief and basemap in ("Dark (Esri)", "Light (OpenStreetMap)"):
        folium.TileLayer(tiles=HILLSHADE[0], attr=HILLSHADE[1], name="Terrain relief (hillshade)", overlay=True,
                         control=True, opacity=0.45 if basemap.startswith("Dark") else 0.3).add_to(m)
    if imagery_date and imagery_layer:
        folium.TileLayer(
            tiles=GIBS_URL.format(layer=GIBS_LAYERS[imagery_layer], date=imagery_date),
            attr="NASA GIBS / EOSDIS", name=f"NASA {imagery_layer} ({imagery_date})",
            overlay=True, control=True, opacity=0.85, max_native_zoom=9, max_zoom=18,
        ).add_to(m)
    if labels:
        lab = LABELS_DARK if basemap.startswith("Dark") else LABELS
        folium.TileLayer(tiles=lab[0], attr=lab[1], name="Places, borders and landmarks", overlay=True,
                         control=True, opacity=0.9).add_to(m)
    return m


def add_fire_pixels(m, det_df, name="Live NASA FIRMS active fires", by_product=True):
    """Live fire pixels coloured by product and sized by fire radiative power."""
    groups = {}
    for _, d in det_df.dropna(subset=["latitude", "longitude"]).iterrows():
        prod = d.get("product", d.get("sensor", "FIRMS"))
        g = groups.setdefault(prod, folium.FeatureGroup(name=f"{prod}" if by_product else name))
        frp = d.get("frp")
        try:
            frp_v = float(frp)
        except (TypeError, ValueError):
            frp_v = None
        radius = 3 if frp_v is None else min(14, 3 + (frp_v ** 0.5) / 2)
        html = (f"<b>{prod}</b> - {d.get('satellite', '')}<br>{d.get('acq_date', '')} {str(d.get('acq_time', '')).zfill(4)} UTC<br>"
                f"FRP: {frp if frp not in (None, '') else 'n/a'} MW - confidence: {d.get('confidence', '')}<br>"
                f"{d.get('daynight', '')}")
        folium.CircleMarker(location=(d["latitude"], d["longitude"]), radius=radius,
                            color=PRODUCT_COLORS.get(prod, "#ff5252"), fill=True, fill_opacity=0.75, weight=0.8,
                            popup=folium.Popup(html, max_width=260)).add_to(g)
    for g in groups.values():
        g.add_to(m)
    return m


def add_product_legend(m, products):
    rows = "<div style='font-weight:600;margin-bottom:2px'>NASA FIRMS product</div>"
    for p in products:
        rows += (f"<div><span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                 f"background:{PRODUCT_COLORS.get(p, '#ff5252')};margin-right:6px'></span>{p}</div>")
    rows += "<div style='color:#aaa'>marker size ~ fire radiative power</div>"
    html = (f"<div style='position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(20,20,24,.9);color:#eee;"
            f"padding:10px 12px;border:1px solid #444;border-radius:6px;font-size:12px;line-height:1.5'>{rows}</div>")
    m.get_root().html.add_child(folium.Element(html))
    return m


def add_claims(m, ver_df, cluster=True):
    layer = MarkerCluster(name="Claim locations", disableClusteringAtZoom=8) if cluster \
        else folium.FeatureGroup(name="Claim locations")
    for _, r in ver_df.dropna(subset=["latitude", "longitude"]).iterrows():
        color = VERDICT_COLORS.get(r["verdict"], "#607d8b")
        html = (f"<b>{VERDICT_LABELS.get(r['verdict'], r['verdict'])}</b><br>"
                f"<i>{short(r['text'], 220)}</i><br>"
                f"Location: {r['llm_location']}<br>Date: {str(r['created_at'])[:16]} UTC<br>"
                + (f"Nearest detection: {r['nearest_detection_km']:.1f} km, {r['nearest_detection_hours']:+.1f} h"
                   if r["verdict"] == "SUPPORTED" else ""))
        folium.CircleMarker(
            location=(r["latitude"], r["longitude"]), radius=7, color=color, fill=True,
            fill_color=color, fill_opacity=0.85, weight=1.5,
            popup=folium.Popup(html, max_width=320), tooltip=short(r["llm_location"], 40),
        ).add_to(layer)
    layer.add_to(m)
    return m


def add_detections(m, det_df, show_unmatched=True):
    matched = folium.FeatureGroup(name="FIRMS detections: matched (<=25 km, <=48 h)")
    other = folium.FeatureGroup(name="FIRMS detections: in search box, not matched", show=show_unmatched)
    for _, d in det_df.dropna(subset=["latitude", "longitude"]).iterrows():
        color = SENSOR_COLORS.get(d["sensor"], "#000")
        html = (f"<b>{d['instrument']} / {d['satellite']}</b><br>Acquired {d['acq_date']} "
                f"{str(d['acq_time']).zfill(4)} UTC<br>FRP: {d['frp']} MW, confidence: {d['confidence']}<br>"
                f"{d['distance_km']:.1f} km from claim, {d['hours_diff']:+.1f} h")
        folium.CircleMarker(
            location=(d["latitude"], d["longitude"]), radius=4 if d["matched"] else 3,
            color=color, fill=True, fill_opacity=0.9 if d["matched"] else 0.35, weight=1,
            popup=folium.Popup(html, max_width=260),
        ).add_to(matched if d["matched"] else other)
    matched.add_to(m)
    if show_unmatched:
        other.add_to(m)
    return m


def add_radius(m, lat, lon, km=25):
    folium.Circle(location=(lat, lon), radius=km * 1000, color="#1565c0", weight=1.5,
                  fill=False, dash_array="6", tooltip=f"{km} km matching radius").add_to(m)
    return m


def add_legend(m, verdicts=True, sensors=True):
    rows = ""
    if verdicts:
        rows += "<div style='font-weight:600;margin-bottom:2px'>Claim verdict</div>"
        for k, c in VERDICT_COLORS.items():
            if k == "NOT_APPLICABLE_NON_FIRE":
                continue
            rows += (f"<div><span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
                     f"background:{c};margin-right:6px'></span>{VERDICT_LABELS[k]}</div>")
    if sensors:
        rows += "<div style='font-weight:600;margin:6px 0 2px'>NASA FIRMS detection</div>"
        for k, c in SENSOR_COLORS.items():
            rows += (f"<div><span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
                     f"background:{c};margin-right:6px'></span>{k} active fire pixel</div>")
        rows += "<div style='color:#aaa'>solid = matched, faded = outside window</div>"
    html = (f"<div style='position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(20,20,24,.9);color:#eee;"
            f"padding:10px 12px;border:1px solid #444;border-radius:6px;font-size:12px;"
            f"line-height:1.5;box-shadow:0 1px 4px rgba(0,0,0,.4)'>{rows}</div>")
    m.get_root().html.add_child(folium.Element(html))
    return m


def finish(m):
    folium.LayerControl(collapsed=False).add_to(m)
    return m
