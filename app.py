"""NASA Geospatial Misinformation Monitor -- Streamlit application.

Run from the repository root:

    streamlit run app.py
"""
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Geospatial Misinformation Monitor", page_icon=":material/public:",
                   layout="wide", initial_sidebar_state="collapsed")

import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dashboard.data import load_summary  # noqa: E402

st.markdown("""
<style>
  #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
  .block-container {padding-top: 0.6rem; padding-bottom: 2rem; max-width: 1400px;}
  [data-testid="stSidebarCollapsedControl"] {display: none;}
  h1 {font-size: 1.55rem; font-weight: 600; margin-bottom: 0.2rem;}
  h2 {font-size: 1.2rem; font-weight: 600; margin-top: 1.4rem;}
  h3, h4 {font-weight: 600;}
  div[data-testid="stMetric"] {padding: 4px 0;}
  div[data-testid="stMetricLabel"] p {font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em; color: #8b95a5;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  blockquote {border-left: 3px solid #3d4756; padding: 4px 12px; margin: 6px 0; color: #d6dae2;}
  .site-header {display:flex; align-items:baseline; justify-content:space-between; border-bottom:1px solid #2a3140;
                padding: 4px 0 10px; margin-bottom: 4px;}
  .site-header .title {font-size: 1.05rem; font-weight: 600; letter-spacing: .01em;}
  .site-header .sub {font-size: 0.8rem; color: #8b95a5;}
  div[data-testid="stCaptionContainer"] {color: #8b95a5;}
</style>
""", unsafe_allow_html=True)

summary = load_summary()
stats = (f"{summary.get('annotated_posts', 0):,} posts · {summary.get('wildfire_posts_verified', 0):,} wildfire claims "
         f"checked · {summary.get('supported', 0):,} with satellite support") if summary else ""
st.markdown(f"<div class='site-header'><span class='title'>NASA Geospatial Misinformation Monitor "
            f"<span class='sub'>&nbsp; undergraduate research project</span></span>"
            f"<span class='sub'>{stats}</span></div>", unsafe_allow_html=True)


NAV = {
    "Project": [
        st.Page("views/overview.py", title="Overview", url_path="overview", default=True),
        st.Page("views/data.py", title="Collected data", url_path="data"),
    ],
    "Satellite evidence": [
        st.Page("views/live.py", title="Live fire map", url_path="live"),
        st.Page("views/map.py", title="Claims map", url_path="map"),
        st.Page("views/dataset.py", title="Annotated dataset", url_path="dataset"),
        st.Page("views/verification.py", title="Verification", url_path="verification"),
        st.Page("views/check.py", title="Check a claim", url_path="check"),
    ],
    "Analysis": [
        st.Page("views/claims.py", title="Claim explorer", url_path="claims"),
        st.Page("views/evaluation.py", title="Evaluation", url_path="evaluation"),
        st.Page("views/cases.py", title="Case studies", url_path="cases"),
    ],
    "Setup": [st.Page("views/sources.py", title="Data sources & keys", url_path="sources")],
}

st.navigation(NAV, position="top").run()
