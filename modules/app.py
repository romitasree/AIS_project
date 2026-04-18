# app.py
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date
from html import escape
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import ee
import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from gemini_integration import get_real_romita_diffs, call_gemini_for_impacts, validate_date_range

# ── GEE init ────────────────────────────────────────────────
try:
    ee.Initialize(project="terratrace-488618")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="terratrace-488618")

# ── Constants ────────────────────────────────────────────────
SUMMARY_REGIONS = ["Celina, TX", "Falcon Lake, TX"]

REGION_COORDS = {
    "Celina":      {"lat": 33.314479, "lon": -96.776550, "zoom": 13},
    "Falcon Lake": {"lat": 26.667417, "lon": -99.159667, "zoom": 12},
    "Dallas":      {"lat": 32.7767,   "lon": -96.7970,   "zoom": 12},
    "Austin":      {"lat": 30.2672,   "lon": -97.7431,   "zoom": 12},
    "Houston":     {"lat": 29.7604,   "lon": -95.3698,   "zoom": 12},
    "San Antonio": {"lat": 29.4241,   "lon": -98.4936,   "zoom": 12},
    "Fort Worth":  {"lat": 32.7555,   "lon": -97.3308,   "zoom": 12},
}

REGION_CSV = {
    "Celina":      "lat33_314479_lonn96_77655",
    "Falcon Lake": "lat26_667417_lonn99_159667",
    "Dallas":      "lat32_7767_lonn96_797",
    "Austin":      "lat30_2672_lonn97_7431",
    "Houston":     "lat29_7604_lonn95_3698",
    "San Antonio": "lat29_4241_lonn98_4936",
    "Fort Worth":  "lat32_7555_lonn97_3308",
}

LAYER_DEFS = {
    "water":              (0, "💧 Water",              "#419bdf"),
    "trees":              (1, "🌲 Trees",              "#397d49"),
    "grass":              (2, "🌿 Grass",              "#88b053"),
    "flooded_vegetation": (3, "🌾 Flooded Vegetation", "#7a87c6"),
    "crops":              (4, "🌽 Crops",              "#e49635"),
    "shrub_and_scrub":    (5, "🌵 Shrubs & Scrubs",    "#dfc35a"),
    "built":              (6, "🏙️ Built",               "#c4281b"),
    "bare":               (7, "🏜️ Bare",               "#a59b8f"),
    "snow_and_ice":       (8, "❄️ Snow & Ice",          "#b39fe1"),
}

DW_CLASSES = ["water","trees","grass","flooded_vegetation","crops",
               "shrub_and_scrub","built","bare","snow_and_ice"]

CITIES = {
    "Dallas":      [32.7767, -96.7970],
    "Austin":      [30.2672, -97.7431],
    "Houston":     [29.7604, -95.3698],
    "San Antonio": [29.4241, -98.4936],
    "Fort Worth":  [32.7555, -97.3308],
    "Frisco":      [33.1507, -96.8236],
    "McKinney":    [33.1976, -96.6153],
    "Celina":      [33.3246, -96.7869],
}

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ── Data helpers ─────────────────────────────────────────────
def load_region_csv(region: str) -> pd.DataFrame | None:
    label = REGION_CSV.get(region)
    if not label:
        return None
    path = OUTPUTS_DIR / f"{label}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def compute_terrascore(df: pd.DataFrame) -> int:
    if df is None or len(df) < 2:
        return 0
    first = df.iloc[0]
    last  = df.iloc[-1]
    built_delta = last["lc_built_pct"] - first["lc_built_pct"]
    tree_loss   = max(0, first["lc_trees_pct"] - last["lc_trees_pct"])
    grass_loss  = max(0, first["lc_grass_pct"] - last["lc_grass_pct"])
    score = min(100, int(built_delta * 2 + tree_loss * 1.5 + grass_loss * 0.5))
    return max(0, score)


# ── GEE tile helper ──────────────────────────────────────────
def _dw_tile_url(start: str, end: str, class_id: int, color: str) -> str | None:
    key = f"tile_{start}_{end}_{class_id}"
    if key in st.session_state:
        return st.session_state[key]
    try:
        tx_rect = ee.Geometry.Rectangle([-106.65, 25.84, -93.51, 36.50])
        img = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterDate(start, end)
            .filterBounds(tx_rect)
            .mode().select("label").eq(class_id).selfMask()
        )
        mid = img.getMapId({"min": 0, "max": 1, "palette": [color.lstrip("#")]})
        url = mid["tile_fetcher"].url_format
        st.session_state[key] = url
        return url
    except Exception:
        return None


def build_map(center_lat, center_lon, zoom, start, end, active_layers):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite", name="Satellite", overlay=False,
    ).add_to(m)
    for key, (class_id, lbl, color) in LAYER_DEFS.items():
        if not active_layers.get(key):
            continue
        url = _dw_tile_url(start, end, class_id, color)
        if url:
            folium.TileLayer(tiles=url, attr="GDW", name=lbl,
                             overlay=True, opacity=0.85).add_to(m)
    for city, coords in CITIES.items():
        folium.Marker(location=coords, popup=city,
                      icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
    return m


# ── Plotly charts ────────────────────────────────────────────
def make_bar_chart(df: pd.DataFrame) -> go.Figure:
    last   = df.iloc[-1]
    values = [last.get(f"lc_{c}_pct", 0) for c in DW_CLASSES]
    colors = [LAYER_DEFS[c][2] for c in DW_CLASSES]
    labels = [LAYER_DEFS[c][1] for c in DW_CLASSES]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#0a1a0f", plot_bgcolor="#0d1f12",
        font=dict(color="#b8d4a0", family="Roboto Mono"),
        margin=dict(t=20, b=40, l=10, r=10),
        yaxis=dict(title="% Coverage", gridcolor="#1a3020", zeroline=False),
        xaxis=dict(tickangle=-30),
        showlegend=False, height=320,
    )
    return fig


def make_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c in DW_CLASSES:
        col = f"lc_{c}_pct"
        if col not in df.columns:
            continue
        lbl = LAYER_DEFS[c][1]
        clr = LAYER_DEFS[c][2]
        fig.add_trace(go.Scatter(
            x=df["year"], y=df[col], mode="lines+markers",
            name=lbl, line=dict(color=clr, width=2), marker=dict(size=5),
        ))
    fig.update_layout(
        paper_bgcolor="#0a1a0f", plot_bgcolor="#0d1f12",
        font=dict(color="#b8d4a0", family="Roboto Mono"),
        margin=dict(t=20, b=40, l=10, r=10),
        yaxis=dict(title="% Coverage", gridcolor="#1a3020", zeroline=False),
        xaxis=dict(title="Year", gridcolor="#1a3020"),
        legend=dict(bgcolor="#0d1f12", bordercolor="#1a3020", borderwidth=1,
                    font=dict(size=10)),
        height=360,
    )
    return fig


def make_terrascore_gauge(score: int) -> go.Figure:
    color = "#5ecf52" if score < 30 else "#e8a020" if score < 60 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#3a6040",
                      tickfont=dict(color="#5a8060")),
            bar=dict(color=color),
            bgcolor="#0d1f12", bordercolor="#1a3020",
            steps=[
                dict(range=[0,  30], color="#0d2a10"),
                dict(range=[30, 60], color="#1a2a0a"),
                dict(range=[60,100], color="#2a1010"),
            ],
            threshold=dict(line=dict(color=color, width=3), value=score),
        ),
        number=dict(font=dict(color=color, size=36, family="Roboto Mono")),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor="#0a1a0f", font=dict(color="#b8d4a0"),
        margin=dict(t=20, b=10, l=20, r=20), height=220,
    )
    return fig


# ════════════════════════════════════════════════════════════
# Page config & CSS
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="TerraTrace", layout="wide", page_icon="🌍")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500;700&display=swap');
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],[data-testid="stMainBlockContainer"],
section.main,section.main>div,.main .block-container{background-color:#0a1a0f !important;}
[data-testid="stHeader"],[data-testid="stDecoration"],
[data-testid="stToolbar"],footer,#MainMenu{display:none !important;}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div,
[data-testid="stSidebarContent"],[data-testid="stSidebarUserContent"]{
  background:#07120a !important;border-right:1px solid #1a3020 !important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,[data-testid="stSidebar"] p{color:#90b880 !important;font-size:13px;}
[data-testid="stSidebar"] [data-baseweb="select"]>div{
  background:#0d1f12 !important;border:1px solid #1e3824 !important;
  border-radius:8px !important;color:#b8d4a0 !important;}
[data-testid="stSidebar"] input{
  background:#0d1f12 !important;color:#b8d4a0 !important;
  border:1px solid #1e3824 !important;border-radius:8px !important;}
.stButton>button{width:100%;background:#0d1f12 !important;color:#5ecf52 !important;
  border:1px solid #2a6030 !important;border-radius:8px !important;
  font-family:'Roboto Mono',monospace !important;font-size:12px !important;padding:8px 0 !important;}
.stButton>button:hover{background:#163a20 !important;border-color:#5ecf52 !important;}
p,li{color:#b8d4a0;}
hr{border-color:#1a3020 !important;margin:8px 0 !important;}
.tt-navbar{display:flex;align-items:center;justify-content:space-between;
  padding:14px 28px 12px;border-bottom:1px solid #1a3020;background:#07120a;margin-bottom:16px;}
.tt-nav-title{font-family:'Roboto Mono',monospace;font-size:20px;font-weight:700;color:#5ecf52;}
.tt-nav-sub{font-size:11px;color:#3a6040;font-style:italic;}
.tt-pill{background:#0d1f12;border:1px solid #1e3824;border-radius:20px;
  padding:4px 14px;font-size:11px;color:#5a8060;font-family:'Roboto Mono',monospace;}
.tt-pill.active{background:#163a20;border-color:#4aaa40;color:#8ddf7e;}
.tt-section{font-family:'Roboto Mono',monospace;font-size:9px;text-transform:uppercase;
  letter-spacing:2.5px;color:#2e5038;padding:14px 0 6px;}
.tt-map-header{font-family:'Roboto Mono',monospace;font-size:11px;font-weight:700;
  color:#5ecf52;text-align:center;padding:6px;background:#0d1f12;
  border:1px solid #1a3020;border-radius:8px;margin-bottom:6px;}
.tt-card{background:#0d1f12;border:1px solid #1a3020;border-radius:10px;
  padding:13px 15px;margin-bottom:9px;}
.tt-label{font-family:'Roboto Mono',monospace;font-size:9px;text-transform:uppercase;
  letter-spacing:2.5px;color:#2e6040;margin-bottom:5px;}
.tt-score-label{font-family:'Roboto Mono',monospace;font-size:10px;text-align:center;
  color:#4a7050;margin-top:4px;}
.impact-card{background:linear-gradient(180deg,rgba(15,31,47,.96),rgba(9,19,31,.96));
  border:1px solid rgba(121,201,255,.55);border-radius:16px;padding:1.1rem;
  box-shadow:0 0 20px rgba(121,201,255,.2);}
.impact-card h4{margin:0 0 .7rem;color:#eaf7ff;font-size:1rem;}
.impact-card p{margin:0 0 .7rem;color:#f4fbff;line-height:1.5;font-size:.85rem;}
.impact-card code{background:rgba(121,201,255,.16);color:#bfe9ff;
  border-radius:999px;padding:.1rem .4rem;}
.impact-empty{color:#b9d9eb;opacity:.9;}
.tt-legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;}
.tt-legend-item{display:flex;align-items:center;gap:5px;background:#0d1f12;
  border:1px solid #1a3020;border-radius:20px;padding:3px 10px;font-size:11px;color:#5a8060;}
.tt-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:Roboto Mono,monospace;font-size:18px;font-weight:700;color:#5ecf52;padding:16px 0 4px">🌍 TerraTrace</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;color:#3a6040;letter-spacing:.5px;margin-bottom:12px">LAND USE MONITOR</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="tt-section">📍 Region</div>', unsafe_allow_html=True)
    region = st.selectbox("Region", options=list(REGION_COORDS.keys()), label_visibility="collapsed")
    sel = REGION_COORDS[region]

    st.markdown('<div class="tt-section">📅 Historical Period</div>', unsafe_allow_html=True)
    hist_start = st.date_input("Hist Start", value=date(2016, 5, 1), key="hist_start", label_visibility="collapsed")
    hist_end   = st.date_input("Hist End",   value=date(2016, 8, 1), key="hist_end",   label_visibility="collapsed")

    st.markdown('<div class="tt-section">📅 Recent Period</div>', unsafe_allow_html=True)
    recent_start = st.date_input("Rec Start", value=date(2024, 5, 1), key="recent_start", label_visibility="collapsed")
    recent_end   = st.date_input("Rec End",   value=date(2024, 8, 1), key="recent_end",   label_visibility="collapsed")

    date_error = None
    valid1, err1 = validate_date_range(hist_start, hist_end)
    valid2, err2 = validate_date_range(recent_start, recent_end)
    if not valid1:
        date_error = f"⚠️ Historical: {err1}"
    elif not valid2:
        date_error = f"⚠️ Recent: {err2}"
    elif recent_start <= hist_end:
        date_error = "⚠️ Recent period must start after the historical period ends."

    if date_error:
        st.error(date_error)
    else:
        st.success(f"✅ {hist_start} → {hist_end}  ↔  {recent_start} → {recent_end}")

    st.markdown('<div class="tt-section">🗺️ Map Layers</div>', unsafe_allow_html=True)
    layer_on = {
        k: st.toggle(lbl, value=(k in ("built", "trees")), key=f"t_{k}")
        for k, (_, lbl, _c) in LAYER_DEFS.items()
    }

    st.divider()
    run_gemini = st.button("🤖 Generate Gemini Overview")

# ════════════════════════════════════════════════════════════
# Navbar
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="tt-navbar">
  <div style="display:flex;align-items:center;gap:12px">
    <span style="font-size:26px">🌍</span>
    <div>
      <div class="tt-nav-title">TerraTrace</div>
      <div class="tt-nav-sub">AI-powered satellite land-use analysis · Texas</div>
    </div>
  </div>
  <div style="display:flex;gap:8px">
    <div class="tt-pill active">{region}</div>
    <div class="tt-pill">{hist_start} → {hist_end}</div>
    <div class="tt-pill">{recent_start} → {recent_end}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# Side-by-side maps
# ════════════════════════════════════════════════════════════
st.markdown('<div class="tt-section" style="padding-bottom:8px">🗺️ Map Comparison</div>', unsafe_allow_html=True)

if not date_error:
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(f'<div class="tt-map-header">◀ HISTORICAL &nbsp; {hist_start} → {hist_end}</div>', unsafe_allow_html=True)
        with st.spinner("Loading…"):
            m_hist = build_map(sel["lat"], sel["lon"], sel["zoom"],
                               str(hist_start), str(hist_end), layer_on)
        st_folium(m_hist, height=500, use_container_width=True, key="map_hist")

    with right_col:
        st.markdown(f'<div class="tt-map-header">RECENT ▶ &nbsp; {recent_start} → {recent_end}</div>', unsafe_allow_html=True)
        with st.spinner("Loading…"):
            m_rec = build_map(sel["lat"], sel["lon"], sel["zoom"],
                              str(recent_start), str(recent_end), layer_on)
        st_folium(m_rec, height=500, use_container_width=True, key="map_rec")

    active_items = [(color, lbl) for k, (_, lbl, color) in LAYER_DEFS.items() if layer_on.get(k)]
    if active_items:
        dots = "".join(
            f'<div class="tt-legend-item"><div class="tt-dot" style="background:{c}"></div>{l}</div>'
            for c, l in active_items
        )
        st.markdown(f'<div class="tt-legend">{dots}</div>', unsafe_allow_html=True)
    else:
        st.info("Toggle on a layer in the sidebar to see land cover overlays on both maps.")
else:
    st.warning("Fix the date errors in the sidebar to load the maps.")

# ════════════════════════════════════════════════════════════
# Analytics — TerraScore + Bar Chart + Timeline
# ════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="tt-section" style="padding-bottom:8px">📊 Analytics</div>', unsafe_allow_html=True)

region_df = load_region_csv(region)

if region_df is not None:
    score_col, bar_col = st.columns([1, 2])

    with score_col:
        score = compute_terrascore(region_df)
        score_label = "🟢 Low Impact" if score < 30 else "🟡 Moderate Impact" if score < 60 else "🔴 High Impact"
        st.markdown('<div class="tt-label">TerraScore™</div>', unsafe_allow_html=True)
        st.plotly_chart(make_terrascore_gauge(score), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="tt-score-label">{score_label}</div>', unsafe_allow_html=True)

    with bar_col:
        st.markdown('<div class="tt-label">Current Land Cover Breakdown (most recent year)</div>', unsafe_allow_html=True)
        st.plotly_chart(make_bar_chart(region_df), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="tt-label" style="margin-top:12px">Historical Land Cover Trends</div>', unsafe_allow_html=True)
    st.plotly_chart(make_timeline(region_df), use_container_width=True, config={"displayModeBar": False})
else:
    st.info(f"No CSV data found for {region}. Run the GEE extractor to generate outputs.")

# ════════════════════════════════════════════════════════════
# Gemini Panel
# ════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="tt-section" style="padding-bottom:8px">🤖 AI Impact Summary</div>', unsafe_allow_html=True)

if run_gemini:
    if date_error:
        st.error("Fix date errors before generating a summary.")
    else:
        with st.spinner("Fetching land-cover changes and generating summary..."):
            try:
                date_range = [
                    {"start": str(hist_start),   "end": str(hist_end)},
                    {"start": str(recent_start), "end": str(recent_end)},
                ]
                romita_diffs = get_real_romita_diffs(date_range=date_range)
                response, api_err = call_gemini_for_impacts(romita_diffs)

                if api_err:
                    st.warning(f"Gemini could not be reached: {api_err}")

                parsed = None
                try:
                    parsed = json.loads(response)
                except json.JSONDecodeError:
                    match = re.search(r"\{[\s\S]*\}", response)
                    if match:
                        try:
                            parsed = json.loads(match.group(0))
                        except json.JSONDecodeError:
                            pass

                if parsed and "results" in parsed:
                    grouped_results = defaultdict(list)
                    for item in parsed["results"]:
                        grouped_results[item["region"]].append(item)

                    summary_cols = st.columns(2)
                    for col, reg in zip(summary_cols, SUMMARY_REGIONS):
                        with col:
                            if grouped_results.get(reg):
                                lines = "".join(
                                    f'<p><code>{escape(item["land_cover"])}</code> '
                                    f'({item["difference_percent"]:+.2f}%): '
                                    f'{escape(item["summary"])}</p>'
                                    for item in grouped_results[reg]
                                )
                            else:
                                lines = '<p class="impact-empty">No impact information available for this location.</p>'
                            st.markdown(
                                f'<div class="impact-card"><h4>{escape(reg)}</h4>{lines}</div>',
                                unsafe_allow_html=True,
                            )
                    if parsed.get("notes"):
                        st.caption(parsed["notes"])
                else:
                    st.write(response)

            except RuntimeError as e:
                st.error(f"Satellite data error: {e}")
else:
    st.markdown(
        '<div class="tt-card"><div class="tt-label">Gemini Overview</div>'
        '<div style="font-size:12px;color:#5a8060;margin-top:4px">'
        'Click <b style="color:#5ecf52">🤖 Generate Gemini Overview</b> in the sidebar to analyse land-cover changes with AI.'
        '</div></div>',
        unsafe_allow_html=True,
    )