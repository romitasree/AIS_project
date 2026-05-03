# app.py
import os
import sys
from datetime import date
from html import escape
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import ee
import textwrap
import folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
from scipy.spatial import ConvexHull

from gemini_integration import get_real_romita_diffs, call_gemini_for_impacts, validate_date_range
from policy import get_policy_brief, THRESHOLDS, build_trajectory_summary, load_forecast_csv, list_available_forecasts

# ── GEE init ────────────────────────────────────────────────
try:
    ee.Initialize(project="terratrace-488618")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="terratrace-488618")

# ── Constants ────────────────────────────────────────────────
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
    "water":              (0, "🟦 Water",              "#419bdf"),
    "trees":              (1, "🟩 Trees",              "#397d49"),
    "grass":              (2, "🟩 Grass",              "#88b053"),
    "flooded_vegetation": (3, "🟪 Flooded Vegetation", "#7a87c6"),
    "crops":              (4, "🟧 Crops",              "#e49635"),
    "shrub_and_scrub":    (5, "🟨 Shrubs & Scrubs",    "#dfc35a"),
    "built":              (6, "🟥 Built",              "#c4281b"),
    "bare":               (7, "🟫 Bare",               "#a59b8f"),
    "snow_and_ice":       (8, "⬜ Snow & Ice",         "#b39fe1"),
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
PIXEL_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "pixel_output"
PRED_CLASSES = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare"]
PRED_FEATURE_COLS = [f"lc_{c}_pct" for c in PRED_CLASSES] + [f"lc_{c}_conf" for c in PRED_CLASSES]
LOOKBACK = 3

FEATURE_COLS = [f"lc_{c}_pct" for c in DW_CLASSES] + [f"lc_{c}_conf" for c in DW_CLASSES]
DIFF_REGION_LABELS = {
    "Celina": "Celina, TX",
    "Falcon Lake": "Falcon Lake, TX",
    "Dallas": "Dallas, TX",
    "Austin": "Austin, TX",
    "Houston": "Houston, TX",
    "San Antonio": "San Antonio, TX",
    "Fort Worth": "Fort Worth, TX",
}

# ── Data helpers ─────────────────────────────────────────────

def forecast_years_until(target_date: date) -> int:
    today = date.today()
    if target_date <= today:
        return 1
    years = round((target_date - today).days / 365.0)
    return max(1, years)

def load_region_csv(region: str) -> pd.DataFrame | None:
    label = REGION_CSV.get(region)
    if not label:
        return None
    path = OUTPUTS_DIR / f"{label}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Build year from end date (fallback to start date)
    if "year" not in df.columns:
        if "end" in df.columns:
            df["year"] = pd.to_datetime(df["end"], errors="coerce").dt.year
        elif "start" in df.columns:
            df["year"] = pd.to_datetime(df["start"], errors="coerce").dt.year

    if "year" in df.columns:
        df = df.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)

    return df


def baseline_row_for_date(df: pd.DataFrame, target_date: date) -> pd.Series:
    if "end" in df.columns:
        end_dates = pd.to_datetime(df["end"], errors="coerce")
        matches = df.loc[end_dates <= pd.Timestamp(target_date)]
        if not matches.empty:
            return matches.iloc[-1]

    if "year" in df.columns:
        matches = df.loc[df["year"].astype(float) <= float(target_date.year)]
        if not matches.empty:
            return matches.iloc[-1]

    return df.iloc[0]


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

def _region_key(region: str) -> str:
    """Convert display name ('Falcon Lake') to file key ('falcon_lake')."""
    return region.lower().replace(" ", "_")


@st.cache_resource
def load_prediction_assets(region: str = ""):
    import pickle
    import tensorflow as tf

    # pixel_output is at terratrace/ — one level above modules/
    pix_dir = Path(__file__).resolve().parents[1] / "pixel_output"
    rkey    = _region_key(region)

    # 1. Per-region model (preferred — no cross-location bleed)
    mp = pix_dir / f"model_{rkey}.keras"
    sp = pix_dir / f"scaler_{rkey}.pkl"
    if mp.exists() and sp.exists():
        model = tf.keras.models.load_model(mp)
        with open(sp, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    # 2. Shared fallback — warns the user
    mp = pix_dir / "model.keras"
    sp = pix_dir / "scaler.pkl"
    if mp.exists() and sp.exists():
        st.warning(
            f"No per-region model found for **{region}**. "
            f"Using shared model — predictions may be inaccurate. "
            f"Run `retrain_per_region.py` to fix this.",
            icon="⚠️",
        )
        model = tf.keras.models.load_model(mp)
        with open(sp, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    raise FileNotFoundError(
        f"No model found in {pix_dir}. Run retrain_per_region.py first."
    )

def stabilize_prediction(row: np.ndarray) -> np.ndarray:
    out = row.astype(np.float32).copy()
    n = len(PRED_CLASSES)

    pct = np.clip(out[:n], 0.0, None)
    total = float(pct.sum())
    if total > 0:
        pct = pct / total * 100.0

    conf = np.clip(pct / 100.0, 0.0, 1.0)

    out[:n] = pct
    out[n:] = conf
    return out


def forecast_recursive(history_df: pd.DataFrame, years: int = 10,
                        region: str = "") -> pd.DataFrame:
    if len(history_df) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} rows of history to forecast.")

    model, scaler = load_prediction_assets(region)

    missing = [c for c in PRED_FEATURE_COLS if c not in history_df.columns]
    if missing:
        raise ValueError(f"History is missing required prediction columns: {missing}")

    # ── Compute per-class historical linear slopes ────────────────────────────
    # These anchor the forecast so it cannot drift opposite to the observed trend.
    # With only ~8 data points an LSTM can easily flip direction — the slope
    # acts as a guardrail, not a replacement.
    n_pred = len(PRED_CLASSES)
    slopes = np.zeros(n_pred, dtype=np.float32)
    last_hist_vals = np.zeros(n_pred, dtype=np.float32)

    for i, cls in enumerate(PRED_CLASSES):
        col = f"lc_{cls}_pct"
        if col in history_df.columns:
            vals = history_df[col].astype(float).values
            if len(vals) >= 2:
                x = np.arange(len(vals), dtype=np.float32)
                slopes[i] = float(np.polyfit(x, vals, 1)[0])  # pp per year
            last_hist_vals[i] = float(history_df[col].iloc[-1])

    history    = history_df[PRED_FEATURE_COLS].astype(np.float32).to_numpy()
    hist_scaled = scaler.transform(history).astype(np.float32)
    window      = hist_scaled[-LOOKBACK:].copy()

    preds = []
    for step in range(years):
        pred_scaled = model.predict(window[np.newaxis, :, :], verbose=0)[0]
        pred_scaled = np.clip(pred_scaled, 0.0, 1.0).astype(np.float32)

        pred_unscaled = scaler.inverse_transform(
            pred_scaled[np.newaxis, :]
        )[0].copy()

        pred_unscaled = stabilize_prediction(pred_unscaled)

        # ── Trend anchor ──────────────────────────────────────────────────────
        # Blend 50/50: LSTM output vs. simple linear extrapolation.
        # This prevents the model from predicting recovery when the satellite
        # record shows a clear, sustained decline (or vice versa).
        TREND_WEIGHT = 0.70
        for i in range(n_pred):
            linear_val = last_hist_vals[i] + slopes[i] * (step + 1)
            linear_val = float(np.clip(linear_val, 0.0, 100.0))
            pred_unscaled[i] = (
                TREND_WEIGHT * linear_val +
                (1 - TREND_WEIGHT) * pred_unscaled[i]
            )

        # Re-stabilize after blending so percentages still sum correctly
        pred_unscaled = stabilize_prediction(pred_unscaled)
        preds.append(pred_unscaled)

        next_scaled = scaler.transform(
            pred_unscaled[np.newaxis, :]
        )[0].astype(np.float32)
        window = np.vstack([window[1:], next_scaled])

    out = pd.DataFrame(preds, columns=PRED_FEATURE_COLS)
    last_year = (
        int(pd.to_datetime(history_df["end"]).dt.year.max())
        if "end" in history_df.columns else 2025
    )
    out.insert(0, "year", [last_year + i for i in range(1, years + 1)])
    return out

# ── GEE helper — predicted change for ANY DW class ──────────────────────────
def _dw_change_tile_url(baseline_start: str, baseline_end: str,
                        class_id: int, delta_pct: float,
                        base_color: str) -> str | None:
    """
    Returns a tile showing the PREDICTED SPATIAL FOOTPRINT of a class:
      Growth  → focal_max dilation  = larger area in the class's own color
      Decline → focal_min erosion   = smaller area in the class's own color
    The tile replaces the baseline — it IS the predicted state, not an overlay.
    """
    key = f"chg6_{baseline_start}_{baseline_end}_{class_id}_{delta_pct:.2f}"
    if key in st.session_state:
        return st.session_state[key]
    try:
        tx_rect = ee.Geometry.Rectangle([-106.65, 25.84, -93.51, 36.50])
        label = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterDate(baseline_start, baseline_end)
            .filterBounds(tx_rect)
            .mode()
            .select("label")
        )
        target  = label.eq(class_id)
        color   = base_color.lstrip("#")
        abs_d   = abs(delta_pct)

        # Radius: how many pixels to grow or shrink.
        # Scaled so a 5pp change = radius 2, 10pp = radius 4, 20pp = radius 8.
        # Capped at 12 to avoid blobs at large scales.
        radius = int(max(1, min(12, abs_d / 2.5)))

        if delta_pct > 0:
            # GROWTH: dilate existing pixels outward → bigger footprint
            predicted = target.focal_max(
                radius=radius, kernelType="circle", units="pixels"
            ).selfMask()
        else:
            # DECLINE: erode existing pixels inward → smaller footprint
            predicted = target.focal_min(
                radius=radius, kernelType="circle", units="pixels"
            ).selfMask()

        mid = predicted.getMapId({"min": 0, "max": 1, "palette": [color]})
        url = mid["tile_fetcher"].url_format
        st.session_state[key] = url
        return url
    except Exception:
        return None    

def build_prediction_map(
    center_lat, center_lon, zoom,
    forecast_df: "pd.DataFrame",
    region_df:   "pd.DataFrame | None" = None,
    baseline_start: str = "2024-05-01",
    baseline_end:   str = "2024-08-01",
    active_layers:  dict | None = None,
):
    """
    Right map shows the PREDICTED SPATIAL STATE, not overlays on current state.

    For every toggled class:
      - If growing:  show dilated footprint  (more area, same color as left map)
      - If declining: show eroded footprint  (less area, same color as left map)
      - If stable:   show baseline unchanged (identical to left map)

    The viewer can directly compare left vs right and see which areas
    physically expanded or contracted.
    """
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles=None)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite", name="Satellite", overlay=False,
    ).add_to(m)

    latest = forecast_df.iloc[-1]

    def _hist_pct(cls):
        if region_df is not None and len(region_df) > 0:
            return float(region_df.iloc[-1].get(f"lc_{cls}_pct", 0.0))
        return 0.0

    layers_to_show = {k for k, v in (active_layers or {}).items() if v}

    for key in layers_to_show:
        if key not in LAYER_DEFS:
            continue
        class_id, lbl, color = LAYER_DEFS[key]
        pred_col = f"lc_{key}_pct"
        delta    = float(latest.get(pred_col, 0.0)) - _hist_pct(key)

        if abs(delta) > 0.5:
            # Show the morphologically-adjusted predicted footprint.
            # This IS the new spatial state — no baseline tile underneath.
            pred_url = _dw_change_tile_url(
                baseline_start, baseline_end, class_id, delta, color
            )
            if pred_url:
                folium.TileLayer(
                    tiles=pred_url,
                    attr="TerraTrace Prediction",
                    name=lbl,
                    overlay=True,
                    opacity=0.85,
                ).add_to(m)
        else:
            # Stable: show baseline unchanged — right map matches left map
            base_url = _dw_tile_url(
                baseline_start, baseline_end, class_id, color
            )
            if base_url:
                folium.TileLayer(
                    tiles=base_url, attr="GDW", name=lbl,
                    overlay=True, opacity=0.85,
                ).add_to(m)

    for city, coords in CITIES.items():
        folium.Marker(
            location=coords, popup=city,
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(m)

    return m

def _resolve_baseline(region_df: "pd.DataFrame | None") -> tuple[str, str]:
    if region_df is not None and "end" in region_df.columns:
        try:
            last_end = pd.to_datetime(region_df["end"]).max()
            last_start = last_end - pd.DateOffset(months=3)
            return str(last_start.date()), str(last_end.date())
        except Exception:
            pass
    return "2024-05-01", "2024-08-01"


@st.cache_data(show_spinner=False)
def load_region_differences(hist_start: str, hist_end: str, recent_start: str, recent_end: str) -> list[dict]:
    date_range = [
        {"start": hist_start, "end": hist_end},
        {"start": recent_start, "end": recent_end},
    ]
    return get_real_romita_diffs(date_range=date_range)


def region_difference_frame(region: str, romita_diffs: list[dict]) -> pd.DataFrame:
    region_label = DIFF_REGION_LABELS.get(region)
    rows = []
    matches = [
        item for item in romita_diffs
        if item.get("region") == region_label
    ]
    diff_lookup = {
        item.get("land_cover"): item.get("difference_percent", 0.0)
        for item in matches
    }

    for land_cover in DW_CLASSES:
        rows.append(
            {
                "land_cover": land_cover,
                "label": LAYER_DEFS[land_cover][1],
                "difference_percent": float(diff_lookup.get(land_cover, 0.0)),
            }
        )

    diff_df = pd.DataFrame(rows)
    diff_df.attrs["has_data"] = bool(matches)
    return diff_df


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
def make_bar_chart(diff_df: pd.DataFrame) -> go.Figure:
    values = diff_df["difference_percent"].tolist()
    labels = diff_df["label"].tolist()
    colors = ["#5ecf52" if value >= 0 else "#e74c3c" for value in values]
    text_positions = ["top center" if value >= 0 else "bottom center" for value in values]

    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{value:+.1f}%" for value in values],
        textposition="none",
        cliponaxis=False,
    ))
    fig.update_layout(
        paper_bgcolor="#0a1a0f", plot_bgcolor="#0d1f12",
        font=dict(color="#b8d4a0", family="Roboto Mono"),
        margin=dict(t=20, b=40, l=10, r=10),
        yaxis=dict(title="% Change", gridcolor="#1a3020", zeroline=True, zerolinecolor="#8aa17d"),
        xaxis=dict(tickangle=-30),
        showlegend=False, height=320,
    )
    fig.add_hline(y=0, line_color="#8aa17d", line_width=1)
    fig.update_traces(textfont=dict(color="#d9ead0", family="Roboto Mono", size=17))

    for label, value, position in zip(labels, values, text_positions):
        fig.add_annotation(
            x=label,
            y=value,
            text=f"{value:+.1f}%",
            showarrow=False,
            yshift=12 if value >= 0 else -12,
            font=dict(color="#d9ead0", family="Roboto Mono", size=17),
            xanchor="center",
            yanchor="bottom" if value >= 0 else "top",
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
        yaxis=dict(title="% Coverage", gridcolor="#1a3020", zeroline=False,
                   tickfont=dict(size=19)),
        xaxis=dict(title="Year", gridcolor="#1a3020",
                   tickfont=dict(size=19)),
        legend=dict(bgcolor="#0d1f12", bordercolor="#1a3020", borderwidth=1,
                    font=dict(size=16)),
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

def make_sparkline(cls: str, hist_df: pd.DataFrame,
                   forecast_df: pd.DataFrame | None,
                   color: str, crit_threshold: float | None = None) -> go.Figure:
    col = f"lc_{cls}_pct"
    fig = go.Figure()

    if col in hist_df.columns:
        fig.add_trace(go.Scatter(
            x=hist_df["year"], y=hist_df[col],
            mode="lines", name="Historical",
            line=dict(color=color, width=2),
            showlegend=False,
        ))

    if forecast_df is not None and col in forecast_df.columns:
        if col in hist_df.columns and len(hist_df):
            bridge_x = [hist_df["year"].iloc[-1], forecast_df["year"].iloc[0]]
            bridge_y = [hist_df[col].iloc[-1],    forecast_df[col].iloc[0]]
            fig.add_trace(go.Scatter(
                x=bridge_x, y=bridge_y, mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=forecast_df["year"], y=forecast_df[col],
            mode="lines", name="Forecast",
            line=dict(color=color, width=1.5, dash="dash"),
            opacity=0.55,
            showlegend=False,
        ))

    if crit_threshold is not None:
        all_years = []
        if col in hist_df.columns:     all_years += hist_df["year"].tolist()
        if forecast_df is not None and col in forecast_df.columns:
            all_years += forecast_df["year"].tolist()
        if all_years:
            fig.add_shape(type="line",
                x0=min(all_years), x1=max(all_years),
                y0=crit_threshold, y1=crit_threshold,
                line=dict(color="#e74c3c", width=1, dash="dot"),
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=4, b=4, l=4, r=4),
        height=80,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, fixedrange=True),
        hovermode=False,
    )
    return fig

def render_subheading(text: str, margin_top: int = 0) -> None:
    style = f"margin-top:{margin_top}px;" if margin_top else ""
    st.markdown(f'<div class="tt-subhead" style="{style}">{text}</div>', unsafe_allow_html=True)


def build_layer_toggle_css() -> str:
    rules = [
        """
<style>
.layer-toggle-marker{display:none;}
"""
    ]

    for layer_name, (_, _, color) in LAYER_DEFS.items():
        rules.append(
            f"""
.layer-toggle-marker[data-layer="{layer_name}"] + div [data-baseweb="switch"] input:checked + div {{
  background-color: {color} !important;
  border-color: {color} !important;
}}
.layer-toggle-marker[data-layer="{layer_name}"] + div [data-baseweb="switch"] input:checked + div > div {{
  background-color: #ffffff !important;
}}
"""
        )

    rules.append("</style>")
    return "".join(rules)


# ════════════════════════════════════════════════════════════
# Page config & CSS
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="TerraTrace", layout="wide", page_icon="🌍", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Averia+Serif+Libre:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&display=swap');

html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],[data-testid="stMainBlockContainer"],
section.main,section.main>div,.main .block-container{background-color:#0a1a0f !important;}
[data-testid="stHeader"],[data-testid="stDecoration"],
[data-testid="stToolbar"],footer,#MainMenu{visibility:hidden !important;}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div,
[data-testid="stSidebarContent"],[data-testid="stSidebarUserContent"]{
  background:#07120a !important;border-right:1px solid #1a3020 !important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,[data-testid="stSidebar"] p{
        color:#90b880 !important;font-size:15px !important;
  font-family:'Averia Serif Libre',serif !important;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] button,
[data-testid="stSidebar"] [data-baseweb="select"] span{
        font-size:15px !important;
  font-family:'Averia Serif Libre',serif !important;}
[data-testid="stSidebar"] [data-baseweb="select"]>div{
  background:#0d1f12 !important;border:1px solid #1e3824 !important;
        border-radius:8px !important;color:#b8d4a0 !important;font-size:15px !important;}
[data-testid="stSidebar"] input{
  background:#0d1f12 !important;color:#b8d4a0 !important;
        border:1px solid #1e3824 !important;border-radius:8px !important;font-size:15px !important;}
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebar"]{max-width:calc(100vw * 0.15 - 15px) !important;}
[data-testid="stAppViewContainer"]{padding-left:0 !important;}
.stButton>button{width:100%;background:#0d1f12 !important;color:#5ecf52 !important;
  border:1px solid #2a6030 !important;border-radius:8px !important;
  font-family:'Averia Serif Libre',serif !important;font-size:13px !important;padding:10px 12px !important;}
.stButton>button:hover{background:#163a20 !important;border-color:#5ecf52 !important;}
p,li{color:#b8d4a0;font-family:'Averia Serif Libre',serif;font-size:14px;}
hr{border-color:#1a3020 !important;margin:8px 0 !important;}
.tt-navbar{display:flex;align-items:center;justify-content:space-between;
  padding:14px 28px 12px;border-bottom:1px solid #1a3020;background:#07120a;margin-bottom:16px;}
.tt-nav-title{font-family:'Averia Serif Libre',serif;font-size:30px;font-weight:700;color:#5ecf52;}
.tt-nav-sub{font-size:12px;color:#3a6040;font-style:italic;font-family:'Averia Serif Libre',serif;}
.tt-pill{background:#0d1f12;border:1px solid #1e3824;border-radius:20px;
    padding:4px 14px;font-size:12px;color:#5a8060;font-family:'Averia Serif Libre',serif;}
.tt-pill.active{background:#163a20;border-color:#4aaa40;color:#8ddf7e;}
.tt-section{font-family:'Averia Serif Libre',serif;font-size:20px;font-weight:700;
  text-transform:uppercase;letter-spacing:2.5px;color:#90b880;padding:14px 0 6px;}
.tt-map-header{font-family:'Averia Serif Libre',serif;font-size:18px;font-weight:700;
  color:#5ecf52;text-align:center;padding:6px;background:#0d1f12;
  border:1px solid #1a3020;border-radius:8px;margin-bottom:6px;}
.tt-subhead{font-family:'Averia Serif Libre',serif;font-size:18px;font-weight:700;
  color:#5ecf52;text-align:center;padding:6px;background:#0d1f12;
  border:1px solid #1a3020;border-radius:8px;margin-bottom:6px;}
.tt-card{background:#0d1f12;border:1px solid #1a3020;border-radius:10px;
  padding:13px 15px;margin-bottom:9px;font-family:'Averia Serif Libre',serif;}
.tt-label{font-family:'Averia Serif Libre',serif;font-size:11px;text-transform:uppercase;
  letter-spacing:2.5px;color:#2e6040;margin-bottom:5px;}
.tt-score-label{font-family:'Averia Serif Libre',serif;font-size:12px;font-weight:700;
  text-align:center;color:#4a7050;margin-top:4px;}
.impact-card{background:linear-gradient(180deg,rgba(13,31,18,.97),rgba(8,20,12,.97));
  border:1px solid rgba(94,207,82,.45);border-radius:16px;padding:1.1rem;
  box-shadow:0 0 20px rgba(94,207,82,.14);}
.impact-card h4{margin:0 0 .7rem;color:#dff6d8;font-size:1.1rem;
  font-family:'Averia Serif Libre',serif;font-weight:700;}
.impact-card p{margin:0 0 .7rem;color:#edf9e7;line-height:1.5;font-size:0.9rem;
  font-family:'Averia Serif Libre',serif;}
.impact-card code{background:rgba(94,207,82,.14);color:#b8f0ad;
  border-radius:999px;padding:.1rem .4rem;}
.impact-empty{color:#a5c99a;opacity:.9;font-family:'Averia Serif Libre',serif;}
.tt-legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;}
.tt-legend-item{display:flex;align-items:center;gap:5px;background:#0d1f12;
  border:1px solid #1a3020;border-radius:20px;padding:3px 10px;
    font-size:15px;color:#5a8060;font-family:'Averia Serif Libre',serif;}
.tt-dot{width:14px;height:14px;border-radius:14px;flex-shrink:0;}
</style>
""", unsafe_allow_html=True)
st.markdown(build_layer_toggle_css(), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:Roboto Mono,monospace;font-size:20px;font-weight:700;color:#5ecf52;padding:16px 0 4px">TerraTrace</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:15px;color:#3a6040;letter-spacing:.5px;margin-bottom:12px">LAND USE MONITOR</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="tt-section" style="font-size:20px">Region</div>', unsafe_allow_html=True)
    region = st.selectbox("Region", options=list(REGION_COORDS.keys()), label_visibility="collapsed")
    sel = REGION_COORDS[region]

    st.markdown('<div class="tt-section" style="font-size:20px">Historical Period</div>', unsafe_allow_html=True)
    hist_start = st.date_input("Hist Start", value=date(2016, 5, 1), key="hist_start", label_visibility="collapsed")
    hist_end   = st.date_input("Hist End",   value=date(2016, 8, 1), key="hist_end",   label_visibility="collapsed")

    st.markdown('<div class="tt-section" style="font-size:20px">Recent/Future Period</div>', unsafe_allow_html=True)
    recent_start = st.date_input("Rec Start", value=date(2024, 5, 1), key="recent_start", label_visibility="collapsed")
    recent_end   = st.date_input("Rec End",   value=date(2024, 8, 1), key="recent_end",   label_visibility="collapsed")

    today = date.today()

    date_error = None
    future_mode = recent_end > today

    valid1, err1 = validate_date_range(hist_start, hist_end)
    if not valid1:
        date_error = f"Historical: {err1}"
    elif recent_start <= hist_end:
        date_error = "Recent period must start after the historical period ends."
    elif recent_start >= recent_end:
        date_error = "Recent period start must be before recent period end."
    elif not future_mode:
        valid2, err2 = validate_date_range(recent_start, recent_end)
        if not valid2:
            date_error = f"Recent: {err2}"

    if date_error:
        st.error(date_error)
    else:
        st.success(f"{hist_start} → {hist_end}  ↔  {recent_start} → {recent_end}")

    st.markdown('<div class="tt-section" style="font-size:20px">Map Layers</div>', unsafe_allow_html=True)
    layer_on = {}
    default_layers_on = set(LAYER_DEFS.keys())
    for k, (_, lbl, _c) in LAYER_DEFS.items():
        st.markdown(f'<div class="layer-toggle-marker" data-layer="{k}"></div>', unsafe_allow_html=True)
        layer_on[k] = st.toggle(lbl, value=(k in default_layers_on), key=f"t_{k}")

    st.divider()

# ════════════════════════════════════════════════════════════
# Navbar
# ════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="tt-navbar">
  <div style="display:flex;align-items:center;gap:12px">
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
region_df = load_region_csv(region)

if region_df is None:
    years = list(range(2016, 2025))
    mock_rows = []
    built = 10.0
    trees = 28.0
    grass = 25.0
    for yr in years:
        built += 1.2
        trees -= 0.6
        grass -= 0.3
        row = {
            "year": yr,
            "lc_built_pct": round(built, 1),
            "lc_trees_pct": round(trees, 1),
            "lc_grass_pct": round(grass, 1),
            "lc_water_pct": 3.0,
            "lc_crops_pct": 12.0,
            "lc_flooded_vegetation_pct": 1.5,
            "lc_shrub_and_scrub_pct": 8.0,
            "lc_bare_pct": 4.0,
            "lc_snow_and_ice_pct": 0.2,
        }
        for c in DW_CLASSES:
            row[f"lc_{c}_conf"] = round(row.get(f"lc_{c}_pct", 0.0) / 100.0, 4)
        mock_rows.append(row)

    region_df = pd.DataFrame(mock_rows)
    st.info(f"Showing estimated data for {region}. Run the GEE extractor to load real satellite data.")

st.markdown('<div class="tt-section" style="padding-bottom:8px;font-size:15px">Map Comparison</div>', unsafe_allow_html=True)

if not date_error:
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(f'<div class="tt-map-header">HISTORICAL &nbsp; {hist_start} → {hist_end}</div>', unsafe_allow_html=True)
        with st.spinner("Loading..."):
            m_hist = build_map(sel["lat"], sel["lon"], sel["zoom"],
                               str(hist_start), str(hist_end), layer_on)
        st_folium(m_hist, height=500, width='stretch', key="map_hist")

    with right_col:
        if future_mode:
            st.markdown(
                f'<div class="tt-map-header">PREDICTED &nbsp; {recent_start} → {recent_end}</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Building forecast map..."):
                forecast_years = forecast_years_until(recent_end)
                forecast_df = forecast_recursive(region_df, years=forecast_years, region=region)
                _bl_start, _bl_end = _resolve_baseline(region_df)
                m_rec = build_prediction_map(
                    sel["lat"], sel["lon"], sel["zoom"],
                    forecast_df, region_df,
                    baseline_start=_bl_start,
                    baseline_end=_bl_end,
                    active_layers=layer_on,
                )
            st_folium(m_rec, height=500, width='stretch', key="map_rec_pred")
        else:
            st.markdown(
                f'<div class="tt-map-header">RECENT &nbsp; {recent_start} → {recent_end}</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Loading..."):
                m_rec = build_map(sel["lat"], sel["lon"], sel["zoom"],
                                str(recent_start), str(recent_end), layer_on)
            st_folium(m_rec, height=500, width='stretch', key="map_rec")
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
st.markdown('<div class="tt-section" style="font-size:15px">Analytics</div>', unsafe_allow_html=True)

region_df = load_region_csv(region)
diff_chart_df = None
diff_chart_error = None

if not date_error:
    try:
        all_diffs = load_region_differences(
            str(hist_start),
            str(hist_end),
            str(recent_start),
            str(recent_end),
        )
        candidate_diff_df = region_difference_frame(region, all_diffs)
        if candidate_diff_df.attrs.get("has_data"):
            diff_chart_df = candidate_diff_df
        elif region in DIFF_REGION_LABELS:
            diff_chart_error = f"No land-cover differences were returned for {region} in the selected date ranges."
        else:
            diff_chart_error = f"Difference chart is currently available for {', '.join(DIFF_REGION_LABELS.keys())}."
    except RuntimeError as exc:
        diff_chart_error = str(exc)

if region_df is None:
    import numpy as np
    years = list(range(2016, 2025))
    mock_rows = []
    built = 10.0
    trees = 28.0
    grass = 25.0
    for yr in years:
        built += 1.2
        trees -= 0.6
        grass -= 0.3
        row = {"year": yr, "lc_built_pct": round(built,1),
               "lc_trees_pct": round(trees,1), "lc_grass_pct": round(grass,1),
               "lc_water_pct": 3.0, "lc_crops_pct": 12.0,
               "lc_flooded_vegetation_pct": 1.5, "lc_shrub_and_scrub_pct": 8.0,
               "lc_bare_pct": 4.0, "lc_snow_and_ice_pct": 0.2}
        mock_rows.append(row)
    region_df = pd.DataFrame(mock_rows)
    st.info(f"Showing estimated data for {region}. Run the GEE extractor to load real satellite data.")

score_col, bar_col = st.columns([1, 2])

with score_col:
    score = compute_terrascore(region_df)
    score_label = "Low Impact" if score < 30 else "Moderate Impact" if score < 60 else "High Impact"
    render_subheading("TerraScore")
    st.plotly_chart(make_terrascore_gauge(score), width='stretch', config={"displayModeBar": False})
    st.markdown(f'<div class="tt-score-label">{score_label}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="tt-card" style="margin-top:8px">
      <div class="tt-subhead">Score Breakdown</div>
      <div style="font-size:11px;color:#5a8060;margin-top:6px;line-height:1.8">
        Built area growth<br>
        Tree cover loss<br>
        Grass cover loss<br>
        <span style="color:#2e6040;font-size:10px">Higher = more urban impact</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with bar_col:
    render_subheading("Change Percentage Over Selected Period")
    if future_mode:
        try:
            last_data_year   = int(region_df["year"].iloc[-1])
            forecast_years_n = max(1, recent_end.year - last_data_year)
            fc_df = forecast_recursive(region_df, years=forecast_years_n, region=region)
            baseline_row = baseline_row_for_date(region_df, hist_end)
            forecast_row = fc_df.iloc[-1]

            pred_rows = []
            for c in DW_CLASSES:
                col = f"lc_{c}_pct"
                base_val = float(baseline_row.get(col, 0.0))
                pred_val = float(forecast_row.get(col, base_val))
                pred_rows.append({
                    "land_cover":         c,
                    "label":              LAYER_DEFS[c][1],
                    "difference_percent": round(pred_val - base_val, 2),
                })
            pred_diff_df = pd.DataFrame(pred_rows)
            pred_diff_df.attrs["has_data"] = True

            render_subheading(
                f"Predicted Change · {int(baseline_row['year'])} → {int(fc_df['year'].iloc[-1])}",
            )
            st.plotly_chart(
                make_bar_chart(pred_diff_df),
                width="stretch",
                config={"displayModeBar": False},
            )
        except Exception as fc_err:
            st.info(f"Predicted change chart unavailable: {fc_err}")
    elif diff_chart_df is not None:
        st.plotly_chart(make_bar_chart(diff_chart_df), width='stretch', config={"displayModeBar": False})
    else:
        st.info(diff_chart_error or "Difference data is unavailable for the current selection.")

render_subheading("Historical Land Cover Trends", margin_top=12)
st.plotly_chart(make_timeline(region_df), width='stretch', config={"displayModeBar": False})

# ════════════════════════════════════════════════════════════
# Policy Actions Panel
# ════════════════════════════════════════════════════════════
render_subheading("Policy Actions", margin_top=12)

run_policy = st.button("Generate Policy Actions", key="run_policy")

if "policy_brief"    not in st.session_state: st.session_state["policy_brief"]    = None
if "policy_brief_err" not in st.session_state: st.session_state["policy_brief_err"] = None
if "policy_forecast" not in st.session_state: st.session_state["policy_forecast"] = None
if "policy_region"   not in st.session_state: st.session_state["policy_region"]   = None

if run_policy:
    with st.spinner("Generating Policy Actions..."):
        try:
            try:
                pf_df = forecast_recursive(region_df, years=10, region=region)
                st.session_state["policy_forecast"] = pf_df
            except Exception as fe:
                pf_df = None
                st.session_state["policy_forecast"] = None
                st.warning(f"Forecast model unavailable: {fe}. Dossier uses historical data only.")

            pct_cols_present = ["year"] + [f"lc_{c}_pct" for c in DW_CLASSES]
            hist_slim = region_df[[c for c in pct_cols_present if c in region_df.columns]].copy()
            if pf_df is not None:
                fore_slim = pf_df[[c for c in pct_cols_present if c in pf_df.columns]].copy()
                full_df = (pd.concat([hist_slim, fore_slim], ignore_index=True)
                             .drop_duplicates(subset="year").sort_values("year"))
            else:
                full_df = hist_slim

            brief, err = get_policy_brief(f"{region}, TX", full_df)
            st.session_state["policy_brief"]     = brief
            st.session_state["policy_brief_err"] = err
            st.session_state["policy_region"]    = region
        except Exception as e:
            st.error(f"Policy engine error: {e}")

# ── Render stored brief ────────────────────────────────────────────────────
brief     = st.session_state.get("policy_brief")
brief_err = st.session_state.get("policy_brief_err")
pf_df     = st.session_state.get("policy_forecast")
br_region = st.session_state.get("policy_region")

if brief and br_region == region:
    if brief_err:
        st.caption(f"{brief_err}")

    alerts  = brief.get("threshold_alerts", [])
    dossier = brief.get("dossier", [])
    note    = brief.get("data_note", "")

    STATUS_COLOR = {
        "CRTICAL":         "#e74c3c",
        "ACTION REQUIRED": "#e8a020",
        "MONITOR":         "#5ecf52",
        "STABLE":          "#2e6040",
    }
    alert_status = {a["land_cover"]: a["status"] for a in alerts}

    st.markdown("""
    <div style="font-family:'Averia Serif Libre',serif;font-size:20px;font-weight:700;
                color:#5a8060;text-align:center;margin-bottom:6px;">
    Projected Effects & Policy Actions
    </div>
    """, unsafe_allow_html=True)

    if not dossier:
        st.info("No significant land-cover changes detected in the selected period.")
    else:
        for item in dossier:
            cls         = item["land_cover"]
            consequence = item.get("consequence", "")
            policies    = item.get("policies", [])
            rate        = item["rate_pp_per_yr"]
            delta       = item["total_delta_pp"]
            accel       = item["acceleration"]
            ty          = item.get("threshold_year")
            color       = LAYER_DEFS.get(cls, (None, None, "#5a8060"))[2]
            lbl         = LAYER_DEFS.get(cls, (None, cls.replace("_", " ").title(), None))[1]
            status      = alert_status.get(cls, "STABLE")
            sc          = STATUS_COLOR.get(status, "#2e6040")

            thresh_map = {
                c: (THRESHOLDS[c][2] if c in THRESHOLDS and THRESHOLDS[c][0] else None)
                for c in DW_CLASSES
            }

            col_txt, col_chart = st.columns([3, 1])

            with col_txt:
                badge = (
                    f'<span style="font-family:Roboto Mono,monospace;font-size:18px;'
                    f'font-weight:700;text-transform:uppercase;letter-spacing:1.5px;'
                    f'color:{sc};padding:1px 7px;border-radius:20px;'
                    f'background:rgba(0,0,0,0.35)">{status}</span>'
                ) if status != "STABLE" else ""

                accel_color = ("#e74c3c" if accel == "accelerating"
                               else "#5ecf52" if accel == "decelerating"
                               else "#5a8060")
                ty_str = (f'<span style="color:#e74c3c;font-family:Roboto Mono,monospace">'
                          f'Critical yr {ty}</span> · ') if ty else ""

                st.markdown(f"""
                <div style="background:#0d1f12;border:1px solid #1a3020;
                            border-left:3px solid {color};
                            border-radius:10px;padding:13px 16px">

                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
                    <span style="font-size:18px;font-weight:600;color:{color}">{lbl}</span>
                    {badge}
                  </div>

                  <div style="font-size:18px;color:#b8d4a0;line-height:1.7;
                              margin-bottom:12px;
                              border-left:2px solid #1a3020;padding-left:10px">
                    {escape(consequence)}
                  </div>

                  <div style="display:flex;flex-wrap:wrap;gap:14px;font-size:18px;
                              font-family:Roboto Mono,monospace;color:#5a8060;
                              margin-bottom:{'14px' if policies else '0'}">
                    <span>Total <b style="color:{color}">{delta:+.1f} pp</b></span>
                    <span>Rate <b style="color:{color}">{rate:+.2f} pp/yr</b></span>
                    <span>Trend <b style="color:{accel_color}">{accel}</b></span>
                    {ty_str}
                  </div>

                  {'<div style="border-top:1px solid #1a3020;margin-top:4px;padding-top:10px">' if policies else ''}
                    {'<div style="font-family:Roboto Mono,monospace;font-size:18px;text-transform:uppercase;letter-spacing:2px;color:#3a7050;margin-bottom:8px;font-weight: bold;">Policy Actions</div>' if policies else ''}
                    {''.join(textwrap.dedent(f'''
<div style="background:#07120a;border:1px solid #1a3020; border-left:2px solid {color}; border-radius:8px;padding:9px 12px;margin-bottom:7px"> <div style="font-family:Roboto Mono,monospace;font-size:18px; text-transform:uppercase;letter-spacing:1px; color:#3a7050;margin-bottom:5px"> {escape(p.get("instrument", ""))} </div> <div style="font-size:18px;color:#90b880;line-height:1.7"> {escape(p.get("action", ""))} </div> </div> ''').strip() for p in policies)}
                  {'</div>' if policies else ''}

                </div>
                """, unsafe_allow_html=True)

            with col_chart:
                pct_col = f"lc_{cls}_pct"
                if pct_col in region_df.columns:
                    crit_t = thresh_map.get(cls)
                    spark  = make_sparkline(cls, region_df, pf_df, color, crit_t)
                    st.plotly_chart(spark, use_container_width=True,
                                    config={"displayModeBar": False},
                                    key=f"spark_{cls}")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:14px;padding:8px 14px;background:#070f09;
                border:1px solid #111f14;border-radius:8px;
                font-size:18px;color:#2e6040;font-family:'Roboto Mono',monospace">
      {escape(note)}
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="tt-card">'
        '<div class="tt-label">Policy Actions</div>'
        '<div style="font-size:18px;color:#5a8060;margin-top:6px;line-height:1.7">'
        'Click <b style="color:#5ecf52">Generate Policy Actions</b> above. '
        'This brief is generated for <b style="color:#b8d4a0">local government officials</b> and'
        'county planners. It synthesises the satellite-derived historical record, the LSTM'
        '10-year forecast, and the current land-cover composition to surface actionable'
        'policy priorities.'
        '</div></div>',
        unsafe_allow_html=True,
    )