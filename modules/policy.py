# policy.py
"""
TerraTrace — Step 3
====================
Reads a city's forecast CSV (produced by forecast.py), builds a plain-language
summary of the projected land-cover trajectory, calls the Gemini API, and returns
a structured policy brief with county-specific recommendations.

Public API
----------
    load_forecast_csv(csv_path)          -> pd.DataFrame
    build_trajectory_summary(label, df)  -> str   (plain-English narrative)
    get_policy_brief(label, df)          -> (dict | str, str | None)
                                            (brief, error_message)
    list_available_forecasts(forecasts_dir) -> dict[label -> Path]
"""

import json
import logging
import os
import re
import socket
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("policy")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FORECASTS_DIR = "outputs/forecasts"

DW_CLASSES = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice",
]

PCT_COLS = [f"lc_{cls}_pct" for cls in DW_CLASSES]

# Only report classes whose absolute change exceeds this threshold (pp)
_CHANGE_THRESHOLD_PP = 1.0

# ---------------------------------------------------------------------------
# Texas-specific consequence descriptions — pure Python, no LLM required
# ---------------------------------------------------------------------------
_TX_CONSEQUENCES = {
    "water": {
        "decline": (
            "The lake or reservoir is shrinking. If this continues, the community could face "
            "water shortages, higher water bills, and the loss of fishing, recreation, and "
            "tourism that local businesses depend on."
        ),
        "increase": (
            "Water coverage is expanding, which could flood roads, homes, and farmland near "
            "the shoreline. Properties in low-lying areas may become uninsurable or unsellable."
        ),
    },
    "trees": {
        "decline": (
            "Trees are being cleared faster than they are being replaced. Without tree cover, "
            "rainwater runs off more quickly — flooding neighborhoods, overwhelming storm drains, "
            "and making summers noticeably hotter for residents."
        ),
        "increase": (
            "Tree cover is growing, which is generally positive, but some of this may be "
            "invasive species like cedar or Chinese tallow that crowd out native plants and "
            "don't provide the same benefits to wildlife or water quality."
        ),
    },
    "grass": {
        "decline": (
            "Native grasslands are disappearing. These open lands act as natural sponges during "
            "heavy rain, and they provide habitat for wildlife. As they vanish, flooding risk "
            "rises and the landscape becomes less resilient to droughts."
        ),
        "increase": (
            "Grassy areas are expanding, which may be recovery after drought — or it could be "
            "invasive grasses spreading where native plants used to grow. Either way, it needs "
            "to be checked in person to understand whether it is a good or bad sign."
        ),
    },
    "flooded_vegetation": {
        "decline": (
            "Wetlands and riverside plant buffers are shrinking. These areas naturally filter "
            "pollution before it reaches the water supply and protect against floods. Losing them "
            "means dirtier water and higher flood damage costs for the whole community."
        ),
        "increase": (
            "Vegetation in flooded areas is expanding. This can sometimes signal invasive plants "
            "like giant cane, which block waterways and make flooding worse. Local officials "
            "should check whether this is native plant recovery or an invasive species problem."
        ),
    },
    "crops": {
        "decline": (
            "Farmland is being converted to other uses at a significant rate. Once high-quality "
            "agricultural soil is paved over or graded flat, it cannot be economically restored. "
            "Local food production capacity and farming-related jobs are being permanently lost."
        ),
        "increase": (
            "More land is being converted to agriculture. While farming can be positive, expanding "
            "crop fields too close to waterways can increase fertilizer runoff into rivers and "
            "lakes, degrading drinking water quality for downstream communities."
        ),
    },
    "shrub_and_scrub": {
        "increase": (
            "Dense brush is spreading across the landscape. This creates a wildfire hazard — "
            "especially for homes and ranches on the outskirts of town. Homeowners in these areas "
            "may already be seeing rising insurance costs or losing coverage altogether."
        ),
        "decline": (
            "Brush is being cleared, but if the land is left bare afterward, it becomes vulnerable "
            "to erosion and can be taken over by invasive grasses that make future wildfires "
            "worse, not better."
        ),
    },
    "built": {
        "increase": (
            "Developed land — roads, parking lots, rooftops — is expanding rapidly. Every new "
            "paved surface sends more stormwater rushing into local creeks and neighborhoods "
            "during heavy rain. Without updated infrastructure, flooding will worsen and road "
            "and drainage repair costs will fall on taxpayers."
        ),
        "decline": (
            "Some built-up areas appear to be clearing or being demolished. If the land is left "
            "exposed without erosion controls, it can clog drainage channels and pollute nearby "
            "water bodies with sediment."
        ),
    },
    "bare": {
        "increase": (
            "Large areas of exposed dirt are expanding. This causes dust problems for nearby "
            "residents, erodes topsoil into local waterways, and usually signals active "
            "construction or land disturbance happening without adequate controls in place."
        ),
        "decline": (
            "Previously bare land is being covered, which is generally positive. Officials should "
            "confirm it is native vegetation or planned development being established — not just "
            "weeds filling in temporarily before further disturbance."
        ),
    },
    "snow_and_ice": {
        "decline": "Reduced snow and ice is expected in Texas and is not a land management concern for this region.",
        "increase": "Any significant ice coverage increase in Texas should be verified — it may be a satellite data classification error rather than a real change.",
    },
}


_TX_POLICIES = {
    "water": {
        "decline": [
            {
                "instrument": "Shoreline Protection Rule",
                "action": (
                    "Pass a local rule that stops new construction and paving within 500 feet "
                    "of the lake or river shoreline. This gives the water body room to recover "
                    "and prevents further shrinkage caused by runoff and erosion from nearby "
                    "development. Citizens can advocate for this at the next city council meeting "
                    "by citing the measured decline rate."
                ),
            },
            {
                "instrument": "Water Conservation District Petition",
                "action": (
                    "Ask the county commissioners to petition the state for minimum water flow "
                    "protections on the affected water body. This legally requires that enough "
                    "water stays in the lake or river to maintain its health — it cannot all be "
                    "diverted for development or agricultural use."
                ),
            },
        ],
        "increase": [
            {
                "instrument": "Flood Risk Mapping Update",
                "action": (
                    "Request that the county update its flood maps to reflect the expanded water "
                    "coverage. This protects homeowners from being surprised by flood insurance "
                    "requirements and helps the county plan roads and drainage correctly."
                ),
            },
        ],
    },
    "trees": {
        "decline": [
            {
                "instrument": "Tree Replacement Rule for Developers",
                "action": (
                    "Require any builder or developer who removes large trees to plant three new "
                    "trees for every one removed, or pay into a community tree fund. Citizens "
                    "can push for this at planning and zoning meetings where new subdivision "
                    "approvals are being voted on."
                ),
            },
            {
                "instrument": "Community Tree Planting Grant",
                "action": (
                    "Apply for state forestry grant money to plant trees along streets, parks, "
                    "and school grounds in the neighborhoods with the least tree cover. This "
                    "reduces flooding, lowers summer temperatures, and improves quality of life "
                    "— and it is partially funded by the state."
                ),
            },
        ],
        "increase": [],
    },
    "grass": {
        "decline": [
            {
                "instrument": "Open Land Preservation Requirement for New Subdivisions",
                "action": (
                    "Require all new housing developments to set aside at least 20% of the land "
                    "as permanently protected native grassland or open space. This land stays "
                    "natural forever, written into the property deed so future owners cannot "
                    "build on it. Citizens can demand this condition before any subdivision "
                    "gets approved."
                ),
            },
            {
                "instrument": "Landowner Tax Incentive for Keeping Native Land",
                "action": (
                    "Work with the county appraisal district to offer property tax reductions "
                    "to landowners who agree not to develop or pave native grassland along "
                    "streams and floodplains. This keeps land open at low cost to the county "
                    "and rewards good stewardship."
                ),
            },
        ],
        "increase": [],
    },
    "flooded_vegetation": {
        "decline": [
            {
                "instrument": "Wetland No-Fill Policy",
                "action": (
                    "Pass a county resolution stating that no permit will be approved for filling "
                    "or draining wetland areas without replacing double the wetland acreage "
                    "somewhere nearby. This protects the community's natural flood protection "
                    "and water filtration at no cost to taxpayers."
                ),
            },
        ],
        "increase": [
            {
                "instrument": "Invasive Plant Removal Budget",
                "action": (
                    "Set aside an annual budget to identify and remove invasive water plants — "
                    "especially giant cane — from local waterways. Left unchecked, these plants "
                    "choke streams and make flooding significantly worse. Citizens can nominate "
                    "problem areas for treatment through a simple reporting system."
                ),
            },
        ],
    },
    "crops": {
        "decline": [
            {
                "instrument": "Farmland Protection Zone",
                "action": (
                    "Designate active farmland as a protected zone on the county land use map, "
                    "requiring a public hearing and a study of the agricultural impact before "
                    "any rezoning application is approved. This gives farmers a voice and slows "
                    "the irreversible loss of productive soil."
                ),
            },
            {
                "instrument": "Soil Restoration Cost-Share Program",
                "action": (
                    "Connect local landowners with federal cost-share programs that pay up to "
                    "75% of the cost to restore degraded farmland using cover crops and "
                    "conservation practices. This keeps land productive, reduces erosion, and "
                    "improves water quality downstream — at minimal cost to the county."
                ),
            },
        ],
        "increase": [],
    },
    "shrub_and_scrub": {
        "increase": [
            {
                "instrument": "Wildfire Safety Clearance Requirement",
                "action": (
                    "Require homeowners and property owners in high-brush areas to maintain a "
                    "100-foot cleared safety zone around any structure. Make this a condition "
                    "of building permits and certificate-of-occupancy inspections. Citizens "
                    "living near dense brush can request a free wildfire risk assessment from "
                    "the state forestry service."
                ),
            },
            {
                "instrument": "Brush Clearing Cost-Share for Landowners",
                "action": (
                    "Establish a program — co-funded with the state — where landowners can get "
                    "help paying for mechanical brush removal on their property, especially "
                    "near remaining native grassland. This reduces wildfire risk for the whole "
                    "neighborhood, not just the individual property."
                ),
            },
        ],
        "decline": [],
    },
    "built": {
        "increase": [
            {
                "instrument": "Developer Infrastructure Fee",
                "action": (
                    "Charge developers a fee for each new acre of pavement or rooftop they "
                    "create, calibrated to cover the actual cost of the extra roads, drainage, "
                    "and water infrastructure that new development requires. This ensures "
                    "growth pays for itself rather than shifting costs to existing residents "
                    "through higher taxes."
                ),
            },
            {
                "instrument": "Green Drainage Requirement for New Buildings",
                "action": (
                    "Require large new commercial or apartment developments to include "
                    "rain gardens, permeable pavement, or rooftop water capture systems that "
                    "reduce runoff by at least 25% compared to a standard paved surface. "
                    "This keeps stormwater out of streets and neighborhoods during heavy rain."
                ),
            },
        ],
        "decline": [],
    },
    "bare": {
        "increase": [
            {
                "instrument": "Construction Site Erosion Inspection",
                "action": (
                    "Direct the city or county engineer to inspect all active construction sites "
                    "for erosion controls — silt fences, straw wattles, and sediment traps. "
                    "Require any site left bare at the end of a season to be seeded with native "
                    "grasses before the next rain season. Citizens who see bare dirt washing "
                    "into a creek can report it to TCEQ online."
                ),
            },
            {
                "instrument": "Dust Control Rule",
                "action": (
                    "Adopt a simple local rule requiring anyone disturbing more than half an "
                    "acre of soil to water it down or apply stabilizer when winds are high. "
                    "This directly protects the air quality and health of nearby residents — "
                    "especially children and elderly people."
                ),
            },
        ],
        "decline": [],
    },
    "snow_and_ice": {"decline": [], "increase": []},
}


def _generate_consequence(cls: str, delta_pct: float) -> str:
    """Return a pre-written Texas-specific consequence sentence from Python templates."""
    direction = "decline" if delta_pct < 0 else "increase"
    return _TX_CONSEQUENCES.get(cls, {}).get(
        direction,
        f"Changes in {cls.replace('_', ' ')} coverage warrant continued monitoring."
    )


def _generate_policy_actions(cls: str, delta_pct: float) -> list[dict]:
    """Return pre-written Texas-specific policy actions from Python templates."""
    direction = "decline" if delta_pct < 0 else "increase"
    return _TX_POLICIES.get(cls, {}).get(direction, [])

# ---------------------------------------------------------------------------
# Threshold definitions — (direction, warning_pct, critical_pct)
# "below" = class is stressed when coverage DROPS below threshold
# "above" = class is stressed when coverage RISES above threshold
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "water":              ("below", 20.0, 12.0),
    "trees":              ("below", 15.0,  8.0),
    "grass":              ("below", 10.0,  5.0),
    "flooded_vegetation": ("below",  8.0,  4.0),
    "crops":              ("below",  8.0,  4.0),
    "shrub_and_scrub":    ("above", 25.0, 38.0),
    "built":              ("above", 28.0, 42.0),
    "bare":               ("above", 12.0, 22.0),
    "snow_and_ice":       (None,    None, None),
}
STATUS_STABLE   = "STABLE"
STATUS_MONITOR  = "MONITOR"
STATUS_ACTION   = "ACTION REQUIRED"
STATUS_CRITICAL = "CRITICAL"


def _compute_threshold_alerts(df: pd.DataFrame) -> list[dict]:
    """
    Pure-Python threshold monitor. No LLM involved.
    Returns a list of alert dicts sorted by urgency then rate magnitude.
    """
    if df.empty:
        return []

    present_pct  = [c for c in PCT_COLS if c in df.columns]
    first        = df.iloc[0]
    last         = df.iloc[-1]
    n_years      = max(1, int(last.get("year", 1)) - int(first.get("year", 0)))
    current_year = int(last.get("year", 2025))

    alerts = []
    for col in present_pct:
        cls = col.replace("lc_", "").replace("_pct", "")
        if cls not in THRESHOLDS or THRESHOLDS[cls][0] is None:
            continue

        direction, warn_thresh, crit_thresh = THRESHOLDS[cls]
        current   = float(last[col])
        start_val = float(first[col])
        rate      = (current - start_val) / n_years        # pp / yr

        # ── Status ────────────────────────────────────────────────────────
        if direction == "below":
            if current <= crit_thresh:          status = STATUS_CRITICAL
            elif current <= warn_thresh:        status = STATUS_ACTION
            elif rate < 0:                      status = STATUS_MONITOR
            else:                               status = STATUS_STABLE
        else:
            if current >= crit_thresh:          status = STATUS_CRITICAL
            elif current >= warn_thresh:        status = STATUS_ACTION
            elif rate > 0:                      status = STATUS_MONITOR
            else:                               status = STATUS_STABLE

        if status == STATUS_STABLE:
            continue

        # ── Years / year of crossing ───────────────────────────────────────
        years_to_critical = None
        threshold_year    = None
        if direction == "below" and rate < 0 and current > crit_thresh:
            years_to_critical = (current - crit_thresh) / abs(rate)
            threshold_year    = current_year + int(years_to_critical)
        elif direction == "above" and rate > 0 and current < crit_thresh:
            years_to_critical = (crit_thresh - current) / rate
            threshold_year    = current_year + int(years_to_critical)

        # ── Progress bar 0–100 % toward critical threshold ─────────────────
        total_span = abs(start_val - crit_thresh) or 1.0
        if direction == "below":
            progress = int(min(100, max(0, (start_val - current) / total_span * 100)))
        else:
            progress = int(min(100, max(0, (current - start_val) / total_span * 100)))

        # ── Acceleration ───────────────────────────────────────────────────
        mid = len(df) // 2
        if mid > 0 and len(df) > 2:
            r1 = (float(df.iloc[mid][col]) - start_val) / max(1, mid)
            r2 = (current - float(df.iloc[mid][col])) / max(1, len(df) - mid)
            accel = ("accelerating" if abs(r2) > abs(r1) * 1.15
                     else "decelerating" if abs(r2) < abs(r1) * 0.85
                     else "steady")
        else:
            accel = "steady"

        alerts.append({
            "land_cover":        cls,
            "direction":         direction,
            "current_pct":       round(current, 1),
            "start_pct":         round(start_val, 1),
            "warn_threshold":    warn_thresh,
            "crit_threshold":    crit_thresh,
            "rate_pp_per_yr":    round(rate, 2),
            "years_to_critical": round(years_to_critical, 1) if years_to_critical else None,
            "threshold_year":    threshold_year,
            "progress":          progress,
            "status":            status,
            "acceleration":      accel,
        })

    order = {STATUS_CRITICAL: 0, STATUS_ACTION: 1, STATUS_MONITOR: 2}
    alerts.sort(key=lambda x: (order.get(x["status"], 3), -abs(x["rate_pp_per_yr"])))
    return alerts

# Gemini model to use
_MODEL_NAME = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Pre-compute hard statistics from the forecast DataFrame
# ---------------------------------------------------------------------------
def _compute_data_stats(df: pd.DataFrame) -> dict:
    """
    Derive hard numbers from the forecast before touching the LLM.
    These are injected verbatim into the prompt so Gemini must cite real figures.
    """
    present_pct = [c for c in PCT_COLS if c in df.columns]
    if df.empty or not present_pct:
        return {}

    first = df.iloc[0]
    last  = df.iloc[-1]
    n_years = max(1, int(last.get("year", 1)) - int(first.get("year", 0)))

    stats = {}
    for col in present_pct:
        cls = col.replace("lc_", "").replace("_pct", "")
        v0  = float(first[col])
        v1  = float(last[col])
        delta = v1 - v0
        rate  = delta / n_years  # pp per year

        # Find the year the class first drops below 10 % (water/trees) or exceeds 25 % (built/shrub)
        threshold_year = None
        low_classes  = {"water", "trees", "flooded_vegetation"}
        high_classes = {"built", "shrub_and_scrub", "bare"}
        if cls in low_classes and v0 > 10.0:
            for _, row in df.iterrows():
                if float(row.get(col, v0)) <= 10.0:
                    threshold_year = int(row.get("year", "?"))
                    break
        elif cls in high_classes and v0 < 25.0:
            for _, row in df.iterrows():
                if float(row.get(col, v0)) >= 25.0:
                    threshold_year = int(row.get("year", "?"))
                    break

        # Detect acceleration: compare first-half rate vs second-half rate
        mid = len(df) // 2
        if mid > 0 and len(df) > 2:
            rate_first = (float(df.iloc[mid][col]) - v0) / max(1, mid)
            rate_second = (v1 - float(df.iloc[mid][col])) / max(1, len(df) - mid)
            accel = "accelerating" if abs(rate_second) > abs(rate_first) * 1.15 else \
                    "decelerating" if abs(rate_second) < abs(rate_first) * 0.85 else "steady"
        else:
            accel = "steady"

        if abs(delta) >= _CHANGE_THRESHOLD_PP:
            stats[cls] = {
                "start_pct":       round(v0, 1),
                "end_pct":         round(v1, 1),
                "total_delta_pp":  round(delta, 1),
                "rate_pp_per_yr":  round(rate, 2),
                "acceleration":    accel,
                "threshold_year":  threshold_year,
            }

    return stats


def _format_stats_for_prompt(stats: dict, label: str, year_start: int, year_end: int) -> str:
    """Render stats as a tightly-structured fact block the LLM must reference."""
    lines = [
        f"REGION: {label}",
        f"FORECAST WINDOW: {year_start} – {year_end}",
        "",
        "PRE-COMPUTED CLASS STATISTICS (use these exact figures — do not invent numbers):",
    ]
    for cls, s in sorted(stats.items(), key=lambda x: abs(x[1]["total_delta_pp"]), reverse=True):
        thresh = f"  ⚠ crosses threshold in {s['threshold_year']}" if s["threshold_year"] else ""
        lines.append(
            f"  {cls.upper():25s}  "
            f"now {s['start_pct']:5.1f}%  →  {s['end_pct']:5.1f}%  "
            f"({s['total_delta_pp']:+.1f} pp total, {s['rate_pp_per_yr']:+.2f} pp/yr, {s['acceleration']}){thresh}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def list_available_forecasts(forecasts_dir: str = FORECASTS_DIR) -> Dict[str, Path]:
    """Return a dict mapping city label → CSV path for all forecasts on disk."""
    d = Path(forecasts_dir)
    if not d.exists():
        return {}
    return {p.stem: p for p in sorted(d.glob("*.csv"))}


def load_forecast_csv(csv_path) -> pd.DataFrame:
    """Load a forecast CSV and return a tidy DataFrame."""
    df = pd.read_csv(csv_path)
    # Keep only columns we care about
    keep = ["year"] + [c for c in PCT_COLS if c in df.columns]
    if "city" in df.columns:
        keep.append("city")
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------
def build_trajectory_summary(label: str, df: pd.DataFrame) -> str:
    """
    Build a plain-language paragraph describing the 10-year land-cover trajectory.

    Only classes whose absolute change exceeds _CHANGE_THRESHOLD_PP are mentioned.
    """
    present_pct = [c for c in PCT_COLS if c in df.columns]
    if df.empty or not present_pct:
        return f"No forecast data available for {label}."

    first = df.iloc[0]
    last  = df.iloc[-1]
    year_start = int(first["year"]) if "year" in df.columns else "?"
    year_end   = int(last["year"])  if "year" in df.columns else "?"

    changes = []
    for col in present_pct:
        cls_name = col.replace("lc_", "").replace("_pct", "").replace("_", " ")
        delta = float(last[col]) - float(first[col])
        if abs(delta) >= _CHANGE_THRESHOLD_PP:
            direction = "increase" if delta > 0 else "decrease"
            changes.append((abs(delta), cls_name, direction, delta))

    # Sort by magnitude descending
    changes.sort(key=lambda x: x[0], reverse=True)

    if not changes:
        return (
            f"The 10-year forecast for {label} ({year_start}–{year_end}) shows "
            "minimal projected change across all land-cover classes."
        )

    parts = []
    for _, cls_name, direction, delta in changes:
        parts.append(f"{cls_name} ({delta:+.1f} pp {direction})")

    summary = (
        f"For {label}, the LSTM model projects the following land-cover shifts "
        f"between {year_start} and {year_end}: {'; '.join(parts)}."
    )
    return summary


def _build_raw_table(df: pd.DataFrame) -> str:
    """Format year-by-year pct columns as a compact text table."""
    present_pct = [c for c in PCT_COLS if c in df.columns]
    cols = (["year"] + present_pct) if "year" in df.columns else present_pct
    return df[cols].to_string(index=False)


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------
def _load_api_key() -> Optional[str]:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def _safe_fallback(label: str, trajectory: str) -> Dict[str, Any]:
    """Return a minimal structured brief when the API is unreachable."""
    return {
        "city": label,
        "headline": "API unavailable — forecast summary only.",
        "trajectory_overview": trajectory,
        "recommendations": [
            {
                "priority": "Medium",
                "land_cover": "general",
                "action": (
                    "Review the trajectory summary above and consult a local "
                    "land-use planner for policy recommendations."
                ),
            }
        ],
        "risks": "Unable to assess risks without AI analysis.",
        "data_note": (
            "Policy brief generated without Gemini. "
            "Connect to the internet and provide a valid API key for full analysis."
        ),
    }

# ---------------------------------------------------------------------------
# Consequence prompt — Gemini writes ONE sentence per class, nothing else
# ---------------------------------------------------------------------------
_CONSEQUENCE_PROMPT = """You are a Texas regional land-use analyst.

Region: {label}
The following land-cover trends were measured from satellite data:
{stats_block}

For EACH class listed, write exactly ONE sentence describing a specific, concrete
physical or economic consequence for local residents or infrastructure if this trend
is not interrupted. Rules you must follow:
- Do NOT mention percentages or numbers of any kind
- Name a specific asset, system, or group at risk (e.g. "the municipal water intake",
  "row-crop livelihoods along FM 3464", "the riparian corridor feeding Falcon Reservoir")
- Do NOT use generic phrases like "ecological impact", "environmental concerns",
  "biodiversity loss", or "sustainability challenges"
- Be specific to the geography: semi-arid Texas borderland / rapidly suburbanising
  North Texas prairie — pick whichever applies to {label}

Return ONLY this JSON with no markdown fences:
{{"consequences": [{{"land_cover": "<exact class name from input>", "consequence": "<one sentence>"}}]}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_policy_brief(
    label: str,
    df: pd.DataFrame,
    model_name: str = _MODEL_NAME,
) -> Tuple[Any, Optional[str]]:
    """
    Returns (brief_dict, error_message_or_None).
    All statistics, consequences, and policy actions are computed in pure Python.
    Gemini is called only to optionally improve the consequence sentence wording.
    """
    stats  = _compute_data_stats(df)
    alerts = _compute_threshold_alerts(df)   # kept for status badge lookup only

    year_start   = int(df.iloc[0]["year"])  if "year" in df.columns else 0
    current_year = date.today().year         # always 2026 — never the forecast end year

    # ── Pure-Python dossier: consequences + policies, zero LLM ───────────
    dossier = []
    for cls, s in sorted(stats.items(),
                         key=lambda x: abs(x[1]["rate_pp_per_yr"]), reverse=True):
        delta = s["total_delta_pp"]
        dossier.append({
            "land_cover":     cls,
            "consequence":    _generate_consequence(cls, delta),
            "policies":       _generate_policy_actions(cls, delta),
            "start_pct":      s["start_pct"],
            "end_pct":        s["end_pct"],
            "total_delta_pp": delta,
            "rate_pp_per_yr": s["rate_pp_per_yr"],
            "acceleration":   s["acceleration"],
            "threshold_year": s.get("threshold_year"),
        })

    # ── Optional Gemini enhancement: replace consequence text only ────────
    api_key = _load_api_key()
    api_err = None

    if api_key:
        try:
            socket.setdefaulttimeout(5)
            socket.getaddrinfo("generativelanguage.googleapis.com", 443)

            year_end    = int(df.iloc[-1]["year"]) if "year" in df.columns else current_year
            stats_block = _format_stats_for_prompt(stats, label, year_start, year_end)

            import google.generativeai as genai
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel(model_name)
            prompt = _CONSEQUENCE_PROMPT.format(label=label, stats_block=stats_block)
            response = m.generate_content(prompt, request_options={"timeout": 45})
            text = re.sub(r"^```(?:json)?\s*", "",
                          (response.text or "").strip(), flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text.strip())
            try:
                parsed   = json.loads(text)
                gem_map  = {i["land_cover"]: i["consequence"]
                            for i in parsed.get("consequences", [])}
                for item in dossier:
                    enhanced = gem_map.get(item["land_cover"], "")
                    if enhanced and len(enhanced) > 20:
                        item["consequence"] = enhanced
            except json.JSONDecodeError:
                api_err = "Gemini response unparseable — built-in consequences used."
        except OSError:
            api_err = "Offline — built-in consequence descriptions used."
        except Exception as exc:
            api_err = f"Gemini optional step failed: {exc} — built-in consequences used."

    brief = {
        "city":             label,
        "threshold_alerts": alerts,
        "dossier":          dossier,
        "data_note": (
            f"Satellite record: {year_start}–{current_year}. "
            f"LSTM forecast: {current_year + 1}–{current_year + 10}. "
            "Policy instruments reference Texas statute as of 2026. "
            "Threshold crossings assume constant rate of change."
        ),
    }
    return brief, api_err
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="TerraTrace — policy brief generator")
    p.add_argument(
        "--forecasts-dir", default=FORECASTS_DIR,
        help="Directory containing forecast CSVs (default: outputs/forecasts)"
    )
    p.add_argument(
        "--city", default=None,
        help="Stem of the forecast CSV to process (omit to process all)"
    )
    p.add_argument(
        "--out-dir", default="outputs/briefs",
        help="Directory to save JSON briefs (default: outputs/briefs)"
    )
    return p.parse_args()


def main():
    args = _parse_args()
    forecasts = list_available_forecasts(args.forecasts_dir)

    if not forecasts:
        log.error("No forecast CSVs found in %s — run forecast.py first.", args.forecasts_dir)
        return

    if args.city:
        if args.city not in forecasts:
            log.error("City '%s' not found. Available: %s", args.city, list(forecasts))
            return
        forecasts = {args.city: forecasts[args.city]}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, csv_path in forecasts.items():
        log.info("Processing %s …", label)
        df = load_forecast_csv(csv_path)
        brief, err = get_policy_brief(label, df)

        if err:
            log.warning("[%s] API warning: %s", label, err)

        out_path = out_dir / f"{label}_brief.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(brief, f, indent=2, ensure_ascii=False)
        log.info("[%s] Brief saved → %s", label, out_path)

    log.info("Done. Briefs written to %s", out_dir)


if __name__ == "__main__":
    main()
