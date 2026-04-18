# policy.py — TerraTrace Step 3
# Reads a city's forecast CSV, builds a plain-language summary of the
# land change trajectory, calls the Claude API, and returns a structured
# policy brief with recommendations specific to that county's projected changes.

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
FORECASTS_DIR = "outputs/forecasts"
CLEANED_DIR   = "outputs/cleaned"

DW_CLASSES = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice",
]

FEATURE_COLS = (
    [f"lc_{cls}_pct"  for cls in DW_CLASSES] +
    [f"lc_{cls}_conf" for cls in DW_CLASSES]
)

# ── Policy prompt ─────────────────────────────────────────────────────────────
POLICY_PROMPT = """\
You are an environmental policy analyst for Texas counties.
Using ONLY the 10-year land-cover forecast data provided below, write a \
structured policy brief.

Do NOT invent statistics, reference other locations, or add information \
not present in the data.

DATA:
{data}

Return JSON strictly in this shape (no markdown fences, no extra keys):
{{
  "city": "<city_label>",
  "forecast_period": "<start_year>–<end_year>",
  "key_trends": [
    "<one concise trend sentence per notable class change>"
  ],
  "environmental_risks": [
    "<specific risk tied to the data, e.g. urban heat island from +X% built>"
  ],
  "policy_recommendations": [
    {{
      "priority": "high",
      "action": "<concrete action>",
      "rationale": "<1-sentence rationale tied to the data>"
    }}
  ],
  "executive_summary": "<2–3 plain-language sentences summarising the forecast \
and top recommendation>"
}}
"""

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_forecast(city_label: str, forecasts_dir: str = FORECASTS_DIR) -> pd.DataFrame | None:
    path = Path(forecasts_dir) / f"{city_label}.csv"
    if not path.exists():
        log.error("Forecast CSV not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        log.info("Loaded forecast for %s — %d rows", city_label, len(df))
        return df
    except Exception as exc:
        log.error("Could not read %s: %s", path, exc)
        return None


def load_historical(city_label: str, cleaned_dir: str = CLEANED_DIR) -> pd.DataFrame | None:
    path = Path(cleaned_dir) / f"{city_label}.csv"
    if not path.exists():
        log.warning("No cleaned historical CSV for %s", city_label)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("Could not read historical %s: %s", path, exc)
        return None


# ── Trajectory builder ────────────────────────────────────────────────────────
def build_trajectory_summary(city_label: str, forecast_df: pd.DataFrame) -> dict:
    """
    Return a compact dict describing per-class change over the forecast window.
    Only classes that change by more than 1 pp are included to keep the
    Claude prompt concise.
    """
    present_pct = [c for c in [f"lc_{cls}_pct" for cls in DW_CLASSES] if c in forecast_df.columns]

    first = forecast_df.iloc[0]
    last  = forecast_df.iloc[-1]

    changes = {}
    for col in present_pct:
        cls_name = col.replace("lc_", "").replace("_pct", "")
        delta = float(last[col]) - float(first[col])
        if abs(delta) > 1.0:          # only meaningful shifts
            changes[cls_name] = {
                "start_pct": round(float(first[col]), 2),
                "end_pct":   round(float(last[col]),  2),
                "change_pp": round(delta, 2),
            }

    return {
        "city":            city_label,
        "forecast_period": f"{int(forecast_df['year'].iloc[0])}–{int(forecast_df['year'].iloc[-1])}",
        "notable_changes": changes,
    }


# ── Claude API call ───────────────────────────────────────────────────────────
def _load_api_key() -> str | None:
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # Try Streamlit secrets as fallback
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def call_claude_for_policy(
    trajectory: dict,
    model: str = "claude-opus-4-5",
) -> tuple[str | None, str | None]:
    """
    Call the Claude API with the trajectory summary.

    Returns (response_text, error_message).
    error_message is None on success.
    """
    api_key = _load_api_key()
    if not api_key:
        msg = (
            "No Anthropic API key found. "
            "Set ANTHROPIC_API_KEY in your environment or Streamlit secrets."
        )
        log.error("call_claude_for_policy: %s", msg)
        return None, msg

    prompt = POLICY_PROMPT.format(data=json.dumps(trajectory, indent=2))

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
        log.info("call_claude_for_policy: success (%d chars)", len(text))
        return text, None

    except Exception as exc:
        exc_str = str(exc).lower()
        if "429" in exc_str or "rate" in exc_str or "quota" in exc_str:
            msg = "Claude API rate limit reached. Wait a moment and retry."
        elif "401" in exc_str or "403" in exc_str or "api_key" in exc_str:
            msg = "Claude API key invalid or lacks permission."
        elif "timeout" in exc_str or "timed out" in exc_str:
            msg = "Claude API request timed out. Check your connection."
        elif "503" in exc_str or "unavailable" in exc_str:
            msg = "Claude API temporarily unavailable. Try again in a few minutes."
        else:
            msg = f"Claude API error: {exc}"
        log.error("call_claude_for_policy: %s", msg)
        return None, msg


# ── JSON parser / normaliser ──────────────────────────────────────────────────
def _parse_policy_json(text: str) -> dict | None:
    clean = text.strip()
    # Strip markdown fences if present
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", clean)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _safe_policy_brief(trajectory: dict, error: str) -> dict:
    """Fallback policy brief when Claude is unavailable."""
    changes = trajectory.get("notable_changes", {})
    trends = [
        f"{cls.replace('_',' ').title()} changed by {info['change_pp']:+.1f} pp "
        f"({info['start_pct']}% → {info['end_pct']}%)"
        for cls, info in changes.items()
    ]
    return {
        "city":           trajectory.get("city", "Unknown"),
        "forecast_period": trajectory.get("forecast_period", ""),
        "key_trends":     trends or ["Insufficient data for trend analysis."],
        "environmental_risks": ["Could not be assessed — Claude API unavailable."],
        "policy_recommendations": [
            {
                "priority":  "N/A",
                "action":    "Manually review the forecast CSV.",
                "rationale": error,
            }
        ],
        "executive_summary": (
            f"API error: {error}. "
            "Review the raw forecast data for manual analysis."
        ),
    }


# ── Public entry point ────────────────────────────────────────────────────────
def generate_policy_brief(
    city_label: str,
    forecasts_dir: str = FORECASTS_DIR,
) -> tuple[dict, str | None]:
    """
    High-level function: load forecast → build trajectory → call Claude → return brief.

    Returns (policy_brief_dict, error_message).
    error_message is None on success.
    """
    df = load_forecast(city_label, forecasts_dir)
    if df is None:
        msg = f"No forecast CSV found for '{city_label}' in {forecasts_dir}."
        return _safe_policy_brief({"city": city_label, "notable_changes": {}}, msg), msg

    trajectory  = build_trajectory_summary(city_label, df)
    raw_text, err = call_claude_for_policy(trajectory)

    if err:
        return _safe_policy_brief(trajectory, err), err

    parsed = _parse_policy_json(raw_text)
    if not parsed:
        msg = "Claude returned an unparseable response."
        log.warning("generate_policy_brief: %s\nRaw: %s", msg, raw_text[:300])
        return _safe_policy_brief(trajectory, msg), msg

    log.info("generate_policy_brief: OK for %s", city_label)
    return parsed, None


def list_available_cities(forecasts_dir: str = FORECASTS_DIR) -> list[str]:
    return [p.stem for p in sorted(Path(forecasts_dir).glob("*.csv"))]


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="TerraTrace — generate policy brief via Claude")
    p.add_argument("--city",          default=None, help="City label (CSV stem)")
    p.add_argument("--forecasts-dir", default=FORECASTS_DIR)
    p.add_argument("--list",          action="store_true", help="List available forecast cities")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.list:
        cities = list_available_cities(args.forecasts_dir)
        if cities:
            print("\n".join(cities))
        else:
            print(f"No forecast CSVs found in {args.forecasts_dir}")
        return

    targets = [args.city] if args.city else list_available_cities(args.forecasts_dir)
    if not targets:
        log.error("No cities to process. Run forecast.py first.")
        sys.exit(1)

    for label in targets:
        brief, err = generate_policy_brief(label, args.forecasts_dir)
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        print(json.dumps(brief, indent=2))
        if err:
            log.warning("[%s] completed with error: %s", label, err)


if __name__ == "__main__":
    main()