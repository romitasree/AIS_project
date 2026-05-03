
import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths (mirror preprocess.py conventions) ──────────────────────────────────
CLEANED_DIR    = "outputs/cleaned"
MODEL_PATH     = "outputs/model.keras"
SCALER_PATH    = "outputs/scaler.pkl"
FORECASTS_DIR  = "outputs/forecasts"

DW_CLASSES = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"
]

FEATURE_COLS = (
    [f"lc_{cls}_pct"  for cls in DW_CLASSES] +
    [f"lc_{cls}_conf" for cls in DW_CLASSES]
)

LOOKBACK      = 3
FORECAST_YEARS = 10


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_model(model_path):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        log.info("Model loaded from %s  (input shape %s)", model_path, model.input_shape)
        return model
    except Exception as exc:
        log.error("Could not load model: %s", exc)
        sys.exit(1)


def load_scaler(scaler_path):
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        log.info("Scaler loaded from %s", scaler_path)
        return scaler
    except Exception as exc:
        log.error("Could not load scaler: %s", exc)
        sys.exit(1)


def load_cleaned_cities(cleaned_dir):
    csvs = sorted(Path(cleaned_dir).glob("*.csv"))
    if not csvs:
        log.error("No cleaned CSVs found in %s", cleaned_dir)
        sys.exit(1)
    cities = {}
    for path in csvs:
        try:
            df = pd.read_csv(path)
            cities[path.stem] = df
            log.info("Loaded %-40s  %d rows", path.name, len(df))
        except Exception as exc:
            log.warning("Could not load %s: %s", path.name, exc)
    log.info("Loaded %d cleaned city files", len(cities))
    return cities


# ── Core forecast logic ───────────────────────────────────────────────────────
def seed_window(df, scaler, lookback):
    """
    Returns the last `lookback` rows as a scaled numpy array of shape
    (lookback, n_features), using only columns present in FEATURE_COLS.
    """
    present = [c for c in FEATURE_COLS if c in df.columns]
    if len(df) < lookback:
        log.warning("Only %d rows — need at least %d for a seed window", len(df), lookback)
        return None, present
    tail = df[present].iloc[-lookback:].values.astype(np.float32)
    scaled = scaler.transform(tail)
    return scaled, present


def forecast_city(label, df, model, scaler, lookback, forecast_years):
    """
    Autoregressively rolls the model forward `forecast_years` steps.
    Returns a DataFrame with columns: year, <feature_cols…>
    """
    window, present = seed_window(df, scaler, lookback)
    if window is None:
        return None

    last_year = int(df["year"].iloc[-1]) if "year" in df.columns else 2023

    # Pad missing features with zeros so the scaler/model always sees full width
    n_full = len(FEATURE_COLS)
    n_present = len(present)

    # Build a full-width buffer; columns not in the cleaned data stay 0
    present_idx = [FEATURE_COLS.index(c) for c in present]

    def to_full(partial_row):
        full = np.zeros(n_full, dtype=np.float32)
        full[present_idx] = partial_row
        return full

    # Seed window → full width
    full_window = np.stack([to_full(row) for row in window])   # (lookback, n_full)

    predictions_scaled = []
    years = []

    for step in range(forecast_years):
        x = full_window[np.newaxis, ...]                        # (1, lookback, n_full)
        pred_scaled = model.predict(x, verbose=0)[0]            # (n_full,)
        pred_scaled = np.clip(pred_scaled, 0.0, 1.0)            # keep in scaler range

        predictions_scaled.append(pred_scaled)
        years.append(last_year + step + 1)

        # Roll the window forward
        full_window = np.vstack([full_window[1:], pred_scaled[np.newaxis, :]])

    # Inverse-transform back to original feature space
    predictions_scaled = np.array(predictions_scaled)           # (forecast_years, n_full)
    predictions = scaler.inverse_transform(predictions_scaled)  # (forecast_years, n_full)

    out_df = pd.DataFrame(predictions, columns=FEATURE_COLS)
    out_df.insert(0, "year", years)
    out_df["city"] = label

    log.info("[%s] Forecast complete — years %d→%d", label, years[0], years[-1])
    return out_df


# ── I/O ───────────────────────────────────────────────────────────────────────
def save_forecast(label, forecast_df, forecasts_dir, dry_run=False):
    if dry_run:
        log.info("[%s] dry-run — skipping write", label)
        return
    out_path = Path(forecasts_dir) / f"{label}.csv"
    forecast_df.to_csv(out_path, index=False)
    log.info("[%s] Forecast → %s", label, out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="TerraTrace — autoregressive LSTM forecast")
    p.add_argument("--cleaned-dir",    default=CLEANED_DIR)
    p.add_argument("--model-path",     default=MODEL_PATH)
    p.add_argument("--scaler-path",    default=SCALER_PATH)
    p.add_argument("--forecasts-dir",  default=FORECASTS_DIR)
    p.add_argument("--lookback",       type=int, default=LOOKBACK)
    p.add_argument("--forecast-years", type=int, default=FORECAST_YEARS)
    p.add_argument("--city",           default=None,
                   help="Forecast a single city only (stem of its CSV filename)")
    p.add_argument("--dry-run",        action="store_true",
                   help="Run everything but don't write output CSVs")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.forecasts_dir).mkdir(parents=True, exist_ok=True)

    model  = load_model(args.model_path)
    scaler = load_scaler(args.scaler_path)
    cities = load_cleaned_cities(args.cleaned_dir)

    if args.city:
        if args.city not in cities:
            log.error("City '%s' not found in %s", args.city, args.cleaned_dir)
            sys.exit(1)
        cities = {args.city: cities[args.city]}

    results = {}
    skipped = []

    for label, df in cities.items():
        forecast_df = forecast_city(
            label, df, model, scaler,
            lookback=args.lookback,
            forecast_years=args.forecast_years,
        )
        if forecast_df is None:
            skipped.append(label)
            continue
        save_forecast(label, forecast_df, args.forecasts_dir, dry_run=args.dry_run)
        results[label] = forecast_df

    log.info("Forecasted %d/%d cities", len(results), len(cities))
    if skipped:
        log.warning("Skipped (insufficient data): %s", skipped)
    log.info("Next: python3 policy.py")


if __name__ == "__main__":
    main()