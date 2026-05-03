"""
retrain_per_region.py
=====================
Trains one LSTM + scaler per region using only that region's own
cleaned CSV from pixel_output/cleaned/.

Run from the modules directory:
    python retrain_per_region.py

Outputs into pixel_output/:
    model_<key>.keras
    scaler_<key>.pkl

Add --force to retrain regions that already have saved models.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
# modules/retrain_per_region.py  →  parents[0]=modules  →  parents[1]=terratrace
ROOT             = Path(__file__).resolve().parents[1]
PIXEL_OUTPUT_DIR = ROOT / "pixel_output"
CLEANED_DIR      = PIXEL_OUTPUT_DIR / "cleaned"
SAVE_DIR         = PIXEL_OUTPUT_DIR   # save per-region models here

# ── Region map: key → CSV stem in pixel_output/cleaned/ ──────────────────────
# Add new regions here as new CSVs become available
REGION_CSV = {
    "celina":       "lat33_314479_lonn96_77655",
    "falcon_lake":  "lat26_667417_lonn99_159667",
    "dallas":       "lat32_7767_lonn96_797",
    "austin":       "lat30_2672_lonn97_7431",
    "houston":      "lat29_7604_lonn95_3698",
    "san_antonio":  "lat29_4241_lonn98_4936",
    "fort_worth":   "lat32_7555_lonn97_3308",
    "frisco":       "lat33_1507_lonn96_8236",
    "mckinney":     "lat33_1976_lonn96_6153",
}

# ── Feature columns ───────────────────────────────────────────────────────────
PRED_CLASSES = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare",
]
FEATURE_COLS = (
    [f"lc_{c}_pct" for c in PRED_CLASSES] +
    [f"lc_{c}_conf" for c in PRED_CLASSES]
)
N_FEATURES = len(FEATURE_COLS)  # 16
LOOKBACK   = 3


def make_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_model(n_features: int, lookback: int):
    import tensorflow as tf
    from tensorflow import keras

    inputs  = keras.Input(shape=(lookback, n_features))
    x       = keras.layers.LSTM(32, return_sequences=False)(inputs)
    x       = keras.layers.Dropout(0.1)(x)
    # Linear activation — lets the model predict genuine decline/growth
    # without sigmoid's pull toward the center of the normalized range
    outputs = keras.layers.Dense(n_features, activation="linear")(x)
    model   = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse")
    return model


def train_region(region_key: str, csv_stem: str, force: bool = False) -> bool:
    model_path  = SAVE_DIR / f"model_{region_key}.keras"
    scaler_path = SAVE_DIR / f"scaler_{region_key}.pkl"

    if model_path.exists() and scaler_path.exists() and not force:
        print(f"  [SKIP] {region_key}: already trained. Use --force to retrain.")
        return True

    # Prefer cleaned CSV; fall back to raw pixel_output CSV
    csv_path = CLEANED_DIR / f"{csv_stem}.csv"
    if not csv_path.exists():
        csv_path = PIXEL_OUTPUT_DIR / f"{csv_stem}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {region_key}: CSV not found ({csv_stem}.csv)")
        return False

    df = pd.read_csv(csv_path)

    # Build year column if absent
    if "year" not in df.columns:
        for col in ["end", "start"]:
            if col in df.columns:
                df["year"] = pd.to_datetime(df[col], errors="coerce").dt.year
                break

    df = df.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)

    # Auto-build confidence columns if absent
    for c in PRED_CLASSES:
        conf_col, pct_col = f"lc_{c}_conf", f"lc_{c}_pct"
        if conf_col not in df.columns and pct_col in df.columns:
            df[conf_col] = df[pct_col] / 100.0

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [SKIP] {region_key}: missing columns {missing}")
        return False

    if len(df) < LOOKBACK + 2:
        print(f"  [SKIP] {region_key}: only {len(df)} rows (need ≥ {LOOKBACK + 2})")
        return False

    data = df[FEATURE_COLS].astype(np.float32).values

    # Fit scaler on THIS region's data only — no cross-location bleed
    scaler      = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data).astype(np.float32)

    X, y = make_sequences(data_scaled, LOOKBACK)
    if len(X) < 2:
        print(f"  [SKIP] {region_key}: not enough sequences ({len(X)})")
        return False

    print(f"  [TRAIN] {region_key}: {len(df)} rows → {len(X)} sequences "
          f"(CSV: {csv_path.name})")

    model = build_model(N_FEATURES, LOOKBACK)
    import tensorflow as tf
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=20, restore_best_weights=True
    )
    model.fit(
        X, y,
        epochs=200,
        batch_size=max(1, len(X) // 2),
        verbose=0,
        callbacks=[early_stop],
        validation_split=0.15 if len(X) >= 5 else 0.0,
    )

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"  [SAVED] {model_path.name}  +  {scaler_path.name}")
    return True


if __name__ == "__main__":
    import sys
    import tensorflow as tf

    force = "--force" in sys.argv
    if force:
        print("--force: retraining all regions from scratch\n")

    print(f"TensorFlow {tf.__version__}")
    print(f"Reading CSVs from : {CLEANED_DIR}")
    print(f"Saving models to  : {SAVE_DIR}\n")

    success = 0
    for key, stem in REGION_CSV.items():
        print(f"→ {key}")
        if train_region(key, stem, force=force):
            success += 1

    print(f"\nDone. {success}/{len(REGION_CSV)} models ready.")
    print(f"\nPer-region files saved:")
    for key in REGION_CSV:
        mp = SAVE_DIR / f"model_{key}.keras"
        sp = SAVE_DIR / f"scaler_{key}.pkl"
        status = "✓" if mp.exists() and sp.exists() else "✗"
        print(f"  {status} {key}")