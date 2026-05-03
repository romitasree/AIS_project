import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

INPUT_DIR    = "outputs"
CLEANED_DIR  = "outputs/cleaned"
WINDOWS_PATH = "outputs/windows.npz"
SCALER_PATH  = "outputs/scaler.pkl"
REPORT_PATH  = "outputs/report.txt"

DW_CLASSES = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"
]

FEATURE_COLS = (
    [f"lc_{cls}_pct"  for cls in DW_CLASSES] +
    [f"lc_{cls}_conf" for cls in DW_CLASSES]
)

LOOKBACK          = 3
MISSING_THRESHOLD = 0.20
TRAIN_FRAC        = 0.80
VAL_FRAC          = 0.10


def load_csvs(input_dir):
    csvs = sorted(Path(input_dir).glob("*.csv"))
    if not csvs:
        log.error("No CSVs found in %s", input_dir)
        sys.exit(1)
    cities = {}
    for path in csvs:
        try:
            df = pd.read_csv(path)
            cities[path.stem] = df
            log.info("Loaded %-40s  %d rows", path.name, len(df))
        except Exception as exc:
            log.warning("Could not load %s: %s", path.name, exc)
    log.info("Loaded %d CSVs total", len(cities))
    return cities


def validate_city(label, df):
    present      = [c for c in FEATURE_COLS if c in df.columns]
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]

    per_col = {}
    for col in present:
        n = int(df[col].isnull().sum())
        per_col[col] = {"count": n, "pct": round(n / len(df) * 100, 1)}

    total_cells = len(df) * len(present) if present else 1
    total_nulls = df[present].isnull().sum().sum() if present else 0
    total_pct   = round(total_nulls / total_cells * 100, 2)
    dupes = int(df.duplicated(subset=["period"]).sum()) if "period" in df.columns else 0

    return {
        "rows": len(df),
        "missing_per_col": per_col,
        "total_missing_pct": total_pct,
        "flagged": total_pct > MISSING_THRESHOLD * 100,
        "duplicate_years": dupes,
        "missing_cols": missing_cols,
    }


def validate_all(cities):
    return {label: validate_city(label, df) for label, df in cities.items()}


def _bar(pct, width=20):
    filled = int(round(pct / 100 * width))
    return f"[{'=' * filled}{'#' * (width - filled)}] {pct:5.1f}%"


def print_report(reports, output_path=None):
    lines = []

    def emit(line=""):
        lines.append(line)
        print(line)

    div = "─" * 70
    flagged = [l for l, r in reports.items() if r["flagged"]]
    emit(div)
    emit("  TerraTrace — Validation Report")
    emit(div)
    emit(f"  Cities: {len(reports)}   Flagged: {len(flagged)}")
    emit()

    for label, r in sorted(reports.items()):
        status = "FLAGGED" if r["flagged"] else "OK"
        emit(div)
        emit(f"  {label:<40} {status}")
        emit(f"  Rows: {r['rows']}   Duplicate years: {r['duplicate_years']}")
        if r["missing_cols"]:
            emit(f"  Missing cols: {r['missing_cols']}")
        emit()
        for col, info in r["missing_per_col"].items():
            marker = " ←" if info["pct"] > 20 else ""
            emit(f"  {col:<35}  {info['count']:>3} rows  {_bar(info['pct'])}{marker}")
        emit()
        emit(f"  Overall: {_bar(r['total_missing_pct'])}")
        emit()

    emit(div)
    if flagged:
        emit("  Cities exceeding missing threshold:")
        for l in flagged:
            emit(f"    • {l}  ({reports[l]['total_missing_pct']}%)")
    else:
        emit("  All cities within threshold.")
    emit(div)

    if output_path:
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        log.info("Report -> %s", output_path)


def clean_city(label, df):
    present = [c for c in FEATURE_COLS if c in df.columns]

    if "period" in df.columns:
        df = df.sort_values("period").drop_duplicates(subset=["period"], keep="first").reset_index(drop=True)
    df[present] = df[present].ffill().bfill()

    remaining = df[present].isnull().sum().sum()
    total     = len(df) * len(present)

    if total > 0 and remaining / total > MISSING_THRESHOLD:
        log.warning("[%s] %.1f%% NaN after fill — excluding", label, remaining / total * 100)
        return None

    if remaining > 0:
        df = df.dropna(subset=present).reset_index(drop=True)

    log.info("[%s] Cleaned -> %d rows", label, len(df))
    return df


def clean_all(cities, cleaned_dir, dry_run=False):
    cleaned = {}
    Path(cleaned_dir).mkdir(parents=True, exist_ok=True)
    for label, df in cities.items():
        result = clean_city(label, df)
        if result is None:
            continue
        cleaned[label] = result
        if not dry_run:
            result.to_csv(Path(cleaned_dir) / f"{label}.csv", index=False)
    log.info("Cleaning: %d/%d cities kept", len(cleaned), len(cities))
    return cleaned


def fit_scaler(cleaned, scaler_path, dry_run=False):
    all_features = pd.concat(
        [df[[c for c in FEATURE_COLS if c in df.columns]] for df in cleaned.values()],
        ignore_index=True,
    )
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_features)
    log.info("Scaler fitted — %d rows x %d features", len(all_features), len(FEATURE_COLS))
    if not dry_run:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        log.info("Scaler → %s", scaler_path)
    return scaler


def make_windows(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_windows(cleaned, scaler, windows_path, dry_run=False):
    X_trains, y_trains = [], []
    X_vals,   y_vals   = [], []
    X_tests,  y_tests  = [], []

    for label, df in cleaned.items():
        present = [c for c in FEATURE_COLS if c in df.columns]
        arr     = scaler.transform(df[present]).astype(np.float32)
        X, y    = make_windows(arr, LOOKBACK)

        n = len(X)
        if n < 3:
            log.warning("[%s] Only %d windows — skipping", label, n)
            continue

        n_train = int(n * TRAIN_FRAC)
        n_val   = int(n * VAL_FRAC)

        X_trains.append(X[:n_train])
        y_trains.append(y[:n_train])
        X_vals.append(X[n_train : n_train + n_val])
        y_vals.append(y[n_train : n_train + n_val])
        X_tests.append(X[n_train + n_val :])
        y_tests.append(y[n_train + n_val :])

        log.info("[%s] train=%d  val=%d  test=%d", label, n_train, n_val, n - n_train - n_val)

    arrays = {
        "X_train": np.concatenate(X_trains), "y_train": np.concatenate(y_trains),
        "X_val":   np.concatenate(X_vals),   "y_val":   np.concatenate(y_vals),
        "X_test":  np.concatenate(X_tests),  "y_test":  np.concatenate(y_tests),
    }

    for k, v in arrays.items():
        log.info("  %-10s  %s", k, v.shape)

    if not dry_run:
        np.savez_compressed(windows_path, **arrays)
        log.info("Windows → %s", windows_path)

    return arrays


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  default=INPUT_DIR)
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--lookback",   type=int,   default=LOOKBACK)
    p.add_argument("--threshold",  type=float, default=MISSING_THRESHOLD)
    return p.parse_args()


def do_preprocess():
    args = parse_args()
    global LOOKBACK, MISSING_THRESHOLD
    LOOKBACK          = args.lookback
    MISSING_THRESHOLD = args.threshold

    cities  = load_csvs(args.input_dir)
    reports = validate_all(cities)
    print_report(reports, output_path=None if args.dry_run else REPORT_PATH)

    if args.dry_run:
        return

    cleaned = clean_all(cities, CLEANED_DIR)
    if not cleaned:
        log.error("No cities survived cleaning.")
        sys.exit(1)

    scaler  = fit_scaler(cleaned, SCALER_PATH)
    arrays  = build_windows(cleaned, scaler, WINDOWS_PATH)

    log.info("Done. X_train=%s  y_train=%s", arrays["X_train"].shape, arrays["y_train"].shape)
    log.info("Next: python3 model.py")