# gee_extractor.py
import logging
import re
import concurrent.futures
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from gee_fetch import get_dw_data, TEXAS_CITIES, DATES
from analyzer import get_landcover_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTDIR   = "outputs"
BUFFER_M = 1500
WORKERS  = 4

DW_CLASSES = [
    "water", "trees", "grass", "flooded_vegetation",
    "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"
]

def _label(coords):
    if "label" in coords:
        return re.sub(r"[^\w\-]", "_", str(coords["label"]))
    lat_str = str(coords["lat"]).replace(".", "_")
    lon_str = str(coords["lon"]).replace(".", "_").replace("-", "n")
    return f"lat{lat_str}_lon{lon_str}"

def _period_label(date: dict) -> str:
    """Turn {"start": "2016-05-01", "end": "2016-08-01"} → 'historical_2016'"""
    year = date["start"][:4]
    idx  = DATES.index(date)
    tag  = "historical" if idx == 0 else "recent"
    return f"{tag}_{year}"

def extract_location(coords, outdir, buffer_m=BUFFER_M):
    label  = _label(coords)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{label}.csv"

    if out_csv.exists():
        log.info("Skip (exists): %s", out_csv.name)
        return out_csv

    entries = get_dw_data([coords], DATES, buffer_m)
    rows = []

    for entry, date in zip(entries, DATES):
        result = get_landcover_analysis(entry)
        row = {
            "label":  label,
            "lat":    coords["lat"],
            "lon":    coords["lon"],
            "period": _period_label(date),
            "start":  date["start"],
            "end":    date["end"],
        }
        for cls in DW_CLASSES:
            if result:
                row[f"lc_{cls}_pct"]  = result["percentages"].get(cls, 0.0)
                row[f"lc_{cls}_conf"] = result["confidence"].get(cls, 0.0)
            else:
                row[f"lc_{cls}_pct"]  = None
                row[f"lc_{cls}_conf"] = None
        rows.append(row)

    if not rows:
        log.warning("No rows for %s", label)
        return None

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log.info("Saved %d rows → %s", len(df), out_csv.name)
    return out_csv

def run_extraction(locations=None, outdir=OUTDIR, workers=WORKERS, buffer_m=BUFFER_M):
    locations = locations or TEXAS_CITIES
    Path(outdir).mkdir(parents=True, exist_ok=True)
    log.info("Extracting %d locations × %d periods", len(locations), len(DATES))

    counts = {"ok": 0, "skipped": 0, "failed": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_location, loc, outdir, buffer_m): loc for loc in locations}
        with tqdm(total=len(locations), unit="location") as bar:
            for fut in concurrent.futures.as_completed(futures):
                bar.update(1)
                loc = futures[fut]
                try:
                    path = fut.result()
                    counts["ok" if path else "skipped"] += 1
                except Exception as exc:
                    counts["failed"] += 1
                    log.error("Failed %s: %s", _label(loc), exc)

    log.info("Done — ok=%d  skipped=%d  failed=%d", counts["ok"], counts["skipped"], counts["failed"])
    return counts