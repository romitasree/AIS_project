"""
Step 1 — GEE Environmental Data Extractor
==========================================
Pulls monthly environmental indicators from Google Earth Engine at
user-specified lat/lon locations and writes one CSV per location,
ready for LSTM ingestion in Step 2.

Each lat/lon point is buffered into a circular region (BUFFER_M metres)
so that GEE's reduceRegion() has an area to aggregate over, not just a
single pixel.

Datasets extracted per location
--------------------------------
  • MODIS MOD13A3   – NDVI, EVI                (monthly composite, 1 km)
  • MODIS MOD11A2   – Land Surface Temp day/night (8-day → monthly mean)
  • CHIRPS Daily    – Precipitation sum          (daily → monthly, ~5 km)
  • MODIS MCD64A1   – Burned-area fraction       (monthly, 500 m)
  • MODIS MCD12Q1   – Land-cover class (IGBP)    (annual mode, 500 m)

Output
------
  <OUTDIR>/
    └── <label>.csv   — one file per location in TEXAS_CITIES

Each CSV row = one calendar month:
  label, lat, lon, year, month,
  ndvi_mean, evi_mean,
  lst_day_mean_c, lst_night_mean_c,
  precip_sum_mm,
  burned_fraction,
  lc_mode, lc_label

Importing
---------
  from gee_extractor import run_extraction, init_gee, extract_location

  init_gee()                        # authenticate once
  run_extraction()                  # extract all TEXAS_CITIES
  df = extract_location(            # extract a single location
      {"lat": 30.267, "lon": -97.743, "label": "Austin"},
      months=month_range(START, END),
      outdir=Path(OUTDIR),
  )
"""

import concurrent.futures
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import ee
import pandas as pd
from tqdm import tqdm
from gee_fetch import TEXAS_CITIES


START: str   = "2014-01-01"   # inclusive, YYYY-MM-DD
END: str     = "2024-01-01"   # exclusive, YYYY-MM-DD
WORKERS: int = 4              # parallel threads (keep ≤ 8 to avoid GEE quota)
OUTDIR: str  = "outputs"      # root directory for all CSVs

# Buffer radius around each point used for reduceRegion (metres).
# 10 km gives a ~314 km² circle — enough pixels for stable statistics.
# Shrink to 1_000 for city-block scale or grow to 50_000 for regional.
BUFFER_M: int = 10_000

# GEE cloud project (optional but recommended for quota; set None to omit)
GEE_PROJECT: str | None = None

# ---------------------------------------------------------------------------
# IGBP land-cover class legend (MCD12Q1 LC_Type1)
# ---------------------------------------------------------------------------
IGBP_CLASSES: dict[int, str] = {
    1:  "Evergreen_Needleleaf_Forest",
    2:  "Evergreen_Broadleaf_Forest",
    3:  "Deciduous_Needleleaf_Forest",
    4:  "Deciduous_Broadleaf_Forest",
    5:  "Mixed_Forest",
    6:  "Closed_Shrublands",
    7:  "Open_Shrublands",
    8:  "Woody_Savannas",
    9:  "Savannas",
    10: "Grasslands",
    11: "Permanent_Wetlands",
    12: "Croplands",
    13: "Urban_and_Built-up",
    14: "Cropland_Natural_Mosaic",
    15: "Snow_and_Ice",
    16: "Barren",
    17: "Water_Bodies",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal GEE reducer constants
# ---------------------------------------------------------------------------
_SCALE_1KM  = 1_000   # metres — used for MODIS 1 km products
_SCALE_500M = 500     # metres — used for 500 m products
_SCALE_5KM  = 5_000   # metres — used for CHIRPS (~5 km native)
_MAX_PIXELS = 1e9


# ===========================================================================
# GEE initialisation
# ===========================================================================

def init_gee(project: str | None = GEE_PROJECT) -> None:
    """Authenticate and initialise the GEE Python client.

    Tries a silent ``ee.Initialize`` first.  If credentials are absent
    (first run on a machine) it launches the browser auth flow, then
    initialises again automatically.

    Parameters
    ----------
    project:
        Optional GEE / Google Cloud project ID.  Pass ``None`` to omit.
    """
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        log.info("GEE initialised ✓")
    except ee.EEException:
        log.warning("GEE credentials not found — launching authentication flow …")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        log.info("GEE initialised after authentication ✓")


# ===========================================================================
# Date helpers
# ===========================================================================

def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Return a list of (year, month) tuples from *start* up to (not including) *end*.

    Parameters
    ----------
    start, end:
        Date strings in ``YYYY-MM-DD`` format.

    Returns
    -------
    list[tuple[int, int]]
        Ordered list of ``(year, month)`` pairs.

    Example
    -------
    >>> month_range("2023-11-01", "2024-02-01")
    [(2023, 11), (2023, 12), (2024, 1)]
    """
    cur  = datetime.strptime(start, "%Y-%m-%d").replace(day=1)
    stop = datetime.strptime(end,   "%Y-%m-%d").replace(day=1)
    months: list[tuple[int, int]] = []
    while cur < stop:
        months.append((cur.year, cur.month))
        cur = cur.replace(month=1, year=cur.year + 1) if cur.month == 12 \
              else cur.replace(month=cur.month + 1)
    return months


def month_window(year: int, month: int) -> tuple[str, str]:
    """Return ``(start_str, end_str)`` covering one full calendar month.

    Parameters
    ----------
    year, month:
        Calendar year and month (1–12).

    Returns
    -------
    tuple[str, str]
        ``(YYYY-MM-DD, YYYY-MM-DD)`` where the end date is the first day of
        the *next* month (exclusive, matching GEE's filterDate convention).
    """
    t0 = datetime(year, month, 1)
    t1 = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")


# ===========================================================================
# Geometry helper
# ===========================================================================

def point_region(lat: float, lon: float, buffer_m: int = BUFFER_M) -> ee.Geometry:
    """Build a GEE circular geometry centred on *lat/lon*.

    Parameters
    ----------
    lat, lon:
        WGS-84 decimal degrees.
    buffer_m:
        Radius of the buffer circle in metres.

    Returns
    -------
    ee.Geometry
        Buffered ``ee.Geometry.Point`` suitable for ``reduceRegion``.
    """
    return ee.Geometry.Point([lon, lat]).buffer(buffer_m)


# ===========================================================================
# Per-band GEE extraction functions
# ===========================================================================

def _safe_get(info: dict, key: str, default=None):
    """Return ``info[key]`` or *default* when the key is absent or ``None``."""
    v = info.get(key)
    return default if v is None else v


def safe_call(fn, *args, retries: int = 3, delay: int = 5, **kwargs) -> dict:
    """Call *fn* with retry / exponential back-off on transient GEE errors.

    Parameters
    ----------
    fn:
        One of the ``extract_*`` functions below.
    retries:
        Maximum number of attempts before giving up and returning ``{}``.
    delay:
        Base sleep duration in seconds; multiplied by attempt number.

    Returns
    -------
    dict
        Band statistics on success, empty dict on all retries exhausted.
    """
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt < retries - 1:
                log.debug(
                    "Retry %d/%d for %s — %s",
                    attempt + 1, retries, fn.__name__, exc,
                )
                time.sleep(delay * (attempt + 1))
            else:
                log.warning("All retries failed for %s: %s", fn.__name__, exc)
    return {}


def extract_ndvi_evi(region: ee.Geometry, year: int, month: int) -> dict:
    """Extract monthly mean NDVI and EVI from MODIS MOD13A3.

    Scale factor ``0.0001`` is applied server-side so returned values are
    in the standard vegetation index range ``[-1, 1]``.

    Parameters
    ----------
    region:
        GEE geometry to aggregate over.
    year, month:
        Target calendar month.

    Returns
    -------
    dict
        Keys: ``ndvi_mean``, ``evi_mean``  (float or None)
    """
    t0, t1 = month_window(year, month)
    img = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterDate(t0, t1)
        .select(["NDVI", "EVI"])
        .mean()
        .multiply(0.0001)
    )
    info = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=_SCALE_1KM,
        maxPixels=_MAX_PIXELS,
        bestEffort=True,
    ).getInfo() or {}
    return {
        "ndvi_mean": _safe_get(info, "NDVI"),
        "evi_mean":  _safe_get(info, "EVI"),
    }


def extract_lst(region: ee.Geometry, year: int, month: int) -> dict:
    """Extract monthly mean Land Surface Temperature from MODIS MOD11A2.

    The product is 8-day; all scenes in the month are averaged.  The raw DN
    is converted to Celsius using: ``value × 0.02 − 273.15``.

    Parameters
    ----------
    region:
        GEE geometry to aggregate over.
    year, month:
        Target calendar month.

    Returns
    -------
    dict
        Keys: ``lst_day_mean_c``, ``lst_night_mean_c``  (float or None)
    """
    t0, t1 = month_window(year, month)
    img = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(t0, t1)
        .select(["LST_Day_1km", "LST_Night_1km"])
        .mean()
        .multiply(0.02)
        .subtract(273.15)
    )
    info = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=_SCALE_1KM,
        maxPixels=_MAX_PIXELS,
        bestEffort=True,
    ).getInfo() or {}
    return {
        "lst_day_mean_c":   _safe_get(info, "LST_Day_1km"),
        "lst_night_mean_c": _safe_get(info, "LST_Night_1km"),
    }


def extract_precip(region: ee.Geometry, year: int, month: int) -> dict:
    """Extract total monthly precipitation from CHIRPS Daily (mm/month).

    Daily images are *summed* (not averaged) to give total monthly rainfall,
    then spatially averaged over the region.

    Parameters
    ----------
    region:
        GEE geometry to aggregate over.
    year, month:
        Target calendar month.

    Returns
    -------
    dict
        Key: ``precip_sum_mm``  (float or None)
    """
    t0, t1 = month_window(year, month)
    img = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(t0, t1)
        .select(["precipitation"])
        .sum()   # accumulate daily → monthly total
    )
    info = img.reduceRegion(
        reducer=ee.Reducer.mean(),   # spatial mean over the region
        geometry=region,
        scale=_SCALE_5KM,
        maxPixels=_MAX_PIXELS,
        bestEffort=True,
    ).getInfo() or {}
    return {"precip_sum_mm": _safe_get(info, "precipitation")}


def extract_burned(region: ee.Geometry, year: int, month: int) -> dict:
    """Extract the fraction of pixels burned in the region from MODIS MCD64A1.

    Returns 0.0 when no MCD64A1 scene exists for the month (common in
    winter months with no fire activity).

    Parameters
    ----------
    region:
        GEE geometry to aggregate over.
    year, month:
        Target calendar month.

    Returns
    -------
    dict
        Key: ``burned_fraction``  (float in ``[0.0, 1.0]``)
    """
    t0, t1 = month_window(year, month)
    col = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(t0, t1)
        .select(["BurnDate"])
    )
    # Short-circuit when no imagery exists for this month
    if col.size().getInfo() == 0:
        return {"burned_fraction": 0.0}

    # BurnDate > 0 marks a burned pixel; mean over the region = fraction burned
    burned_mask = col.first().gt(0).rename("burned")
    info = burned_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=_SCALE_500M,
        maxPixels=_MAX_PIXELS,
        bestEffort=True,
    ).getInfo() or {}
    return {"burned_fraction": _safe_get(info, "burned", 0.0)}


def extract_land_cover(region: ee.Geometry, year: int) -> dict:
    """Extract the dominant IGBP land-cover class from MODIS MCD12Q1.

    MCD12Q1 is an annual product released with a ~6-month lag.  Years beyond
    2022 are clamped to 2022 until newer imagery is available.

    ``ee.Reducer.mode()`` is used because land-cover is categorical — mean
    would produce meaningless fractional class IDs.

    Parameters
    ----------
    region:
        GEE geometry to aggregate over.
    year:
        Calendar year to fetch.

    Returns
    -------
    dict
        Key: ``lc_mode``  (int IGBP class or None)
    """
    lc_year = min(year, 2022)
    col = (
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .filterDate(f"{lc_year}-01-01", f"{lc_year}-12-31")
        .select(["LC_Type1"])
    )
    if col.size().getInfo() == 0:
        return {"lc_mode": None}
    info = col.first().reduceRegion(
        reducer=ee.Reducer.mode(),
        geometry=region,
        scale=_SCALE_500M,
        maxPixels=_MAX_PIXELS,
        bestEffort=True,
    ).getInfo() or {}
    return {"lc_mode": _safe_get(info, "LC_Type1")}


# ===========================================================================
# Per-location extraction
# ===========================================================================

def _location_label(entry: dict) -> str:
    """Derive a filesystem-safe label string from a location dict.

    Uses the ``"label"`` key if present, otherwise falls back to
    ``"lat{lat}_lon{lon}"`` with dots replaced by underscores.
    """
    if "label" in entry:
        return re.sub(r"[^\w\-]", "_", str(entry["label"]))
    lat_str = str(entry["lat"]).replace(".", "_")
    lon_str = str(entry["lon"]).replace(".", "_").replace("-", "n")
    return f"lat{lat_str}_lon{lon_str}"


def extract_location(
    entry: dict,
    months: list[tuple[int, int]],
    outdir: Path,
    buffer_m: int = BUFFER_M,
) -> Path | None:
    """Extract all monthly rows for a single lat/lon location and write a CSV.

    This is the core per-location worker.  It is safe to call from multiple
    threads concurrently — each location writes to its own file and they do
    not share mutable state.

    Parameters
    ----------
    entry:
        Dict with keys ``"lat"``, ``"lon"``, and optionally ``"label"``.
    months:
        Ordered list of ``(year, month)`` tuples produced by
        :func:`month_range`.
    outdir:
        Root output directory.  Will be created if absent.
    buffer_m:
        Radius of the circular buffer around the point (metres).

    Returns
    -------
    Path
        Path to the written CSV on success.
    None
        When the location produces no rows (all GEE calls failed) or the CSV
        already existed and was skipped.
    """
    lat   = entry["lat"]
    lon   = entry["lon"]
    label = _location_label(entry)

    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{label}.csv"

    if out_csv.exists():
        log.debug("Skip (already done): %s", out_csv.name)
        return out_csv

    region = point_region(lat, lon, buffer_m)
    rows: list[dict] = []

    # Land-cover is annual — cache the GEE call per year to avoid 12× redundancy
    lc_cache: dict[int, dict] = {}

    for year, month in months:
        row: dict = {
            "label": label,
            "lat":   lat,
            "lon":   lon,
            "year":  year,
            "month": month,
        }

        row.update(safe_call(extract_ndvi_evi,  region, year, month))
        row.update(safe_call(extract_lst,        region, year, month))
        row.update(safe_call(extract_precip,     region, year, month))
        row.update(safe_call(extract_burned,     region, year, month))

        if year not in lc_cache:
            lc_cache[year] = safe_call(extract_land_cover, region, year)
        row.update(lc_cache[year])

        lc_int = row.get("lc_mode")
        row["lc_label"] = IGBP_CLASSES.get(lc_int, "Unknown") if lc_int is not None else None

        rows.append(row)

    if not rows:
        log.warning("No rows produced for %s — skipping", label)
        return None

    df = pd.DataFrame(rows)
    # Enforce a consistent column order for downstream consumers
    col_order = [
        "label", "lat", "lon", "year", "month",
        "ndvi_mean", "evi_mean",
        "lst_day_mean_c", "lst_night_mean_c",
        "precip_sum_mm",
        "burned_fraction",
        "lc_mode", "lc_label",
    ]
    df = df.reindex(columns=[c for c in col_order if c in df.columns])
    df.to_csv(out_csv, index=False)
    log.info("Saved %d rows → %s", len(df), out_csv)
    return out_csv


# ===========================================================================
# Top-level orchestrator
# ===========================================================================

def run_extraction(
    locations: list[dict] = TEXAS_CITIES,
    start: str            = START,
    end: str              = END,
    outdir: str | Path    = OUTDIR,
    workers: int          = WORKERS,
    buffer_m: int         = BUFFER_M,
) -> dict[str, int]:
    """Extract environmental time-series for all *locations* and write CSVs.

    This is the primary entry point when importing this module.
    ``init_gee()`` **must** be called before ``run_extraction()``.

    Parameters
    ----------
    locations:
        List of dicts, each with ``"lat"``, ``"lon"``, and optional
        ``"label"`` keys.  Defaults to the module-level ``TEXAS_CITIES``.
    start, end:
        Date range in ``YYYY-MM-DD`` format.  *end* is exclusive.
    outdir:
        Root directory for output CSVs.
    workers:
        Number of parallel threads.  Keep at or below 8 to avoid GEE
        per-user quota errors.
    buffer_m:
        Buffer radius in metres passed to each :func:`extract_location` call.

    Returns
    -------
    dict[str, int]
        Summary counts: ``{"ok": N, "skipped": N, "failed": N}``.

    Example
    -------
    >>> from gee_extractor import init_gee, run_extraction
    >>> init_gee()
    >>> summary = run_extraction()
    >>> print(summary)
    {'ok': 2, 'skipped': 0, 'failed': 0}
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    months = month_range(start, end)
    log.info(
        "Date range: %s → %s  (%d months)",
        start, end, len(months),
    )
    log.info(
        "Extracting %d locations × %d months with %d workers (buffer=%dm) …",
        len(locations), len(months), workers, buffer_m,
    )

    results: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0}

    def _worker(entry: dict) -> Path | None:
        return extract_location(entry, months, outdir, buffer_m)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, loc): loc for loc in locations}
        with tqdm(total=len(locations), unit="location") as bar:
            for fut in concurrent.futures.as_completed(futures):
                bar.update(1)
                loc = futures[fut]
                try:
                    path = fut.result()
                    if path:
                        results["ok"] += 1
                    else:
                        results["skipped"] += 1
                except Exception as exc:
                    results["failed"] += 1
                    log.error(
                        "Location %s failed: %s",
                        _location_label(loc), exc,
                    )

    log.info(
        "Done. ✓ written=%d  ↷ skipped=%d  ✗ failed=%d",
        results["ok"], results["skipped"], results["failed"],
    )
    return results
