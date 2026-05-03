import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import preprocess as base_preprocess

MAIN_DW_CLASSES = base_preprocess.DW_CLASSES[:8]


def _configure_base(
    input_dir: str,
    output_dir: str,
    lookback: int,
    threshold: float,
    use_main8: bool,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "cleaned").mkdir(parents=True, exist_ok=True)

    base_preprocess.INPUT_DIR = input_dir
    base_preprocess.CLEANED_DIR = str(output_path / "cleaned")
    base_preprocess.WINDOWS_PATH = str(output_path / "windows.npz")
    base_preprocess.SCALER_PATH = str(output_path / "scaler.pkl")
    base_preprocess.REPORT_PATH = str(output_path / "report.txt")

    base_preprocess.LOOKBACK = lookback
    base_preprocess.MISSING_THRESHOLD = threshold

    if use_main8:
        base_preprocess.DW_CLASSES = MAIN_DW_CLASSES
    base_preprocess.FEATURE_COLS = (
        [f"lc_{cls}_pct" for cls in base_preprocess.DW_CLASSES]
        + [f"lc_{cls}_conf" for cls in base_preprocess.DW_CLASSES]
    )


def do_preprocess_pixel(
    input_dir: str = "pixel_output",
    output_dir: str = "pixel_output",
    lookback: int = 3,
    threshold: float = 0.20,
    dry_run: bool = False,
    use_main8: bool = True,
):
    _configure_base(
        input_dir=input_dir,
        output_dir=output_dir,
        lookback=lookback,
        threshold=threshold,
        use_main8=use_main8,
    )

    cities = base_preprocess.load_csvs(base_preprocess.INPUT_DIR)
    reports = base_preprocess.validate_all(cities)
    base_preprocess.print_report(
        reports,
        output_path=None if dry_run else base_preprocess.REPORT_PATH,
    )

    if dry_run:
        return None

    cleaned = base_preprocess.clean_all(cities, base_preprocess.CLEANED_DIR)
    if not cleaned:
        raise RuntimeError("No locations survived cleaning in pixel preprocess pipeline.")

    scaler = base_preprocess.fit_scaler(cleaned, base_preprocess.SCALER_PATH)
    arrays = base_preprocess.build_windows(cleaned, scaler, base_preprocess.WINDOWS_PATH)
    return arrays


def _parse_args():
    p = argparse.ArgumentParser(description="Pixel pipeline preprocess wrapper")
    p.add_argument("--input-dir", default="pixel_output")
    p.add_argument("--output-dir", default="pixel_output")
    p.add_argument("--lookback", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--all-9-classes", action="store_true", help="Use all 9 Dynamic World classes")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    do_preprocess_pixel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lookback=args.lookback,
        threshold=args.threshold,
        dry_run=args.dry_run,
        use_main8=not args.all_9_classes,
    )
