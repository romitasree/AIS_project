import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gee_extractor as base_extractor

# Default to the first 8 Dynamic World classes for pixel pipeline outputs.
MAIN_DW_CLASSES = base_extractor.DW_CLASSES[:8]


def run_pixel_extraction(
    outdir: str = "pixel_output",
    workers: int = 4,
    buffer_m: int = 1500,
    use_main8: bool = True,
):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if use_main8:
        base_extractor.DW_CLASSES = MAIN_DW_CLASSES

    return base_extractor.run_extraction(
        locations=base_extractor.TEXAS_CITIES,
        outdir=outdir,
        workers=workers,
        buffer_m=buffer_m,
    )


def _parse_args():
    p = argparse.ArgumentParser(description="Pixel pipeline extractor wrapper")
    p.add_argument("--outdir", default="pixel_output")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--buffer-m", type=int, default=1500)
    p.add_argument("--all-9-classes", action="store_true", help="Use all 9 Dynamic World classes")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pixel_extraction(
        outdir=args.outdir,
        workers=args.workers,
        buffer_m=args.buffer_m,
        use_main8=not args.all_9_classes,
    )
