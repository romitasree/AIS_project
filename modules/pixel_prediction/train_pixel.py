import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from modules.pixel_prediction.gee_extractor_pixel import run_pixel_extraction
from modules.pixel_prediction.preprocess_pixel import MAIN_DW_CLASSES, do_preprocess_pixel


def train_pixel_model(
    output_dir: str = "pixel_output",
    run_extract: bool = True,
    run_preprocess: bool = True,
    lookback: int = 3,
    threshold: float = 0.20,
    epochs: int = 100,
    batch_size: int = 8,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if run_extract:
        run_pixel_extraction(outdir=str(out), use_main8=True)

    if run_preprocess:
        do_preprocess_pixel(
            input_dir=str(out),
            output_dir=str(out),
            lookback=lookback,
            threshold=threshold,
            dry_run=False,
            use_main8=True,
        )

    data = np.load(out / "windows.npz")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(y_train.shape[1])(x)

    model = Model(inputs, output)
    model.compile(optimizer="adam", loss="mae")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(str(out / "model.keras"), save_best_only=True),
    ]

    if len(X_val) > 0:
        fit_kwargs = {"validation_data": (X_val, y_val)}
    else:
        fit_kwargs = {"validation_split": 0.1}

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        **fit_kwargs,
    )

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history.get("val_loss", []), label="val")
    plt.legend()
    plt.title("Pixel pipeline loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.savefig(out / "loss_curve.png")
    plt.close()

    test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {test_mae:.4f}")

    feature_names = (
        [f"lc_{cls}_pct" for cls in MAIN_DW_CLASSES]
        + [f"lc_{cls}_conf" for cls in MAIN_DW_CLASSES]
    )
    y_pred = model.predict(X_test, verbose=0)

    print("Per-feature MAE:")
    for i, name in enumerate(feature_names):
        mae = abs(y_pred[:, i] - y_test[:, i]).mean()
        print(f"  {name:<35} {mae:.4f}")


def _parse_args():
    p = argparse.ArgumentParser(description="Train pixel-output model using wrapped pipeline")
    p.add_argument("--output-dir", default="pixel_output")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--lookback", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_pixel_model(
        output_dir=args.output_dir,
        run_extract=not args.skip_extract,
        run_preprocess=not args.skip_preprocess,
        lookback=args.lookback,
        threshold=args.threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
