import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from gee_extractor import run_extraction
from preprocess import do_preprocess

run_extraction()
do_preprocess()

os.makedirs("outputs", exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
data = np.load("outputs/windows.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# ── Model ─────────────────────────────────────────────────────────────────────
inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = layers.LSTM(64, return_sequences=True)(inputs)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(32)(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(y_train.shape[1])(x)

model = Model(inputs, output)
model.compile(optimizer="adam", loss="mae")
model.summary()
# ── Training ──────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("outputs/model.keras", save_best_only=True),  # native format
]

if len(X_val) > 0:
    val_data = (X_val, y_val)
    fit_kwargs = {"validation_data": val_data}
else:
    print("Warning: X_val is empty — using validation_split=0.1 instead")
    fit_kwargs = {"validation_split": 0.1}

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    callbacks=callbacks,
    verbose=1,
    **fit_kwargs,
)

# ── Loss curve ────────────────────────────────────────────────────────────────
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.title("Loss curve")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.savefig("outputs/loss_curve.png")
plt.close()

# ── Evaluation ────────────────────────────────────────────────────────────────
test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE: {test_mae:.4f}")

feature_names = (
    [f"lc_{cls}_pct"  for cls in ["water","trees","grass","flooded_vegetation","crops","shrub_and_scrub","built","bare","snow_and_ice"]] +
    [f"lc_{cls}_conf" for cls in ["water","trees","grass","flooded_vegetation","crops","shrub_and_scrub","built","bare","snow_and_ice"]]
)

y_pred = model.predict(X_test)
print("\nPer-feature MAE:")
for i, name in enumerate(feature_names):
    mae = abs(y_pred[:, i] - y_test[:, i]).mean()
    print(f"  {name:<35}  {mae:.4f}")