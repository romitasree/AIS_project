import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

data = np.load("outputs/windows.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = layers.LSTM(64, return_sequences=True)(inputs)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(32)(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(y_train.shape[1])(x)

model = Model(inputs, output)
model.compile(optimizer="adam", loss="mae")
model.summary()

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
)

model.save("outputs/model.keras")
