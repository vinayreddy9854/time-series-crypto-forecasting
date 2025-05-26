import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

model = load_model("models/lstm_model.keras")
y_scaler = joblib.load("models/y_scaler.save")

X_val = np.load("data/preprocessed/X_val.npy")
y_val = np.load("data/preprocessed/y_val.npy")
y_pred = model.predict(X_val)
y_pred_inv = y_scaler.inverse_transform(y_pred)
y_val_inv = y_scaler.inverse_transform(y_val.reshape(-1, 1))

plt.figure(figsize=(10, 5))
plt.plot(y_val_inv, label="Actual Price", linewidth=2)
plt.plot(y_pred_inv, label="Predicted Price", linestyle="--")
plt.title("Actual vs Predicted Crypto Price")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


