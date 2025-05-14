import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Load data (X_train, y_train, X_val, y_val)
X_train = np.load('data/preprocessed/X_train.npy')
y_train = np.load('data/preprocessed/y_train.npy')
X_val = np.load('data/preprocessed/X_val.npy')
y_val = np.load('data/preprocessed/y_val.npy')

# Check for NaN values in the data
print("NaN values in X_train:", np.isnan(X_train).sum())
print("NaN values in y_train:", np.isnan(y_train).sum())
print("NaN values in X_val:", np.isnan(X_val).sum())
print("NaN values in y_val:", np.isnan(y_val).sum())

# Replace NaN values with zeros (optional)
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_val = np.nan_to_num(X_val)
y_val = np.nan_to_num(y_val)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))

# Save the model
model.save("models/lstm_model.keras")


print("✅ Model training complete!")





