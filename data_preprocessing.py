import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib  # for saving the y_scaler

# Load your dataset
data = pd.read_csv("data/features/crypto_features.csv")  # Updated dataset path

# Feature engineering: create lag and rolling statistics
data['lag_1'] = data['price'].shift(1)
data['rolling_mean_3'] = data['price'].rolling(window=3).mean()
data['rolling_std_3'] = data['price'].rolling(window=3).std()
data['rolling_mean_7'] = data['price'].rolling(window=7).mean()
data['rolling_std_7'] = data['price'].rolling(window=7).std()

# Drop rows with NaN values after rolling calculations
data.dropna(inplace=True)

# Prepare features and target
X = data[['price', 'lag_1', 'rolling_mean_3', 'rolling_std_3', 'rolling_mean_7', 'rolling_std_7']].values
y = data['price'].shift(-1).dropna().values.reshape(-1, 1)  # Reshape for scaler compatibility

# Ensure X and y have the same length by trimming the last row of X
X = X[:-1]  # Match X with y's length

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale X using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Scale y using MinMaxScaler for LSTM stability
y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)
y_val = y_scaler.transform(y_val)

# Save y_scaler for inverse transformation during prediction
joblib.dump(y_scaler, 'models/y_scaler.save')

# Reshape X for LSTM: (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

# Save preprocessed data
np.save('data/preprocessed/X_train.npy', X_train)
np.save('data/preprocessed/y_train.npy', y_train)
np.save('data/preprocessed/X_val.npy', X_val)
np.save('data/preprocessed/y_val.npy', y_val)

print("✅ Data preprocessing complete and saved.")
