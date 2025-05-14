import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace with your actual dataset path)
data = pd.read_csv("data/features/crypto_features.csv", parse_dates=["Date"], index_col="Date")

# Feature engineering (e.g., adding rolling averages, volatility)
data['Rolling_Avg'] = data['Close'].rolling(window=10).mean()

# Train-test split (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Normalize the data (if required for machine learning models)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data[['Close', 'Rolling_Avg']])
test_scaled = scaler.transform(test_data[['Close', 'Rolling_Avg']])
