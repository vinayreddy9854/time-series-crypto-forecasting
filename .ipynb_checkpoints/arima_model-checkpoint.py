import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace with your actual dataset path)
data = pd.read_csv("crypto_data.csv", parse_dates=["Date"], index_col="Date")

# Feature engineering (e.g., adding rolling averages, volatility)
data['Rolling_Avg'] = data['Close'].rolling(window=10).mean()

# Train-test split (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Normalize the data (if required for machine learning models)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data[['Close', 'Rolling_Avg']])
test_scaled = scaler.transform(test_data[['Close', 'Rolling_Avg']])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics
mae_arima = mean_absolute_error(test_data['Close'], forecast)
rmse_arima = mean_squared_error(test_data['Close'], forecast, squared=False)
r2_arima = r2_score(test_data['Close'], forecast)

print(f"ARIMA MAE: {mae_arima}, RMSE: {rmse_arima}, R²: {r2_arima}")

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(test_data.index, test_data['Close'], label='True', color='blue')
plt.plot(test_data.index, forecast, label='Predicted (ARIMA)', color='red')
plt.legend()
plt.title("ARIMA Model Forecast vs Actual")
plt.show()


