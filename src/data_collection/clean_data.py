import pandas as pd

# Load the data
data = pd.read_csv('crypto_prices_30_days.csv')

# Convert timestamp to datetime format (no 'unit' because it's already a string)
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Sort data by timestamp
data = data.sort_values('timestamp')

# Remove duplicates if any
data = data.drop_duplicates()

# Save cleaned data
data.to_csv('cleaned_crypto_prices.csv', index=False)

print("Data cleaned and saved to cleaned_crypto_prices.csv")

