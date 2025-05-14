import pandas as pd

# Load cleaned data
df = pd.read_csv('data/processed/cleaned_crypto_prices.csv', parse_dates=['timestamp'])

# Sort by timestamp just in case
df.sort_values('timestamp', inplace=True)

# Create features
df['daily_return'] = df['price'].pct_change()
df['ma_7'] = df['price'].rolling(window=7).mean()
df['ma_14'] = df['price'].rolling(window=14).mean()
df['volatility_7'] = df['price'].rolling(window=7).std()

# Drop NA values from rolling calculations
df.dropna(inplace=True)

# Save to features directory
df.to_csv('data/features/crypto_features.csv', index=False)
print("Features generated and saved to crypto_features.csv")

