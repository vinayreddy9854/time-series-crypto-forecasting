import pandas as pd
df = pd.read_csv('data/processed/cleaned_crypto_prices.csv', parse_dates=['timestamp'])
df.sort_values('timestamp', inplace=True)
df['daily_return'] = df['price'].pct_change()
df['ma_7'] = df['price'].rolling(window=7).mean()
df['ma_14'] = df['price'].rolling(window=14).mean()
df['volatility_7'] = df['price'].rolling(window=7).std()
df.dropna(inplace=True)
df.to_csv('data/features/crypto_features.csv', index=False)
print("Features generated and saved to crypto_features.csv")

