import pandas as pd

data = pd.read_csv('crypto_prices_30_days.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values('timestamp')
data = data.drop_duplicates()
data.to_csv('cleaned_crypto_prices.csv', index=False)

print("Data cleaned and saved to cleaned_crypto_prices.csv")

