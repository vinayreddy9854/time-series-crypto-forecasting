import requests
import pandas as pd
from datetime import datetime

def fetch_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '30',
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        print("API Response:", response.json())  
        data = response.json()
        timestamps = [datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d %H:%M:%S') for item in data["prices"]]
        prices = [item[1] for item in data["prices"]]
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })
        print(df.head())
        df.to_csv('crypto_prices_30_days.csv', index=False)
        print("Data saved to crypto_prices_30_days.csv")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
fetch_crypto_data()






