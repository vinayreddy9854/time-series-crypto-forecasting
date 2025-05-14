import requests
import pandas as pd
from datetime import datetime

def fetch_crypto_data():
    # API endpoint to fetch historical data from CoinGecko
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '30',
        'interval': 'daily'
    }
    
    # Request data from the API
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("API Response:", response.json())  # Print the full response
        
        # Parse the JSON response
        data = response.json()

        # Extract timestamps and prices from the response
        timestamps = [datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d %H:%M:%S') for item in data["prices"]]
        prices = [item[1] for item in data["prices"]]

        # Create a DataFrame from the extracted data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })

        # Print the first few rows of the DataFrame to verify
        print(df.head())

        # Save the DataFrame to a CSV file
        df.to_csv('crypto_prices_30_days.csv', index=False)
        print("Data saved to crypto_prices_30_days.csv")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

# Run the function to fetch and save data
fetch_crypto_data()






