import requests
import json

# Define the API URL
url = "http://127.0.0.1:5000/predict"  # Adjust if you deploy to a different URL

# Example features (ensure these match the model's expected input)
data = {
    'features': [0.1, 0.3, 0.4, 0.2, 0.5]  # Replace with your actual feature data
}

# Send a POST request with the data
response = requests.post(url, json=data)

# Check the response
try:
    response_data = response.json()
    print(response_data)
except Exception as e:
    print(f"Error: {str(e)}")


