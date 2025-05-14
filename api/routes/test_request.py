import requests
import json

# Localhost URL where your Flask API is running
url = "http://127.0.0.1:5000/predict"

# Sample feature data – ensure this matches the number and order of features your model expects
data = {
    "features": [0.45, 0.67, 0.32, 0.81, 0.56]  # Example 5 features
}

# Make the POST request to the API
response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})

# Print the response from the API
print(response.json())



