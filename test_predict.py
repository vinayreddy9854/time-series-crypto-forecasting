import requests
import json
url = "http://127.0.0.1:5000/predict"  
data = {
    'features': [0.1, 0.3, 0.4, 0.2, 0.5]  
}
response = requests.post(url, json=data)
try:
    response_data = response.json()
    print(response_data)
except Exception as e:
    print(f"Error: {str(e)}")


