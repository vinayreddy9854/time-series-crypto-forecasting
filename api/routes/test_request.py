import requests
import json
url = "http://127.0.0.1:5000/predict"
data = {
    "features": [0.45, 0.67, 0.32, 0.81, 0.56]  
}
response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})

print(response.json())



