import base64
import requests
import json

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

url = 'https://pos-model-api.onrender.com/predict_class'
image_path = '/Users/fahadpatel/Downloads/t1.jpg'
image_base64 = image_to_base64(image_path)
headers = {'Content-Type': 'application/json'}
data = json.dumps({"image": image_base64})

try:
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        try:
            response_data = response.json()
            print("Response from API:", response_data)
        except ValueError:
            print("Received non-JSON response:", response.text)
    else:
        print(f"Error {response.status_code}: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
