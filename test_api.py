import base64
import requests
import json

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

url = 'http://localhost:5000/predict_class'
url = 'https://pos-model-api.onrender.com/predict_class'

image_path = '/Users/fahadpatel/Downloads/t1.jpg'
image_base64 = image_to_base64(image_path)

data = json.dumps({"image": image_base64})
headers = {'Content-Type': 'application/json'}

response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    print("Response from API:", response.json())
else:
    print(f"Error {response.status_code}: Failed to get a response from the API")

