import base64
import requests
import json

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

url = 'http://127.0.0.1:5000/predict_class'

image_path = '/Users/fahadpatel/Downloads/t4.jpeg'
image_base64 = image_to_base64(image_path)

data = json.dumps({"image": image_base64})
headers = {'Content-Type': 'application/json'}

response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    if 'application/json' in response.headers.get('Content-Type', ''):
        print("Response from API:", response.json())
    else:
        print("Received non-JSON response:", response.text)
else:
    print(f"Error {response.status_code} {response.json()}: Failed to get a response from the API")

