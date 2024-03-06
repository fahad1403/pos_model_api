from flask import Flask, request, jsonify
from PIL import Image
import torch
import os
import gzip
import shutil
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

class POSCNN(nn.Module):
    def __init__(self):
        super(POSCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = None

def load_model(compressed_model_path, model_class = POSCNN):
    model_temp_path = 'temp_model.pth'
    with gzip.open(compressed_model_path, 'rb') as f_in:
        with open(model_temp_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    model = model_class()
    model.load_state_dict(torch.load(model_temp_path))
    model.eval()
    os.remove(model_temp_path)
    return model

def predict_image(image, model, transform):
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    classes = ['genuine', 'pos']
    predicted_class = classes[predicted.item()]

    return predicted_class

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    image = Image.open(file.stream).convert('RGB')
    predicted_class = predict_image(image, model, transform)
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    model = load_model('advanced_pos_model.pth.gz')
    app.run(debug=True, port=5000)
