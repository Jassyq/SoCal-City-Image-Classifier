import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CITIES = ['Anaheim', 'Bakersfield', 'Los_Angeles', 'Riverside', 'San_Diego', 'SLO']

def predict(image_path):
    # Device
    device = torch.device('cpu')

    # Load model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CITIES))
    )

    # Load saved weights
    weights_path = os.path.join(os.path.dirname(__file__), 'weights.pt')
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.eval()

    # Same transform as validation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    predictions = {}

    for fname in os.listdir(image_path):
        if fname.endswith('.jpg'):
            img = Image.open(os.path.join(image_path, fname)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                predictions[fname] = CITIES[predicted.item()]

    return predictions