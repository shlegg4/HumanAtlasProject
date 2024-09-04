import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ..utils import log_message
import numpy as np


class FeatureExtractor:
    def __init__(self, model_name='resnet50', layer='avgpool'):
        # Load the pre-trained model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image):
        # Load an image and preprocess it
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image).unsqueeze(0)

        # Extract features
        with torch.no_grad():
            features = self.model(image)

        # Flatten the features to a vector
        features = features.view(features.size(0), -1).numpy()
        return features
