import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ..utils import log_message  # Assuming you have a custom logging utility

class FeatureExtractor:
    def __init__(self, model_name='resnet50', layer='avgpool'):
        log_message('info', 'Initializing FeatureExtractor...')
        try:
            # Load the pre-trained model
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            # Set model to evaluation mode
            self.model.eval()

            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])

            # Define the image transformation pipeline
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            log_message('info', 'Image transformation pipeline set up successfully.')
        except Exception as e:
            log_message('error', f'Error during initialization of FeatureExtractor: {str(e)}')
            raise e

    def extract_features(self, image):
        try:

            # Convert the image array to a PIL image and log it
            image = Image.fromarray(np.uint8(image))

            # Apply transformations and log the shape after transformation
            image = self.transform(image).unsqueeze(0)

            # Extract features and ensure no gradients are computed
            with torch.no_grad():
                features = self.model(image)
            # Flatten the features and log the shape
            features = features.view(features.size(0), -1).numpy()

            return features
        except Exception as e:
            log_message('error', f'Error during feature extraction: {str(e)}')
            raise e
