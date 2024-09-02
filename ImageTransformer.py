import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
from torch.nn.functional import cosine_similarity

# Function to process image and get last hidden states
def get_image_representation(url):
    image = Image.open(requests.get(url, stream=True).raw)
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]

# URLs of two images
url1 = 'http://images.proteinatlas.org/46705/108451_A_3_6.jpg'
url2 = 'http://images.proteinatlas.org/46705/108451_A_1_3.jpg'

# Get the representations for both images
image1_rep = get_image_representation(url1)
image2_rep = get_image_representation(url2)

# Compute cosine similarity for each corresponding patch
cos_similarities = cosine_similarity(image1_rep[0], image2_rep[0])  # [sequence_length]

# Print the cosine similarities for each patch
for i, sim in enumerate(cos_similarities):
    print(f"Patch {i}: Cosine Similarity = {sim.item()}")
