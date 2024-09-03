# Convert image to superpixels -> get feature vectors from superpixels -> perform PCA on feature vectors -> Cluster reduced vectors

from .SuperpixelSegmenter import SuperpixelSegmenter
from .FeatureExtraction import FeatureExtractor
from .DimensionalityReduction import PCAProcessor
from .Clustering import ClusteringProcessor
import numpy as np
import os



def main():
    image = '108450_A_1_7'
    # 1. Take image from outputs and perform superpixel segmentation
    image_path = f'\outputs/{image}.jpg'
    segmenter = SuperpixelSegmenter(image_path, n_segments=1000, compactness=10, sigma=1)
    segmenter.segment_and_save()

    # 2. Convert superpixels into feature vectors
    # Usage
    segment_dir = f'./segments/{image}'
    extractor = FeatureExtractor()
    pca_processor = PCAProcessor()
    
    all_features = []

    # Iterate over all image segments in the directory
    for root, dirs, files in os.walk(segment_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Adjust based on your image formats
                image_path = os.path.join(root, file)
                features = extractor.extract_features(image_path)
                all_features.append(features)

    all_features = np.vstack(all_features)  # Stack all features into a numpy array

    # 3. Perform PCA on feature vectors
    reduced_features = pca_processor.fit_transform(all_features)
    
    # 4. Cluster Reduced features
    # Assume `reduced_features` is the result from PCA
    clustering_processor = ClusteringProcessor(n_clusters=5)
    labels = clustering_processor.fit(reduced_features)

    # Optional: If your data is 2D (e.g., using only 2 principal components), you can plot the clusters
    # If reduced_features is more than 2D, consider using only the first two components for visualization
    clustering_processor.plot_clusters(reduced_features[:, :2], labels)

    