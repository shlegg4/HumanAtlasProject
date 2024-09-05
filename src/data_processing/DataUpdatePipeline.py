import numpy as np
from . import SuperpixelSegmenter, FeatureExtractor, PCAProcessor, ClusteringProcessor
from ..services import MilvusHandler
from ..utils import log_message, Segment, WorkItem



class DataUpdatePipeline:
    def __init__(self, image_name: str):
        self.image_name = image_name
        self.image_path = f'./outputs/{image_name}.jpg'
        self.segment_dir = f'./segments/{image_name}'
        self.db_handler = MilvusHandler(collection_name='pathology_slides2')
        self.feature_extractor = FeatureExtractor()
        self.pca_processor = PCAProcessor(model_path='../dependencies/pca')

    def update_database(self):
        # Create new workitem 
        workItem = WorkItem()

        # 1. Perform superpixel segmentation
        log_message('info', 'segmentation started')
        segmenter = SuperpixelSegmenter(self.image_path, n_segments=1000, compactness=10, sigma=1)
        segments = segmenter.segment_and_save()  # Assuming this method returns the segmented paths as numpy arrays

        # 2. Convert superpixels into feature vectors
        log_message('info', 'feature extraction started')
        

        all_features = []
        paths = []

        # Iterate over all image segments in the directory
        for segment in segments:
            path = segment['path']  # segment contains the path as a 2D numpy array
            image = segment['image']  # Assuming each segment also has a path to the segment image
            features = self.feature_extractor.extract_features(image)
            all_features.append(features)
            paths.append(path)

        all_features = np.vstack(all_features)  # Stack all features into a numpy array

        # 3. Perform PCA on feature vectors
        log_message('info', 'Started PCA')
        
        reduced_features = self.pca_processor.fit_transform(all_features)

        # Store segments in the database
        log_message('info', 'Started Saving Vectors to Database')
        for i, reduced_feature in enumerate(reduced_features):
            segment = Segment(vector=reduced_feature.tolist(), path=paths[i])
            self.db_handler.insert_segment(segment=segment)

        # Remember to close the database connection when done
        self.db_handler.close_connection()
        