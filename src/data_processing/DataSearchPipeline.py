import os
import numpy as np
from . import SuperpixelSegmenter, FeatureExtractor, PCAProcessor, ClusteringProcessor
from ..services import MilvusHandler
from ..utils import log_message, Segment, WorkItem



class DataSearchPipeline:
    def __init__(self):
        self.db_handler = MilvusHandler(collection_name='pathology_slides2')
        self.feature_extractor = FeatureExtractor()

    def search(self, image):
        # Create new workitem 
        workItem = WorkItem()
        
        # Extract the image feature
        log_message('info', 'feature extraction started')
        features = self.feature_extractor.extract_features(image)
       

        # 3. Perform PCA on feature vectors
        log_message('info', 'Started PCA')
        pca_processor = PCAProcessor(model_path='../dependencies/pca')
        reduced_features = pca_processor.transform(features)
        reduced_features = reduced_features.flatten()
        #4. Perform search on database
        print(reduced_features)
        result = self.db_handler.find_by_vector(reduced_features.tolist())
        
        return result.to_dict()
       
    