from . import FeatureExtractor, PCAProcessor
from ..services import MilvusHandler
from ..utils import log_message, Segment, WorkItem
from skimage import io


class DataSearchPipeline:
    def __init__(self, db_Handler, feature_extractor, pca_processor):
        log_message('info', 'started data search pipeline')
        self.db_handler = db_Handler
        self.feature_extractor = feature_extractor
        self.pca_processor = pca_processor

    def search(self, image_path):
        # Connect to milvus
        self.db_handler.connect()

        # Create new workitem 
        workItem = WorkItem()
        
        image = io.imread(image_path)
        
        # Extract the image feature
        log_message('info', 'feature extraction started')
        features = self.feature_extractor.extract_features(image)
       

        # 3. Perform PCA on feature vectors
        log_message('info', 'Started PCA')
        reduced_features = self.pca_processor.transform(features)
        reduced_features = reduced_features.flatten()
        #4. Perform search on database
        print(reduced_features)
        result = self.db_handler.find_by_vector(reduced_features.tolist())
        
        # Close connection to milvus
        self.db_handler.close_connection()

        return result.to_dict()
       
    