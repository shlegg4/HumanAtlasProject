from ..utils import log_message, download_image, crop_image

class DataSearchPipeline:
    def __init__(self, db_Handler, feature_extractor, pca_processor):
        log_message('info', 'started data search pipeline')
        self.db_handler = db_Handler
        self.feature_extractor = feature_extractor
        self.pca_processor = pca_processor

    def search(self, image_url, boundary):
       

        # Download and crop image
        image = download_image(image_url=image_url)
        image = crop_image(image, boundary=boundary)
        
        # Extract the image feature
        log_message('info', f'feature extraction started {image}')
        features = self.feature_extractor.extract_features(image)
       

        # 3. Perform PCA on feature vectors
        log_message('info', 'Started PCA')
        reduced_features = self.pca_processor.transform(features)
        reduced_features = reduced_features.flatten()
        #4. Perform search on database
        print(reduced_features)
        result = self.db_handler.find_by_vector(reduced_features.tolist())
        
        return result.to_dict()
       
    