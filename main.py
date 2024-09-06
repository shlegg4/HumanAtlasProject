from concurrent import futures
import grpc
import HAP_pb2
import HAP_pb2_grpc
from src.data_processing import DataUpdatePipeline, DataSearchPipeline
from src.utils.logging import setup_logger
from src.services import MilvusHandler
from src.data_processing import FeatureExtractor, SuperpixelSegmenter, PCAProcessor
import json


class HAPService(HAP_pb2_grpc.HAPServiceServicer):
    def __init__(self):
        # Load the model when the server starts
        collection_name = 'thursday'
        model_path = '../dependencies/pca'

        db_handler = MilvusHandler(collection_name=collection_name)
        feature_extractor =FeatureExtractor()
        superpixel_segmenter = SuperpixelSegmenter()
        pca_processor = PCAProcessor(model_path=model_path)
        self.update_pipeline = DataUpdatePipeline(db_handler=db_handler, feature_extractor=feature_extractor, superpixel_segmenter=superpixel_segmenter, pca_processor=pca_processor)
        self.search_pipeline = DataSearchPipeline(db_Handler=db_handler, feature_extractor=feature_extractor, pca_processor=pca_processor)
        
    def Search(self, request, context):
        # Use the loaded model to make a prediction
        image_path = request.image_path
        prediction = self.search_pipeline.search(image_path)
        prediction_json = json.dumps(prediction)
        return HAP_pb2.PredictionResponse(prediction=prediction_json)

    def Update(self, request, context):
        image_path = request.image_path
        status = self.update_pipeline.update_database(image_path=image_path)
        return HAP_pb2.UpdateStatus(status=status)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    HAP_pb2_grpc.add_HAPServiceServicer_to_server(HAPService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server running on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    setup_logger()
    serve()
    
