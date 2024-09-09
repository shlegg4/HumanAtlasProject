
import aio_pika
from celery import Celery
from src.services import MilvusHandler
from ..data_processing import FeatureExtractor, SuperpixelSegmenter, PCAProcessor, DataUpdatePipeline, DataSearchPipeline
import json
import asyncio

app = Celery('tasks')
app.config_from_object('src.workers.celeryconfig')

# Load models and services
collection_name = 'thursday'
model_path = '../dependencies/pca'

db_handler = MilvusHandler(collection_name=collection_name)
feature_extractor = FeatureExtractor()
superpixel_segmenter = SuperpixelSegmenter()
pca_processor = PCAProcessor(model_path=model_path)
update_pipeline = DataUpdatePipeline(db_handler=db_handler, feature_extractor=feature_extractor, superpixel_segmenter=superpixel_segmenter, pca_processor=pca_processor)
search_pipeline = DataSearchPipeline(db_Handler=db_handler, feature_extractor=feature_extractor, pca_processor=pca_processor)




async def send_status_update(task_id, status):
    # Set up asynchronous RabbitMQ connection and channel
    connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq:5672/")
    async with connection:
        channel = await connection.channel()

        # Declare exchange if it doesn't exist (durable=True)
        exchange = await channel.declare_exchange("task_updates", aio_pika.ExchangeType.FANOUT, durable=True)

        # Publish task status to RabbitMQ exchange
        message = aio_pika.Message(body=f"Task {task_id}: {status}".encode())
        await exchange.publish(message, routing_key="")

        # Close the connection
        await connection.close()

@app.task(bind=True)
def search_task(self, image_path):
    # Send "STARTED" status
    asyncio.run(send_status_update(self.request.id, "STARTED"))

    # Task logic (e.g., image search)
    prediction = search_pipeline.search(image_path)

    # Send "SUCCESS" status
    asyncio.run(send_status_update(self.request.id, "SUCCESS"))

    return json.dumps(prediction)

@app.task(bind=True)
def update_task(self, image_path):
   

    # Send "STARTED" status
    asyncio.run(send_status_update(self.request.id, "STARTED"))

    # Task logic (e.g., database update)
    status = update_pipeline.update_database(image_path=image_path)

    # Send "SUCCESS" status
    asyncio.run(send_status_update(self.request.id, "SUCCESS"))

    return json.dumps(status)