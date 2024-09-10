
import aio_pika
from celery import Celery
from celery.signals import worker_init, worker_shutdown
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

# Code to run when worker is initialized
@worker_init.connect
def on_worker_init(**kwargs):
    print("Worker is starting up!")
    # Your custom startup code here
    # e.g., Initialize connections, load models, etc.
    # You can also log messages or perform other startup tasks.
    db_handler.connect()

# Code to run when worker shuts down
@worker_shutdown.connect
def on_worker_shutdown(**kwargs):
    print("Worker is shutting down!")
    # Your custom shutdown code here
    # e.g., Clean up connections, release resources, etc.
    # This is where you'd handle graceful shutdown operations.
    db_handler.close_connection()



async def send_status_update(task_id, status):
    # Set up asynchronous RabbitMQ connection and channel
    message_body = {
        "task_id": task_id,
        "status": status
    }

    # Convert the dictionary to a JSON-encoded string
    json_message = json.dumps(message_body)


    connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq:5672/")
    async with connection:
        channel = await connection.channel()

        # Declare exchange if it doesn't exist (durable=True)
        exchange = await channel.declare_exchange("task_updates", aio_pika.ExchangeType.FANOUT, durable=True)

        # Publish task status to RabbitMQ exchange
        message = aio_pika.Message(json_message.encode())
        await exchange.publish(message, routing_key="")

        # Close the connection
        await connection.close()

@app.task(bind=True)
def search_task(self, image_url, boundary):
    # Send "STARTED" status
    asyncio.run(send_status_update(self.request.id, "STARTED"))

    # Task logic (e.g., image search)
    prediction = search_pipeline.search(image_url, boundary)

    # Send "SUCCESS" status
    asyncio.run(send_status_update(self.request.id, "SUCCESS"))

    return json.dumps(prediction)

@app.task(bind=True)
def update_task(self, image_url):
   

    # Send "STARTED" status
    asyncio.run(send_status_update(self.request.id, "STARTED"))

    # Task logic (e.g., database update)
    status = update_pipeline.update_database(image_url=image_url)

    # Send "SUCCESS" status
    asyncio.run(send_status_update(self.request.id, "SUCCESS"))

    return json.dumps(status)