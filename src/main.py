from fastapi import FastAPI, WebSocket
from src.workers.tasks import search_task, update_task
import aio_pika
from .utils import log_message
import json

app = FastAPI()


@app.post("/search")
async def search_endpoint(item: dict):
    result = search_task.delay(item.get('image_url'), item.get('boundary'))
    return {"task_id": result.id}

@app.get("/search/result")
async def get_result(task_id: str):
    result = search_task.AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return {"prediction": result.result}
    else:
        return {"status": result.state}

@app.post("/update")
async def update_endpoint(item: dict):
    result = update_task.delay(item.get('image_url'))
    return {"task_id": result.id}

@app.get("/update/status")
async def get_update_status(task_id: str):
    result = update_task.AsyncResult(task_id)
    print(result.get(timeout=10))
    if result.state == 'SUCCESS':
        return {"status": result.result}
    else:
        return {"status": result.state}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Set up asynchronous RabbitMQ connection
    connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq:5672/")
    channel = await connection.channel()

    # Declare the fanout exchange
    exchange = await channel.declare_exchange("task_updates", aio_pika.ExchangeType.FANOUT, durable=True)

    # Declare a unique, temporary, auto-deleted queue for this WebSocket connection
    queue = await channel.declare_queue("status_message", exclusive=False, auto_delete=True)

    # Bind the queue to the exchange
    await queue.bind(exchange)

    # Create an asynchronous consumer to receive messages
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                # Try to decode and parse the message as JSON
                try:
                    decoded_message = message.body.decode()
                    parsed_message = json.loads(decoded_message)
                    await websocket.send_json(parsed_message)
                except json.JSONDecodeError:
                    # If the message is not valid JSON, send it as plain text
                    await websocket.send_text(decoded_message)

    # Close the WebSocket connection
    await channel.close()
    await connection.close()
    await websocket.close()
