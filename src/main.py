from fastapi import FastAPI, WebSocket
from src.workers.tasks import search_task, update_task
import aio_pika
import json

app = FastAPI()


@app.post("/search")
async def search_endpoint(item: dict):
    result = search_task.delay(item.get('image_path'))
    return {"task_id": result.id}

@app.get("/result")
async def get_result(task_id: str):
    result = search_task.AsyncResult(task_id)
    if result.state == 'SUCCESS':
        return {"prediction": result.result}
    else:
        return {"status": result.state}

@app.post("/update")
async def update_endpoint(item: dict):
    result = update_task.delay(item.get('image_path'))
    return {"task_id": result.id}

@app.get("/update-status")
async def get_update_status(task_id: str):
    result = update_task.AsyncResult(task_id)
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

    # Declare the exchange and queue
    exchange = await channel.declare_exchange("task_updates", aio_pika.ExchangeType.FANOUT, durable=True)
    queue = await channel.declare_queue("shared_task_updates_queue", durable=True)
    await queue.bind(exchange)

    # Create an asynchronous consumer to receive messages
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                # Send RabbitMQ message to the WebSocket client
                await websocket.send_text(message.body.decode())

    # Close the WebSocket connection
    await websocket.close()