import asyncio
from victoria.fetcher import fetch_metric_data
from model.trainer import train_lstm_model
from storage.database import save_model

async def run_training_queue(queue, config):
    workers = []
    for _ in range(config["infrastructure"]["workers"]):
        workers.append(asyncio.create_task(process_metric(queue, config)))
    await asyncio.gather(*workers)

async def process_metric(queue, config):
    while True:
        metric = await queue.get()
        print(f"Start training for metric: {metric['name']}")
        data = await fetch_metric_data(metric["name"], config)
        model, hyperparams = await train_lstm_model(data, config)
        await save_model(metric["id"], model, hyperparams)
        queue.task_done()
