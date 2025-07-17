import asyncio
import yaml
from db.listener import listen_for_metrics
from queue.worker import run_training_queue

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

async def main():
    queue = asyncio.Queue()
    await asyncio.gather(
        listen_for_metrics(queue, CONFIG),
        run_training_queue(queue, CONFIG)
    )

if __name__ == "__main__":
    asyncio.run(main())
