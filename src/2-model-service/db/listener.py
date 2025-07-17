import asyncpg
import asyncio
import json


async def listen_for_metrics(queue, config):
    async def handle_notify(conn, pid, channel, payload):
        print(f"📡 Notification received on {channel}: {payload}")
        try:
            metric_id = int(payload)
        except ValueError:
            print("⚠️ Invalid payload:", payload)
            return

        # Получаем данные по метрике из БД (можно оптимизировать)
        row = await conn.fetchrow("""
            SELECT id, name FROM metrics 
            WHERE id = $1 AND is_active = true 
              AND id NOT IN (SELECT metric_id FROM metric_models)
        """, metric_id)

        if row:
            await queue.put(dict(row))
            print(f"✅ Metric {row['name']} added to queue")
        else:
            print(f"⏭ Metric {metric_id} already processed or inactive")

    conn = await asyncpg.connect(dsn="postgresql://mad:secretPASSW0rd@80.93.60.49:30000/ml_models")
    await conn.add_listener('new_active_metric', lambda *args: asyncio.create_task(handle_notify(*args, queue=queue)))
    print("👂 Listening for 'new_active_metric' notifications...")

    while True:
        await asyncio.sleep(3600)  # Просто поддерживаем соединение активным