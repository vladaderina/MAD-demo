import asyncpg
import pickle
from datetime import datetime

async def save_model(metric_id, model, hyperparams):
    conn = await asyncpg.connect(dsn="postgresql://user:password@host:port/db")
    data = pickle.dumps(model)
    await conn.execute("""
        INSERT INTO models(name, model_data, created_at, last_updated, hyperparams, status, version)
        VALUES($1, $2, $3, $4, $5, $6, $7)
    """, f"model_{metric_id}", data, datetime.utcnow(), datetime.utcnow(), hyperparams, "trained", "1.0")
    await conn.close()
