import asyncio
import asyncpg
import aiohttp
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

# === 1. Загрузка конфигурации ===
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

async def create_db_pool():
    return await asyncpg.create_pool(
        user="mad",
        password="secretPASSW0rd",
        database="ml_models",
        host="80.93.60.49",
        port=30000,
    )

# === 2. Получение данных из VictoriaMetrics ===
async def fetch_metric_data(metric_name: str, db_pool, config: dict) -> pd.DataFrame:
    """
    Получает данные из VictoriaMetrics для всех запросов, связанных с заданной метрикой.
    Возвращает многомерный DataFrame, где каждая колонка — один временной ряд.
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT r.query, m.start_monitoring, m.end_monitoring, m.step
            FROM metrics m
            JOIN metric_requests mr ON m.id = mr.metric_id
            JOIN request r ON mr.request_id = r.id
            WHERE m.name = $1
        """, metric_name)

    if not rows:
        raise ValueError(f"No queries found for metric '{metric_name}'")

    dfs = []

    async with aiohttp.ClientSession() as session:
        for i, row in enumerate(rows):
            query, start, end, step = row["query"], row["start"], row["end"], row["step"]

            url = f"{config['victoriametrics']['url']}/api/v1/query_range"
            params = {
                "query": query,
                "start": start,
                "end": end,
                "step": step or "60"
            }

            async with session.get(url, params=params) as resp:
                result = await resp.json()

            try:
                values = result["data"]["result"][0]["values"]
            except (KeyError, IndexError):
                raise RuntimeError(f"No data returned for query {i+1}: {query}")

            df = pd.DataFrame(values, columns=["timestamp", f"value_{i}"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df[f"value_{i}"] = df[f"value_{i}"].astype(float)
            dfs.append(df.set_index("timestamp"))

    # Объединяем все временные ряды по timestamp
    merged_df = pd.concat(dfs, axis=1).reset_index()
    return merged_df

# === 3. Обучение LSTM модели ===
async def train_lstm_model(dataframe: pd.DataFrame, config: dict):
    """
    Обучает LSTM модель на одномерных или многомерных временных рядах.
    
    Args:
        dataframe: DataFrame с колонками ['timestamp', 'value'] (одномерный) 
                  или ['timestamp', 'value_0', 'value_1', ...] (многомерный)
        config: Конфигурация модели
        
    Returns:
        Обученная модель и дополнительные параметры
    """
    # Подготовка данных
    if 'value' in dataframe.columns:
        # Одномерный случай
        value_cols = ['value']
        is_multivariate = False
    else:
        # Многомерный случай
        value_cols = [col for col in dataframe.columns if col.startswith('value_')]
        if not value_cols:
            raise ValueError("No value columns found in dataframe")
        is_multivariate = True
    
    window_size = config["data"]["window_size"]
    n_features = len(value_cols)
    
    # Нормализация данных
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(dataframe[value_cols])
    
    # Создание оконных данных
    X, y = [], []
    for i in range(len(scaled_values) - window_size):
        X.append(scaled_values[i:i + window_size])
        y.append(scaled_values[i + window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    # Разделение на train/validation
    split_idx = int(len(X) * (1 - config["models"][0]["validation_split"]))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Построение модели
    model_config = config["models"][0]
    model = Sequential()
    
    # Первый LSTM слой
    first_layer_units = model_config["layers"][0]["units"]
    model.add(LSTM(
        first_layer_units,
        input_shape=(window_size, n_features),
        return_sequences=len(model_config["layers"]) > 1
    ))
    
    # Последующие слои
    for layer in model_config["layers"][1:]:
        if layer["type"] == "LSTM":
            model.add(LSTM(
                layer["units"],
                return_sequences=layer.get("return_sequences", False)
            ))
        elif layer["type"] == "Dense":
            model.add(Dense(layer["units"]))
    
    if model_config.get("dropout"):
        model.add(Dropout(model_config["dropout"]))
    
    # Выходной слой
    model.add(Dense(n_features))
    
    # Компиляция
    optimizer = Adam(learning_rate=model_config["learning_rate"]) if model_config["optimizer"] == "adam" \
        else RMSprop(learning_rate=model_config["learning_rate"])
    model.compile(loss=model_config["loss"], optimizer=optimizer)
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        epochs=model_config["epochs"],
        batch_size=model_config["batch_size"],
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return {
        "model": model,
        "config": model_config,
        "scaler": scaler,
        "history": history.history,
        "n_features": n_features,
        "is_multivariate": is_multivariate,
        "value_columns": value_cols
    }


# === 4. Сохранение модели ===
async def save_model(metric_id, model, model_info):
    """
    Сохраняет модель и все связанные параметры в базу данных
    
    Args:
        metric_id: ID метрики
        model: Обученная модель Keras
        model_info: Словарь с информацией о модели, содержащий:
            - model_config: гиперпараметры модели
            - data_config: параметры данных (scaler, n_features и др.)
    """
    conn = await asyncpg.connect(dsn="postgresql://mad:secretPASSW0rd@80.93.60.49:30000/ml_models")
    
    # Сериализация компонентов
    model_data = pickle.dumps({
        'model': model,
        'scaler': model_info['data_config']['scaler'],
        'window_size': model_info['data_config']['window_size']
    })
    
    hyperparams = {
        'model_config': model_info['model_config'],
        'data_config': {
            'n_features': model_info['data_config']['n_features'],
            'is_multivariate': model_info['data_config']['is_multivariate'],
            'value_columns': model_info['data_config']['value_columns']
        }
    }
    
    await conn.execute("""
        INSERT INTO models(name, model_data, created_at, last_updated, hyperparams, status, version)
        VALUES($1, $2, $3, $4, $5, $6, $7)
    """, 
    f"model_{metric_id}", 
    model_data,
    datetime.utcnow(),
    datetime.utcnow(),
    hyperparams,
    "trained",
    "1.0")
    
    await conn.close()

# === 5. Обработка метрики из очереди ===
async def process_metric(queue, config, db_pool):
    while True:
        metric = await queue.get()
        print(f"🚀 Start training for metric: {metric['name']}")
        data = await fetch_metric_data(metric["name"], db_pool, config)
        training_result = await train_lstm_model(data, config)
        await save_model(
            metric["id"], 
            training_result["model"], 
            {
                "hyperparams": training_result["config"],
                "scaler": training_result["scaler"],
                "n_features": training_result["n_features"],
                "is_multivariate": training_result["is_multivariate"],
                "value_columns": training_result["value_columns"]
            }
        )
        print(f"✅ Model saved for metric: {metric['name']}")
        queue.task_done()


# === 6. Запуск воркеров ===
async def run_training_queue(queue, config, db_pool):
    workers = [asyncio.create_task(process_metric(queue, config, db_pool)) 
               for _ in range(config["infrastructure"]["workers"])]
    await asyncio.gather(*workers)

# === 7. Подписка на события PostgreSQL ===
async def listen_for_metrics(queue, config, db_pool):
    async def handle_notify(conn, pid, channel, payload):
        print(f"📡 Notification received on {channel}: {payload}")
        try:
            metric_id = int(payload)
        except ValueError:
            print("⚠️ Invalid payload:", payload)
            return

        row = await conn.fetchrow("""
            SELECT id, name FROM metrics 
            WHERE id = $1 AND status = active 
              AND id NOT IN (SELECT metric_id FROM metric_models)
        """, metric_id)

        if row:
            await queue.put(dict(row))
            print(f"📥 Metric {row['name']} added to queue")
        else:
            print(f"⏭ Metric {metric_id} already processed or inactive")

    conn = await asyncpg.connect(dsn="postgresql://mad:secretPASSW0rd@80.93.60.49:30000/ml_models")
    await conn.add_listener('new_active_metric', lambda *args: asyncio.create_task(handle_notify(*args, queue=queue)))
    print("👂 Listening for 'new_active_metric' notifications...")

    while True:
        await asyncio.sleep(3600)  # Просто держим соединение открытым


# === 8. Главный запуск ===
async def main():
    config = load_config()
    db_pool = await create_db_pool()
    
    queue = asyncio.Queue()
    await asyncio.gather(
        listen_for_metrics(queue, config, db_pool),
        run_training_queue(queue, config, db_pool)
    )
if __name__ == "__main__":
    asyncio.run(main())
