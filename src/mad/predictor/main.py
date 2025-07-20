import asyncio
import asyncpg
import aiohttp
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop

async def create_db_pool():
    return await asyncpg.create_pool(
        user="mad",
        password="secretPASSW0rd",
        database="ml_models",
        host="80.93.60.49",
        port=30000,
    )

def iso_to_timestamp(iso_date):
    if isinstance(iso_date, datetime):
        dt = iso_date
    else:
        dt = datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%SZ")
    return int(dt.timestamp() * 1000)

# === 1. Загрузка конфигурации ===
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# === 2. Получение данных из VictoriaMetrics ===
async def fetch_metric_data(metric_name: str, db_pool, config: dict) -> pd.DataFrame:
    """
    Получает данные из VictoriaMetrics для метрики.
    Возвращает DataFrame с временными рядами.
    """
    async with db_pool.acquire() as conn:
        metric = await conn.fetchrow("""
            SELECT id, query, step FROM metrics 
            WHERE name = $1
        """, metric_name)

    if not metric:
        raise ValueError(f"Metric '{metric_name}' not found in database")

    url = f"{config['system']['victoriametrics_url']}/query_range"
    params = {
        "query": metric["query"],
        "start": iso_to_timestamp(datetime.now(timezone.utc) - 3600 * 1000),  # Последний час
        "end": iso_to_timestamp(datetime.now(timezone.utc)),
        "step": f"{metric['step']}s"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            result = await resp.json()

        try:
            values = result["data"]["result"][0]["values"]
        except (KeyError, IndexError):
            raise RuntimeError(f"No data returned for metric {metric_name}")

        df = pd.DataFrame(values, columns=["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["value"] = df["value"].astype(float)
        return df.set_index("timestamp")

# === 3. Обучение LSTM модели ===
async def train_lstm_model(dataframe: pd.DataFrame, model_config: dict):
    # Подготовка данных
    window_size = model_config.get("window_size", 10)
    
    # Нормализация данных
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(dataframe[["value"]])
    
    # Создание оконных данных
    X, y = [], []
    for i in range(len(scaled_values) - window_size):
        X.append(scaled_values[i:i + window_size])
        y.append(scaled_values[i + window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    # Разделение на train/validation
    split_idx = int(len(X) * (1 - model_config.get("validation_split", 0.2)))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Построение модели
    model = Sequential()
    
    # Первый LSTM слой
    first_layer_units = model_config["layers"][0]["units"]
    model.add(LSTM(
        first_layer_units,
        input_shape=(window_size, 1),
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
    model.add(Dense(1))
    
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
        "window_size": window_size
    }

# === 4. Сохранение модели ===
async def save_model(model_name: str, model, model_info, db_pool):
    """
    Сохраняет модель и все связанные параметры в базу данных
    """
    async with db_pool.acquire() as conn:
        # Получаем ID модели
        model_id = await conn.fetchval("""
            SELECT id FROM models WHERE name = $1
        """, model_name)
        
        if not model_id:
            # Создаем новую модель если не существует
            model_id = await conn.fetchval("""
                INSERT INTO models(
                    name, 
                    max_stored_versions, 
                    hyperparams_mode, 
                    status, 
                    active_version,
                    training_start,
                    training_end
                )
                VALUES($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """, 
            model_name,
            5,  # default max_stored_versions
            "manual",  # или "optuna" в зависимости от конфига
            "active",
            "1.0",
            datetime.now(timezone.utc),
            datetime.now(timezone.utc))

        # Сериализация модели и scaler
        model_data = pickle.dumps({
            'model': model,
            'scaler': model_info['scaler'],
            'window_size': model_info['window_size']
        })
        
        # Сохранение версии модели
        await conn.execute("""
            INSERT INTO models_version(
                model_data, 
                version, 
                model_id, 
                hyperparams
            )
            VALUES($1, $2, $3, $4)
        """, 
        model_data,
        "1.0",  # Можно реализовать логику версионирования
        model_id,
        json.dumps(model_info['config']))

# === 5. Обработка модели из очереди ===
async def process_model(queue, config, db_pool):
    while True:
        model_name = await queue.get()
        print(f"🚀 Start training for model: {model_name}")
        
        try:
            # Получаем конфигурацию модели
            model_config = next(
                m for m in config["mad_predictor"]["models"] 
                if m["name"] == model_name
            )
            
            # Получаем основную метрику
            main_metric = model_config["main_metric"]
            
            # Получаем данные для обучения
            data = await fetch_metric_data(main_metric, db_pool, config)
            
            # Обучаем модель
            training_result = await train_lstm_model(
                data, 
                model_config.get("manual_params", {})
            )
            
            # Сохраняем модель
            await save_model(
                model_name,
                training_result["model"],
                {
                    "config": training_result["config"],
                    "scaler": training_result["scaler"],
                    "window_size": training_result["window_size"]
                },
                db_pool
            )
            
            print(f"✅ Model saved for: {model_name}")
            
        except Exception as e:
            print(f"❌ Error processing model {model_name}: {str(e)}")
            
        finally:
            queue.task_done()

# === 6. Запуск воркеров ===
async def run_training_queue(queue, config, db_pool):
    workers = [
        asyncio.create_task(process_model(queue, config, db_pool)) 
        for _ in range(config["system"].get("workers", 3))
    ]
    await asyncio.gather(*workers)

# === 7. Подписка на события PostgreSQL ===
async def listen_for_models(queue, config, db_pool):
    async def handle_notify(conn, pid, channel, payload):
        print(f"📡 Notification received on {channel}: {payload}")
        try:
            model_id = int(payload)
        except ValueError:
            print("⚠️ Invalid payload:", payload)
            return
            
        async with db_pool.acquire() as conn:
            model = await conn.fetchrow("""
                SELECT id, name FROM models 
                WHERE id = $1 AND status = 'active'
            """, model_id)

            if model:
                await queue.put(model["name"])
                print(f"📥 Model {model['name']} added to queue")
            else:
                print(f"⏭ Model {model_id} is inactive or not found")

    conn = await asyncpg.connect(dsn=config["system"]["db_conn_string"])
    await conn.add_listener('new_active_model', lambda conn, pid, channel, payload:
        asyncio.create_task(handle_notify(conn, pid, channel, payload)))

    print("👂 Listening for 'new_active_model' notifications...")

    while True:
        await asyncio.sleep(3600)  # Поддержка живого соединения

# === 8. Главный запуск ===
async def main():
    config = load_config()
    db_pool = await create_db_pool()
    
    queue = asyncio.Queue()
    
    # Добавляем модели из конфига в очередь для первоначального обучения
    for model in config["mad_predictor"]["models"]:
        await queue.put(model["name"])
    
    await asyncio.gather(
        listen_for_models(queue, config, db_pool),
        run_training_queue(queue, config, db_pool)
    )

if __name__ == "__main__":
    asyncio.run(main())