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
    print("VM get metricsssssssssssssss")
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
async def save_model(
    model_id: int,
    model: object,
    model_info: dict,
    db_pool
) -> None:
    """
    Saves the model and all related parameters to the database according to the new schema
    """
    async with db_pool.acquire() as conn:
        try:
            # Start transaction
            async with conn.transaction():
                # Get the next version number
                version = await conn.fetchval("""
                    SELECT COALESCE(MAX(version::integer), 0) + 1 
                    FROM models 
                    WHERE model_id = $1
                """, model_id)
                
                print(f"📦 Saving model {model_id} as version {version}")
                
                # Serialize model, scaler and related data
                model_data = pickle.dumps({
                    'model': model,
                    'scaler': model_info['scaler'],
                    'window_size': model_info['window_size']
                })
                
                # Save new model version
                await conn.execute("""
                    INSERT INTO models (
                        model_id,
                        model_data,
                        status,
                        version,
                        hyperparams,
                        created_at
                    ) VALUES ($1, $2, 'active', $3, $4, NOW())
                """, 
                model_id,
                model_data,
                str(version),
                json.dumps(model_info['config']))
                
                print(f"💾 Saved model {model_id} version {version}")
                
                # Update active version in models_info if this is the first version
                if version == 1:
                    await conn.execute("""
                        UPDATE models_info
                        SET active_version = $1
                        WHERE id = $2
                    """, str(version), model_id)
                
                # Clean up old versions if exceeding max_stored_versions
                await conn.execute("""
                    DELETE FROM models 
                    WHERE id IN (
                        SELECT id FROM models 
                        WHERE model_id = $1 
                        AND id NOT IN (
                            SELECT id FROM models 
                            WHERE model_id = $1 
                            ORDER BY created_at DESC 
                            LIMIT (SELECT max_stored_versions FROM models_info WHERE id = $1)
                        )
                    )
                """, model_id)
                
                print(f"🧹 Cleaned up old versions for model {model_id}")
        
        except Exception as e:
            print(f"❌ Failed to save model {model_id}: {str(e)}")
            raise

# === 5. Обработка модели из очереди ===
async def process_model(queue, db_pool):
    while True:
        model_id = await queue.get()
        print(f"🚀 Start processing model with ID: {model_id}")
        
        async with db_pool.acquire() as conn:
            try:
                # Start transaction
                async with conn.transaction():
                    # Get model info and update status to 'training'
                    model_info = await conn.fetchrow("""
                        UPDATE models_info 
                        SET status = 'training'
                        WHERE id = $1 
                        RETURNING id, name, metric_id, hyperparams_mode, 
                                  training_start, training_end, status
                    """, model_id)
                    
                    if not model_info:
                        print(f"⏭ Model info {model_id} not found")
                        continue
                    
                    # Get main metric
                    main_metric = await conn.fetchrow("""
                        SELECT id, name, query, step 
                        FROM metrics 
                        WHERE id = $1 AND status = 'active'
                    """, model_info['metric_id'])
                    
                    if not main_metric:
                        print(f"⏭ Main metric for model {model_info['name']} not found or inactive")
                        await conn.execute("""
                            UPDATE models_info 
                            SET status = 'deactive' 
                            WHERE id = $1
                        """, model_id)
                        continue
                    
                    # Get related features (metrics)
                    features = await conn.fetch("""
                        SELECT m.id, m.name, m.query, m.step 
                        FROM features f
                        JOIN metrics m ON f.metric_id = m.id
                        WHERE f.model_id = $1 AND m.status = 'active'
                    """, model_id)
                    
                    # Combine all metrics (main + features)
                    all_metrics = [main_metric] + [dict(f) for f in features]
                    
                    # Get manual parameters if needed
                    manual_params = {}
                    if model_info['hyperparams_mode'] == 'manual':
                        active_model = await conn.fetchrow("""
                            SELECT hyperparams 
                            FROM models 
                            WHERE model_id = $1 AND status = 'waiting'
                            ORDER BY created_at DESC 
                            LIMIT 1
                        """, model_id)
                        manual_params = dict(active_model['hyperparams']) if active_model else {}
                    
                    # Get data for all metrics
                    metric_data = []
                    for metric in all_metrics:
                        data = await fetch_metric_data(
                            metric['query'],
                            metric['step'],
                            model_info['training_start'],
                            model_info['training_end'],
                            db_pool
                        )
                        metric_data.append(data)
                    
                    # Train model
                    training_result = await train_lstm_model(
                        metric_data,
                        manual_params
                    )
                    
                    # Save the model (creates new entry in models table)
                    await save_model(
                        model_id,
                        training_result["model"],
                        {
                            "config": training_result["config"],
                            "scaler": training_result["scaler"],
                            "window_size": training_result["window_size"]
                        },
                        db_pool
                    )
                    
                    # Update model info status and active version
                    await conn.execute("""
                        UPDATE models_info 
                        SET status = 'active', 
                            active_version = $2,
                            training_start = NOW(),
                            training_end = NOW()
                        WHERE id = $1
                    """, model_id, str(training_result["version"]))
                    
                    print(f"✅ Model {model_info['name']} (ID: {model_id}) successfully trained and saved")
            
            except Exception as e:
                print(f"❌ Error processing model {model_id}: {str(e)}")
                # On error, revert status appropriately
                try:
                    await conn.execute("""
                        UPDATE models_info 
                        SET status = CASE 
                            WHEN status = 'training' THEN 'active' 
                            ELSE status 
                        END
                        WHERE id = $1
                    """, model_id)
                except Exception as update_err:
                    print(f"⚠️ Failed to update model status: {update_err}")
            
            finally:
                queue.task_done()

# === 6. Запуск воркеров ===
async def run_training_queue(queue, config, db_pool):
    print("122")
    workers = [
        asyncio.create_task(process_model(queue, db_pool)) 
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
                SELECT id FROM models 
                WHERE id = $1 AND status = 'waiting'
            """, model_id)

            if model:
                await queue.put(model["id"])
                print(f"📥 Model {model['id']} added to queue")
            else:
                print(f"⏭ Model {model_id} is inactive or not found")

    conn = await asyncpg.connect(dsn=config["system"]["db_conn_string"])
    await conn.add_listener('model_training_queue', lambda conn, pid, channel, payload:
        asyncio.create_task(handle_notify(conn, pid, channel, payload)))

    print("👂 Listening for 'model_training_queue' notifications...")

    while True:
        await asyncio.sleep(3600)  # Поддержка живого соединения

# === 8. Главный запуск ===
async def main():
    config = load_config()
    db_pool = await create_db_pool()
    
    queue = asyncio.Queue()

    await asyncio.gather(
        listen_for_models(queue, config, db_pool),
        run_training_queue(queue, config, db_pool)
    )

if __name__ == "__main__":
    asyncio.run(main())