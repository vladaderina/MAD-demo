import logging
from logging.handlers import RotatingFileHandler
import asyncio
import json
import os
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple

import asyncpg
import aiohttp
import yaml
import numpy as np
import pandas as pd
from aiohttp import web
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Константы
DEFAULT_WINDOW_SIZE = 24
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
MODEL_STATUS_ACTIVE = 'active'
MODEL_STATUS_WAITING = 'waiting'
MODEL_STATUS_INACTIVE = 'inactive'
DEFAULT_RETRAIN_INTERVAL = 24  # часов

# Настройка логгера
logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Исключение для ошибок конфигурации"""
    pass

class ModelTrainer:
    """Сервис обучения и переобучения моделей обнаружения аномалий."""

    def __init__(self, trainer_config_path: str = "trainer-config.yaml"):
        """Инициализация сервиса."""
        self.victoriametrics_url = self._get_required_env('VICTORIAMETRICS_URL')
        self.db_conn_string = self._get_required_env('DB_CONN_STRING')
        self.log_path = os.getenv('LOG_PATH', './anomaly_detection.log')
        
        self.trainer_config = self._load_trainer_config(trainer_config_path)
        self.db_pool = None
        self._setup_logging()
        self.task_queue = asyncio.Queue()
        self.app = web.Application()
        self.app.add_routes([
            web.post('/api/v1/train', self.handle_train_request),
            web.post('/api/v1/retrain', self.handle_retrain_request),
            web.get('/health', self.handle_health_check)
        ])
        
    def _get_required_env(self, var_name: str) -> str:
        """Получение обязательной переменной окружения."""
        value = os.getenv(var_name)
        if not value:
            raise ConfigError(f"Требуется переменная окружения {var_name}")
        return value
        
    def _setup_logging(self) -> None:
        """Настройка системы логирования."""
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        try:
            file_handler = RotatingFileHandler(
                self.log_path,
                maxBytes=5*1024*1024,
                backupCount=3
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except IOError as e:
            logger.error(f"Не удалось открыть файл логов {self.log_path}: {str(e)}")
            
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    def _load_trainer_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации для тренера."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            return config
        except FileNotFoundError:
            logger.warning(f"Файл конфигурации тренера не найден: {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Ошибка парсинга YAML конфигурации тренера: {str(e)}")
            return {}
    
    def _get_model_config(self, model_name: str) -> Optional[Dict]:
        """Получение конфигурации модели по имени."""
        # Теперь модели хранятся только в конфиге тренера
        return next(
            (m for m in self.trainer_config.get('models', []) 
             if m['name'] == model_name),
            None
        )
    
    def _get_metric_config(self, metric_name: str) -> Optional[Dict]:
        """Получение конфигурации метрики по имени."""
        # Метрики теперь хранятся только в конфиге тренера
        return next(
            (m for m in self.trainer_config.get('metrics', []) 
             if m['name'] == metric_name),
            None
        )
    
    async def _create_db_pool(self) -> asyncpg.Pool:
        """Создание пула подключений к PostgreSQL."""
        try:
            db_config = self._parse_db_conn_string(self.db_conn_string)
            return await asyncpg.create_pool(**db_config)
        except Exception as e:
            logger.error(f"Ошибка создания пула подключений: {str(e)}")
            raise
    
    def _parse_db_conn_string(self, conn_string: str) -> Dict:
        """Парсинг строки подключения к БД."""
        try:
            parts = conn_string.split('://')[1].split('@')
            user_pass = parts[0].split(':')
            host_port_db = parts[1].split('/')
            host_port = host_port_db[0].split(':')
            
            return {
                'user': user_pass[0],
                'password': user_pass[1],
                'database': host_port_db[1],
                'host': host_port[0],
                'port': int(host_port[1])
            }
        except (IndexError, ValueError) as e:
            logger.error(f"Неверный формат строки подключения к БД: {conn_string}")
            raise ConfigError("Неверный формат строки подключения к БД")
    
    async def handle_health_check(self, request: web.Request) -> web.Response:
        """Проверка здоровья сервиса."""
        return web.json_response({"status": "ok"})
    
    async def handle_train_request(self, request: web.Request) -> web.Response:
        """Обработчик запроса на обучение новой модели."""
        try:
            data = await request.json()
            model_id = data.get('model_id')
            
            if not model_id:
                return web.json_response(
                    {"error": "model_id is required"},
                    status=400
                )
            
            # Проверяем, что модель существует
            async with self.db_pool.acquire() as conn:
                exists = await conn.fetchval(
                    "SELECT 1 FROM models WHERE id = $1 AND status = 'waiting'",
                    model_id
                )
                if not exists:
                    return web.json_response(
                        {"error": "Model not found or not in waiting status"},
                        status=404
                    )
            
            await self.task_queue.put(model_id)
            logger.info(f"Получен запрос на обучение модели ID {model_id}")
            
            return web.json_response(
                {"status": "queued", "model_id": model_id},
                status=202
            )
            
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON"},
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {str(e)}")
            return web.json_response(
                {"error": f"Internal server error: {str(e)}"},
                status=500
            )
    
    async def handle_retrain_request(self, request: web.Request) -> web.Response:
        """Обработчик запроса на переобучение модели."""
        try:
            data = await request.json()
            model_id = data.get('model_id')
            
            if not model_id:
                return web.json_response(
                    {"error": "model_id is required"},
                    status=400
                )
            
            # Проверяем, что модель существует и активна
            async with self.db_pool.acquire() as conn:
                model = await conn.fetchrow("""
                    SELECT mi.id, m.version 
                    FROM models_info mi
                    JOIN models m ON m.model_id = mi.id
                    WHERE mi.id = $1 AND m.status = 'active'
                    ORDER BY m.created_at DESC LIMIT 1
                """, model_id)
                
                if not model:
                    return web.json_response(
                        {"error": "Active model not found"},
                        status=404
                    )
            
            # Создаем новую версию модели
            async with self.db_pool.acquire() as conn:
                new_version = await conn.fetchval("""
                    INSERT INTO models (
                        model_id, status, version, hyperparams
                    ) VALUES (
                        $1, 'waiting', 
                        (SELECT COALESCE(MAX(version::float), 0) + 1 FROM models WHERE model_id = $1)::text,
                        (SELECT hyperparams FROM models WHERE model_id = $1 AND status = 'active' ORDER BY created_at DESC LIMIT 1)
                    )
                    RETURNING id
                """, model_id)
                
                logger.info(f"Создана новая версия модели ID {model_id} для переобучения")
                await self.task_queue.put(new_version)
                
                return web.json_response(
                    {"status": "queued", "model_id": model_id, "new_version_id": new_version},
                    status=202
                )
                
        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Invalid JSON"},
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {str(e)}")
            return web.json_response(
                {"error": f"Internal server error: {str(e)}"},
                status=500
            )
    
    async def _notify_model_ready(self, model_id: int) -> None:
        """Отправка уведомления о готовности модели."""
        detector_service_url = os.getenv('DETECTOR_SERVICE_URL')
        if not detector_service_url:
            logger.debug("URL сервиса обнаружения не задан, пропускаем отправку")
            return
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{detector_service_url}/api/v1/model_ready",
                    json={'model_id': model_id},
                    timeout=5
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Ошибка отправки уведомления: HTTP {resp.status}")
                    else:
                        logger.info(f"Уведомление о готовности модели {model_id} отправлено")
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления: {str(e)}")
    
    async def _load_waiting_models(self) -> None:
        """Загрузка моделей со статусом 'waiting' при старте сервиса."""
        try:
            async with self.db_pool.acquire() as conn:
                models = await conn.fetch("""
                    SELECT id FROM models 
                    WHERE status = 'waiting'
                    AND created_at > NOW() - INTERVAL '7 days'
                    ORDER BY created_at ASC
                """)
                
                for model in models:
                    await self.task_queue.put(model['id'])
                    logger.info(f"Добавлена в очередь модель ID {model['id']} (при старте сервиса)")
                
                if models:
                    logger.info(f"Загружено {len(models)} моделей для обработки")
                    
        except Exception as e:
            logger.error(f"Ошибка загрузки ожидающих моделей: {str(e)}")
    
    async def _fetch_metric_data(self, metric_name: str, time_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Получение данных метрики из VictoriaMetrics."""
        metric_config = self._get_metric_config(metric_name)
        if not metric_config:
            raise ValueError(f"Метрика '{metric_name}' не найдена в конфигурации")

        start_time, end_time = time_range
        url = f"{self.victoriametrics_url}/query_range"
        params = {
            "query": metric_config["query"],
            "start": int(start_time.timestamp()),
            "end": int(end_time.timestamp()),
            "step": "15s"  # По умолчанию, можно добавить в конфиг метрики
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    result = await resp.json()

                values = result["data"]["result"][0]["values"]
                df = pd.DataFrame(values, columns=["timestamp", "value"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df["value"] = pd.to_numeric(df["value"])
                return df.set_index("timestamp")
                
        except Exception as e:
            logger.error(f"Ошибка получения данных метрики: {str(e)}")
            raise
    
    async def _get_training_period(self, model_config: Dict) -> Tuple[datetime, datetime]:
        """Получение периода обучения для модели."""
        training_period = model_config.get('training_period', {})
        
        if 'fixed_range' in training_period:
            fixed = training_period['fixed_range']
            start = datetime.fromisoformat(fixed['start'])
            end = datetime.fromisoformat(fixed['end'])
            return start, end
        elif 'auto_range' in training_period:
            auto = training_period['auto_range']
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=int(auto['lookback_period'][:-1]))
            return start, end
        
        # По умолчанию - последние 7 дней
        end = datetime.now(timezone.utc)
        return end - timedelta(days=7), end
    
    async def _process_model(self, model_id: int) -> None:
        """Обработка модели из очереди на обучение."""
        logger.info(f"Начало обработки модели ID {model_id}")
        
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Получаем информацию о модели
                    model = await conn.fetchrow("""
                        SELECT m.id, m.model_id, m.status, m.version,
                               m.hyperparams::jsonb as hyperparams,
                               mi.name as model_name,
                               mi.hyperparameter_mode as hyperparams_mode,
                               met.name as metric_name
                        FROM models m
                        JOIN models_info mi ON m.model_id = mi.id
                        JOIN metrics met ON mi.metric_id = met.id
                        WHERE m.id = $1 AND m.status = 'waiting'
                        LIMIT 1
                    """, model_id)
                    
                    if not model:
                        logger.warning(f"Модель ID {model_id} не найдена или не в статусе 'waiting'")
                        return

                    logger.info(f"Обработка модели: {model['model_name']} (v{model['version']})")

                    # Получаем конфигурацию модели
                    model_config = self._get_model_config(model['model_name'])
                    if not model_config:
                        raise ValueError(f"Конфигурация модели {model['model_name']} не найдена")

                    # Получаем период обучения
                    training_start, training_end = await self._get_training_period(model_config)

                    # Получаем данные для обучения
                    try:
                        metric_data = await self._fetch_metric_data(
                            model['metric_name'],
                            (training_start, training_end)
                        )
                        if metric_data.empty:
                            raise ValueError("Нет данных для обучения")
                        logger.info(f"Получено {len(metric_data)} точек данных")
                    except Exception as e:
                        logger.error(f"Ошибка получения данных метрики: {str(e)}")
                        await conn.execute("""
                            UPDATE models 
                            SET status = 'inactive'
                            WHERE id = $1
                        """, model['id'])
                        return

                    # Обучаем модель
                    try:
                        logger.info("Запуск обучения модели...")
                        
                        # Получаем гиперпараметры в зависимости от режима
                        if model['hyperparams_mode'] == 'optuna' and self.trainer_config.get('optuna_tuning'):
                            hyperparams = await self._optimize_hyperparameters(
                                metric_data, 
                                model_config,
                                model['model_name']
                            )
                        else:
                            hyperparams = model_config.get('manual_params', {})
                        
                        training_result = await asyncio.to_thread(
                            self._train_lstm_model, 
                            metric_data, 
                            hyperparams,
                            model['model_name']
                        )
                        logger.info("Обучение успешно завершено")
                    except Exception as e:
                        logger.error(f"Ошибка обучения: {str(e)}")
                        await conn.execute("""
                            UPDATE models 
                            SET status = 'inactive'
                            WHERE id = $1
                        """, model['id'])
                        return

                    # Сохраняем модель
                    try:
                        model_data = pickle.dumps({
                            'model': training_result["model"],
                            'scaler': training_result["scaler"],
                            'window_size': training_result["window_size"]
                        })
                        
                        await conn.execute("""
                            UPDATE models SET
                                model_data = $1,
                                status = 'active',
                                hyperparams = $2,
                                created_at = NOW()
                            WHERE id = $3
                        """, model_data, json.dumps(training_result["config"]), model['id'])
                        
                        # Обновляем активную версию
                        await conn.execute("""
                            UPDATE models_info
                            SET active_version = $1,
                                training_start = NOW(),
                                training_end = NOW()
                            WHERE id = $2
                        """, model['version'], model['model_id'])
                        
                        logger.info("Модель успешно сохранена")
                        
                        # Отправляем уведомление о готовности модели
                        await self._notify_model_ready(model['model_id'])
                            
                    except Exception as e:
                        logger.error(f"Ошибка сохранения модели: {str(e)}")
                        await conn.execute("""
                            UPDATE models 
                            SET status = 'inactive'
                            WHERE id = $1
                        """, model['id'])
                        return

                    logger.info(f"Модель {model_id} успешно обработана")

        except Exception as e:
            logger.error(f"Критическая ошибка обработки модели {model_id}: {str(e)}")
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE models 
                        SET status = 'inactive'
                        WHERE id = $1
                    """, model_id)
            except Exception as e:
                logger.error(f"Ошибка обновления статуса модели: {str(e)}")
    
    async def _optimize_hyperparameters(self, data: pd.DataFrame, 
                                      model_config: Dict, model_name: str) -> Dict:
        """Оптимизация гиперпараметров с использованием Optuna."""
        if not self.trainer_config.get('optuna_tuning'):
            return {}
            
        optuna_config = self.trainer_config['optuna_tuning']
        
        def objective(trial):
            # Параметры из конфигурации Optuna
            params = {
                'window_size': trial.suggest_int('window_size', 12, 48, step=6),
                'layers': [],
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    optuna_config['hyperparameter_ranges']['learning_rate']['min'],
                    optuna_config['hyperparameter_ranges']['learning_rate']['max'],
                    log=optuna_config['hyperparameter_ranges']['learning_rate'].get('log', False)
                ),
                'batch_size': trial.suggest_categorical(
                    'batch_size',
                    optuna_config['hyperparameter_ranges']['batch_size']
                ),
                'dropout': trial.suggest_float(
                    'dropout',
                    optuna_config['hyperparameter_ranges']['dropout']['min'],
                    optuna_config['hyperparameter_ranges']['dropout']['max']
                ),
                'epochs': trial.suggest_int(
                    'epochs',
                    optuna_config['hyperparameter_ranges']['epochs']['min'],
                    optuna_config['hyperparameter_ranges']['epochs']['max']
                )
            }
            
            # Добавляем слои из конфигурации
            for layer_cfg in optuna_config['hyperparameter_ranges']['layers']:
                if layer_cfg['type'] == 'LSTM':
                    params['layers'].append({
                        'type': 'LSTM',
                        'units': trial.suggest_int(
                            f"lstm_units_{len(params['layers'])}",
                            layer_cfg['units']['min'],
                            layer_cfg['units']['max'],
                            step=layer_cfg['units'].get('step', 32)
                        ),
                        'return_sequences': trial.suggest_categorical(
                            f"lstm_return_seq_{len(params['layers'])}",
                            layer_cfg['return_sequences']
                        )
                    })
                elif layer_cfg['type'] == 'Dense':
                    params['layers'].append({
                        'type': 'Dense',
                        'units': trial.suggest_int(
                            f"dense_units_{len(params['layers'])}",
                            layer_cfg['units']['min'],
                            layer_cfg['units']['max']
                        )
                    })
            
            # Фиксированные параметры
            params.update({
                'loss': optuna_config['fixed_parameters']['loss'],
                'optimizer': optuna_config['fixed_parameters']['optimizer'],
                'validation_split': optuna_config['fixed_parameters']['validation_split']
            })
            
            # Обучаем модель с текущими параметрами
            try:
                result = self._train_lstm_model(data, params, f"{model_name}_trial_{trial.number}")
                return result['history']['val_loss'][-1]
            except Exception as e:
                logger.error(f"Ошибка в trial {trial.number}: {str(e)}")
                return float('inf')
        
        study = optuna.create_study(
            direction=optuna_config['direction'],
            sampler=TPESampler(),
            pruner=HyperbandPruner()
        )
        
        logger.info(f"Начало оптимизации гиперпараметров для модели {model_name}")
        study.optimize(objective, n_trials=optuna_config['n_trials'])
        logger.info(f"Оптимизация завершена. Лучшие параметры: {study.best_params}")
        
        return study.best_params
    
    def _train_lstm_model(self, dataframe: pd.DataFrame, hyperparams: Dict, model_name: str) -> Dict:
        """Обучение LSTM модели."""
        try:
            logger.info(f"Начало обучения модели {model_name}")
            
            # Устанавливаем параметры по умолчанию
            config_params = {
                "window_size": DEFAULT_WINDOW_SIZE,
                "validation_split": 0.2,
                "layers": [
                    {"type": "LSTM", "units": 64, "return_sequences": False}
                ],
                "epochs": DEFAULT_EPOCHS,
                "batch_size": DEFAULT_BATCH_SIZE,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "optimizer": "adam",
                "loss": "mse",
                "dropout": 0.2
            }
            
            if isinstance(hyperparams, dict):
                config_params.update(hyperparams)
            
            window_size = config_params["window_size"]
            n_features = len(dataframe.columns)
            
            # Масштабирование данных
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(dataframe)
            
            # Подготовка данных
            X, y = [], []
            for i in range(len(scaled_values) - window_size):
                X.append(scaled_values[i:i + window_size])
                y.append(scaled_values[i + window_size, 0])
            
            X = np.array(X)
            y = np.array(y)

            # Разделение данных
            split_idx = int(len(X) * (1 - config_params["validation_split"]))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Создание модели
            model = Sequential()
            
            for i, layer in enumerate(config_params["layers"]):
                layer_type = layer["type"].lower()
                units = layer.get("units", 64)
                return_sequences = layer.get("return_sequences", i < len(config_params["layers"]) - 1)
                
                if layer_type == "lstm":
                    if i == 0:
                        model.add(LSTM(
                            units,
                            input_shape=(window_size, n_features),
                            return_sequences=return_sequences
                        ))
                    else:
                        model.add(LSTM(
                            units,
                            return_sequences=return_sequences
                        ))
                elif layer_type == "dense":
                    model.add(Dense(units))
                
                if "dropout" in layer:
                    model.add(Dropout(layer["dropout"]))
            
            model.add(Dense(1))
            
            if "dropout" in config_params:
                model.add(Dropout(config_params["dropout"]))
            
            # Компиляция модели
            optimizer = Adam(learning_rate=config_params["learning_rate"]) \
                if config_params["optimizer"] == "adam" \
                else RMSprop(learning_rate=config_params["learning_rate"])
            model.compile(loss=config_params["loss"], optimizer=optimizer)
            
            # Обучение модели
            history = model.fit(
                X_train, y_train,
                epochs=config_params["epochs"],
                batch_size=config_params["batch_size"],
                validation_data=(X_val, y_val),
                verbose=1
            )
            
            logger.info(f"Обучение модели {model_name} завершено")
            
            return {
                "model": model,
                "config": config_params,
                "scaler": scaler,
                "history": history.history,
                "window_size": window_size,
                "n_features": n_features
            }
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {str(e)}")
            raise RuntimeError(f"Ошибка обучения модели: {str(e)}")
    
    async def _check_retrain_required(self) -> List[int]:
        """Проверка необходимости переобучения моделей."""
        models_for_retrain = []
        
        try:
            async with self.db_pool.acquire() as conn:
                models = await conn.fetch("""
                    SELECT mi.id, mi.name, mi.training_end, 
                           (SELECT MAX(version::float) FROM models WHERE model_id = mi.id) as max_version,
                           mi.max_stored_versions
                    FROM models_info mi
                    WHERE EXISTS (
                        SELECT 1 FROM models 
                        WHERE model_id = mi.id AND status = 'active'
                    )
                """)
                
                for model in models:
                    model_config = self._get_model_config(model['name'])
                    if not model_config or not model_config.get('retrain', {}).get('enabled', False):
                        continue
                    
                    interval = model_config['retrain'].get('interval', f"{DEFAULT_RETRAIN_INTERVAL}h")
                    interval_hours = self._parse_interval(interval)
                    
                    if not model['training_end'] or \
                       (datetime.now(timezone.utc) >= model['training_end'] + timedelta(hours=interval_hours)):
                        models_for_retrain.append(model['id'])
                        logger.info(f"Модель {model['name']} требует переобучения")
                        
        except Exception as e:
            logger.error(f"Ошибка проверки переобучения: {str(e)}")
        
        return models_for_retrain
    
    def _parse_interval(self, interval_str: str) -> int:
        """Парсинг строки интервала в часы."""
        if interval_str.endswith('h'):
            return int(interval_str[:-1])
        elif interval_str.endswith('d'):
            return int(interval_str[:-1]) * 24
        return DEFAULT_RETRAIN_INTERVAL
    
    async def _retrain_checker(self) -> None:
        """Периодическая проверка необходимости переобучения моделей."""
        while True:
            try:
                models_to_retrain = await self._check_retrain_required()
                for model_id in models_to_retrain:
                    await self.handle_retrain_request(web.Request(
                        method='POST',
                        path='/api/v1/retrain',
                        match_info={},
                        headers={},
                        content_type='application/json',
                        body=json.dumps({'model_id': model_id}).encode()
                    ))
                await asyncio.sleep(3600)  # Проверка каждый час
            except Exception as e:
                logger.error(f"Ошибка в планировщике переобучения: {str(e)}")
                await asyncio.sleep(60)
    
    async def _worker_loop(self) -> None:
        """Цикл обработки задач воркерами."""
        while True:
            try:
                model_id = await self.task_queue.get()
                try:
                    await self._process_model(model_id)
                except Exception as e:
                    logger.error(f"Ошибка обработки задачи: {e}", exc_info=True)
                finally:
                    self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка воркера: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def start(self) -> None:
        """Запуск сервиса."""
        try:
            logger.info("Запуск сервиса обучения моделей")
            
            # Инициализация подключения к БД
            self.db_pool = await self._create_db_pool()
            
            # Загрузка существующих моделей в статусе 'waiting'
            await self._load_waiting_models()
            
            # Запуск HTTP сервера
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            host = os.getenv('HOST', '0.0.0.0')
            port = int(os.getenv('PORT', 8080))
            
            site = web.TCPSite(runner, host, port)
            await site.start()
            logger.info(f"HTTP сервер запущен на {host}:{port}")
            
            # Запуск воркеров
            workers = [
                asyncio.create_task(self._worker_loop()) 
                for _ in range(int(os.getenv('WORKERS', 3)))
            ]
            
            # Запуск планировщика переобучения
            checker = asyncio.create_task(self._retrain_checker())
            
            await asyncio.gather(*workers, checker)
            
        except Exception as e:
            logger.error(f"Ошибка сервиса: {str(e)}", exc_info=True)
            raise
        finally:
            if self.db_pool:
                await self.db_pool.close()
            logger.info("Сервис остановлен")

async def main():
    """Основная функция для запуска сервиса."""
    try:
        trainer_config_path = os.getenv('TRAINER_CONFIG_PATH', 'mad-trainer-config.yaml')
        
        service = ModelTrainer(trainer_config_path)
        await service.start()
    except ConfigError as e:
        logger.error(f"Ошибка конфигурации: {str(e)}")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания, остановка сервиса")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)