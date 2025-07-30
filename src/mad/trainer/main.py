import logging
from logging.handlers import RotatingFileHandler
import asyncio
import json
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

import asyncpg
import aiohttp
import yaml
import numpy as np
import pandas as pd
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


class ModelTrainer:
    """Сервис обучения и переобучения моделей обнаружения аномалий."""

    def __init__(self, config_path: str = "config.yaml"):
        """Инициализация сервиса.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config = self._load_config(config_path)
        self.db_pool = None
        self._setup_logging()
        self.task_queue = asyncio.Queue()
        
    def _setup_logging(self) -> None:
        """Настройка системы логирования."""
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Файловый обработчик с ротацией
        file_handler = RotatingFileHandler(
            self.config['system'].get('log_path', 'service.log'),
            maxBytes=5*1024*1024,
            backupCount=3
        )
        file_handler.setFormatter(formatter)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурационного файла.
        
        Args:
            config_path: Путь к конфигурационному файлу
            
        Returns:
            Словарь с конфигурацией
            
        Raises:
            FileNotFoundError: Если файл конфигурации не найден
            yaml.YAMLError: Если ошибка парсинга YAML
        """
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Файл конфигурации не найден: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Ошибка парсинга YAML: {str(e)}")
            raise
    
    async def _create_db_pool(self) -> asyncpg.Pool:
        """Создание пула подключений к PostgreSQL.
        
        Returns:
            Пул подключений к БД
            
        Raises:
            asyncpg.PostgresError: При ошибках подключения к БД
        """
        try:
            db_config = self._parse_db_conn_string(self.config['system']['db_conn_string'])
            return await asyncpg.create_pool(**db_config)
        except Exception as e:
            logger.error(f"Ошибка создания пула подключений: {str(e)}")
            raise
    
    def _parse_db_conn_string(self, conn_string: str) -> Dict:
        """Парсинг строки подключения к БД.
        
        Args:
            conn_string: Строка подключения в формате postgresql://user:password@host:port/database
            
        Returns:
            Словарь с параметрами подключения
        """
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
    
    async def _fetch_metric_data(self, metric_name: str) -> pd.DataFrame:
        """Получение данных метрики из VictoriaMetrics.
        
        Args:
            metric_name: Название метрики
            
        Returns:
            DataFrame с данными метрики
            
        Raises:
            ValueError: Если метрика не найдена или нет данных
            aiohttp.ClientError: При ошибках HTTP запроса
        """
        try:
            async with self.db_pool.acquire() as conn:
                metric = await conn.fetchrow("""
                    SELECT id, query, step FROM metrics 
                    WHERE name = $1 AND status = 'active'
                """, metric_name)
            
            if not metric:
                raise ValueError(f"Метрика '{metric_name}' не найдена")

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=1)
            
            url = f"{self.config['system']['victoriametrics_url']}/query_range"
            params = {
                "query": metric["query"],
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "step": f"{metric['step']}s"
            }

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
    
    def _get_model_config(self, model_name: str) -> Optional[Dict]:
        """Получение конфигурации модели по имени.
        
        Args:
            model_name: Название модели
            
        Returns:
            Словарь с конфигурацией модели или None, если модель не найдена
        """
        return next(
            (m for m in self.config.get('mad_predictor', {}).get('models', []) 
             if m['name'] == model_name),
            None
        )
    
    def _get_model_architecture(self, model_config: Dict) -> Optional[Dict]:
        """Получение архитектуры модели из конфигурации.
        
        Args:
            model_config: Конфигурация модели
            
        Returns:
            Словарь с параметрами модели или None, если требуется оптимизация
        """
        if model_config.get('hyperparameter_mode') == 'manual' and 'manual_params' in model_config:
            return model_config['manual_params']
        if model_config.get('hyperparameter_mode') == 'optuna':
            return None
        
        # Значения по умолчанию
        return {
            "layers": [
                {"type": "LSTM", "units": 64, "return_sequences": True},
                {"type": "LSTM", "units": 32, "return_sequences": False},
                {"type": "Dense", "units": 1}
            ],
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout": 0.2,
            "epochs": 50,
            "loss": "mean_squared_error",
            "optimizer": "adam",
            "validation_split": 0.2,
            "window_size": 24
        }
    
    def _create_optuna_study(self) -> optuna.Study:
        """Создание объекта Study для Optuna.
        
        Returns:
            Объект Study для оптимизации гиперпараметров
        """
        optuna_config = self.config.get('mad_predictor', {}).get('optuna_tuning', {})
        
        study_params = {
            'direction': 'minimize',
            'sampler': TPESampler(),
            'pruner': HyperbandPruner(),
        }
        
        if 'direction' in optuna_config:
            study_params['direction'] = optuna_config['direction']
        if 'sampler' in optuna_config and optuna_config['sampler'] == 'TPE':
            study_params['sampler'] = TPESampler()
        if 'pruner' in optuna_config and optuna_config['pruner'] == 'Hyperband':
            study_params['pruner'] = HyperbandPruner()
        
        return optuna.create_study(**study_params)
    
    def _get_optuna_params(self, trial: optuna.Trial) -> Dict:
        """Генерация параметров для Optuna.
        
        Args:
            trial: Объект Trial из Optuna
            
        Returns:
            Словарь с параметрами модели
        """
        optuna_config = self.config.get('mad_predictor', {}).get('optuna_tuning', {})
        params = {'layers': []}
        
        # Обработка слоев LSTM
        lstm_layers = [layer for layer in optuna_config.get('hyperparameter_ranges', {}).get('layers', []) 
                      if layer['type'] == 'LSTM']
        for i, layer_ranges in enumerate(lstm_layers):
            layer = {'type': 'LSTM'}
            if 'units' in layer_ranges:
                layer['units'] = trial.suggest_int(
                    f'lstm_{i}_units', 
                    layer_ranges['units']['min'], 
                    layer_ranges['units']['max'],
                    step=layer_ranges['units'].get('step', 32)
                )
            if 'return_sequences' in layer_ranges:
                layer['return_sequences'] = trial.suggest_categorical(
                    f'lstm_{i}_return_seq', 
                    layer_ranges['return_sequences']
                )
            params['layers'].append(layer)
        
        # Обработка слоев Dense
        dense_layers = [layer for layer in optuna_config.get('hyperparameter_ranges', {}).get('layers', []) 
                      if layer['type'] == 'Dense']
        for i, layer_ranges in enumerate(dense_layers):
            layer = {'type': 'Dense'}
            if 'units' in layer_ranges:
                layer['units'] = trial.suggest_int(
                    f'dense_{i}_units', 
                    layer_ranges['units']['min'], 
                    layer_ranges['units']['max']
                )
            params['layers'].append(layer)
        
        # Обработка других параметров
        if 'learning_rate' in optuna_config.get('hyperparameter_ranges', {}):
            ranges = optuna_config['hyperparameter_ranges']['learning_rate']
            params['learning_rate'] = trial.suggest_float(
                'learning_rate', 
                ranges['min'], 
                ranges['max'], 
                log=ranges.get('log', True)
            )
        
        if 'batch_size' in optuna_config.get('hyperparameter_ranges', {}):
            params['batch_size'] = trial.suggest_categorical(
                'batch_size', 
                optuna_config['hyperparameter_ranges']['batch_size']
            )
        
        if 'dropout' in optuna_config.get('hyperparameter_ranges', {}):
            ranges = optuna_config['hyperparameter_ranges']['dropout']
            params['dropout'] = trial.suggest_float(
                'dropout', 
                ranges['min'], 
                ranges['max']
            )
        
        if 'epochs' in optuna_config.get('hyperparameter_ranges', {}):
            ranges = optuna_config['hyperparameter_ranges']['epochs']
            params['epochs'] = trial.suggest_int(
                'epochs', 
                ranges['min'], 
                ranges['max']
            )
        
        # Добавляем фиксированные параметры
        if 'fixed_parameters' in optuna_config:
            params.update(optuna_config['fixed_parameters'])
        
        return params
    
    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Функция цели для Optuna.
        
        Args:
            trial: Объект Trial из Optuna
            X_train: Обучающие данные
            y_train: Целевые значения для обучения
            X_val: Валидационные данные
            y_val: Целевые значения для валидации
            
        Returns:
            Значение функции потерь на валидационных данных
        """
        params = self._get_optuna_params(trial)
        model = Sequential()
        n_features = X_train.shape[2]
        window_size = params.get('window_size', DEFAULT_WINDOW_SIZE)
        
        for i, layer in enumerate(params['layers']):
            layer_type = layer['type'].lower()
            units = layer['units']
            return_sequences = layer.get('return_sequences', i < len(params['layers']) - 1)
            
            if layer_type == 'lstm':
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
            elif layer_type == 'dense':
                model.add(Dense(units))
            
            if 'dropout' in params and i < len(params['layers']) - 1:
                model.add(Dropout(params['dropout']))
        
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=params['learning_rate']) if params.get('optimizer', 'adam') == 'adam' \
            else RMSprop(learning_rate=params['learning_rate'])
        model.compile(loss=params['loss'], optimizer=optimizer)
        
        history = model.fit(
            X_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        return min(history.history['val_loss'])
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Оптимизация гиперпараметров модели с помощью Optuna.
        
        Args:
            X_train: Обучающие данные
            y_train: Целевые значения для обучения
            X_val: Валидационные данные
            y_val: Целевые значения для валидации
            
        Returns:
            Словарь с оптимальными параметрами
        """
        study = self._create_optuna_study()
        optuna_config = self.config.get('mad_predictor', {}).get('optuna_tuning', {})
        n_trials = optuna_config.get('n_trials', 20)
        
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=None
        )
        
        return study.best_params
    
    def _train_lstm_model(self, dataframe: pd.DataFrame, hyperparams: Optional[Dict] = None, 
                         model_name: Optional[str] = None) -> Dict:
        """Обучение LSTM модели.
        
        Args:
            dataframe: Данные для обучения
            hyperparams: Гиперпараметры модели (если None, будут взяты из конфига)
            model_name: Имя модели (для поиска в конфиге)
            
        Returns:
            Словарь с результатами обучения:
            - model: обученная модель
            - config: параметры модели
            - scaler: объект масштабирования
            - history: история обучения
            - window_size: размер окна
            - n_features: количество признаков
            
        Raises:
            RuntimeError: При ошибках обучения модели
        """
        try:
            logger.info(f"Начало обучения модели {model_name or 'без имени'}")
            
            # Получаем архитектуру модели из конфига, если указано имя
            if model_name:
                model_config = self._get_model_config(model_name)
                if model_config:
                    architecture = self._get_model_architecture(model_config)
                    if architecture is None:  # Оптимизация гиперпараметров
                        logger.info("Запуск подбора гиперпараметров с помощью Optuna")
                        
                        window_size = DEFAULT_WINDOW_SIZE
                        n_features = len(dataframe.columns)
                        scaler = StandardScaler()
                        scaled_values = scaler.fit_transform(dataframe)
                        
                        X, y = [], []
                        for i in range(len(scaled_values) - window_size):
                            X.append(scaled_values[i:i + window_size])
                            y.append(scaled_values[i + window_size, 0])
                        
                        X = np.array(X)
                        y = np.array(y)
                        
                        split_idx = int(len(X) * 0.8)
                        X_train, X_val = X[:split_idx], X[split_idx:]
                        y_train, y_val = y[:split_idx], y[split_idx:]
                        
                        best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
                        hyperparams = best_params
                    elif hyperparams is None:
                        hyperparams = architecture
            
            # Устанавливаем параметры по умолчанию, если не заданы
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
            
            logger.info(f"Обучение модели {model_name or 'без имени'} завершено")
            
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
    
    async def _save_model(self, model_id: int, model: object, model_info: Dict, 
                         existing_model_id: Optional[int] = None, is_retrain: bool = False) -> None:
        """Сохранение модели в БД.
        
        Args:
            model_id: ID модели в таблице models_info
            model: Объект обученной модели
            model_info: Дополнительная информация о модели
            existing_model_id: ID существующей модели для обновления
            is_retrain: Флаг переобучения модели
            
        Raises:
            asyncpg.PostgresError: При ошибках работы с БД
        """
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    model_data = pickle.dumps({
                        'model': model,
                        'scaler': model_info['scaler'],
                        'window_size': model_info['window_size']
                    })
                    
                    if existing_model_id:
                        # Обновление существующей модели
                        await conn.execute("""
                            UPDATE models SET
                                model_data = $1,
                                status = 'active',
                                hyperparams = $2,
                                created_at = NOW()
                            WHERE id = $3
                        """, model_data, json.dumps(model_info['config']), existing_model_id)
                    else:
                        # Создание новой версии модели
                        version = await conn.fetchval("""
                            SELECT COALESCE(MAX(version::float), 0) + 1 
                            FROM models 
                            WHERE model_id = $1
                        """, model_id)
                        
                        await conn.execute("""
                            INSERT INTO models (
                                model_id, model_data, status,
                                version, hyperparams, created_at
                            ) VALUES ($1, $2, 'active', $3::text, $4, NOW())
                        """, model_id, model_data, str(version), json.dumps(model_info['config']))
                    
                    # Обновление активной версии
                    update_query = """
                        UPDATE models_info
                        SET active_version = (
                            SELECT version FROM models 
                            WHERE model_id = $1 AND status = 'active'
                            ORDER BY created_at DESC LIMIT 1
                        )
                    """
                    if is_retrain:
                        update_query += ", training_start = NOW(), training_end = NOW()"
                    
                    await conn.execute(update_query + " WHERE id = $1", model_id)
                    
                    # Удаление старых версий
                    await conn.execute("""
                        DELETE FROM models 
                        WHERE id IN (
                            SELECT id FROM models 
                            WHERE model_id = $1 
                            ORDER BY created_at DESC 
                            OFFSET (SELECT max_stored_versions FROM models_info WHERE id = $1)
                        )
                    """, model_id)
                    
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {str(e)}")
            raise
    
    async def _check_retrain_required(self) -> List[int]:
        """Проверка необходимости переобучения моделей.
        
        Returns:
            Список ID моделей, требующих переобучения
        """
        models_for_retrain = []
        
        for model_config in self.config.get('mad_predictor', {}).get('models', []):
            if not model_config.get('retrain', {}).get('enabled', False):
                continue
                
            try:
                async with self.db_pool.acquire() as conn:
                    model = await conn.fetchrow("""
                        SELECT m.id, mi.training_end 
                        FROM models m
                        JOIN models_info mi ON m.model_id = mi.id
                        WHERE mi.name = $1 AND m.status = 'active'
                        ORDER BY m.created_at DESC LIMIT 1
                    """, model_config['name'])
                    
                    if model:
                        last_train = model['training_end']
                        interval = model_config['retrain'].get('interval', f"{DEFAULT_RETRAIN_INTERVAL}h")
                        interval_hours = self._parse_interval(interval)
                        
                        if not last_train or (datetime.now(timezone.utc) >= last_train + timedelta(hours=interval_hours)):
                            models_for_retrain.append(model['id'])
            except Exception as e:
                logger.error(f"Ошибка проверки переобучения модели {model_config['name']}: {str(e)}")
        
        return models_for_retrain
    
    def _parse_interval(self, interval_str: str) -> int:
        """Парсинг строки интервала в часы.
        
        Args:
            interval_str: Строка интервала (например, "24h" или "7d")
            
        Returns:
            Количество часов
        """
        if interval_str.endswith('h'):
            return int(interval_str[:-1])
        elif interval_str.endswith('d'):
            return int(interval_str[:-1]) * 24
        return DEFAULT_RETRAIN_INTERVAL
    
    async def _process_model(self, model_id: int) -> None:
        """Обработка модели из очереди на обучение.
        
        Args:
            model_id: ID модели для обработки
        """
        logger.info(f"Начало обработки модели ID {model_id}")
        
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Получаем информацию о модели
                    model = await conn.fetchrow("""
                        SELECT m.id, m.model_id, m.status, m.version,
                               m.hyperparams::jsonb as hyperparams,
                               mi.name as model_name,
                               mi.hyperparams_mode
                        FROM models m
                        JOIN models_info mi ON m.model_id = mi.id
                        WHERE m.id = $1 AND m.status = 'waiting'
                        LIMIT 1
                    """, model_id)
                    
                    if not model:
                        logger.warning(f"Модель ID {model_id} не найдена или не в статусе 'waiting'")
                        return

                    logger.info(f"Обработка модели: {model['model_name']} (v{model['version']})")

                    # Получаем метрику для обучения
                    main_metric = await conn.fetchrow("""
                        SELECT name, query, step 
                        FROM metrics 
                        WHERE id = $1 AND status = 'active'
                    """, (await conn.fetchval("""
                        SELECT metric_id FROM models_info WHERE id = $1
                    """, model['model_id'])))
                    
                    if not main_metric:
                        logger.error(f"Для модели {model_id} не найдена активная метрика")
                        await conn.execute("""
                            UPDATE models 
                            SET status = 'deactive'
                            WHERE id = $1
                        """, model['id'])
                        return

                    logger.info(f"Используется метрика: {main_metric['name']}")

                    # Получаем данные для обучения
                    try:
                        metric_data = await self._fetch_metric_data(main_metric['name'])
                        if metric_data.empty:
                            raise ValueError("Нет данных для обучения")
                        logger.info(f"Получено {len(metric_data)} точек данных")
                    except Exception as e:
                        logger.error(f"Ошибка получения данных метрики: {str(e)}")
                        await conn.execute("""
                            UPDATE models 
                            SET status = 'deactive'
                            WHERE id = $1
                        """, model['id'])
                        return

                    # Подготавливаем гиперпараметры
                    hyperparams = None
                    if model['hyperparams_mode'] == 'manual' and model['hyperparams']:
                        try:
                            hyperparams = model['hyperparams']
                            logger.info("Используются ручные гиперпараметры")
                        except Exception as e:
                            logger.error(f"Ошибка парсинга гиперпараметров: {str(e)}")
                            hyperparams = None

                    # Обучаем модель
                    try:
                        logger.info("Запуск обучения модели...")
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
                            SET status = 'deactive'
                            WHERE id = $1
                        """, model['id'])
                        return

                    # Определяем, является ли это плановым переобучением
                    is_retrain = False
                    model_config = self._get_model_config(model['model_name'])
                    
                    if model_config and model_config.get('retrain', {}).get('enabled', False):
                        last_train = await conn.fetchval("""
                            SELECT training_end FROM models_info WHERE id = $1
                        """, model['model_id'])
                        
                        if last_train:
                            interval = model_config['retrain'].get('interval', f"{DEFAULT_RETRAIN_INTERVAL}h")
                            interval_hours = self._parse_interval(interval)
                            next_train_time = last_train + timedelta(hours=interval_hours)
                            is_retrain = datetime.now(timezone.utc) >= next_train_time
                        else:
                            is_retrain = True

                    # Сохраняем модель
                    try:
                        await self._save_model(
                            model['model_id'],
                            training_result["model"],
                            {
                                "config": training_result["config"],
                                "scaler": training_result["scaler"],
                                "window_size": training_result["window_size"]
                            },
                            existing_model_id=model['id'],
                            is_retrain=is_retrain
                        )
                        logger.info("Модель успешно сохранена")
                    except Exception as e:
                        logger.error(f"Ошибка сохранения модели: {str(e)}")
                        await conn.execute("""
                            UPDATE models 
                            SET status = 'deactive'
                            WHERE id = $1
                        """, model['id'])
                        return

                    # Обновляем статус модели
                    await conn.execute("""
                        UPDATE models 
                        SET status = 'active'
                        WHERE id = $1
                    """, model['id'])

                    logger.info(f"Модель {model_id} успешно обработана")

        except Exception as e:
            logger.error(f"Критическая ошибка обработки модели {model_id}: {str(e)}")
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE models 
                        SET status = 'deactive'
                        WHERE model_id = $1
                    """, model_id)
            except Exception as e:
                logger.error(f"Ошибка обновления статуса модели: {str(e)}")
    
    async def _listen_for_models(self) -> None:
        """Слушание уведомлений о новых моделях для обучения."""
        async def handle_notify(conn, pid, channel, payload):
            await self.task_queue.put(int(payload))

        try:
            db_config = self._parse_db_conn_string(self.config['system']['db_conn_string'])
            conn = await asyncpg.connect(**db_config)
            await conn.add_listener('model_training_queue', handle_notify)
            
            logger.info("Слушатель уведомлений запущен")
            
            while True:
                await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"Ошибка в слушателе уведомлений: {str(e)}")
    
    async def _load_existing_waiting_models(self) -> None:
        """Загрузка существующих моделей в статусе 'waiting'."""
        try:
            async with self.db_pool.acquire() as conn:
                models = await conn.fetch("SELECT id FROM models WHERE status = 'waiting'")
                for model in models:
                    await self.task_queue.put(model['id'])
                    logger.info(f"Добавлена в очередь модель ID {model['id']}")
        except Exception as e:
            logger.error(f"Ошибка загрузки ожидающих моделей: {str(e)}")
    
    async def _retrain_checker(self) -> None:
        """Периодическая проверка необходимости переобучения моделей."""
        while True:
            try:
                models_to_retrain = await self._check_retrain_required()
                for model_id in models_to_retrain:
                    await self.task_queue.put(model_id)
                    logger.info(f"Модель ID {model_id} добавлена в очередь для переобучения")
            except Exception as e:
                logger.error(f"Ошибка проверки переобучения: {str(e)}")
            
            await asyncio.sleep(3600)  # Проверка каждый час
    
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
            
            # Загрузка существующих задач
            await self._load_existing_waiting_models()
            
            # Запуск воркеров
            workers = [
                asyncio.create_task(self._worker_loop()) 
                for _ in range(self.config["system"].get("workers", 3))
            ]
            
            # Запуск слушателя уведомлений
            listener = asyncio.create_task(self._listen_for_models())
            
            # Запуск планировщика переобучения
            checker = asyncio.create_task(self._retrain_checker())
            
            await asyncio.gather(listener, *workers, checker)
            
        except Exception as e:
            logger.error(f"Ошибка сервиса: {str(e)}", exc_info=True)
            raise
        finally:
            if self.db_pool:
                await self.db_pool.close()
            logger.info("Сервис остановлен")


async def main():
    """Основная функция для запуска сервиса."""
    service = ModelTrainer()
    await service.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания, остановка сервиса")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)