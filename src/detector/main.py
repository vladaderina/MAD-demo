import logging
from logging.handlers import RotatingFileHandler
import os
import threading
import queue
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import json
import pickle
import select
import time
import aiohttp

import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.extensions
import requests
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Константы
DEFAULT_WORKER_THREADS = 4
DEFAULT_POLL_INTERVAL = 5  # секунд
DEFAULT_METRIC_STEP = 60  # секунд
ANOMALY_THRESHOLD_MULTIPLIER = 3  # множитель для порога аномалии

# Настройка логгера
logger = logging.getLogger(__name__)

class AnomalyDetectionService:
    """Сервис обнаружения аномалий на основе моделей машинного обучения."""

    def __init__(self, config_path: str = "minimal_config.yaml"):
        """Инициализация сервиса."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._init_service_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурационного файла."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                
            # Валидация обязательных параметров
            if 'mad-detector' not in config:
                raise ValueError("Отсутствует обязательный раздел конфига: mad-detector")
                
            return config
        except FileNotFoundError:
            logger.error(f"Файл конфигурации не найден: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Ошибка парсинга YAML: {str(e)}")
            raise
    
    def _setup_logging(self) -> None:
        """Настройка системы логирования."""
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Файловый обработчик с ротацией
        file_handler = RotatingFileHandler(
            os.getenv('LOG_PATH', 'anomaly_detection.log'),
            maxBytes=5*1024*1024,
            backupCount=3
        )
        file_handler.setFormatter(formatter)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def _init_service_components(self) -> None:
        """Инициализация компонентов сервиса."""
        # Парсинг строки подключения к БД
        db_conn_string = os.getenv('DB_CONN_STRING')
        if not db_conn_string:
            raise ValueError("Переменная DB_CONN_STRING обязательна")
        
        try:
            # Парсинг строки подключения
            conn_parts = db_conn_string.split('://')[1].split('@')
            user_pass = conn_parts[0].split(':')
            host_port_db = conn_parts[1].split('/')
            host_port = host_port_db[0].split(':')
            
            self.db_config = {
                'host': host_port[0],
                'database': host_port_db[1],
                'user': user_pass[0],
                'password': user_pass[1],
                'port': host_port[1] if len(host_port) > 1 else '5432'
            }
        except Exception as e:
            raise ValueError(f"Неправильный формат переменной DB_CONN_STRING: {str(e)}")
        
        # URL для подключения к VictoriaMetrics
        self.vm_url = os.getenv('VICTORIAMETRICS_URL')
        if not self.vm_url:
            raise ValueError("VICTORIAMETRICS_URL environment variable is required")
        
        # URL сервиса уведомлений
        self.notifier_url = os.getenv('NOTIFIER_SERVICE_URL')
        
        # Кэш загруженных моделей
        self.models_cache: Dict[int, Any] = {}
        # Активные модели (метаданные)
        self.active_models: Dict[int, Dict] = {}
        
        # Очередь задач для обработки
        self.task_queue = queue.Queue()
        # Пул воркеров
        self.workers: List[threading.Thread] = []
        # Флаг для остановки сервиса
        self.stop_event = threading.Event()
        
        # Параметры для определения аномалий
        self.anomaly_params = {
            'local': self.config['mad-detector']['system_anomaly']['min_confirmations']['local_anomaly'],
            'group': self.config['mad-detector']['system_anomaly']['min_confirmations']['group_anomaly'],
            'global': self.config['mad-detector']['system_anomaly']['min_confirmations']['global_anomaly']
        }
    
    async def _notify_anomaly(self, anomaly_data: Dict) -> None:
        """Отправка данных об аномалии в сервис уведомлений."""
        if not hasattr(self, 'notifier_url') or not self.notifier_url:
            logger.debug("URL сервиса уведомлений не задан, пропускаем отправку")
            return
            
        try:
            # Преобразуем строки времени в datetime объекты при необходимости
            if isinstance(anomaly_data.get('start_time'), str):
                anomaly_data['start_time'] = datetime.fromisoformat(anomaly_data['start_time'])
            
            if 'end_time' in anomaly_data and isinstance(anomaly_data['end_time'], str):
                anomaly_data['end_time'] = datetime.fromisoformat(anomaly_data['end_time'])
            
            # Преобразуем datetime объекты обратно в строки для отправки
            send_data = anomaly_data.copy()
            if isinstance(send_data.get('start_time'), datetime):
                send_data['start_time'] = send_data['start_time'].isoformat()
            
            if isinstance(send_data.get('end_time'), datetime):
                send_data['end_time'] = send_data['end_time'].isoformat()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.notifier_url}/api/v1/anomalies",
                    json=send_data,
                    timeout=5
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Ошибка отправки уведомления: HTTP {resp.status}")
                    else:
                        logger.info(f"Уведомление об аномалии {anomaly_data['id']} отправлено")
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления: {str(e)}")
    
    def _get_db_connection(self) -> psycopg2.extensions.connection:
        """Получение нового соединения с БД."""
        try:
            return psycopg2.connect(**self.db_config)
        except psycopg2.OperationalError as e:
            logger.error(f"Ошибка подключения к БД: {str(e)}")
            raise
    
    def start(self) -> None:
        """Запуск сервиса."""
        try:
            logger.info("Запуск сервиса обнаружения аномалий")
            
            # Запуск воркеров
            num_workers = int(os.getenv('WORKER_THREADS', DEFAULT_WORKER_THREADS))
            self._start_workers(num_workers)
            
            logger.info("Сервис успешно запущен")
            
            # Основной цикл
            while not self.stop_event.is_set():
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Ошибка запуска сервиса: {str(e)}", exc_info=True)
            raise
        finally:
            self.stop()
            logger.info("Сервис остановлен")
    
    def stop(self) -> None:
        """Остановка сервиса."""
        self.stop_event.set()
        logger.info("Сервис обнаружения аномалий остановлен")
    
    def _start_workers(self, num_workers: int) -> None:
        """Запуск воркеров для обработки задач."""
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AnomalyWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            logger.info(f"Запущен воркер {worker.name}")
    
    def _worker_loop(self) -> None:
        """Цикл обработки задач воркерами."""
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                try:
                    self._process_task(task)
                except Exception as e:
                    logger.error(f"Ошибка обработки задачи: {str(e)}", exc_info=True)
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка воркера: {str(e)}", exc_info=True)
                time.sleep(5)
    
    def _process_task(self, task: Dict) -> None:
        """Обработка задачи детектирования аномалий."""
        model_id = task['model_id']
        
        try:
            # 1. Проверка активности модели
            if model_id not in self.active_models:
                logger.warning(f"Модель {model_id} не активна, пропускаем")
                return
            
            model_data = self.active_models[model_id]
            metric_id = model_data['metric_id']
            
            # 2. Проверка актуальности версии модели
            if not self._check_model_version_active(model_data['model_info_id'], model_data['version']):
                self._remove_active_model(model_id)
                return

            # 3. Закрытие старых аномалий перед обработкой новых данных
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    self._close_anomalies(cursor, metric_id)
                    conn.commit()

            # 4. Получение данных метрики с учетом window_size
            metric_data = self._get_metric_data(
                metric_id=metric_id,
                model_data=model_data
            )
            
            if not metric_data:
                logger.warning(f"Не удалось получить данные для модели {model_id} (метрика {metric_id})")
                return

            logger.info(f"Получено {len(metric_data)} точек для модели {model_id}")

            # 4. Загрузка модели (с кэшированием)
            model = self._load_model(model_id)
            if not model:
                logger.error(f"Не удалось загрузить модель {model_id}")
                return

            # 5. Выполнение предсказаний
            window_size = model_data['hyperparams'].get('window_size', len(metric_data))
            predictions = self._make_predictions(metric_data, model, window_size)
            
            if predictions.size == 0:
                logger.warning(f"Не сгенерировано предсказаний для модели {model_id}")
                return

            # 6. Обработка и сохранение результатов
            self._process_and_save_results(
                actual=metric_data,
                predicted=predictions,
                model_id=model_id,
                model_data=model_data
            )

        except Exception as e:
            logger.error(f"Ошибка обработки модели {model_id}: {str(e)}", exc_info=True)

    def _get_threshold_for_metric(self, model_id: int, metric_query: str) -> float:
        """Определение порога для метрики с учетом конфига."""
        # Ищем конфигурацию для этой метрики
        metric_config = None
        for metric_cfg in self.config['mad-detector'].get('points_anomaly', []):
            if metric_cfg.get('metric') == metric_query:
                metric_config = metric_cfg
                break
        
        if metric_config:
            delta_threshold = metric_config.get('delta_threshold', 'auto')
            median_window = metric_config.get('median_window', '3d')
            
            if delta_threshold != 'auto':
                try:
                    return float(delta_threshold)
                except (ValueError, TypeError):
                    logger.warning(f"Некорректное значение delta_threshold для {metric_query}, используем расчетный порог")
            
            # Если порог 'auto', передаем median_window в функцию расчета
            return self._get_anomaly_threshold(model_id, median_window)
        
        # Если не нашли конфига для метрики, используем дефолтные значения
        return self._get_anomaly_threshold(model_id, '3d')

    def _get_metric_query(self, metric_id: int) -> str:
        """Получение query метрики по ID."""
        with self._get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT query FROM metrics WHERE id = %s", (metric_id,))
                result = cursor.fetchone()
                return result[0] if result else ""

    def _process_and_save_results(self, actual: List[float], predicted: np.ndarray,
                                model_id: int, model_data: Dict) -> None:
        """Обработка и сохранение результатов предсказаний."""
        conn = None
        try:
            # 1. Подготовка данных
            predicted = predicted.flatten()
            actual_values = np.array(actual[-len(predicted):])
            min_length = min(len(predicted), len(actual_values))
            predicted = predicted[:min_length]
            actual_values = actual_values[:min_length]
            errors = np.abs(actual_values - predicted)
            
            # 2. Определение порога аномалии
            metric_query = self._get_metric_query(model_data['metric_id'])
            threshold = self._get_threshold_for_metric(model_id, metric_query)
            
            metric_id = model_data['metric_id']
            metric_step = model_data.get('metric_step', DEFAULT_METRIC_STEP)
            
            # 3. Открытие соединения с БД
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # 4. Получение времени последней сохраненной точки
            cursor.execute("""
                SELECT MAX(timestamp) 
                FROM prediction_errors 
                WHERE model_id = %s AND metric_id = %s
            """, (model_id, metric_id))
            last_saved_time = cursor.fetchone()[0]
            
            # 5. Получение количества ранее сохраненных точек
            cursor.execute("""
                SELECT COUNT(*) 
                FROM prediction_errors 
                WHERE model_id = %s AND metric_id = %s
            """, (model_id, metric_id))
            existing_points_count = cursor.fetchone()[0]
            
            # 6. Сохранение новых точек
            saved_count = 0
            anomaly_count = 0
            
            for i in range(len(errors)):
                timestamp = datetime.now(timezone.utc) - timedelta(
                    seconds=(len(errors) - i - 1) * metric_step)
                
                # Пропуск уже сохраненных точек
                if last_saved_time and timestamp <= last_saved_time:
                    continue
                    
                error_value = float(errors[i])
                predicted_value = float(predicted[i])
                actual_value = float(actual_values[i])

                # Сохранение ошибки предсказания
                cursor.execute("""
                    INSERT INTO prediction_errors 
                    (model_id, metric_id, timestamp, error_value, 
                    predicted_value, actual_value, anomaly_threshold)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    model_id, metric_id, timestamp,
                    error_value, predicted_value, 
                    actual_value, float(threshold)
                ))

                if row := cursor.fetchone():
                    saved_count += 1
                    error_id = row[0]
                    
                    # Проверка на аномалию (если не первое окно)
                    if error_value > threshold and existing_points_count > 0:
                        cursor.execute("""
                            INSERT INTO anomaly_points
                            (model_id, metric_id, timestamp, prediction_error_id)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (model_id, metric_id, timestamp) DO NOTHING
                        """, (model_id, metric_id, timestamp, error_id))
                        anomaly_count += 1

            # 7. Проверка системных аномалий (если не первое окно)
            if saved_count > 0 and existing_points_count > 0:
                self._check_system_anomalies(cursor, model_id, metric_id)

            conn.commit()

            if existing_points_count == 0:
                logger.info(f"Начальное окно: сохранено {saved_count} точек для модели {model_id}")
            else:
                logger.info(f"Сохранено {saved_count} предсказаний ({anomaly_count} аномалий) для модели {model_id}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения результатов для модели {model_id}: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def _check_system_anomalies(self, cursor: Any, model_id: int, metric_id: int) -> None:
        current_time = datetime.now(timezone.utc)
        model_info_id = self.active_models[model_id]['model_info_id']
        
        try:
            # 1. Получаем имя метрики
            cursor.execute("SELECT name FROM metrics WHERE id = %s", (metric_id,))
            metric_name = cursor.fetchone()[0]
            
            # 2. Проверяем, есть ли уже активная аномалия для этой метрики
            cursor.execute("""
                SELECT id FROM anomaly_system 
                WHERE metric_id = %s AND end_time IS NULL
                LIMIT 1
            """, (metric_id,))
            if cursor.fetchone():
                logger.debug(f"Активная аномалия для метрики {metric_id} уже существует")
                return
            
            # 3. Проверка количества аномалий за последний час
            cursor.execute("""
                SELECT COUNT(*) 
                FROM anomaly_points 
                WHERE model_id = %s AND metric_id = %s
                AND timestamp > %s - INTERVAL '1 hour'
            """, (model_id, metric_id, current_time))
            recent_anomalies = cursor.fetchone()[0]
            
            # 4. Определение типа аномалии
            if recent_anomalies >= self.anomaly_params['group']:
                anomaly_type = 'group'
            elif recent_anomalies >= self.anomaly_params['local']:
                anomaly_type = 'local'
            else:
                return
            
            # 5. Сохранение в БД с обработкой возможного конфликта
            try:
                cursor.execute("""
                    INSERT INTO anomaly_system
                    (start_time, anomaly_type, average_anom_score, metric_id, description)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    current_time, 
                    anomaly_type,
                    0,  # Заглушка для average_anom_score
                    metric_id,
                    f"Обнаружена {anomaly_type} аномалия"
                ))
                anomaly_id = cursor.fetchone()[0]
                
                # 6. Отправка уведомления о начале аномалии
                anomaly_data = {
                    'action': 'start',
                    'id': anomaly_id,
                    'anomaly_type': anomaly_type,
                    'metric_id': metric_id,
                    'metric_name': metric_name,
                    'start_time': current_time.isoformat(),
                    'description': f"Обнаружена {anomaly_type} аномалия",
                    'average_anom_score': 0
                }
                
                threading.Thread(
                    target=lambda: asyncio.run(self._notify_anomaly(anomaly_data)),
                    daemon=True
                ).start()
                
            except psycopg2.errors.UniqueViolation:
                logger.debug(f"Аномалия для метрики {metric_id} уже зарегистрирована")
                conn.rollback()
                return
                
        except Exception as e:
            logger.error(f"Ошибка проверки системных аномалий: {str(e)}")
            raise
            
    def _close_anomalies(self, cursor: Any, metric_id: int) -> None:
        try:
            # 1. Находим аномалии без end_time
            dict_cursor = cursor.connection.cursor(cursor_factory=RealDictCursor)
            dict_cursor.execute("""
                SELECT 
                    a.id, a.anomaly_type, a.start_time, a.description,
                    m.name as metric_name
                FROM anomaly_system a
                JOIN metrics m ON a.metric_id = m.id
                WHERE a.metric_id = %s AND a.end_time IS NULL
            """, (metric_id,))
            
            for anomaly in dict_cursor.fetchall():
                # 2. Проверяем, есть ли свежие точки
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM anomaly_points 
                    WHERE metric_id = %s
                    AND timestamp > NOW() - INTERVAL '10 minutes'
                """, (metric_id,))
                recent_points = cursor.fetchone()[0]
                
                if recent_points == 0:
                    # 3. Закрываем аномалию
                    end_time = datetime.now(timezone.utc)
                    cursor.execute("""
                        UPDATE anomaly_system
                        SET end_time = %s
                        WHERE id = %s
                        RETURNING id
                    """, (end_time, anomaly['id']))
                    
                    # 4. Подготавливаем данные для уведомления
                    start_time = anomaly['start_time']
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    
                    anomaly_data = {
                        'action': 'end',
                        'id': anomaly['id'],
                        'anomaly_type': anomaly['anomaly_type'],
                        'metric_id': metric_id,
                        'metric_name': anomaly['metric_name'],
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat(),
                        'duration': str(end_time - start_time),
                        'description': anomaly['description'],
                        'average_anom_score': 0
                    }
                    
                    threading.Thread(
                        target=lambda: asyncio.run(self._notify_anomaly(anomaly_data)),
                        daemon=True
                    ).start()
                dict_cursor.close()
        except Exception as e:
            logger.error(f"Ошибка при закрытии аномалий: {str(e)}")
    
    def _get_metric_data(self, metric_id: int, model_data: Dict) -> Optional[List[float]]:
        """Получение данных метрики из VictoriaMetrics с учетом window_size."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT query, step FROM metrics WHERE id = %s
                    """, (metric_id,))
                    if metric := cursor.fetchone():
                        # Получаем параметры из модели
                        window_size = model_data['hyperparams'].get('window_size', 60)
                        metric_step = metric['step']
                        
                        # Рассчитываем временной диапазон для запроса
                        end_time = datetime.now(timezone.utc)
                        start_time = end_time - timedelta(seconds=window_size * metric_step)
                        
                        # Формируем запрос к VictoriaMetrics
                        response = requests.get(
                            f"{self.vm_url}/api/v1/query_range", 
                            params={
                                'query': metric['query'],
                                'start': start_time.timestamp(),
                                'end': end_time.timestamp(),
                                'step': f"{metric_step}s"
                            },
                            timeout=30
                        )
                        response.raise_for_status()
                        
                        # Обрабатываем ответ
                        if data := response.json().get('data', {}).get('result'):
                            return [float(point[1]) for point in data[0]['values']]
            return None
        except Exception as e:
            logger.error(f"Ошибка получения данных метрики {metric_id}: {str(e)}")
            return None
    
    def _load_model(self, model_id: int) -> Optional[Any]:
        """Загрузка модели из БД с кэшированием."""
        if model_id in self.models_cache:
            return self.models_cache[model_id]
            
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT model_data, hyperparams FROM models WHERE id = %s
                    """, (model_id,))
                    if model_data := cursor.fetchone():
                        model = pickle.loads(model_data[0])
                        if isinstance(model, dict):
                            model = model['model']  # Извлечение модели если сохранена в словаре
                        
                        self.models_cache[model_id] = model
                        return model
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_id}: {str(e)}")
        
        return None
    
    def _make_predictions(self, data: List[float], model: Any, 
                         window_size: int) -> np.ndarray:
        """Выполнение предсказаний на основе данных."""
        if not data or len(data) < 2:
            logger.warning("Недостаточно данных для предсказания")
            return np.zeros((1, 1))
        
        try:
            # Масштабирование данных
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
            
            # Подготовка данных для модели
            X = scaled_data[-window_size:].reshape(1, -1)
            
            # Выполнение предсказания
            predictions = model.predict(X)
            
            # Приведение к правильной форме
            if predictions.ndim > 2:
                predictions = predictions.reshape(-1, 1)
            
            # Обратное масштабирование
            return scaler.inverse_transform(predictions)
        except Exception as e:
            logger.error(f"Ошибка предсказания: {str(e)}")
            return np.zeros((1, 1))
    
    def _parse_time_window(self, window_str: str) -> timedelta:
        """Преобразует строку '3d', '24h' и т.д. в timedelta."""
        if not window_str:
            return timedelta(days=3)  # Значение по умолчанию
        
        try:
            value = int(window_str[:-1])
            unit = window_str[-1].lower()
            
            if unit == 'd':
                return timedelta(days=value)
            elif unit == 'h':
                return timedelta(hours=value)
            elif unit == 'm':
                return timedelta(minutes=value)
            else:
                raise ValueError(f"Unknown time unit: {unit}")
        except (ValueError, IndexError):
            logger.warning(f"Invalid time window format: '{window_str}'. Using default 3d.")
            return timedelta(days=3)

    def _get_anomaly_threshold(self, model_id: int, window_str: str = '3d') -> float:
        """Расчет порога аномалии на основе медианы и MAD с параметризованным окном."""
        try:
            window_delta = self._parse_time_window(window_str)
            interval_str = f"INTERVAL '{window_delta.total_seconds()} seconds'"
            
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. Получение количества точек за указанный период
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM prediction_errors 
                        WHERE model_id = %s 
                        AND timestamp > NOW() - {interval_str}
                    """, (model_id,))
                    recent_points_count = cursor.fetchone()[0]
                    
                    # 2. Расчет медианы ошибок
                    cursor.execute(f"""
                        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY error_value) as median
                        FROM prediction_errors
                        WHERE model_id = %s 
                        AND timestamp > NOW() - {interval_str}
                    """, (model_id,))
                    median = cursor.fetchone()[0] or 0.0
                    
                    # 3. Расчет медианного абсолютного отклонения (MAD)
                    cursor.execute(f"""
                        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (
                            ORDER BY ABS(error_value - %s)
                        ) as mad
                        FROM prediction_errors
                        WHERE model_id = %s 
                        AND timestamp > NOW() - {interval_str}
                    """, (median, model_id))
                    mad = cursor.fetchone()[0] or 0.0
                    
                    # 4. Расчет порога (медиана + 3 * 1.4826 * MAD)
                    threshold = median + ANOMALY_THRESHOLD_MULTIPLIER * 1.4826 * mad
                    
                    logger.debug(
                        f"Порог для модели {model_id}: "
                        f"медиана={median:.4f}, mad={mad:.4f}, порог={threshold:.4f} "
                        f"(окно: {window_str})"
                    )
                    return threshold
                    
        except Exception as e:
            logger.error(f"Ошибка расчета порога для модели {model_id}: {str(e)}")
            return 0.0
    
    def _get_last_prediction_time(self, model_id: int, metric_id: int) -> Optional[datetime]:
        """Получение времени последнего предсказания."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT MAX(timestamp) AT TIME ZONE 'UTC'
                        FROM prediction_errors 
                        WHERE model_id = %s AND metric_id = %s
                    """, (model_id, metric_id))
                    result = cursor.fetchone()[0]
                    return result.replace(tzinfo=timezone.utc) if result else None
        except Exception as e:
            logger.error(f"Ошибка получения времени последнего предсказания: {str(e)}")
            return None
    
    def _check_model_version_active(self, model_info_id: int, version: str) -> bool:
        """Проверка активности версии модели."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT active_version FROM models_info 
                        WHERE id = %s AND active_version = %s
                    """, (model_info_id, version))
                    return bool(cursor.fetchone())
        except Exception as e:
            logger.error(f"Ошибка проверки версии модели: {str(e)}")
            return False
    
    def _add_active_model(self, model_id: int) -> None:
        """Добавление активной модели в кэш."""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT 
                            m.id, m.version, m.hyperparams,
                            mi.id as model_info_id, mi.name as model_name, 
                            mi.metric_id, mi.step as prediction_step,
                            mt.query as metric_query, mt.step as metric_step
                        FROM models m
                        JOIN models_info mi ON m.model_id = mi.id
                        JOIN metrics mt ON mi.metric_id = mt.id
                        WHERE m.id = %s AND m.status = 'active'
                    """, (model_id,))
                    if model := cursor.fetchone():
                        self.active_models[model_id] = dict(model)
                        logger.info(f"Добавлена активная модель {model['model_name']} (ID: {model_id})")
        except Exception as e:
            logger.error(f"Ошибка добавления активной модели: {str(e)}")
    
    def _remove_active_model(self, model_id: int) -> None:
        """Удаление модели из кэша."""
        if model_id in self.active_models:
            model_name = self.active_models[model_id]['model_name']
            del self.active_models[model_id]
            if model_id in self.models_cache:
                del self.models_cache[model_id]
            logger.info(f"Удалена модель {model_name} (ID: {model_id})")
    
    def activate_model(self, model_id: int) -> None:
        """Активация модели для детектирования."""
        self._add_active_model(model_id)
    
    def deactivate_model(self, model_id: int) -> None:
        """Деактивация модели для детектирования."""
        self._remove_active_model(model_id)
    
    def process_model(self, model_id: int) -> None:
        """Добавление задачи обработки модели в очередь."""
        self.task_queue.put({'model_id': model_id})


if __name__ == "__main__":
    try:
        detector = AnomalyDetectionService()
        detector.start()
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания, остановка сервиса")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)