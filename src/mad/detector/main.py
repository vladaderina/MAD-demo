import logging
from logging.handlers import RotatingFileHandler
import os
import threading
import queue
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import json
import pickle
import select
import time

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
ANOMALY_THRESHOLD_MULTIPLIER = 3  # множитель для порога аномалии (3 * MAD)

# Настройка логгера
logger = logging.getLogger(__name__)

class AnomalyDetectionService:
    """Сервис обнаружения аномалий на основе моделей машинного обучения."""

    def __init__(self, config_path: str = "config.yaml"):
        """Инициализация сервиса.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._init_service_components()
        
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
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Валидация обязательных параметров
            required_keys = ['system', 'mad_detector']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Отсутствует обязательный раздел конфига: {key}")
                    
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
            self.config['system'].get('log_path', 'anomaly_detection.log'),
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
        db_conn_string = self.config['system']['db_conn_string']
        conn_parts = db_conn_string.split('://')[1].split('@')
        user_pass = conn_parts[0].split(':')
        host_port_db = conn_parts[1].split('/')
        host_port = host_port_db[0].split(':')
        
        self.db_config = {
            'host': host_port[0],
            'database': host_port_db[1],
            'user': user_pass[0],
            'password': user_pass[1],
            'port': host_port[1]
        }
        
        # URL для подключения к VictoriaMetrics
        self.vm_url = self.config['system']['victoriametrics_url'].split('/api/v1')[0]
        
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
        
        # Подключение для слушания уведомлений от БД
        self.notify_conn = self._create_notify_connection()
        
        # Параметры для определения аномалий
        self.anomaly_params = {
            'local': self.config['mad_detector']['system_anomaly']['min_confirmations']['local_anomaly'],
            'group': self.config['mad_detector']['system_anomaly']['min_confirmations']['group_anomaly'],
            'global': self.config['mad_detector']['system_anomaly']['min_confirmations']['global_anomaly']
        }
    
    def _create_notify_connection(self) -> psycopg2.extensions.connection:
        """Создание подключения для получения уведомлений от PostgreSQL.
        
        Returns:
            Подключение к PostgreSQL с поддержкой уведомлений
            
        Raises:
            psycopg2.OperationalError: При ошибках подключения к БД
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            return conn
        except psycopg2.OperationalError as e:
            logger.error(f"Ошибка подключения к PostgreSQL: {str(e)}")
            raise
    
    def _get_db_connection(self) -> psycopg2.extensions.connection:
        """Получение нового соединения с БД.
        
        Returns:
            Соединение с PostgreSQL
            
        Raises:
            psycopg2.OperationalError: При ошибках подключения
        """
        try:
            return psycopg2.connect(**self.db_config)
        except psycopg2.OperationalError as e:
            logger.error(f"Ошибка подключения к БД: {str(e)}")
            raise
    
    def start(self) -> None:
        """Запуск сервиса."""
        try:
            logger.info("Запуск сервиса обнаружения аномалий")
            
            # Загрузка активных моделей
            self._load_active_models()
            
            # Запуск воркеров
            num_workers = int(os.getenv('WORKER_THREADS', DEFAULT_WORKER_THREADS))
            self._start_workers(num_workers)
            
            # Запуск планировщика задач
            scheduler_thread = threading.Thread(
                target=self._schedule_detections, 
                daemon=True
            )
            scheduler_thread.start()
            
            # Запуск слушателя изменений
            listener_thread = threading.Thread(
                target=self._listen_for_changes, 
                daemon=True
            )
            listener_thread.start()
            
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
        
        # Закрытие подключения для уведомлений
        if hasattr(self, 'notify_conn') and self.notify_conn:
            self.notify_conn.close()
            logger.info("Соединение для уведомлений закрыто")
        
        logger.info("Сервис обнаружения аномалий остановлен")
    
    def _load_active_models(self) -> None:
        """Загрузка метаданных активных моделей из БД."""
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
                        WHERE m.status = 'active' AND mi.active_version = m.version
                    """)
                    for model in cursor.fetchall():
                        model_id = model['id']
                        self.active_models[model_id] = dict(model)
                        logger.info(f"Загружена активная модель {model['model_name']} (ID: {model_id})")
        except Exception as e:
            logger.error(f"Ошибка загрузки активных моделей: {str(e)}", exc_info=True)
    
    def _start_workers(self, num_workers: int) -> None:
        """Запуск воркеров для обработки задач.
        
        Args:
            num_workers: Количество воркеров
        """
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
        """Обработка задачи детектирования аномалий.
        
        Args:
            task: Словарь с данными задачи (должен содержать 'model_id')
        """
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

            # 3. Получение времени последнего предсказания
            last_pred_time = self._get_last_prediction_time(model_id, metric_id)
            
            # 4. Получение новых данных метрики
            metric_data = self._get_metric_data(
                metric_id=metric_id,
                model_data=model_data,
                since=last_pred_time
            )
            
            if not metric_data or len(metric_data) < 2:
                logger.debug(f"Нет новых данных для модели {model_id} (метрика {metric_id})")
                return

            logger.info(f"Обработка {len(metric_data)} новых точек для модели {model_id}")

            # 5. Загрузка модели (с кэшированием)
            model = self._load_model(model_id)
            if not model:
                logger.error(f"Не удалось загрузить модель {model_id}")
                return

            # 6. Выполнение предсказаний
            window_size = model_data['hyperparams'].get('window_size', len(metric_data))
            predictions = self._make_predictions(metric_data, model, window_size)
            
            if predictions.size == 0:
                logger.warning(f"Не сгенерировано предсказаний для модели {model_id}")
                return

            # 7. Обработка и сохранение результатов
            self._process_and_save_results(
                actual=metric_data,
                predicted=predictions,
                model_id=model_id,
                model_data=model_data
            )

        except Exception as e:
            logger.error(f"Ошибка обработки модели {model_id}: {str(e)}", exc_info=True)
    
    def _process_and_save_results(self, actual: List[float], predicted: np.ndarray,
                                model_id: int, model_data: Dict) -> None:
        """Обработка и сохранение результатов предсказаний.
        
        Args:
            actual: Фактические значения метрики
            predicted: Предсказанные значения
            model_id: ID модели
            model_data: Метаданные модели
        """
        conn = None
        try:
            # 1. Подготовка данных
            predicted = predicted.flatten()
            actual_values = np.array(actual[-len(predicted):])
            min_length = min(len(predicted), len(actual_values))
            predicted = predicted[:min_length]
            actual_values = actual_values[:min_length]
            errors = np.abs(actual_values - predicted)
            
            # 2. Получение порога аномалии
            threshold = self._get_anomaly_threshold(model_id)
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
    
    def _get_metric_data(self, metric_id: int, model_data: Dict, 
                        since: Optional[datetime] = None) -> Optional[List[float]]:
        """Получение данных метрики из VictoriaMetrics.
        
        Args:
            metric_id: ID метрики
            model_data: Метаданные модели
            since: Время, начиная с которого нужно получить данные
            
        Returns:
            Список значений метрики или None при ошибке
        """
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT query, step FROM metrics WHERE id = %s
                    """, (metric_id,))
                    if metric := cursor.fetchone():
                        window_size = model_data['hyperparams'].get('window_size', 60)
                        end_time = datetime.now(timezone.utc)
                        start_time = since if since else end_time - timedelta(
                            seconds=window_size * metric['step'])
                        
                        response = requests.get(
                            f"{self.vm_url}/api/v1/query_range", 
                            params={
                                'query': metric['query'],
                                'start': start_time.timestamp(),
                                'end': end_time.timestamp(),
                                'step': f"{metric['step']}s"
                            },
                            timeout=30
                        )
                        response.raise_for_status()
                        
                        if data := response.json().get('data', {}).get('result'):
                            return [float(point[1]) for point in data[0]['values']]
            return None
        except Exception as e:
            logger.error(f"Ошибка получения данных метрики {metric_id}: {str(e)}")
            return None
    
    def _load_model(self, model_id: int) -> Optional[Any]:
        """Загрузка модели из БД с кэшированием.
        
        Args:
            model_id: ID модели
            
        Returns:
            Загруженная модель или None при ошибке
        """
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
        """Выполнение предсказаний на основе данных.
        
        Args:
            data: Входные данные
            model: Модель для предсказаний
            window_size: Размер окна
            
        Returns:
            Массив с предсказаниями
        """
        if not data or len(data) < 2:
            logger.warning("Недостаточно данных для предсказания")
            return np.zeros((1, 1))
        
        try:
            # Масштабирование данных
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
            
            # Корректировка размера окна под доступные данные
            if len(scaled_data) < window_size:
                logger.warning(f"Недостаточно данных ({len(scaled_data)} точек), используем доступные")
                window_size = len(scaled_data)
            
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
    
    def _get_anomaly_threshold(self, model_id: int) -> float:
        """Расчет порога аномалии на основе медианы и MAD.
        
        Args:
            model_id: ID модели
            
        Returns:
            Рассчитанный порог аномалии
        """
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. Получение количества точек за последние 3 дня
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM prediction_errors 
                        WHERE model_id = %s 
                        AND timestamp > NOW() - INTERVAL '3 days'
                    """, (model_id,))
                    recent_points_count = cursor.fetchone()[0]
                    
                    # 2. Расчет медианы ошибок
                    cursor.execute("""
                        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY error_value) as median
                        FROM prediction_errors
                        WHERE model_id = %s 
                        AND timestamp > NOW() - INTERVAL '3 days'
                    """, (model_id,))
                    median = cursor.fetchone()[0] or 0.0
                    
                    # 3. Расчет медианного абсолютного отклонения (MAD)
                    cursor.execute("""
                        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (
                            ORDER BY ABS(error_value - %s)
                        ) as mad
                        FROM prediction_errors
                        WHERE model_id = %s 
                        AND timestamp > NOW() - INTERVAL '3 days'
                    """, (median, model_id))
                    mad = cursor.fetchone()[0] or 0.0
                    
                    # 4. Расчет порога (медиана + 3 * 1.4826 * MAD)
                    threshold = median + ANOMALY_THRESHOLD_MULTIPLIER * 1.4826 * mad
                    
                    logger.debug(
                        f"Порог для модели {model_id}: "
                        f"медиана={median:.4f}, mad={mad:.4f}, порог={threshold:.4f}"
                    )
                    return threshold
                    
        except Exception as e:
            logger.error(f"Ошибка расчета порога для модели {model_id}: {str(e)}")
            return 0.0
    
    def _get_last_prediction_time(self, model_id: int, metric_id: int) -> Optional[datetime]:
        """Получение времени последнего предсказания.
        
        Args:
            model_id: ID модели
            metric_id: ID метрики
            
        Returns:
            Время последнего предсказания или None
        """
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
        """Проверка активности версии модели.
        
        Args:
            model_info_id: ID информации о модели
            version: Версия модели
            
        Returns:
            True если версия активна, иначе False
        """
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
        """Добавление активной модели в кэш.
        
        Args:
            model_id: ID модели
        """
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
        """Удаление модели из кэша.
        
        Args:
            model_id: ID модели
        """
        if model_id in self.active_models:
            model_name = self.active_models[model_id]['model_name']
            del self.active_models[model_id]
            if model_id in self.models_cache:
                del self.models_cache[model_id]
            logger.info(f"Удалена модель {model_name} (ID: {model_id})")
    
    def _check_system_anomalies(self, cursor: Any, model_id: int, metric_id: int) -> None:
        """Проверка и сохранение системных аномалий.
        
        Args:
            cursor: Курсор БД
            model_id: ID модели
            metric_id: ID метрики
        """
        current_time = datetime.now(timezone.utc)
        model_info_id = self.active_models[model_id]['model_info_id']
        
        # 1. Проверка количества аномалий за последний час
        cursor.execute("""
            SELECT COUNT(*) 
            FROM anomaly_points 
            WHERE model_id = %s AND metric_id = %s
            AND timestamp > %s - INTERVAL '1 hour'
        """, (model_id, metric_id, current_time))
        recent_anomalies = cursor.fetchone()[0]
        
        # 2. Проверка глобальных аномалий (основная метрика + связанные)
        is_main_metric = (metric_id == self.active_models[model_id]['metric_id'])
        if is_main_metric:
            cursor.execute("""
                SELECT COUNT(DISTINCT ap.metric_id) 
                FROM anomaly_points ap
                JOIN features f ON ap.metric_id = f.metric_id
                WHERE f.model_id = %s
                AND ap.timestamp > %s - INTERVAL '5 minutes'
            """, (model_info_id, current_time))
            affected_metrics = cursor.fetchone()[0]
            
            if affected_metrics >= 2:  # Основная + хотя бы одна связанная
                cursor.execute("""
                    INSERT INTO anomaly_system
                    (start_time, anomaly_type, average_anom_score, metric_id, description)
                    VALUES (%s, 'global', %s, %s, %s)
                    ON CONFLICT (metric_id) WHERE end_time IS NULL AND anomaly_type = 'global' 
                    DO UPDATE SET
                        average_anom_score = (anomaly_system.average_anom_score + EXCLUDED.average_anom_score) / 2
                """, (
                    current_time, 
                    0,  # Заглушка для average_anom_score
                    metric_id,
                    f"Обнаружена глобальная аномалия по {affected_metrics} метрикам"
                ))
                return
        
        # 3. Определение типа аномалии (group или local)
        if recent_anomalies >= self.anomaly_params['group']:
            anomaly_type = 'group'
        elif recent_anomalies >= self.anomaly_params['local']:
            anomaly_type = 'local'
        else:
            return
        
        # 4. Сохранение системной аномалии
        cursor.execute("""
            INSERT INTO anomaly_system
            (start_time, anomaly_type, average_anom_score, metric_id, description)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (metric_id) WHERE end_time IS NULL AND anomaly_type = %s
            DO UPDATE SET
                average_anom_score = (anomaly_system.average_anom_score + EXCLUDED.average_anom_score) / 2,
                start_time = LEAST(anomaly_system.start_time, EXCLUDED.start_time)
        """, (
            current_time, 
            anomaly_type,
            0,  # Заглушка для average_anom_score
            metric_id,
            f"Обнаружена {anomaly_type} аномалия",
            anomaly_type
        ))

        # 5. Проверка на закрытие аномалии (если новых точек не было)
        metric_step = self.active_models[model_id].get('metric_step', DEFAULT_METRIC_STEP)
        max_age_sec = metric_step * 10
        cursor.execute("""
            SELECT COUNT(*) 
            FROM anomaly_points 
            WHERE model_id = %s AND metric_id = %s
            AND timestamp > NOW() - INTERVAL '%s seconds'
        """, (model_id, metric_id, max_age_sec))
        recent_count = cursor.fetchone()[0]

        if recent_count == 0:
            cursor.execute("""
                UPDATE anomaly_system
                SET end_time = NOW()
                WHERE metric_id = %s AND end_time IS NULL
            """, (metric_id,))
            logger.info(f"Закрыты записи anomaly_system для метрики {metric_id} из-за неактивности")
    
    def _listen_for_changes(self) -> None:
        """Прослушивание уведомлений об изменениях моделей."""
        try:
            with self.notify_conn.cursor() as cursor:
                cursor.execute("LISTEN model_changes")
                logger.info("Слушатель изменений моделей запущен")
                
                while not self.stop_event.is_set():
                    # Проверка уведомлений с таймаутом
                    if select.select([self.notify_conn], [], [], DEFAULT_POLL_INTERVAL) == ([], [], []):
                        continue
                    
                    self.notify_conn.poll()
                    while self.notify_conn.notifies:
                        notify = self.notify_conn.notifies.pop(0)
                        self._handle_model_change(notify.payload)
                        
        except Exception as e:
            logger.error(f"Ошибка в слушателе изменений: {str(e)}", exc_info=True)
            if not self.stop_event.is_set():
                time.sleep(5)
                self._listen_for_changes()
    
    def _handle_model_change(self, payload: str) -> None:
        """Обработка уведомления об изменении модели.
        
        Args:
            payload: JSON-строка с данными об изменении
        """
        try:
            data = json.loads(payload)
            action = data.get('action')
            model_id = data.get('model_id')
            model_info_id = data.get('model_info_id')
            
            if action == 'activate':
                self._add_active_model(model_id)
            elif action == 'deactivate':
                self._remove_active_model(model_id)
            elif action == 'version_change' and model_info_id:
                self._handle_version_change(model_info_id)
                
        except Exception as e:
            logger.error(f"Ошибка обработки изменения модели: {str(e)}")
    
    def _handle_version_change(self, model_info_id: int) -> None:
        """Обработка изменения версии модели.
        
        Args:
            model_info_id: ID информации о модели
        """
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Получение новой активной модели
                    cursor.execute("""
                        SELECT id FROM models 
                        WHERE model_id = %s AND status = 'active'
                        ORDER BY created_at DESC LIMIT 1
                    """, (model_info_id,))
                    if new_active := cursor.fetchone():
                        # Удаление старых версий из кэша
                        for model_id in list(self.active_models.keys()):
                            if self.active_models[model_id]['model_info_id'] == model_info_id:
                                self._remove_active_model(model_id)
                        
                        # Добавление новой версии
                        self._add_active_model(new_active['id'])
        except Exception as e:
            logger.error(f"Ошибка обработки изменения версии: {str(e)}")
    
    def _schedule_detections(self) -> None:
        """Планирование задач детектирования."""
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now(timezone.utc)
                
                for model_id, model_data in self.active_models.items():
                    last_run = model_data.get('last_run')
                    prediction_step = model_data['prediction_step']
                    
                    if not last_run or (current_time - last_run).total_seconds() >= prediction_step:
                        self.task_queue.put({'model_id': model_id})
                        self.active_models[model_id]['last_run'] = current_time
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Ошибка планирования задач: {str(e)}", exc_info=True)
                time.sleep(10)


if __name__ == "__main__":
    try:
        detector = AnomalyDetectionService()
        detector.start()
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания, остановка сервиса")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)