import argparse
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple, Any

import psycopg2
import schedule
import yaml
from psycopg2.extras import Json

# Константы
DEFAULT_LOG_PATH = 'db_manager.log'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3
DEFAULT_RETENTION_PERIOD = '30d'  # 30 дней
DEFAULT_CLEANUP_INTERVAL = '1d'   # 1 день
MIN_TRAINING_PERIOD = '5d'        # Минимальный период обучения


class DatabaseManager:
    """Сервис управления базой данных для системы обнаружения аномалий."""

    def __init__(self, config_path: str):
        """Инициализация сервиса.
        
        Args:
            config_path: Путь к конфигурационному файлу
            
        Raises:
            FileNotFoundError: Если файл конфигурации не найден
            yaml.YAMLError: Если файл содержит невалидный YAML
            psycopg2.OperationalError: При ошибке подключения к БД
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.db_conn = self._init_db_connection()
        self.logger.info("Сервис управления БД инициализирован")

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка и валидация конфигурационного файла.
        
        Args:
            config_path: Путь к YAML конфигурационному файлу
            
        Returns:
            Загруженная конфигурация
            
        Raises:
            FileNotFoundError: Если файл конфигурации не найден
            yaml.YAMLError: Если файл содержит невалидный YAML
            ValueError: Если конфигурация не содержит обязательных параметров
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Валидация обязательных параметров
            required_sections = ['system', 'metrics', 'mad_predictor']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Отсутствует обязательная секция конфигурации: {section}")
                    
            if 'db_conn_string' not in config['system']:
                raise ValueError("Отсутствует обязательный параметр: system.db_conn_string")
                
            return config
            
        except FileNotFoundError as e:
            self.logger.error(f"Конфигурационный файл не найден: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Ошибка парсинга YAML: {str(e)}")
            raise

    def _setup_logging(self) -> None:
        """Настройка системы логирования с ротацией логов."""
        log_path = self.config['system'].get('log_path', DEFAULT_LOG_PATH)
        
        # Основной логгер
        self.logger = logging.getLogger('DBManager')
        self.logger.setLevel(logging.INFO)
        
        # Форматтер
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Обработчики
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _init_db_connection(self) -> 'psycopg2.connection':
        """Установка соединения с PostgreSQL базой данных.
        
        Returns:
            Активное соединение с БД
            
        Raises:
            psycopg2.OperationalError: При ошибке подключения к БД
        """
        try:
            conn = psycopg2.connect(self.config['system']['db_conn_string'])
            self.logger.info("Успешное подключение к базе данных")
            return conn
        except psycopg2.OperationalError as e:
            self.logger.error(f"Ошибка подключения к базе данных: {str(e)}")
            raise

    def _parse_duration(self, duration_str: str) -> int:
        """Преобразование строки длительности в секунды.
        
        Args:
            duration_str: Строка длительности (например, '1d', '2h', '30m')
            
        Returns:
            Длительность в секундах
            
        Raises:
            ValueError: Если строка имеет неверный формат
        """
        try:
            if duration_str.endswith('d'):
                return int(duration_str[:-1]) * 86400  # Дни
            elif duration_str.endswith('h'):
                return int(duration_str[:-1]) * 3600   # Часы
            elif duration_str.endswith('m'):
                return int(duration_str[:-1]) * 60     # Минуты
            elif duration_str.endswith('s'):
                return int(duration_str[:-1])          # Секунды
            else:
                return int(duration_str)               # По умолчанию считаем как секунды
        except ValueError as e:
            self.logger.error(f"Неверный формат длительности: {duration_str}")
            raise ValueError(f"Неверный формат длительности: {duration_str}")

    def _insert_exclude_periods(self, cursor: 'psycopg2.cursor', metric_id: int, metric_config: Dict) -> None:
        """Добавление периодов исключения для метрик.
        
        Args:
            cursor: Курсор базы данных
            metric_id: ID метрики
            metric_config: Конфигурация метрики
        """
        if 'exclude_periods' not in metric_config:
            return
            
        for period in metric_config['exclude_periods']:
            # Проверяем существование периода аномалии
            cursor.execute(
                "SELECT id FROM anomaly_system WHERE metric_id = %s AND start_time = %s",
                (metric_id, period['start']))
            
            if not cursor.fetchone():
                # Вставляем новый период системной аномалии
                cursor.execute(
                    """
                    INSERT INTO anomaly_system (
                        start_time, end_time, anomaly_type, 
                        average_anom_score, metric_id, description
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        period['start'], 
                        period['end'],
                        period.get('anomaly_type', 'global'),
                        100,  # Максимальный уровень аномальности
                        metric_id,
                        period.get('reason', 'Предопределенный исключаемый период')
                    )
                )
                self.logger.info(
                    f"Добавлен период аномалии для метрики {metric_id}: "
                    f"{period['start']} - {period['end']}"
                )

    def _get_hyperparams(self, model_config: Dict) -> Dict:
        """Получение гиперпараметров модели.
        
        Args:
            model_config: Конфигурация модели
            
        Returns:
            Словарь с гиперпараметрами
        """
        if model_config['hyperparameter_mode'] == 'manual':
            return model_config['manual_params']
        else:
            # Параметры по умолчанию для оптимизации
            return {
                'layers': [
                    {'type': 'LSTM', 'units': 64, 'return_sequences': True},
                    {'type': 'LSTM', 'units': 32, 'return_sequences': False},
                    {'type': 'Dense', 'units': 1}
                ],
                'learning_rate': 0.001,
                'batch_size': 32,
                'dropout': 0.2,
                'epochs': 50,
                'loss': 'mean_squared_error',
                'optimizer': 'adam',
                'validation_split': 0.2
            }
    
    def init_database(self) -> None:
        """Инициализация базы данных - создание метрик и моделей.
        
        Raises:
            psycopg2.DatabaseError: При ошибке работы с БД
        """
        self.logger.info("Начало инициализации базы данных")
        
        try:
            with self.db_conn.cursor() as cursor:
                # Создание метрик из конфигурации
                for metric in self.config['metrics']:
                    # Проверка существования метрики
                    cursor.execute(
                        "SELECT id FROM metrics WHERE name = %s",
                        (metric['name'],))
                    
                    if cursor.fetchone() is None:
                        # Добавление новой метрики
                        cursor.execute(
                            """
                            INSERT INTO metrics (name, status, query)
                            VALUES (%s, 'active', %s)
                            RETURNING id
                            """,
                            (metric['name'], metric['query']))
                        
                        metric_id = cursor.fetchone()[0]
                        self._insert_exclude_periods(cursor, metric_id, metric)
                        self.logger.info(f"Добавлена метрика {metric['name']} с id {metric_id}")

                # Создание моделей из конфигурации
                for model in self.config['mad_predictor']['models']:
                    # Получение ID основной метрики
                    cursor.execute(
                        "SELECT id FROM metrics WHERE name = %s",
                        (model['main_metric'],))
                    metric_id = cursor.fetchone()[0]
                    
                    # Проверка существования модели
                    cursor.execute(
                        "SELECT id FROM models_info WHERE name = %s",
                        (model['name'],))
                    
                    if cursor.fetchone() is None:
                        # Определение периода обучения
                        training_period = model['training_period']
                        start, end, step = self._calculate_training_period(training_period)
                        
                        # Добавление информации о модели
                        cursor.execute(
                            """
                            INSERT INTO models_info (
                                name, metric_id, max_stored_versions, 
                                hyperparams_mode, active_version,
                                training_start, training_end, step
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (
                                model['name'], 
                                metric_id, 
                                model['version_history'],
                                model['hyperparameter_mode'], 
                                '1',
                                start, 
                                end, 
                                int(self._parse_duration(step))
                            ))
                        
                        model_info_id = cursor.fetchone()[0]

                        # Добавление самой модели
                        version = '1.0'
                        hyperparams = self._get_hyperparams(model)

                        cursor.execute(
                            """
                            INSERT INTO models (
                                status, version, model_id, hyperparams
                            ) VALUES (%s, %s, %s, %s)
                            RETURNING id
                            """,
                            ('waiting', version, model_info_id, Json(hyperparams)))
                        
                        model_id = cursor.fetchone()[0]
                        self.logger.info(f"Добавлена модель {model['name']} версии {version}")

                        # Добавление дополнительных метрик
                        if 'additional_metrics' in model:
                            for add_metric in model['additional_metrics']:
                                cursor.execute(
                                    "SELECT id FROM metrics WHERE name = %s",
                                    (add_metric,))
                                
                                add_metric_id = cursor.fetchone()[0]
                                cursor.execute(
                                    """
                                    INSERT INTO features (model_id, metric_id) 
                                    VALUES (%s, %s)
                                    """,
                                    (model_info_id, add_metric_id))
            
            self.db_conn.commit()
            self.logger.info("Инициализация базы данных завершена")
            
        except psycopg2.DatabaseError as e:
            self.db_conn.rollback()
            self.logger.error(f"Ошибка инициализации БД: {str(e)}")
            raise

    def _calculate_training_period(self, training_period: Dict) -> Tuple[str, str, str]:
        """Расчет периода обучения модели.
        
        Args:
            training_period: Конфигурация периода обучения
            
        Returns:
            Кортеж (start, end, step)
        """
        if 'fixed_range' in training_period:
            start = training_period['fixed_range']['start']
            end = training_period['fixed_range']['end']
            step = training_period['fixed_range'].get('step', '1m')
        elif 'auto_range' in training_period:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(
                seconds=self._parse_duration(training_period['auto_range']['lookback_period'])
            )
            start = start_dt.isoformat(timespec='microseconds').replace('+00:00', 'Z')
            end = end_dt.isoformat(timespec='microseconds').replace('+00:00', 'Z')
            step = training_period['auto_range'].get('step', '1m')
        else:
            raise ValueError("Не указан тип периода обучения")
        
        return start, end, step
    
    def retrain_marker(self, model_name: Optional[str] = None, strategy_type: Optional[str] = None) -> None:
        """Пометка моделей для переобучения.
        
        Args:
            model_name: Имя модели для переобучения
            strategy_type: Тип стратегии ('sliding_window' или 'expanding_window')
            
        Raises:
            psycopg2.DatabaseError: При ошибке работы с БД
            ValueError: Если модель не найдена
        """
        self.logger.info(f"Пометка для переобучения. Модель: {model_name}, стратегия: {strategy_type}")
        
        try:
            with self.db_conn.cursor() as cursor:
                # Получение данных модели
                cursor.execute(
                    """
                    SELECT mi.id, mi.name, mi.training_start, mi.training_end, 
                        mi.active_version, mi.max_stored_versions
                    FROM models_info mi
                    WHERE mi.name = %s
                    """,
                    (model_name,))
                
                model_data = cursor.fetchone()
                
                if not model_data:
                    self.logger.warning(f"Модель {model_name} не найдена")
                    raise ValueError(f"Модель {model_name} не найдена")
                
                model_id, name, training_start, training_end, active_version, max_versions = model_data
                
                # Преобразование дат
                if isinstance(training_start, str):
                    training_start = datetime.fromisoformat(training_start.replace('Z', '+00:00'))
                if isinstance(training_end, str):
                    training_end = datetime.fromisoformat(training_end.replace('Z', '+00:00'))
                
                # Поиск конфигурации модели
                model_config = next(
                    (m for m in self.config['mad_predictor']['models'] if m['name'] == name),
                    None
                )
                
                if not model_config:
                    self.logger.warning(f"Конфигурация модели {name} не найдена")
                    raise ValueError(f"Конфигурация модели {name} не найдена")
                
                # Расчет нового периода обучения
                new_start, new_end = self._calculate_new_period(
                    strategy_type,
                    training_start,
                    training_end,
                    model_config
                )
                
                # Генерация новой версии
                new_version = self._generate_next_version(active_version)
                
                # Создание новой версии модели
                hyperparams = self._get_hyperparams(model_config)
                cursor.execute(
                    """
                    INSERT INTO models (
                        model_data, created_at, status, version, model_id, hyperparams
                    ) VALUES (%s, %s, 'waiting', %s, %s, %s)
                    """,
                    (
                        b'placeholder',
                        datetime.now(timezone.utc),
                        new_version,
                        model_id,
                        Json(hyperparams)
                    ))
                
                # Обновление информации о модели
                cursor.execute(
                    """
                    UPDATE models_info
                    SET training_start = %s,
                        training_end = %s,
                        active_version = %s
                    WHERE id = %s
                    """,
                    (
                        new_start.astimezone(timezone.utc),
                        new_end.astimezone(timezone.utc),
                        new_version,
                        model_id
                    ))
                
                # Очистка старых версий
                cursor.execute(
                    """
                    DELETE FROM models 
                    WHERE model_id = %s AND id NOT IN (
                        SELECT id FROM models 
                        WHERE model_id = %s 
                        ORDER BY created_at DESC 
                        LIMIT %s
                    )
                    """,
                    (model_id, model_id, max_versions))
                
                self.logger.info(f"Модель {name} помечена для переобучения. Версия: {new_version}")
            
            self.db_conn.commit()
            
        except psycopg2.DatabaseError as e:
            self.db_conn.rollback()
            self.logger.error(f"Ошибка пометки для переобучения: {str(e)}")
            raise

    def _calculate_new_period(self, strategy: Optional[str], 
                            current_start: datetime, 
                            current_end: datetime,
                            model_config: Dict) -> Tuple[datetime, datetime]:
        """Расчет нового периода обучения.
        
        Args:
            strategy: Тип стратегии ('sliding_window' или 'expanding_window')
            current_start: Текущая начальная дата периода
            current_end: Текущая конечная дата периода
            model_config: Конфигурация модели
            
        Returns:
            Кортеж (new_start, new_end)
        """
        now = datetime.now(timezone.utc)
        
        if strategy == 'sliding_window':
            # Скользящее окно - сохраняем длительность
            duration = current_end - current_start
            new_end = now
            new_start = new_end - duration
            
        elif strategy == 'expanding_window':
            # Расширяющееся окно - сохраняем начало
            new_start = current_start
            new_end = now
            
            # Проверка максимальной длительности
            max_duration = self._parse_duration(model_config.get('max_training_period', '365d'))
            if (new_end - new_start) > timedelta(seconds=max_duration):
                new_start = new_end - timedelta(seconds=max_duration)
        else:
            # Если стратегия не указана, используем исходный период
            new_start, new_end = current_start, current_end
        
        # Проверка минимальной длительности (5 дней по умолчанию)
        min_duration = timedelta(seconds=self._parse_duration(
            model_config.get('min_training_period', MIN_TRAINING_PERIOD)
        ))
        if (new_end - new_start) < min_duration:
            new_start = new_end - min_duration
        
        return new_start, new_end
    
    def _generate_next_version(self, current_version: str) -> str:
        """Генерация номера следующей версии.
        
        Args:
            current_version: Текущая версия
            
        Returns:
            Новая версия
        """
        try:
            return str(int(current_version) + 1)
        except ValueError:
            return datetime.utcnow().strftime("%Y%m%d%H%M%S")
    
    def clean_anomaly_points(self) -> None:
        """Очистка старых точек аномалий.
        
        Raises:
            psycopg2.DatabaseError: При ошибке работы с БД
        """
        self.logger.info("Очистка точек аномалий")
        
        try:
            with self.db_conn.cursor() as cursor:
                # Получение периода хранения из конфига
                retention_period = self.config.get('mad_detector', {}).get('points_retention', DEFAULT_RETENTION_PERIOD)
                retention_seconds = self._parse_duration(retention_period)
                cutoff_time = datetime.utcnow() - timedelta(seconds=retention_seconds)
                
                # Удаление старых данных
                cursor.execute(
                    "DELETE FROM anomaly_points WHERE timestamp < %s",
                    (cutoff_time,))
                deleted_count = cursor.rowcount
                
                cursor.execute(
                    "DELETE FROM anomaly_system WHERE end_time IS NOT NULL AND end_time < %s",
                    (cutoff_time,))
                deleted_system_count = cursor.rowcount
                
                self.db_conn.commit()
                self.logger.info(
                    f"Очищено {deleted_count} точек аномалий и {deleted_system_count} системных аномалий"
                )
                
        except psycopg2.DatabaseError as e:
            self.db_conn.rollback()
            self.logger.error(f"Ошибка очистки: {str(e)}")
            raise
    
    def start_scheduled_tasks(self) -> None:
        """Запуск задач по расписанию."""
        self.logger.info("Запуск запланированных задач")
        
        # Планирование переобучения моделей
        for model in self.config['mad_predictor']['models']:
            if model['retrain']['enabled']:
                interval = model['retrain']['interval']
                strategy = model['retrain']['strategy']
                
                if interval.endswith('d'):
                    days = int(interval[:-1])
                    schedule.every(days).days.do(
                        self.retrain_marker, 
                        model_name=model['name'],
                        strategy_type=strategy
                    )
                elif interval.endswith('h'):
                    hours = int(interval[:-1])
                    schedule.every(hours).hours.do(
                        self.retrain_marker,
                        model_name=model['name'],
                        strategy_type=strategy
                    )
                
                self.logger.info(
                    f"Запланировано переобучение модели {model['name']} "
                    f"каждые {interval} со стратегией {strategy}"
                )
        
        # Планирование очистки аномалий
        cleanup_interval = self.config.get('system', {}).get('cleanup_interval', DEFAULT_CLEANUP_INTERVAL)
        if cleanup_interval.endswith('d'):
            days = int(cleanup_interval[:-1])
            schedule.every(days).days.do(self.clean_anomaly_points)
        elif cleanup_interval.endswith('h'):
            hours = int(cleanup_interval[:-1])
            schedule.every(hours).hours.do(self.clean_anomaly_points)
        
        self.logger.info(f"Запланирована очистка аномалий каждые {cleanup_interval}")
        
        # Запуск планировщика в фоне
        def run_scheduler() -> None:
            """Функция для выполнения задач по расписанию."""
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        self.logger.info("Планировщик запущен")


def main() -> None:
    """Основная функция для обработки команд."""
    parser = argparse.ArgumentParser(description="Сервис управления базой данных для системы обнаружения аномалий")
    parser.add_argument('--config', type=str, required=True, help="Путь к конфигурационному файлу")
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Команды
    init_parser = subparsers.add_parser('init', help='Инициализация БД')
    retrain_parser = subparsers.add_parser('retrain', help='Пометка для переобучения')
    retrain_parser.add_argument('--model', type=str, required=True, help='Имя модели')
    retrain_parser.add_argument('--strategy', type=str, required=True,
                               choices=['sliding_window', 'expanding_window'], 
                               help='Стратегия переобучения')
    clean_parser = subparsers.add_parser('clean-points', help='Очистка аномалий')
    start_parser = subparsers.add_parser('start', help='Запуск сервиса')
    
    args = parser.parse_args()
    
    try:
        service = DatabaseManager(args.config)
        
        if args.command == 'init':
            service.init_database()
        elif args.command == 'retrain':
            service.retrain_marker(args.model, args.strategy)
        elif args.command == 'clean-points':
            service.clean_anomaly_points()
        elif args.command == 'start':
            service.start_scheduled_tasks()
            # Бесконечный цикл для работы сервиса
            while True:
                time.sleep(1)
    
    except Exception as e:
        logging.error(f"Ошибка сервиса: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()