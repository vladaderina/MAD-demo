import os
import argparse
import logging
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml
import psycopg2
from psycopg2.extras import Json, RealDictCursor

# Константы
DEFAULT_LOG_PATH = '/var/log/db_manager.log'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3
DEFAULT_CONFIG_DIR = '/app/config'
DEFAULT_CONFIG_PATH = f'{DEFAULT_CONFIG_DIR}/default_config.yaml'
USER_CONFIG_PATH = f'{DEFAULT_CONFIG_DIR}/config.yaml'
DEFAULT_DB_PORT = '5432'

class DatabaseManager:
    """Сервис управления базой данных для системы обнаружения аномалий."""

    def __init__(self, config_path: Optional[str] = None):
        """Инициализация сервиса."""
        self._setup_logging()
        self.config = self._load_config(config_path)
        self._validate_config()
        self.db_conn = self._init_db_connection()
        self.logger.info("Сервис управления БД инициализирован")

    def _setup_logging(self) -> None:
        """Настройка системы логирования."""
        log_path = self.config.get('general', {}).get('log_path', DEFAULT_LOG_PATH)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        self.logger = logging.getLogger('DBManager')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Файловый обработчик с ротацией
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Загрузка конфигурации с дефолтами."""
        try:
            # Создаем директорию для конфигов если нужно
            os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
            
            # Загружаем дефолтный конфиг
            default_config = {}
            if Path(DEFAULT_CONFIG_PATH).exists():
                with open(DEFAULT_CONFIG_PATH) as f:
                    default_config = yaml.safe_load(f) or {}
            
            # Определяем путь к пользовательскому конфигу
            user_config_path = config_path if config_path else USER_CONFIG_PATH
            user_config = {}
            
            if Path(user_config_path).exists():
                with open(user_config_path) as f:
                    user_config = yaml.safe_load(f) or {}
            
            # Глубокое слияние конфигов
            config = self._deep_merge(default_config, user_config)
            
            # Применяем дефолты для моделей
            if 'models' in config.get('general', {}):
                for model in config['general']['models']:
                    self._apply_model_defaults(model)
            
            return config
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
            raise

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Рекурсивное слияние словарей."""
        for key, value in update.items():
            if key in base and isinstance(value, dict) and isinstance(base[key], dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _apply_model_defaults(self, model: Dict) -> None:
        """Применение дефолтных параметров к модели."""
        # Дефолтные параметры для retrain
        if 'retrain' not in model:
            model['retrain'] = {}
        model['retrain'].setdefault('enabled', True)
        
        # Дефолтные параметры для training_period
        if 'training_period' not in model:
            model['training_period'] = {'auto_range': {'lookback_period': '7d'}}
        
        # Дефолтные параметры для hyperparameter_mode
        model.setdefault('hyperparameter_mode', 'optuna')
        model.setdefault('version_history', 3)

    def _validate_config(self) -> None:
        """Проверка конфигурации."""
        if not self.config.get('general', {}).get('metrics'):
            self.logger.warning("Конфигурация не содержит метрик")
        
        if not self.config.get('general', {}).get('models'):
            self.logger.warning("Конфигурация не содержит моделей")

    def _init_db_connection(self) -> 'psycopg2.connection':
        """Установка соединения с PostgreSQL."""
        conn_string = self.config['general'].get('db_conn_string')
        if not conn_string:
            raise ValueError("Необходимо указать db_conn_string в конфигурации")
        
        try:
            conn = psycopg2.connect(**self._parse_db_conn_string(conn_string))
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as e:
            self.logger.error(f"Ошибка подключения к БД: {str(e)}")
            raise

    def _parse_db_conn_string(self, conn_string: str) -> Dict:
        """Парсинг строки подключения к PostgreSQL."""
        try:
            if '://' in conn_string:
                conn_string = conn_string.split('://')[1]
            
            if '@' in conn_string:
                user_part, host_part = conn_string.split('@', 1)
                user, password = user_part.split(':', 1)
            else:
                host_part = conn_string
                user = password = None
            
            if '/' in host_part:
                host_port, database = host_part.split('/', 1)
            else:
                host_port = host_part
                database = None
            
            if ':' in host_port:
                host, port = host_port.split(':', 1)
            else:
                host = host_port
                port = DEFAULT_DB_PORT
            
            return {
                'host': host,
                'database': database,
                'user': user,
                'password': password,
                'port': port
            }
        except Exception as e:
            raise ValueError(f"Неверный формат строки подключения: {str(e)}")

    def _parse_duration(self, duration_str: str) -> int:
        """Преобразование строки длительности в секунды."""
        try:
            if duration_str.endswith('d'):
                return int(duration_str[:-1]) * 86400
            elif duration_str.endswith('h'):
                return int(duration_str[:-1]) * 3600
            elif duration_str.endswith('m'):
                return int(duration_str[:-1]) * 60
            elif duration_str.endswith('s'):
                return int(duration_str[:-1])
            return int(duration_str)
        except ValueError:
            raise ValueError(f"Неверный формат длительности: {duration_str}")

    def _insert_exclude_periods(self, cursor: 'psycopg2.cursor', metric_id: int, metric_config: Dict) -> None:
        """Добавление периодов исключения для метрик с проверкой ошибок."""
        if 'exclude_periods' not in metric_config:
            self.logger.debug(f"Для метрики ID {metric_id} нет периодов исключения")
            return
            
        if not isinstance(metric_config['exclude_periods'], list):
            self.logger.error(f"Некорректный формат exclude_periods для метрики ID {metric_id}")
            return
            
        for period in metric_config['exclude_periods']:
            try:
                # Валидация обязательных полей
                if not all(key in period for key in ['start', 'end']):
                    self.logger.error(f"Пропущен обязательный параметр start/end в периоде для метрики ID {metric_id}")
                    continue
                    
                # Преобразование времени
                start_time = datetime.fromisoformat(period['start'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(period['end'].replace('Z', '+00:00'))
                
                # Проверка на существование периода
                cursor.execute(
                    """SELECT id FROM anomaly_system 
                    WHERE metric_id = %s AND start_time = %s AND end_time = %s""",
                    (metric_id, start_time, end_time)
                )
                
                if cursor.fetchone() is None:
                    cursor.execute(
                        """
                        INSERT INTO anomaly_system (
                            start_time, end_time, anomaly_type, 
                            average_anom_score, metric_id, description
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            start_time, 
                            end_time,
                            period.get('anomaly_type', 'global'),
                            100,  # Максимальный уровень аномальности
                            metric_id,
                            period.get('reason', 'Предопределенный исключаемый период')
                        )
                    )
                    self.logger.info(
                        f"Добавлен период исключения для метрики {metric_id}: "
                        f"{start_time} - {end_time}"
                    )
                    
            except ValueError as e:
                self.logger.error(f"Ошибка формата времени в периоде исключения: {str(e)}")
            except Exception as e:
                self.logger.error(f"Ошибка добавления периода исключения: {str(e)}")

    def init_database(self) -> None:
        """Инициализация структуры базы данных."""
        if not self.config.get('general', {}).get('metrics') or not self.config.get('general', {}).get('models'):
            raise ValueError("Конфигурация должна содержать метрики и модели")
        
        self.logger.info("Начало инициализации БД")
        
        try:
            with self.db_conn.cursor() as cursor:
                # Создание таблицы метрик
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL UNIQUE,
                        status VARCHAR(50) DEFAULT 'active',
                        query TEXT NOT NULL,
                        interpolation VARCHAR(50),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Добавление метрик
                for metric in self.config['general']['metrics']:
                    cursor.execute(
                        """
                        INSERT INTO metrics (name, status, query, interpolation)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE 
                        SET status = EXCLUDED.status, 
                            query = EXCLUDED.query,
                            interpolation = EXCLUDED.interpolation
                        RETURNING id
                        """,
                        (
                            metric['name'], 
                            metric.get('status', 'active'), 
                            metric['query'],
                            metric.get('interpolation', 'linear')
                        )
                    )
                    metric_id = cursor.fetchone()[0]
                    self._insert_exclude_periods(cursor, metric_id, metric)
                    self.logger.info(f"Добавлена метрика {metric['name']} (ID: {metric_id})")
                
                # Создание таблицы информации о моделях
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS models_info (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL UNIQUE,
                        main_metric_id INTEGER REFERENCES metrics(id),
                        version_history INTEGER DEFAULT 3,
                        hyperparameter_mode VARCHAR(50) DEFAULT 'auto',
                        retrain_enabled BOOLEAN DEFAULT TRUE,
                        retrain_strategy VARCHAR(50),
                        retrain_interval VARCHAR(50),
                        training_period_type VARCHAR(50),
                        training_period_start TIMESTAMP,
                        training_period_end TIMESTAMP,
                        training_period_step VARCHAR(50),
                        training_period_lookback VARCHAR(50),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Создание таблицы дополнительных метрик для моделей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_additional_metrics (
                        model_id INTEGER REFERENCES models_info(id),
                        metric_id INTEGER REFERENCES metrics(id),
                        PRIMARY KEY (model_id, metric_id)
                    )
                """)
                
                # Создание таблицы версий моделей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS models (
                        id SERIAL PRIMARY KEY,
                        model_id INTEGER REFERENCES models_info(id),
                        status VARCHAR(50) DEFAULT 'waiting',
                        version VARCHAR(50) NOT NULL,
                        hyperparams JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Добавление моделей
                for model in self.config['general']['models']:
                    # Получаем ID основной метрики
                    cursor.execute(
                        "SELECT id FROM metrics WHERE name = %s",
                        (model['main_metric'],)
                    )
                    metric_id = cursor.fetchone()[0]
                    
                    # Определяем параметры обучения
                    training_period = model.get('training_period', {})
                    training_period_type = 'fixed_range' if 'fixed_range' in training_period else 'auto_range'
                    
                    # Добавляем информацию о модели
                    cursor.execute(
                        """
                        INSERT INTO models_info (
                            name, main_metric_id, version_history,
                            hyperparameter_mode, retrain_enabled,
                            retrain_strategy, retrain_interval,
                            training_period_type, training_period_start,
                            training_period_end, training_period_step,
                            training_period_lookback
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE
                        SET main_metric_id = EXCLUDED.main_metric_id,
                            version_history = EXCLUDED.version_history,
                            hyperparameter_mode = EXCLUDED.hyperparameter_mode,
                            retrain_enabled = EXCLUDED.retrain_enabled,
                            retrain_strategy = EXCLUDED.retrain_strategy,
                            retrain_interval = EXCLUDED.retrain_interval,
                            training_period_type = EXCLUDED.training_period_type,
                            training_period_start = EXCLUDED.training_period_start,
                            training_period_end = EXCLUDED.training_period_end,
                            training_period_step = EXCLUDED.training_period_step,
                            training_period_lookback = EXCLUDED.training_period_lookback
                        RETURNING id
                        """,
                        (
                            model['name'],
                            metric_id,
                            model.get('version_history', 3),
                            model.get('hyperparameter_mode', 'optuna'),
                            model.get('retrain', {}).get('enabled', True),
                            model.get('retrain', {}).get('strategy'),
                            model.get('retrain', {}).get('interval'),
                            training_period_type,
                            training_period.get('fixed_range', {}).get('start'),
                            training_period.get('fixed_range', {}).get('end'),
                            training_period.get('fixed_range', {}).get('step'),
                            training_period.get('auto_range', {}).get('lookback_period')
                        )
                    )
                    model_id = cursor.fetchone()[0]
                    
                    # Добавляем дополнительные метрики
                    for metric_name in model.get('additional_metrics', []):
                        cursor.execute(
                            "SELECT id FROM metrics WHERE name = %s",
                            (metric_name,)
                        )
                        additional_metric_id = cursor.fetchone()[0]
                        
                        cursor.execute(
                            """
                            INSERT INTO model_additional_metrics (model_id, metric_id)
                            VALUES (%s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (model_id, additional_metric_id)
                        )
                    
                    # Добавляем начальную версию модели
                    hyperparams = self._get_hyperparams(model)
                    cursor.execute(
                        """
                        INSERT INTO models (
                            model_id, status, version, hyperparams
                        ) VALUES (%s, %s, %s, %s)
                        """,
                        (model_id, 'waiting', '1.0', Json(hyperparams))
                    )
                    
                    self.logger.info(f"Добавлена модель {model['name']} (ID: {model_id})")
                
                # Создаем таблицу для настроек детектора
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detector_settings (
                        id SERIAL PRIMARY KEY,
                        metric_id INTEGER REFERENCES metrics(id),
                        delta_threshold VARCHAR(50),
                        median_window VARCHAR(50),
                        percentile_threshold FLOAT,
                        group_anomaly_confirmations INTEGER,
                        local_anomaly_confirmations INTEGER,
                        global_anomaly_confirmations INTEGER,
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(metric_id)
                    )
                """)
                
                # Добавляем настройки детектора
                if 'mad-detector' in self.config.get('mad-components', {}):
                    detector_config = self.config['mad-components']['mad-detector']
                    system_anomaly = detector_config.get('system_anomaly', {})
                    
                    # Для точечных аномалий
                    for point_anomaly in detector_config.get('points_anomaly', []):
                        metric_name = point_anomaly['metric']
                        cursor.execute(
                            "SELECT id FROM metrics WHERE name = %s",
                            (metric_name,)
                        )
                        result = cursor.fetchone()
                        if not result:
                            continue
                            
                        metric_id = result[0]
                        
                        cursor.execute(
                            """
                            INSERT INTO detector_settings (
                                metric_id, delta_threshold, median_window,
                                percentile_threshold, group_anomaly_confirmations,
                                local_anomaly_confirmations, global_anomaly_confirmations
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (metric_id) DO UPDATE
                            SET delta_threshold = EXCLUDED.delta_threshold,
                                median_window = EXCLUDED.median_window,
                                percentile_threshold = EXCLUDED.percentile_threshold,
                                group_anomaly_confirmations = EXCLUDED.group_anomaly_confirmations,
                                local_anomaly_confirmations = EXCLUDED.local_anomaly_confirmations,
                                global_anomaly_confirmations = EXCLUDED.global_anomaly_confirmations
                            """,
                            (
                                metric_id,
                                str(point_anomaly.get('delta_threshold', 'auto')),
                                point_anomaly.get('median_window', '30m'),
                                system_anomaly.get('percentile_threshold', 0.95),
                                system_anomaly.get('min_confirmations', {}).get('group_anomaly', 20),
                                system_anomaly.get('min_confirmations', {}).get('local_anomaly', 3),
                                system_anomaly.get('min_confirmations', {}).get('global_anomaly', 1)
                            )
                        )
                
                self.db_conn.commit()
                self.logger.info("Инициализация БД успешно завершена")
                
        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Ошибка инициализации БД: {str(e)}")
            raise

    def _get_hyperparams(self, model_config: Dict) -> Dict:
        """Получение гиперпараметров модели с проверкой"""
        if model_config.get('hyperparameter_mode') == 'manual':
            if 'manual_params' not in model_config:
                raise ValueError("manual_params must be specified when hyperparameter_mode=manual")
            if not isinstance(model_config['manual_params'], dict):
                raise ValueError("manual_params must be a dictionary")
            return model_config['manual_params']
        
        # Возвращаем дефолтные параметры для optuna
        return {
            'direction': 'minimize',
            'metric': 'val_loss',
            'n_trials': 20,
            'sampler': 'TPE',
            'pruner': 'Hyperband',
            'fixed_parameters': {
                'loss': 'mean_squared_error',
                'optimizer': 'adam',
                'validation_split': 0.2
            }
        }

    def clean_old_data(self) -> None:
        """Очистка старых данных с учетом конфигурации каждой метрики."""
        self.logger.info("Начало очистки данных с учетом конфигурации метрик")
        
        try:
            with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # 1. Получаем все активные метрики
                cursor.execute("""
                    SELECT m.id, m.name, ds.delta_threshold, ds.median_window,
                           ds.percentile_threshold, ds.group_anomaly_confirmations
                    FROM metrics m
                    LEFT JOIN detector_settings ds ON m.id = ds.metric_id
                    WHERE m.status = 'active'
                """)
                metrics = cursor.fetchall()
                
                if not metrics:
                    self.logger.warning("Нет активных метрик для очистки")
                    return
                
                total_points_deleted = 0
                total_anomalies_deleted = 0
                
                # 2. Для каждой метрики определяем период хранения
                for metric in metrics:
                    metric_id = metric['id']
                    metric_name = metric['name']
                    
                    # Определяем период хранения
                    if metric['delta_threshold'] == 'auto':
                        # Для auto берем максимальное из median_window и периода для групповой аномалии
                        median_window = metric.get('median_window', '30m')
                        median_seconds = self._parse_duration(median_window)
                        
                        # Период для групповой аномалии (confirmations * 60 секунд)
                        group_seconds = metric.get('group_anomaly_confirmations', 20) * 60
                        
                        retention_seconds = max(median_seconds, group_seconds)
                    else:
                        # Для фиксированного delta_threshold храним только для групповой аномалии
                        retention_seconds = metric.get('group_anomaly_confirmations', 20) * 60
                    
                    cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=retention_seconds)
                    
                    self.logger.info(
                        f"Очистка данных для метрики {metric_name} (ID: {metric_id}) "
                        f"старше {retention_seconds//60} минут (до {cutoff_time})"
                    )
                    
                    # Удаляем старые точки аномалий для этой метрики
                    cursor.execute(
                        "DELETE FROM anomaly_points WHERE metric_id = %s AND timestamp < %s",
                        (metric_id, cutoff_time)
                    )
                    points_deleted = cursor.rowcount
                    
                    # Удаляем старые системные аномалии для этой метрики
                    cursor.execute(
                        """DELETE FROM anomaly_system 
                        WHERE metric_id = %s AND end_time IS NOT NULL AND end_time < %s""",
                        (metric_id, cutoff_time)
                    )
                    anomalies_deleted = cursor.rowcount
                    
                    total_points_deleted += points_deleted
                    total_anomalies_deleted += anomalies_deleted
                    
                    self.logger.info(
                        f"Для метрики {metric_name} удалено: "
                        f"{points_deleted} точек, {anomalies_deleted} аномалий"
                    )
                
                self.db_conn.commit()
                self.logger.info(
                    f"Очистка завершена. Всего удалено: "
                    f"{total_points_deleted} точек аномалий, "
                    f"{total_anomalies_deleted} системных аномалий"
                )
                
        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Ошибка при очистке данных: {str(e)}")
            raise

    def close(self) -> None:
        """Корректное закрытие соединений."""
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()
            self.logger.info("Соединение с БД закрыто")

def main():
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Сервис управления базой данных для системы обнаружения аномалий"
    )
    parser.add_argument('--config', help="Путь к конфигурационному файлу")
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Команда init
    init_parser = subparsers.add_parser('init', help='Инициализация БД')
    
    # Команда clean
    clean_parser = subparsers.add_parser('clean', help='Очистка старых данных')
    
    args = parser.parse_args()
    
    manager = None
    try:
        manager = DatabaseManager(args.config)
        
        if args.command == 'init':
            manager.init_database()
        elif args.command == 'clean':
            manager.clean_old_data()
            
    except Exception as e:
        logging.error(f"Ошибка выполнения команды: {str(e)}", exc_info=True)
        exit(1)
    finally:
        if manager:
            manager.close()

if __name__ == "__main__":
    main()