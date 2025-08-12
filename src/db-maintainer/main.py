import os
import argparse
import logging
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml
import psycopg2
from psycopg2.extras import Json

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
        log_path = os.getenv('LOG_PATH', DEFAULT_LOG_PATH)
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
            if 'trainer' in config and 'models' in config['trainer']:
                model_defaults = config.get('defaults', {}).get('model', {})
                for model in config['trainer']['models']:
                    self._apply_model_defaults(model, model_defaults)
            
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

    def _apply_model_defaults(self, model: Dict, defaults: Dict) -> None:
        """Применение дефолтных параметров к модели."""
        for key, value in defaults.items():
            if key not in model:
                model[key] = value
            elif isinstance(value, dict) and isinstance(model[key], dict):
                self._apply_model_defaults(model[key], value)

    def _validate_config(self) -> None:
        """Проверка конфигурации."""
        if not self.config.get('metrics'):
            self.logger.warning("Конфигурация не содержит метрик")
        
        if not self.config.get('trainer', {}).get('models'):
            self.logger.warning("Конфигурация не содержит моделей")

    def _init_db_connection(self) -> 'psycopg2.connection':
        """Установка соединения с PostgreSQL."""
        conn_string = os.getenv('DB_CONN_STRING')
        if not conn_string:
            raise ValueError("Необходимо указать DB_CONN_STRING в переменных окружения")
        
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
        if not self.config.get('metrics') or not self.config.get('trainer', {}).get('models'):
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
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Добавление метрик
                for metric in self.config['metrics']:
                    cursor.execute(
                        """
                        INSERT INTO metrics (name, status, query)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (name) DO UPDATE 
                        SET status = EXCLUDED.status, query = EXCLUDED.query
                        RETURNING id
                        """,
                        (metric['name'], metric.get('status', 'active'), metric['query'])
                    )
                    metric_id = cursor.fetchone()[0]
                    self._insert_exclude_periods(cursor, metric_id, metric)
                    self.logger.info(f"Добавлена метрика {metric['name']} (ID: {metric_id})")
                
                # Создание таблицы информации о моделях (упрощенная версия)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS models_info (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL UNIQUE,
                        metric_id INTEGER REFERENCES metrics(id),
                        max_stored_versions INTEGER DEFAULT 3,
                        hyperparams_mode VARCHAR(50) DEFAULT 'auto',
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Создание таблицы версий моделей (упрощенная версия)
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
                
                # Добавление моделей (упрощенная версия)
                for model in self.config['trainer']['models']:
                    # Получаем ID основной метрики
                    cursor.execute(
                        "SELECT id FROM metrics WHERE name = %s",
                        (model['main_metric'],)
                    )
                    metric_id = cursor.fetchone()[0]
                    
                    # Добавляем модель
                    cursor.execute(
                        """
                        INSERT INTO models_info (
                            name, metric_id, max_stored_versions,
                            hyperparams_mode
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE
                        SET metric_id = EXCLUDED.metric_id,
                            max_stored_versions = EXCLUDED.max_stored_versions,
                            hyperparams_mode = EXCLUDED.hyperparams_mode
                        RETURNING id
                        """,
                        (
                            model['name'],
                            metric_id,
                            model.get('version_history', 3),
                            model.get('hyperparameter_mode', 'auto')
                        )
                    )
                    model_id = cursor.fetchone()[0]
                    
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
        
        # Возвращаем дефолтные параметры
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

    def clean_old_data(self, retention_period: str = "30d") -> None:
        """Очистка старых данных."""
        retention_seconds = self._parse_duration(retention_period)
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=retention_seconds)
        
        self.logger.info(f"Очистка данных старше {retention_period} (до {cutoff_time})")
        
        try:
            with self.db_conn.cursor() as cursor:
                # Удаляем старые точки аномалий
                cursor.execute(
                    "DELETE FROM anomaly_points WHERE timestamp < %s",
                    (cutoff_time,)
                )
                points_deleted = cursor.rowcount
                
                # Удаляем старые системные аномалии
                cursor.execute(
                    "DELETE FROM anomaly_system WHERE end_time IS NOT NULL AND end_time < %s",
                    (cutoff_time,)
                )
                anomalies_deleted = cursor.rowcount
                
                self.db_conn.commit()
                self.logger.info(
                    f"Очистка завершена. Удалено: "
                    f"{points_deleted} точек аномалий, "
                    f"{anomalies_deleted} системных аномалий"
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
    clean_parser.add_argument('--retention', default='30d',
                            help='Период хранения данных (например, 30d, 1h)')
    
    args = parser.parse_args()
    
    manager = None
    try:
        manager = DatabaseManager(args.config)
        
        if args.command == 'init':
            manager.init_database()
        elif args.command == 'clean':
            manager.clean_old_data(args.retention)
            
    except Exception as e:
        logging.error(f"Ошибка выполнения команды: {str(e)}", exc_info=True)
        exit(1)
    finally:
        if manager:
            manager.close()

if __name__ == "__main__":
    main()