import argparse
import asyncio
import json
import logging
import os
import pickle
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

import aiohttp
import asyncpg
import numpy as np
import yaml
from aiohttp import web

# Константы
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8080
DEFAULT_LOG_PATH = 'anomaly_api.log'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3
MODEL_STATUS_ACTIVE = 'active'
MODEL_STATUS_INACTIVE = 'inactive'
MODEL_STATUS_WAITING = 'waiting'

# Настройка логгера
logger = logging.getLogger(__name__)


class AnomalyAPIService:
    """API сервис для управления моделями обнаружения аномалий."""

    def __init__(self, config_path: str):
        """Инициализация сервиса.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.db_pool = None
        self.app = web.Application()
        self._setup_routes()
        self.model_storage_path = self.config['system'].get('model_storage_path', './models')

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурационного файла.
        
        Args:
            config_path: Путь к YAML конфигурационному файлу
            
        Returns:
            Загруженная конфигурация
            
        Raises:
            FileNotFoundError: Если файл конфигурации не найден
            yaml.YAMLError: Если файл содержит невалидный YAML
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self._validate_config(config)
                return config
        except FileNotFoundError:
            logger.error(f"Конфигурационный файл не найден: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Ошибка парсинга YAML: {str(e)}")
            raise

    def _validate_config(self, config: Dict) -> None:
        """Валидация обязательных параметров конфигурации.
        
        Args:
            config: Загруженная конфигурация
            
        Raises:
            ValueError: Если отсутствуют обязательные параметры
        """
        required_sections = ['system', 'database']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Отсутствует обязательная секция конфигурации: {section}")
                
        if 'db_conn_string' not in config['system']:
            raise ValueError("Отсутствует обязательный параметр: system.db_conn_string")

    def _setup_logging(self) -> None:
        """Настройка системы логирования."""
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Файловый обработчик с ротацией
        log_path = self.config['system'].get('log_path', DEFAULT_LOG_PATH)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def _setup_routes(self) -> None:
        """Настройка маршрутов API."""
        # Маршруты для работы с моделями
        self.app.router.add_route('POST', '/models/retrain', self.retrain_model)
        self.app.router.add_route('DELETE', '/models/{model_name}/{version}', self.delete_model)
        self.app.router.add_route('PUT', '/models/{model_name}/{version}/activate', self.activate_model)
        self.app.router.add_route('GET', '/models/{model_name}', self.list_models)
        self.app.router.add_route('POST', '/models', self.add_model)
        self.app.router.add_route('GET', '/models/{model_name}/{version}', self.show_model)
        self.app.router.add_route('GET', '/models/{model_name}/{version}/export', self.export_model)
        self.app.router.add_route('POST', '/models/{model_name}/import', self.import_model)

        # Маршруты для работы с метриками
        self.app.router.add_route('POST', '/metrics', self.add_metric)
        self.app.router.add_route('DELETE', '/metrics/{metric_name}', self.delete_metric)
        self.app.router.add_route('GET', '/metrics', self.list_metrics)
        self.app.router.add_route('GET', '/metrics/{metric_name}', self.show_metric)

        # Маршруты для работы с аномалиями
        self.app.router.add_route('GET', '/anomalies', self.list_anomalies)
        self.app.router.add_route('POST', '/anomalies/system', self.add_system_anomaly)

        # Маршруты для диагностики
        self.app.router.add_route('GET', '/diagnostics/models/{model_name}/status', self.model_status)

    async def _connect_db(self) -> None:
        """Установка соединения с базой данных.
        
        Raises:
            asyncpg.PostgresError: При ошибке подключения к БД
        """
        try:
            db_config = self._parse_db_conn_string(self.config['system']['db_conn_string'])
            self.db_pool = await asyncpg.create_pool(**db_config)
            logger.info("Успешное подключение к базе данных")
        except asyncpg.PostgresError as e:
            logger.error(f"Ошибка подключения к БД: {str(e)}")
            raise

    def _parse_db_conn_string(self, conn_string: str) -> Dict:
        """Парсинг строки подключения к БД.
        
        Args:
            conn_string: Строка подключения в формате postgresql://user:password@host:port/database
            
        Returns:
            Словарь с параметрами подключения
            
        Raises:
            ValueError: Если строка подключения имеет неверный формат
        """
        try:
            # Формат строки: postgresql://user:password@host:port/database
            db_parts = conn_string.split('://')[1].split('@')
            creds, host_port_db = db_parts[0].split(':'), db_parts[1].split('/')
            host, port = host_port_db[0].split(':')
            
            return {
                'user': creds[0],
                'password': creds[1],
                'host': host,
                'port': int(port),
                'database': host_port_db[1]
            }
        except (IndexError, ValueError) as e:
            logger.error(f"Неверный формат строки подключения: {conn_string}")
            raise ValueError("Неверный формат строки подключения к БД")

    async def start(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        """Запуск API сервиса.
        
        Args:
            host: Хост для запуска сервера
            port: Порт для запуска сервера
            
        Raises:
            Exception: При ошибке запуска сервиса
        """
        try:
            logger.info("Запуск Anomaly API Service")
            
            # Инициализация подключения к БД
            await self._connect_db()
            
            # Запуск сервера
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            
            logger.info(f"Сервис запущен по адресу http://{host}:{port}")
            await asyncio.Event().wait()
            
        except Exception as e:
            logger.error(f"Ошибка запуска сервиса: {str(e)}", exc_info=True)
            raise
        finally:
            if self.db_pool:
                await self.db_pool.close()
            logger.info("Сервис остановлен")

    # === Методы для работы с моделями ===

    async def retrain_model(self, request: web.Request) -> web.Response:
        """Пометка модели для переобучения.
        
        Args:
            request: Запрос с параметрами модели
            
        Returns:
            Ответ с результатом операции
        """
        try:
            data = await request.json()
            model_name = data.get('model_name')
            
            if not model_name:
                return web.json_response(
                    {"error": "Не указано имя модели"},
                    status=400
                )
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE models 
                    SET status = $1 
                    WHERE model_id = (
                        SELECT id FROM models_info WHERE name = $2
                    )
                    ORDER BY created_at DESC
                    LIMIT 1
                """, MODEL_STATUS_WAITING, model_name)
                
                logger.info(f"Модель {model_name} помечена для переобучения")
                return web.json_response({
                    "status": "success",
                    "message": f"Модель {model_name} помечена для переобучения"
                })
                
        except Exception as e:
            logger.error(f"Ошибка при пометке модели для переобучения: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def delete_model(self, request: web.Request) -> web.Response:
        """Удаление конкретной версии модели.
        
        Args:
            request: Запрос с параметрами модели
            
        Returns:
            Ответ с результатом операции
        """
        try:
            model_name = request.match_info['model_name']
            version = request.match_info['version']
            
            async with self.db_pool.acquire() as conn:
                # Проверяем, что модель существует
                model_exists = await conn.fetchval("""
                    SELECT 1 FROM models 
                    WHERE model_id = (
                        SELECT id FROM models_info WHERE name = $1
                    ) AND version = $2
                """, model_name, version)
                
                if not model_exists:
                    return web.json_response(
                        {"error": "Модель не найдена"},
                        status=404
                    )
                
                # Удаляем модель
                await conn.execute("""
                    DELETE FROM models 
                    WHERE model_id = (
                        SELECT id FROM models_info WHERE name = $1
                    ) AND version = $2
                """, model_name, version)
                
                logger.info(f"Версия {version} модели {model_name} удалена")
                return web.json_response({
                    "status": "success",
                    "message": f"Версия {version} модели {model_name} удалена"
                })
                
        except Exception as e:
            logger.error(f"Ошибка при удалении модели: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def activate_model(self, request: web.Request) -> web.Response:
        """Активация конкретной версии модели.
        
        Args:
            request: Запрос с параметрами модели
            
        Returns:
            Ответ с результатом операции
        """
        try:
            model_name = request.match_info['model_name']
            version = request.match_info['version']
            
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Проверяем, что модель существует
                    model_exists = await conn.fetchval("""
                        SELECT 1 FROM models 
                        WHERE version = $1 AND model_id = (
                            SELECT id FROM models_info WHERE name = $2
                        )
                    """, version, model_name)
                    
                    if not model_exists:
                        return web.json_response(
                            {"error": "Модель не найдена"},
                            status=404
                        )
                    
                    # Обновляем активную версию в models_info
                    await conn.execute("""
                        UPDATE models_info 
                        SET active_version = $1 
                        WHERE id = (
                            SELECT model_id FROM models 
                            WHERE version = $1 AND model_id = (
                                SELECT id FROM models_info WHERE name = $2
                            )
                        )
                    """, version, model_name)
                    
                    # Обновляем статусы всех версий модели
                    await conn.execute("""
                        UPDATE models 
                        SET status = CASE 
                            WHEN version = $1 THEN $2
                            ELSE $3 
                        END
                        WHERE model_id = (
                            SELECT id FROM models_info WHERE name = $4
                        )
                    """, version, MODEL_STATUS_ACTIVE, MODEL_STATUS_INACTIVE, model_name)
                    
                    logger.info(f"Версия {version} модели {model_name} активирована")
                    return web.json_response({
                        "status": "success",
                        "message": f"Версия {version} модели {model_name} активирована"
                    })
                    
        except Exception as e:
            logger.error(f"Ошибка при активации модели: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def add_model(self, request: web.Request) -> web.Response:
        """Добавление новой модели.
        
        Args:
            request: Запрос с параметрами модели
            
        Returns:
            Ответ с результатом операции
        """
        try:
            data = await request.json()
            
            # Проверка обязательных полей
            required_fields = ['name', 'main_metric', 'hyperparameter_mode']
            if not all(field in data for field in required_fields):
                return web.json_response(
                    {"error": f"Отсутствуют обязательные поля: {required_fields}"},
                    status=400
                )
            
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Проверяем, что модель с таким именем не существует
                    model_exists = await conn.fetchval(
                        "SELECT 1 FROM models_info WHERE name = $1",
                        data['name']
                    )
                    
                    if model_exists:
                        return web.json_response(
                            {"error": f"Модель с именем {data['name']} уже существует"},
                            status=400
                        )
                    
                    # Получаем ID основной метрики
                    metric_id = await conn.fetchval(
                        "SELECT id FROM metrics WHERE name = $1",
                        data['main_metric']
                    )
                    
                    if not metric_id:
                        return web.json_response(
                            {"error": f"Метрика {data['main_metric']} не найдена"},
                            status=404
                        )
                    
                    # Добавляем информацию о модели
                    await conn.execute("""
                        INSERT INTO models_info (
                            name, metric_id, max_stored_versions, 
                            hyperparameter_mode, active_version
                        )
                        VALUES ($1, $2, $3, $4, $5)
                    """, 
                    data['name'],
                    metric_id,
                    data.get('version_history', 3),
                    data['hyperparameter_mode'],
                    '1')
                    
                    # Получаем ID добавленной модели
                    model_info_id = await conn.fetchval(
                        "SELECT id FROM models_info WHERE name = $1",
                        data['name']
                    )
                    
                    # Определяем гиперпараметры в зависимости от режима
                    if data['hyperparameter_mode'] == 'manual':
                        hyperparams = data.get('manual_params', {})
                    else:
                        hyperparams = self._get_default_hyperparams()
                    
                    # Добавляем начальную версию модели
                    await conn.execute("""
                        INSERT INTO models (
                            model_id, version, status, hyperparams
                        )
                        VALUES ($1, $2, $3, $4)
                    """, 
                    model_info_id,
                    '1',
                    MODEL_STATUS_WAITING,
                    json.dumps(hyperparams))
                    
                    # Добавляем дополнительные метрики, если они указаны
                    for metric_name in data.get('additional_metrics', []):
                        metric_id = await conn.fetchval(
                            "SELECT id FROM metrics WHERE name = $1",
                            metric_name
                        )
                        if metric_id:
                            await conn.execute("""
                                INSERT INTO features (model_id, metric_id)
                                VALUES ($1, $2)
                                ON CONFLICT DO NOTHING
                            """, model_info_id, metric_id)
                    
                    logger.info(f"Модель {data['name']} успешно добавлена")
                    return web.json_response({
                        "status": "success",
                        "message": f"Модель {data['name']} успешно добавлена"
                    })
                    
        except Exception as e:
            logger.error(f"Ошибка при добавлении модели: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    def _get_default_hyperparams(self) -> Dict:
        """Возвращает гиперпараметры по умолчанию.
        
        Returns:
            Словарь с гиперпараметрами по умолчанию
        """
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

    async def list_models(self, request: web.Request) -> web.Response:
        """Получение списка версий модели.
        
        Args:
            request: Запрос с параметрами модели
            
        Returns:
            Ответ со списком версий модели
        """
        try:
            model_name = request.match_info['model_name']
            
            async with self.db_pool.acquire() as conn:
                models = await conn.fetch("""
                    SELECT m.version, m.status, m.created_at, 
                           m.hyperparams, mi.active_version
                    FROM models m
                    JOIN models_info mi ON m.model_id = mi.id
                    WHERE mi.name = $1
                    ORDER BY m.created_at DESC
                """, model_name)
                
                if not models:
                    return web.json_response(
                        {"error": "Модель не найдена"},
                        status=404
                    )
                
                result = [dict(model) for model in models]
                return web.json_response(result)
                
        except Exception as e:
            logger.error(f"Ошибка при получении списка моделей: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def show_model(self, request: web.Request) -> web.Response:
        """Получение информации о конкретной версии модели.
        
        Args:
            request: Запрос с параметрами модели
            
        Returns:
            Ответ с информацией о модели
        """
        try:
            model_name = request.match_info['model_name']
            version = request.match_info['version']
            
            async with self.db_pool.acquire() as conn:
                model = await conn.fetchrow("""
                    SELECT m.version, m.status, m.created_at, 
                           m.hyperparams, mi.active_version,
                           (SELECT name FROM metrics WHERE id = mi.metric_id) as main_metric
                    FROM models m
                    JOIN models_info mi ON m.model_id = mi.id
                    WHERE mi.name = $1 AND m.version = $2
                """, model_name, version)
                
                if not model:
                    return web.json_response(
                        {"error": "Модель не найдена"},
                        status=404
                    )
                
                return web.json_response(dict(model))
                
        except Exception as e:
            logger.error(f"Ошибка при получении информации о модели: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def export_model(self, request: web.Request) -> web.Response:
        """Экспорт модели в файл.
        
        Args:
            request: Запрос с параметрами модели
            
        Returns:
            Ответ с файлом модели или сообщением об ошибке
        """
        try:
            model_name = request.match_info['model_name']
            version = request.match_info['version']
            
            model_path = os.path.join(self.model_storage_path, f"{model_name}_{version}.pkl")
            
            if not os.path.exists(model_path):
                return web.json_response(
                    {"error": "Файл модели не найден"},
                    status=404
                )
            
            return web.FileResponse(model_path)
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте модели: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def import_model(self, request: web.Request) -> web.Response:
        """Импорт модели из файла.
        
        Args:
            request: Запрос с файлом модели
            
        Returns:
            Ответ с результатом операции
        """
        try:
            model_name = request.match_info['model_name']
            
            # Проверяем, что запрос содержит файл
            if not request.has_body:
                return web.json_response(
                    {"error": "Запрос не содержит файл модели"},
                    status=400
                )
            
            # Создаем директорию для моделей, если ее нет
            os.makedirs(self.model_storage_path, exist_ok=True)
            
            # Читаем данные файла
            reader = await request.multipart()
            field = await reader.next()
            
            if field.name != 'model_file':
                return web.json_response(
                    {"error": "Неверное имя поля файла"},
                    status=400
                )
            
            # Генерируем имя файла
            version = str(uuid.uuid4())
            model_path = os.path.join(self.model_storage_path, f"{model_name}_{version}.pkl")
            
            # Сохраняем файл
            with open(model_path, 'wb') as f:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
            
            logger.info(f"Модель {model_name} версии {version} успешно импортирована")
            return web.json_response({
                "status": "success",
                "message": f"Модель {model_name} успешно импортирована",
                "version": version
            })
            
        except Exception as e:
            logger.error(f"Ошибка при импорте модели: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    # === Методы для работы с метриками ===

    async def add_metric(self, request: web.Request) -> web.Response:
        """Добавление новой метрики.
        
        Args:
            request: Запрос с параметрами метрики
            
        Returns:
            Ответ с результатом операции
        """
        try:
            data = await request.json()
            
            if not all(field in data for field in ['name', 'query']):
                return web.json_response(
                    {"error": "Необходимо указать name и query"},
                    status=400
                )
            
            async with self.db_pool.acquire() as conn:
                # Добавляем или обновляем метрику
                await conn.execute("""
                    INSERT INTO metrics (name, status, query, step)
                    VALUES ($1, 'active', $2, $3)
                    ON CONFLICT (name) DO UPDATE
                    SET query = EXCLUDED.query, step = EXCLUDED.step
                """, 
                data['name'], 
                data['query'], 
                data.get('step', 60))
                
                # Добавляем периоды исключения, если они указаны
                if 'exclude_periods' in data:
                    metric_id = await conn.fetchval(
                        "SELECT id FROM metrics WHERE name = $1",
                        data['name']
                    )
                    for period in data['exclude_periods']:
                        await conn.execute("""
                            INSERT INTO anomaly_system (
                                start_time, end_time, anomaly_type, 
                                metric_id, description
                            )
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT DO NOTHING
                        """, 
                        period.get('start'),
                        period.get('end'),
                        period.get('anomaly_type', 'maintenance'),
                        metric_id,
                        period.get('description', 'Системный исключаемый период'))
                
                logger.info(f"Метрика {data['name']} добавлена/обновлена")
                return web.json_response({
                    "status": "success",
                    "message": f"Метрика {data['name']} добавлена/обновлена"
                })
                
        except Exception as e:
            logger.error(f"Ошибка при добавлении метрики: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def delete_metric(self, request: web.Request) -> web.Response:
        """Удаление метрики.
        
        Args:
            request: Запрос с именем метрики
            
        Returns:
            Ответ с результатом операции
        """
        try:
            metric_name = request.match_info['metric_name']
            
            async with self.db_pool.acquire() as conn:
                # Проверяем, что метрика существует
                metric_exists = await conn.fetchval(
                    "SELECT 1 FROM metrics WHERE name = $1",
                    metric_name
                )
                
                if not metric_exists:
                    return web.json_response(
                        {"error": "Метрика не найдена"},
                        status=404
                    )
                
                # Удаляем метрику
                await conn.execute(
                    "DELETE FROM metrics WHERE name = $1",
                    metric_name
                )
                
                logger.info(f"Метрика {metric_name} удалена")
                return web.json_response({
                    "status": "success",
                    "message": f"Метрика {metric_name} удалена"
                })
                
        except Exception as e:
            logger.error(f"Ошибка при удалении метрики: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def list_metrics(self, request: web.Request) -> web.Response:
        """Получение списка всех метрик.
        
        Args:
            request: Запрос на получение списка
            
        Returns:
            Ответ со списком метрик
        """
        try:
            async with self.db_pool.acquire() as conn:
                metrics = await conn.fetch("""
                    SELECT id, name, status, query, step 
                    FROM metrics
                    ORDER BY name
                """)
                
                return web.json_response([dict(metric) for metric in metrics])
                
        except Exception as e:
            logger.error(f"Ошибка при получении списка метрик: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def show_metric(self, request: web.Request) -> web.Response:
        """Получение информации о метрике.
        
        Args:
            request: Запрос с именем метрики
            
        Returns:
            Ответ с информацией о метрике
        """
        try:
            metric_name = request.match_info['metric_name']
            
            async with self.db_pool.acquire() as conn:
                metric = await conn.fetchrow("""
                    SELECT id, name, status, query, step 
                    FROM metrics
                    WHERE name = $1
                """, metric_name)
                
                if not metric:
                    return web.json_response(
                        {"error": "Метрика не найдена"},
                        status=404
                    )
                
                return web.json_response(dict(metric))
                
        except Exception as e:
            logger.error(f"Ошибка при получении информации о метрике: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    # === Методы для работы с аномалиями ===

    async def list_anomalies(self, request: web.Request) -> web.Response:
        """Получение списка аномалий.
        
        Args:
            request: Запрос на получение списка
            
        Returns:
            Ответ со списком аномалий
        """
        try:
            async with self.db_pool.acquire() as conn:
                anomalies = await conn.fetch("""
                    SELECT ap.id, ap.timestamp, 
                           mi.name as model_name, m.name as metric_name
                    FROM anomaly_points ap
                    JOIN models m ON ap.model_id = m.id
                    JOIN models_info mi ON m.model_id = mi.id
                    JOIN metrics m ON ap.metric_id = m.id
                    ORDER BY ap.timestamp DESC
                    LIMIT 100
                """)
                
                return web.json_response([dict(anomaly) for anomaly in anomalies])
                
        except Exception as e:
            logger.error(f"Ошибка при получении списка аномалий: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    async def add_system_anomaly(self, request: web.Request) -> web.Response:
        """Добавление системной аномалии.
        
        Args:
            request: Запрос с параметрами аномалии
            
        Returns:
            Ответ с результатом операции
        """
        try:
            data = await request.json()
            
            required_fields = ['start_time', 'anomaly_type', 'metric_name']
            if not all(field in data for field in required_fields):
                return web.json_response(
                    {"error": "Необходимо указать start_time, anomaly_type и metric_name"}, 
                    status=400
                )
            
            async with self.db_pool.acquire() as conn:
                # Получаем ID метрики
                metric_id = await conn.fetchval(
                    "SELECT id FROM metrics WHERE name = $1",
                    data['metric_name']
                )
                
                if not metric_id:
                    return web.json_response(
                        {"error": "Метрика не найдена"},
                        status=404
                    )
                
                # Добавляем аномалию
                await conn.execute("""
                    INSERT INTO anomaly_system (
                        start_time, end_time, anomaly_type, 
                        average_anom_score, metric_id, description
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                data['start_time'],
                data.get('end_time'),
                data['anomaly_type'],
                data.get('average_anom_score', 100),
                metric_id,
                data.get('description', 'Обнаружено системой MAD'))
                
                logger.info(f"Системная аномалия для метрики {data['metric_name']} добавлена")
                return web.json_response({
                    "status": "success",
                    "message": "Системная аномалия добавлена"
                })
                
        except Exception as e:
            logger.error(f"Ошибка при добавлении системной аномалии: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )

    # === Методы диагностики ===

    async def model_status(self, request: web.Request) -> web.Response:
        """Получение статуса модели.
        
        Args:
            request: Запрос с именем модели
            
        Returns:
            Ответ с информацией о статусе модели
        """
        try:
            model_name = request.match_info['model_name']
            
            async with self.db_pool.acquire() as conn:
                status = await conn.fetchrow("""
                    SELECT m.status, m.version, m.created_at,
                           COUNT(ap.id) as anomaly_count,
                           MAX(ap.timestamp) as last_anomaly
                    FROM models m
                    JOIN models_info mi ON m.model_id = mi.id
                    LEFT JOIN anomaly_points ap ON ap.model_id = m.id
                    WHERE mi.name = $1 AND m.status = $2
                    GROUP BY m.status, m.version, m.created_at
                """, model_name, MODEL_STATUS_ACTIVE)
                
                if not status:
                    return web.json_response(
                        {"error": "Активная модель не найдена"},
                        status=404
                    )
                
                return web.json_response(dict(status))
                
        except Exception as e:
            logger.error(f"Ошибка при получении статуса модели: {str(e)}", exc_info=True)
            return web.json_response(
                {"error": "Внутренняя ошибка сервера"},
                status=500
            )


async def main():
    """Основная функция для запуска сервиса."""
    parser = argparse.ArgumentParser(description="API сервис для обнаружения аномалий")
    parser.add_argument('--config', type=str, required=True, help="Путь к конфигурационному файлу")
    parser.add_argument('--host', type=str, default=DEFAULT_HOST, help="Хост для запуска API")
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help="Порт для запуска API")
    args = parser.parse_args()
    
    # Инициализация и запуск сервиса
    service = AnomalyAPIService(args.config)
    try:
        await service.start(args.host, args.port)
    except Exception as e:
        logger.critical(f"Критическая ошибка при работе сервиса: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())