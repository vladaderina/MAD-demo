import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

import aiohttp
import asyncpg
from aiohttp import web

# Константы
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8080
DEFAULT_LOG_PATH = '/var/log/anomaly_api.log'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3

# Настройка логгера
logger = logging.getLogger(__name__)


class AnomalyAPIService:
    """API сервис для управления системой обнаружения аномалий."""

    def __init__(self):
        """Инициализация сервиса."""
        self._validate_environment()
        self._setup_logging()
        self.db_pool = None
        self.http_session = None
        self.app = web.Application()
        self._setup_routes()

    def _validate_environment(self) -> None:
        """Проверка обязательных переменных среды."""
        required_vars = [
            'DB_CONN_STRING'
        ]
        
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    def _setup_logging(self) -> None:
        """Настройка системы логирования."""
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        log_path = os.getenv('LOG_PATH', DEFAULT_LOG_PATH)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def _setup_routes(self) -> None:
        """Настройка маршрутов API."""
        self.app.router.add_routes([
            # Модели
            web.get('/models', self.list_models),
            web.post('/models', self.create_model),
            web.get('/models/{model_id}', self.get_model),
            web.put('/models/{model_id}', self.update_model),
            web.delete('/models/{model_id}', self.delete_model),
            web.post('/models/{model_id}/retrain', self.retrain_model),
            web.post('/models/{model_id}/activate', self.activate_model),
            
            # Метрики
            web.get('/metrics', self.list_metrics),
            web.post('/metrics', self.create_metric),
            web.get('/metrics/{metric_id}', self.get_metric),
            web.put('/metrics/{metric_id}', self.update_metric),
            web.delete('/metrics/{metric_id}', self.delete_metric),
            
            # Аномалии
            web.get('/anomalies', self.list_anomalies),
            web.get('/anomalies/{anomaly_id}', self.get_anomaly),
            
            # Диагностика
            web.get('/health', self.health_check),
            web.get('/status', self.system_status),
        ])

    async def _connect_db(self) -> None:
        """Установка соединения с базой данных."""
        db_conn_string = os.environ['DB_CONN_STRING']
        
        try:
            self.db_pool = await asyncpg.create_pool(db_conn_string)
            logger.info("Успешное подключение к базе данных")
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {str(e)}")
            raise

    async def start(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        """Запуск API сервиса."""
        try:
            logger.info("Запуск Anomaly API Service")
            
            await self._connect_db()
            self.http_session = aiohttp.ClientSession()
            
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
            if self.http_session:
                await self.http_session.close()
            logger.info("Сервис остановлен")

    # ===== Методы для работы с моделями =====

    async def list_models(self, request: web.Request) -> web.Response:
        """Получение списка всех моделей."""
        try:
            async with self.db_pool.acquire() as conn:
                models = await conn.fetch(
                    "SELECT id, name, status, active_version, created_at FROM models_info ORDER BY created_at DESC"
                )
                return web.json_response([dict(model) for model in models])
        except Exception as e:
            logger.error(f"Ошибка при получении списка моделей: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def create_model(self, request: web.Request) -> web.Response:
        """Создание новой модели."""
        try:
            data = await request.json()
            
            # Валидация входных данных
            required_fields = ['name', 'metric_id', 'hyperparameter_mode']
            if not all(field in data for field in required_fields):
                return web.json_response(
                    {"error": f"Missing required fields: {required_fields}"},
                    status=400
                )
            
            # Проверка режима гиперпараметров
            if data['hyperparameter_mode'] == 'manual' and 'hyperparams' not in data:
                return web.json_response(
                    {"error": "Hyperparameters are required for manual mode"},
                    status=400
                )
            
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Проверка существования метрики
                    metric_exists = await conn.fetchval(
                        "SELECT 1 FROM metrics WHERE id = $1", 
                        data['metric_id']
                    )
                    if not metric_exists:
                        return web.json_response(
                            {"error": "Metric not found"},
                            status=404
                        )
                    
                    # Создание модели
                    model_id = await conn.fetchval(
                        """
                        INSERT INTO models_info (
                            name, metric_id, hyperparameter_mode, 
                            max_stored_versions, active_version
                        )
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id
                        """,
                        data['name'],
                        data['metric_id'],
                        data['hyperparameter_mode'],
                        data.get('max_stored_versions', 3),
                        '1.0'
                    )
                    
                    # Создание начальной версии модели
                    await conn.execute(
                        """
                        INSERT INTO models (
                            model_id, version, status, hyperparams
                        )
                        VALUES ($1, $2, $3, $4)
                        """,
                        model_id,
                        '1.0',
                        'waiting',
                        json.dumps(data.get('hyperparams', {})))
                    
                    return web.json_response({
                        "status": "success",
                        "model_id": model_id
                    }, status=201)
                    
        except Exception as e:
            return web.json_response(
                {"error": f"Server error: {str(e)}"},
                status=500
            )

    async def get_model(self, request: web.Request) -> web.Response:
        """Получение информации о конкретной модели."""
        try:
            model_id = request.match_info['model_id']
            
            async with self.db_pool.acquire() as conn:
                # Получение основной информации о модели
                model = await conn.fetchrow(
                    """
                    SELECT mi.id, mi.name, mi.status, mi.active_version, 
                           mi.created_at, m.name as metric_name
                    FROM models_info mi
                    JOIN metrics m ON mi.metric_id = m.id
                    WHERE mi.id = $1
                    """,
                    int(model_id)
                )
                
                if not model:
                    return web.json_response(
                        {"error": "Model not found"},
                        status=404
                    )
                
                # Получение списка версий модели
                versions = await conn.fetch(
                    "SELECT id, version, status, created_at FROM models WHERE model_id = $1 ORDER BY created_at DESC",
                    int(model_id)
                )
                
                result = dict(model)
                result['versions'] = [dict(version) for version in versions]
                
                return web.json_response(result)
                
        except ValueError:
            return web.json_response(
                {"error": "Invalid model ID"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка при получении модели: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def update_model(self, request: web.Request) -> web.Response:
        """Обновление информации о модели."""
        try:
            model_id = request.match_info['model_id']
            data = await request.json()
            
            async with self.db_pool.acquire() as conn:
                # Проверка существования модели
                model_exists = await conn.fetchval(
                    "SELECT 1 FROM models_info WHERE id = $1", 
                    int(model_id)
                )
                if not model_exists:
                    return web.json_response(
                        {"error": "Model not found"},
                        status=404
                    )
                
                # Обновление модели
                await conn.execute(
                    """
                    UPDATE models_info
                    SET name = COALESCE($1, name),
                        metric_id = COALESCE($2, metric_id),
                        max_stored_versions = COALESCE($3, max_stored_versions)
                    WHERE id = $4
                    """,
                    data.get('name'),
                    data.get('metric_id'),
                    data.get('max_stored_versions'),
                    int(model_id)
                )
                
                return web.json_response({
                    "status": "success",
                    "message": "Model updated"
                })
                
        except ValueError:
            return web.json_response(
                {"error": "Invalid model ID"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка при обновлении модели: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def delete_model(self, request: web.Request) -> web.Response:
        """Удаление модели и всех её версий."""
        try:
            model_id = request.match_info['model_id']
            
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Удаление всех версий модели
                    await conn.execute(
                        "DELETE FROM models WHERE model_id = $1",
                        int(model_id)
                    )
                    
                    # Удаление самой модели
                    await conn.execute(
                        "DELETE FROM models_info WHERE id = $1",
                        int(model_id)
                    )
                    
                    return web.json_response({
                        "status": "success",
                        "message": "Model deleted"
                    })
                    
        except ValueError:
            return web.json_response(
                {"error": "Invalid model ID"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка при удалении модели: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def retrain_model(self, request: web.Request) -> web.Response:
        """Запрос на переобучение модели."""
        try:
            model_id = request.match_info['model_id']
            trainer_url = os.getenv('TRAINER_SERVICE_URL')
            
            if not trainer_url:
                return web.json_response(
                    {"error": "Trainer service not configured"},
                    status=500
                )
            
            async with self.http_session.post(
                f"{trainer_url}/models/{model_id}/retrain"
            ) as resp:
                if resp.status != 200:
                    return web.json_response(
                        await resp.json(),
                        status=resp.status
                    )
                
                return web.json_response({
                    "status": "success",
                    "message": "Retraining started"
                })
                
        except Exception as e:
            logger.error(f"Ошибка при запросе переобучения: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def activate_model(self, request: web.Request) -> web.Response:
        """Активация конкретной версии модели."""
        try:
            model_id = request.match_info['model_id']
            data = await request.json()
            
            if 'version' not in data:
                return web.json_response(
                    {"error": "Version not specified"},
                    status=400
                )
            
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Проверка существования версии
                    version_exists = await conn.fetchval(
                        """
                        SELECT 1 FROM models 
                        WHERE model_id = $1 AND version = $2
                        """,
                        int(model_id),
                        data['version']
                    )
                    
                    if not version_exists:
                        return web.json_response(
                            {"error": "Version not found"},
                            status=404
                        )
                    
                    # Активация версии
                    await conn.execute(
                        """
                        UPDATE models_info
                        SET active_version = $1
                        WHERE id = $2
                        """,
                        data['version'],
                        int(model_id))
                    
                    # Обновление статусов
                    await conn.execute(
                        """
                        UPDATE models
                        SET status = CASE
                            WHEN version = $1 THEN 'active'
                            ELSE 'inactive'
                        END
                        WHERE model_id = $2
                        """,
                        data['version'],
                        int(model_id))
                    
                    return web.json_response({
                        "status": "success",
                        "message": "Model version activated"
                    })
                    
        except ValueError:
            return web.json_response(
                {"error": "Invalid model ID"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка при активации модели: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    # ===== Методы для работы с метриками =====

    async def list_metrics(self, request: web.Request) -> web.Response:
        """Получение списка всех метрик."""
        try:
            async with self.db_pool.acquire() as conn:
                metrics = await conn.fetch(
                    "SELECT id, name, status, query, step, created_at FROM metrics ORDER BY name"
                )
                return web.json_response([dict(metric) for metric in metrics])
        except Exception as e:
            logger.error(f"Ошибка при получении списка метрик: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def create_metric(self, request: web.Request) -> web.Response:
        """Создание новой метрики."""
        try:
            data = await request.json()
            
            # Валидация входных данных
            required_fields = ['name', 'query']
            if not all(field in data for field in required_fields):
                return web.json_response(
                    {"error": f"Missing required fields: {required_fields}"},
                    status=400
                )
            
            async with self.db_pool.acquire() as conn:
                metric_id = await conn.fetchval(
                    """
                    INSERT INTO metrics (name, query, step, status)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    data['name'],
                    data['query'],
                    data.get('step', 60),
                    data.get('status', 'active')
                )
                
                return web.json_response({
                    "status": "success",
                    "metric_id": metric_id
                }, status=201)
                
        except asyncpg.UniqueViolationError:
            return web.json_response(
                {"error": "Metric with this name already exists"},
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка при создании метрики: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def get_metric(self, request: web.Request) -> web.Response:
        """Получение информации о конкретной метрике."""
        try:
            metric_id = request.match_info['metric_id']
            
            async with self.db_pool.acquire() as conn:
                metric = await conn.fetchrow(
                    "SELECT id, name, status, query, step, created_at FROM metrics WHERE id = $1",
                    int(metric_id)
                
                if not metric:
                    return web.json_response(
                        {"error": "Metric not found"},
                        status=404
                    )
                
                return web.json_response(dict(metric))
                
        except ValueError:
            return web.json_response(
                {"error": "Invalid metric ID"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка при получении метрики: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def update_metric(self, request: web.Request) -> web.Response:
        """Обновление информации о метрике."""
        try:
            metric_id = request.match_info['metric_id']
            data = await request.json()
            
            async with self.db_pool.acquire() as conn:
                # Проверка существования метрики
                metric_exists = await conn.fetchval(
                    "SELECT 1 FROM metrics WHERE id = $1", 
                    int(metric_id)
                )
                if not metric_exists:
                    return web.json_response(
                        {"error": "Metric not found"},
                        status=404
                    )
                
                # Обновление метрики
                await conn.execute(
                    """
                    UPDATE metrics
                    SET name = COALESCE($1, name),
                        query = COALESCE($2, query),
                        step = COALESCE($3, step),
                        status = COALESCE($4, status)
                    WHERE id = $5
                    """,
                    data.get('name'),
                    data.get('query'),
                    data.get('step'),
                    data.get('status'),
                    int(metric_id)
                )
                
                return web.json_response({
                    "status": "success",
                    "message": "Metric updated"
                })
                
        except ValueError:
            return web.json_response(
                {"error": "Invalid metric ID"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка при обновлении метрики: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def delete_metric(self, request: web.Request) -> web.Response:
        """Удаление метрики."""
        try:
            metric_id = request.match_info['metric_id']
            
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Проверка использования метрики
                    used_in_models = await conn.fetchval(
                        "SELECT 1 FROM models_info WHERE metric_id = $1 LIMIT 1",
                        int(metric_id)
                    )
                    
                    if used_in_models:
                        return web.json_response(
                            {"error": "Metric is used in models and cannot be deleted"},
                            status=400
                        )
                    
                    # Удаление метрики
                    await conn.execute(
                        "DELETE FROM metrics WHERE id = $1",
                        int(metric_id)
                    )
                    
                    return web.json_response({
                        "status": "success",
                        "message": "Metric deleted"
                    })
                    
        except ValueError:
            return web.json_response(
                {"error": "Invalid metric ID"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка при удалении метрики: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    # ===== Методы для работы с аномалиями =====

    async def list_anomalies(self, request: web.Request) -> web.Response:
        """Получение списка аномалий."""
        try:
            limit = int(request.query.get('limit', 100))
            offset = int(request.query.get('offset', 0))
            
            async with self.db_pool.acquire() as conn:
                anomalies = await conn.fetch(
                    """
                    SELECT a.id, a.start_time, a.end_time, a.anomaly_type,
                           m.name as metric_name, a.description,
                           a.average_anom_score
                    FROM anomaly_system a
                    JOIN metrics m ON a.metric_id = m.id
                    ORDER BY a.start_time DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset
                )
                
                total = await conn.fetchval("SELECT COUNT(*) FROM anomaly_system")
                
                return web.json_response({
                    "items": [dict(anomaly) for anomaly in anomalies],
                    "total": total
                })
                
        except Exception as e:
            logger.error(f"Ошибка при получении списка аномалий: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def get_anomaly(self, request: web.Request) -> web.Response:
        """Получение информации о конкретной аномалии."""
        try:
            anomaly_id = request.match_info['anomaly_id']
            
            async with self.db_pool.acquire() as conn:
                anomaly = await conn.fetchrow(
                    """
                    SELECT a.id, a.start_time, a.end_time, a.anomaly_type,
                           m.name as metric_name, a.description,
                           a.average_anom_score
                    FROM anomaly_system a
                    JOIN metrics m ON a.metric_id = m.id
                    WHERE a.id = $1
                    """,
                    int(anomaly_id)
                )
                
                if not anomaly:
                    return web.json_response(
                        {"error": "Anomaly not found"},
                        status=404
                    )
                
                return web.json_response(dict(anomaly))
                
        except ValueError:
            return web.json_response(
                {"error": "Invalid anomaly ID"}, 
                status=400
            )
        except Exception as e:
            logger.error(f"Ошибка при получении аномалии: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    async def create_system_anomaly(self, request: web.Request) -> web.Response:
        """Создание системной аномалии."""
        try:
            data = await request.json()
            
            # Валидация входных данных
            required_fields = ['start_time', 'metric_id', 'anomaly_type']
            if not all(field in data for field in required_fields):
                return web.json_response(
                    {"error": f"Missing required fields: {required_fields}"},
                    status=400
                )
            
            async with self.db_pool.acquire() as conn:
                # Проверка существования метрики
                metric_exists = await conn.fetchval(
                    "SELECT 1 FROM metrics WHERE id = $1", 
                    data['metric_id']
                )
                if not metric_exists:
                    return web.json_response(
                        {"error": "Metric not found"},
                        status=404
                    )
                
                # Создание аномалии
                anomaly_id = await conn.fetchval(
                    """
                    INSERT INTO anomaly_system (
                        start_time, end_time, anomaly_type,
                        metric_id, description, average_anom_score
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    data['start_time'],
                    data.get('end_time'),
                    data['anomaly_type'],
                    data['metric_id'],
                    data.get('description', 'Manual addition'),
                    data.get('average_anom_score', 100)
                )
                
                return web.json_response({
                    "status": "success",
                    "anomaly_id": anomaly_id
                }, status=201)
                
        except Exception as e:
            logger.error(f"Ошибка при создании аномалии: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )

    # ===== Методы диагностики =====

    async def health_check(self, request: web.Request) -> web.Response:
        """Проверка здоровья сервиса."""
        status = {'status': 'ok'}
        
        # Проверка БД
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
                status['database'] = 'ok'
        except Exception as e:
            status['database'] = 'error'
            status['database_error'] = str(e)
        
        # Проверка зависимых сервисов
        services = {
            'trainer': os.getenv('TRAINER_SERVICE_URL')
        }
        
        for name, url in services.items():
            if not url:
                status[f"{name}_service"] = "not configured"
                continue
            
            try:
                async with self.http_session.get(f"{url}/health") as resp:
                    status[f"{name}_service"] = "ok" if resp.status == 200 else "error"
            except Exception as e:
                status[f"{name}_service"] = "error"
                status[f"{name}_error"] = str(e)
        
        return web.json_response(status)

    async def system_status(self, request: web.Request) -> web.Response:
        """Получение общего статуса системы."""
        try:
            async with self.db_pool.acquire() as conn:
                stats = {
                    'models_count': await conn.fetchval("SELECT COUNT(*) FROM models_info"),
                    'metrics_count': await conn.fetchval("SELECT COUNT(*) FROM metrics"),
                    'active_anomalies': await conn.fetchval(
                        "SELECT COUNT(*) FROM anomaly_system WHERE end_time IS NULL"
                    ),
                    'last_anomaly': await conn.fetchval(
                        "SELECT MAX(start_time) FROM anomaly_system"
                    )
                }
                
                return web.json_response(stats)
                
        except Exception as e:
            logger.error(f"Ошибка при получении статуса системы: {str(e)}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )


async def main():
    """Основная функция для запуска сервиса."""
    parser = argparse.ArgumentParser(description="API сервис для обнаружения аномалий")
    parser.add_argument('--host', type=str, default=DEFAULT_HOST, help="Хост для запуска API")
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help="Порт для запуска API")
    args = parser.parse_args()
    
    service = AnomalyAPIService()
    try:
        await service.start(args.host, args.port)
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())