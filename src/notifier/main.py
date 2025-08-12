import asyncio
import argparse
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Dict, Optional, List

from aiohttp import web
import aiohttp
from jinja2 import Environment

# Константы
DEFAULT_PORT = 8000  # Порт по умолчанию
DEFAULT_LOG_PATH = 'anomaly_notifier.log'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3

# Настройка логгера
logger = logging.getLogger(__name__)

class AnomalyNotifier:
    """Сервис уведомлений об аномалиях с HTTP API."""
    
    def __init__(self):
        """Инициализация сервиса."""
        self.jinja_env = Environment()
        self.jinja_env.filters['datetimeformat'] = self._format_datetime
        
        self.config = self._load_config_from_env()
        self._setup_logging()
        self._init_templates()
        
        # Инициализация HTTP сервера
        self.app = web.Application()
        self.app.add_routes([
            web.post('/api/v1/anomalies', self.handle_anomaly),
            web.get('/health', self.handle_healthcheck)
        ])
        
        self.http_session = None  # Для отправки в Telegram
        self.runner = None
        self.site = None

    def _init_templates(self):
        """Инициализация шаблонов сообщений."""
        self.templates = {
            'start': self.jinja_env.from_string("""
                🚨 *Обнаружена аномалия* 🚨

                *Тип:* `{{ anomaly_type }}`
                *Метрика:* `{{ metric_name }}`
                *Время начала:* `{{ start_time|datetimeformat }}`
                *Описание:* {{ description }}

                {% if average_anom_score %}*Средний уровень аномальности:* `{{ "%.2f"|format(average_anom_score) }}`{% endif %}
                """.strip()),
            'end': self.jinja_env.from_string("""
                ✅ *Аномалия завершена* ✅

                *Тип:* `{{ anomaly_type }}`
                *Метрика:* `{{ metric_name }}`
                *Начало:* `{{ start_time|datetimeformat }}`
                *Конец:* `{{ end_time|datetimeformat }}`
                *Длительность:* {{ duration }}

                {% if average_anom_score %}*Средний уровень аномальности:* `{{ "%.2f"|format(average_anom_score) }}`{% endif %}
                """.strip())
        }

    async def handle_healthcheck(self, request):
        """Обработчик healthcheck запросов."""
        return web.json_response({"status": "ok"})

    async def handle_anomaly(self, request):
        """Обработчик HTTP запросов с данными об аномалиях."""
        try:
            data = await request.json()
            logger.info(f"Получены данные об аномалии: {data}")
            
            # Валидация обязательных полей
            required_fields = ['action', 'id', 'anomaly_type', 'metric_name']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Отсутствует обязательное поле: {field}")
            
            await self._process_anomaly(data)
            return web.json_response({"status": "success"})
        except json.JSONDecodeError:
            logger.error("Невалидный JSON в запросе")
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except ValueError as e:
            logger.error(f"Ошибка валидации: {str(e)}")
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {str(e)}")
            return web.json_response({"error": "Internal server error"}, status=500)

    def _load_config_from_env(self) -> Dict:
        """Загрузка конфигурации из переменных окружения."""
        config = {
            'system': {
                'debug': os.getenv('DEBUG', 'false').lower() == 'true',
                'port': int(os.getenv('PORT', DEFAULT_PORT)),
                'log_path': os.getenv('LOG_PATH', DEFAULT_LOG_PATH)
            },
            'mad-notifier': {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID')
            }
        }
        
        self._validate_config(config)
        return config

    def _validate_config(self, config: Dict) -> None:
        """Валидация обязательных параметров конфигурации."""
        if not config['mad-notifier'].get('bot_token'):
            raise ValueError("Необходимо указать TELEGRAM_BOT_TOKEN в переменных окружения")
        
        if not config['mad-notifier'].get('chat_id'):
            raise ValueError("Необходимо указать TELEGRAM_CHAT_ID в переменных окружения")

    def _setup_logging(self) -> None:
        """Настройка системы логирования."""
        log_level = logging.DEBUG if self.config['system'].get('debug', False) else logging.INFO
        logger.setLevel(log_level)
        
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        log_path = self.config['system'].get('log_path', DEFAULT_LOG_PATH)
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

    def _format_datetime(self, value) -> str:
        """Форматирование datetime для шаблонов."""
        if not value:
            return "не указано"
        
        # Преобразуем строку в datetime при необходимости
        if isinstance(value, str):
            try:
                if value.endswith('Z'):
                    value = value[:-1] + '+00:00'
                value = datetime.fromisoformat(value)
            except ValueError as e:
                logger.error(f"Ошибка преобразования строки в datetime: {str(e)}")
                return "неверный формат времени"
        
        if not isinstance(value, datetime):
            return "неверный тип времени"
        
        try:
            # Если время без временной зоны, считаем что это UTC
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            
            dt_str = value.astimezone(timezone.utc).strftime('%Y\-%m\-%d %H:%M:%S UTC')
            return dt_str.replace(".", "\.").replace("-", "\-").replace("(", "\(").replace(")", "\)")
        except Exception as e:
            logger.error(f"Ошибка форматирования времени: {str(e)}")
            return "ошибка формата времени"

    def _prepare_telegram_text(self, text: str) -> str:
        """Подготовка текста для Telegram с экранированием и проверкой."""
        if not text or len(text.strip()) == 0:
            logger.error("Пустое сообщение")
            return ""

        markdown_chars = '_*[]()~`>#+-=|{}.!'
        escaped_text = ''.join(f'\\{char}' if char in markdown_chars else char for char in text)
        
        if len(escaped_text) > 4096:
            logger.error(f"Сообщение слишком длинное ({len(escaped_text)} символов)")
            return escaped_text[:4000] + "... [сообщение сокращено]"
        
        return escaped_text

    async def _send_telegram_message(self, text: str) -> bool:
        """Отправка сообщений в Telegram."""
        try:
            bot_token = self.config['mad-notifier'].get('bot_token')
            chat_id = self.config['mad-notifier'].get('chat_id')
            
            if not bot_token or not chat_id:
                logger.error("Не указан TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID в переменных окружения")
                return False

            safe_text = self._prepare_telegram_text(text)
            if not safe_text:
                return False

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            params = {
                'chat_id': chat_id,
                'text': safe_text,
                'parse_mode': 'MarkdownV2',
                'disable_web_page_preview': True
            }

            async with self.http_session.post(url, json=params) as resp:
                response = await resp.json()
                
                if resp.status == 200 and response.get('ok'):
                    logger.debug(f"Сообщение успешно отправлено в чат {chat_id}")
                    return True
                
                error_msg = response.get('description', 'Unknown error')
                logger.error(f"Ошибка Telegram API: {error_msg}")
                return False

        except aiohttp.ClientError as e:
            logger.error(f"Сетевая ошибка при отправке в Telegram: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при отправке сообщения: {str(e)}")
            return False

    async def _process_anomaly(self, anomaly_data: Dict) -> None:
        """Обработка данных об аномалии."""
        try:
            # Преобразуем строковые временные метки в datetime объекты
            if isinstance(anomaly_data.get('start_time'), str):
                anomaly_data['start_time'] = datetime.fromisoformat(
                    anomaly_data['start_time'].replace('Z', '+00:00')
                )
            
            if isinstance(anomaly_data.get('end_time'), str):
                anomaly_data['end_time'] = datetime.fromisoformat(
                    anomaly_data['end_time'].replace('Z', '+00:00')
                )
            
            template_type = 'start' if anomaly_data['action'] == 'start' else 'end'
            message = self.templates[template_type].render(**anomaly_data)
            
            if await self._send_telegram_message(message):
                action = "начала" if template_type == 'start' else "завершения"
                logger.info(f"Уведомление о {action} аномалии {anomaly_data['id']} отправлено")
                
        except Exception as e:
            logger.error(f"Ошибка обработки аномалии: {str(e)}")

    async def start_server(self, port: int = None):
        """Запуск HTTP сервера."""
        port = port or self.config['system'].get('port', DEFAULT_PORT)
        
        # Инициализация клиента для Telegram
        self.http_session = aiohttp.ClientSession()
        
        # Запуск сервера
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, '0.0.0.0', port)
        await self.site.start()
        
        logger.info(f"Сервис уведомлений запущен на порту {port}")

    async def stop_server(self):
        """Остановка HTTP сервера."""
        if self.http_session:
            await self.http_session.close()
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Сервис уведомлений остановлен")


async def main():
    """Основная функция для запуска сервиса."""
    parser = argparse.ArgumentParser(description='Anomaly Notifier Service')
    parser.add_argument('--port', type=int, help='Port to listen on')
    args = parser.parse_args()

    notifier = AnomalyNotifier()
    
    try:
        await notifier.start_server(port=args.port)
        while True:
            await asyncio.sleep(3600)  # Бесконечный цикл
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания")
    except Exception as e:
        logger.error(f"Ошибка при работе сервиса: {str(e)}")
    finally:
        await notifier.stop_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Сервис завершил работу")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")