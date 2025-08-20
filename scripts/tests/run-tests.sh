#!/bin/bash
set -e

SERVICE_NAME=$1

echo "Running tests for service: $SERVICE_NAME"

# Устанавливаем зависимости
pip install -r requirements.txt
pip install -r $SERVICE_NAME/requirements-test.txt
pip install pytest pytest-cov pytest-asyncio

# Копируем тесты
cp -r $SERVICE_NAME/* tests/ 2>/dev/null || echo "No specific tests found, using default"

echo "Test structure:"
find tests/ -type f -name "*.py" | head -10

# Запускаем тесты
python -m pytest tests/ -v --cov=. --cov-report=xml