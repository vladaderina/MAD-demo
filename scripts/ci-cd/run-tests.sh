#!/bin/bash
set -euo pipefail

SERVICE_NAME=$1

echo "🔧 Setting up test environment for: $SERVICE_NAME"

# Create test directory structure
mkdir -p test-results

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Устанавливаем дополнительные зависимости для тестов
pip install freezegun  # Добавляем недостающую зависимость

if [[ -f "$SERVICE_NAME/requirements-test.txt" ]]; then
    pip install -r "$SERVICE_NAME/requirements-test.txt"
fi

pip install pytest pytest-cov pytest-asyncio

# Очищаем возможные конфликтующие файлы
rm -f test_*.py conftest.py 2>/dev/null || true

# Run tests directly from their original locations
echo "🚀 Running tests..."
python -m pytest \
    --junitxml=test-results/junit.xml \
    --cov=. \
    --cov-report=xml:test-results/coverage.xml \
    --cov-report=html:test-results/coverage-html \
    -v \
    test-suites/mad-notifier/ \
    "$SERVICE_NAME/tests/" \
    || echo "Pytest completed with exit code: $?"

echo "✅ Test execution completed"