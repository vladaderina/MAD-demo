#!/bin/bash
set -euo pipefail

SERVICE_NAME=$1

echo "🔧 Setting up test environment for: $SERVICE_NAME"

# Create test directory structure
mkdir -p test-results

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

if [[ -f "$SERVICE_NAME/requirements-test.txt" ]]; then
    pip install -r "$SERVICE_NAME/requirements-test.txt"
fi

# Только необходимые зависимости для тестов
pip install pytest pytest-cov pytest-asyncio

# Prepare test files
if [[ -d "$SERVICE_NAME/tests" ]]; then
    cp -r "$SERVICE_NAME/tests" ./
fi

if [[ -d "test-suites/mad-notifier" ]]; then
    cp -r test-suites/mad-notifier/* ./
fi

# Run tests sequentially
echo "🚀 Running tests..."
python -m pytest \
    --junitxml=test-results/junit.xml \
    --cov=. \
    --cov-report=xml:test-results/coverage.xml \
    --cov-report=html:test-results/coverage-html \
    -v \
    test-suites/$SERVICE_NAME test_*.py *test*.py

echo "✅ Tests completed successfully"