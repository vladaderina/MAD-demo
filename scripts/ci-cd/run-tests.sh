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

pip install pytest pytest-cov pytest-asyncio pytest-xdist

# Prepare test files
if [[ -d "$SERVICE_NAME/tests" ]]; then
    cp -r "$SERVICE_NAME/tests" ./
fi

if [[ -d "test-suites/mad-notifier" ]]; then
    cp -r test-suites/mad-notifier/* ./
fi

# Run tests
echo "🚀 Running tests..."
python -m pytest \
    --junitxml=test-results/junit.xml \
    --cov=. \
    --cov-report=xml:test-results/coverage.xml \
    --cov-report=html:test-results/coverage-html \
    -n auto \
    -v \
    tests/ test_*.py *test*.py

echo "✅ Tests completed successfully"