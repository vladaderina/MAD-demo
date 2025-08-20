#!/bin/bash
set -e

echo "Installing linting tools"
pip install black flake8 isort mypy

echo "Running Black check"
black --check --diff .

echo "Running flake8"
flake8 . --count --show-source --statistics --max-line-length=88

echo "Running isort"
isort --check-only --diff .

echo "Running mypy"
mypy --ignore-missing-imports .