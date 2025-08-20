#!/bin/bash
set -e

echo "Installing security tools"
pip install safety bandit

echo "Running safety check"
safety check --full-report

echo "Running bandit"
bandit -r . -ll