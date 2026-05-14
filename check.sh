#!/usr/bin/env bash
set -euo pipefail

echo "=== ruff ==="
python -m ruff check src scripts tests 2>&1 | tail -10

echo ""
echo "=== mypy ==="
python -m mypy src scripts tests 2>&1 | tail -10

echo ""
echo "=== bandit ==="
python -m bandit -r src scripts -q 2>&1 | tail -10

echo ""
echo "=== pyright ==="
pyright src scripts tests 2>&1 | tail -20
