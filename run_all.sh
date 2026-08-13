#!/bin/bash
# VIPER — one-shot setup + run script for vast.ai / Kaggle / any Linux GPU box.
set -e

echo "=== [1/3] Installing dependencies ==="
pip install -q -r requirements.txt

echo "=== [2/3] Smoke test (should finish in ~1-2 minutes) ==="
python main.py --smoke-test

echo "=== [2/3] Smoke test passed. Starting FULL run (approx 4-5 hours) ==="
python main.py

echo "=== [3/3] Zipping results ==="
cd "$(dirname "$0")"
zip -r viper_results.zip results/ logs/
echo "Done. See results/*.csv and viper_results.zip"
