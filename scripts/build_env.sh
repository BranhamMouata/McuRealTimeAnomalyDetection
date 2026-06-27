#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -x "$PROJECT_ROOT/mcu_anomaly_env/bin/python" ]]; then
	echo "Environment already exists. Skipping environment creation."
	exit 0
fi

PYTHON_BIN="python3"
cd "$PROJECT_ROOT"

echo "Creating virtual environment..."
"$PYTHON_BIN" -m venv mcu_anomaly_env

PYTHON_BIN="$PROJECT_ROOT/mcu_anomaly_env/bin/python"
echo "Installing dependencies..."
"$PYTHON_BIN" -m pip install -r requirements.txt


echo "Environment setup completed."
