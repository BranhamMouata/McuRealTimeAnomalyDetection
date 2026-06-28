#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -x "$PROJECT_ROOT/mcu_anomaly_env/bin/python" ]]; then
	PYTHON_BIN="$PROJECT_ROOT/mcu_anomaly_env/bin/python"
else
	PYTHON_BIN="python3"
fi
cd $PROJECT_ROOT
echo "Exporting error threshold to header file..."
"$PYTHON_BIN" python/convert_error_threshold_to_c.py
echo "Standard scaler created in ${PROJECT_ROOT}/include/error_threshold.h"
