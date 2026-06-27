#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -x "$PROJECT_ROOT/mcu_anomaly_env/bin/python" ]]; then
	PYTHON_BIN="$PROJECT_ROOT/mcu_anomaly_env/bin/python"
else
	PYTHON_BIN="python3"
fi

cd "$PROJECT_ROOT"

echo "Fetching dataset..."
"$PYTHON_BIN" python/download_dataset.py

echo "Formatting dataset..."
"$PYTHON_BIN" python/format_data.py

echo "Extracting features..."
"$PYTHON_BIN" python/extract_features.py

echo "Training model..."
"$PYTHON_BIN" python/train_model.py

echo "Converting model to litert format with full integer quantization..."
"$PYTHON_BIN" python/convert_to_litert.py

echo "Evaluating model..."
"$PYTHON_BIN" python/evaluate.py

echo "Converting model to C array..."
"$PROJECT_ROOT/scripts/convert_model_to_carray.sh"

echo "Model build pipeline completed."
