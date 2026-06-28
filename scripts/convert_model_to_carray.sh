#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root from script location.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
MODEL_DIR="${REPO_ROOT}/model"

if [[ ! -f "${MODEL_DIR}/cwru_autoencoder_quant8.tflite" ]]; then
	echo "Error: model file not found at ${MODEL_DIR}/cwru_autoencoder_quant8.tflite" >&2
	exit 1
fi

cd "${MODEL_DIR}"

FILE_NAME="cwru_autoencoder_quant8.h"
xxd -i "cwru_autoencoder_quant8.tflite" > "${FILE_NAME}" 

# Ensure fixed-width integer types used in the transformed array declaration are available.
sed -i '1i #include <cstdint>' "${FILE_NAME}"

if [[ ! -f "${MODEL_DIR}/${FILE_NAME}" ]]; then
	echo "Error: Conversion failed" >&2
	exit 1
fi

#Read the model cc model file and add a alignas(4) to the model array definition
sed -i 's/unsigned char/alignas(4) inline constexpr uint8_t/' "${FILE_NAME}"
cp "${FILE_NAME}" "../include/"
echo "Model converted to C array at ${MODEL_DIR}/${FILE_NAME}"