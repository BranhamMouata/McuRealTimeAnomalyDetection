#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root from script location.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
TFLM_DIR="${REPO_ROOT}/tflite-micro"
OUT_DIR="${REPO_ROOT}/lib/static_lib"

if [[ ! -f "${TFLM_DIR}/tensorflow/lite/micro/tools/make/Makefile" ]]; then
	echo "Error: tflite-micro directory not found or incomplete at ${TFLM_DIR}" >&2
	exit 1
fi

cd "${TFLM_DIR}"

make -j8 -f tensorflow/lite/micro/tools/make/Makefile \
TARGET=cortex_m_generic \
TARGET_ARCH=cortex-m4 \
OPTIMIZED_KERNEL_DIR=cmsis_nn \
LDFLAGS="--specs=nosys.specs" \
microlite

LIB_DIR="${TFLM_DIR}/gen/cortex_m_generic_cortex-m4_default_cmsis_nn_gcc/lib"
if [[ ! -d "${LIB_DIR}" ]]; then
	echo "Error: expected library output directory not found: ${LIB_DIR}" >&2
	exit 1
fi

mkdir -p "${OUT_DIR}"
cp "${LIB_DIR}"/* "${OUT_DIR}/"
echo "Library built and copied to ${OUT_DIR}"