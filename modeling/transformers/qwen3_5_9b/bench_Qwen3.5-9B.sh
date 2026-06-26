#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

# Benchmark script for Qwen3.5-9B
# Compares baseline vs PyPTO backend

set -e

MODEL_ID="Qwen3.5-9B"
MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/Qwen3.5-9B}"
INPUT_FILE="$(dirname $0)/sample_inputs.txt"
OUTPUT_LENGTH=100
SUMMARY_FILE="$(dirname $0)/benchmark_summary.txt"

echo "========================================"
echo "  Qwen3.5-9B Performance Benchmark"
echo "========================================"
echo ""
echo "Model: ${MODEL_ID}"
echo "Model path: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Output length: ${OUTPUT_LENGTH} tokens"
echo ""

# Clean previous results
rm -f ${SUMMARY_FILE}

echo "Phase 1: Running baseline (no PyPTO)..."
python3 $(dirname $0)/ask_Qwen3.5-9B.py \
    --model-path ${MODEL_PATH} \
    --sentence_file ${INPUT_FILE} \
    --output_length ${OUTPUT_LENGTH}

echo ""
echo "Phase 2: Running with PyPTO..."
python3 $(dirname $0)/ask_Qwen3.5-9B.py \
    --model-path ${MODEL_PATH} \
    --sentence_file ${INPUT_FILE} \
    --output_length ${OUTPUT_LENGTH} \
    --use_pypto

echo ""
echo "========================================"
echo "  Benchmark Complete"
echo "========================================"
