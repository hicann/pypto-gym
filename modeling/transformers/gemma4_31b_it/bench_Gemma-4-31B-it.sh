#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

# Benchmark script for Gemma-4-31B-it
# Compares baseline vs PyPTO backend

set -e

MODEL_ID="Gemma-4-31B-it"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/gemma-4-31b-it}"
INPUT_FILE="$(dirname $0)/sample_inputs.txt"
OUTPUT_LENGTH=100

echo "========================================"
echo "  Gemma-4-31B-it Performance Benchmark"
echo "========================================"
echo ""
echo "Model: ${MODEL_ID}"
echo "Model path: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Output length: ${OUTPUT_LENGTH} tokens"
echo ""

echo "Phase 1: Running baseline (no PyPTO)..."
python3 $(dirname $0)/ask_Gemma-4-31B-it.py \
    --model-path ${MODEL_PATH} \
    --sentence_file ${INPUT_FILE} \
    --output_length ${OUTPUT_LENGTH} \
    --report_file $(dirname $0)/bench_baseline.json

echo ""
echo "Phase 2: Running with PyPTO..."
python3 $(dirname $0)/ask_Gemma-4-31B-it.py \
    --model-path ${MODEL_PATH} \
    --sentence_file ${INPUT_FILE} \
    --output_length ${OUTPUT_LENGTH} \
    --use_pypto \
    --report_file $(dirname $0)/bench_pypto.json

echo ""
echo "========================================"
echo "  Benchmark Complete"
echo "========================================"
echo "Results: bench_baseline.json, bench_pypto.json"
