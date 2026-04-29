#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0

# Benchmark script for Qwen3-1.7B model
# Compares PyTorch baseline vs PyPTO fused kernels

set -e

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/gitCode/cann/network/Qwen3-1.7B}"
DEVICE="${DEVICE:-0}"
PROMPT="${PROMPT:-你好}"

echo "========================================"
echo "  Qwen3-1.7B Inference Benchmark"
echo "========================================"
echo ""
echo "Model: ${MODEL_PATH}"
echo "Device: ${DEVICE}"
echo "Prompt: ${PROMPT}"
echo ""

echo "========================================"
echo "  PyTorch Eager Baseline"
echo "========================================"
python infer.py \
    --model-path ${MODEL_PATH} \
    --device ${DEVICE} \
    --prompt "${PROMPT}"

echo ""
echo "========================================"
echo "  PyPTO Optimized"
echo "========================================"
python infer.py \
    --model-path ${MODEL_PATH} \
    --device ${DEVICE} \
    --prompt "${PROMPT}" \
    --use-pto

echo ""
echo "========================================"
echo "  Done"
echo "========================================"
