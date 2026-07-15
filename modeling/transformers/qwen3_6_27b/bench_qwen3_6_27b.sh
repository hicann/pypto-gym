#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# E2E generation benchmark for Qwen3.6-27B (Ascend 910B3): whole-model greedy generation,
# eager vs PyPTO GatedDeltaRule. Per the benchmarking standard the timed run is 1 warmup +
# 1 measured generate(); this driver runs the eager baseline then the PyPTO path and writes a
# JSON report for each. bench_baseline.json / bench_pypto.json are runtime outputs,
# not committed -- the deployable perf table in README.md is the source of truth.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_ID="qwen3_6_27b"
MODEL_PATH="${MODEL_PATH:-/path/to/models/Qwen3.6-27B}"
DEVICE="${DEVICE:-0}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-100}"
LONG_SEQ="${LONG_SEQ:-256}"
KERNEL_PATH="${KERNEL_PATH:-/tmp/qwen_kernels}"
BASELINE_REPORT="${SCRIPT_DIR}/bench_baseline.json"
PYPTO_REPORT="${SCRIPT_DIR}/bench_pypto.json"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

echo "========================================"
echo "  Qwen3.6-27B E2E Benchmark (single 910B3)"
echo "========================================"
echo "Model: ${MODEL_PATH}  |  device: ${DEVICE}  |  output_length: ${OUTPUT_LENGTH}  |  long_seq: ${LONG_SEQ}"
echo ""

echo ">>> Phase 1: eager baseline (no PyPTO)"
python3 "${SCRIPT_DIR}/bench_qwen3_6_27b.py" \
    --model "${MODEL_ID}" --device "${DEVICE}" --model-path "${MODEL_PATH}" \
    --output_length "${OUTPUT_LENGTH}" --long-seq "${LONG_SEQ}" \
    --report-file "${BASELINE_REPORT}"
echo ""

echo ">>> Phase 2: PyPTO fused GatedDeltaRule"
python3 "${SCRIPT_DIR}/bench_qwen3_6_27b.py" \
    --model "${MODEL_ID}" --device "${DEVICE}" --model-path "${MODEL_PATH}" \
    --output_length "${OUTPUT_LENGTH}" --long-seq "${LONG_SEQ}" \
    --kernel-path "${KERNEL_PATH}" --use_pypto \
    --report-file "${PYPTO_REPORT}"
echo ""

echo "========================================"
echo "  Benchmark Complete"
echo "  baseline -> ${BASELINE_REPORT}"
echo "  pypto    -> ${PYPTO_REPORT}"
echo "========================================"
