#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Two-phase benchmark for Kimi-Linear-48B-A3B-Instruct on Ascend NPU.
#   Phase 1: baseline (torch chunked KDA via kimi_fla_compat)
#   Phase 2: pypto    (--use_pypto; PyPTO fused KDA chunk kernel)
# Parses both JSON reports and prints a comparison table.
#
# NOTE(multi-NPU coverage): this is SINGLE-PROCESS (device_map shard). PyPTO binds
# to one NPU/process, so Phase 2 accelerates only the bound NPU's KDA layers
# (partial, ~6/20 with 4 NPUs) — the speedup here UNDER-states the kernel and is
# NOT the full-model 20/20 number. The ask script prints a [PyPTO coverage] warning.
# Note (future work): for full 20/20 e2e, use a multi-process (one-process-per-NPU) bench.

set -e

MODEL_ID="Kimi-Linear-48B-A3B-Instruct"
MODEL_PATH="${MODEL_PATH:-/data/models/Kimi-Linear-48B-A3B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASK="$SCRIPT_DIR/ask_Kimi-Linear-48B-A3B.py"
INPUT_FILE="$SCRIPT_DIR/sample_inputs.txt"
NPUS="${NPUS:-4}"
OUTLEN="${OUTLEN:-100}"
SUMMARY_DIR="${SUMMARY_DIR:-$SCRIPT_DIR}"
mkdir -p "$SUMMARY_DIR"

echo "========================================"
echo "  ${MODEL_ID} Performance Benchmark"
echo "========================================"
echo "Model path: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Output length: ${OUTLEN} tokens, NPUs: ${NPUS}"
echo ""

echo "=== Phase 1: baseline (no PyPTO) ==="
python3 "$ASK" --model-path "$MODEL_PATH" \
    --sentence_file "$INPUT_FILE" \
    --output_length "$OUTLEN" --num_npus "$NPUS" \
    --report-file "$SUMMARY_DIR/baseline_report.json"

echo ""
echo "=== Phase 2: pypto (--use_pypto) ==="
python3 "$ASK" --model-path "$MODEL_PATH" \
    --sentence_file "$INPUT_FILE" \
    --output_length "$OUTLEN" --num_npus "$NPUS" --use_pypto \
    --report-file "$SUMMARY_DIR/pypto_report.json"

echo ""
echo "=== Comparison ==="
python3 - "$SUMMARY_DIR/baseline_report.json" "$SUMMARY_DIR/pypto_report.json" <<'PY'
import json, sys
b = json.load(open(sys.argv[1]))
p = json.load(open(sys.argv[2]))
def peak(r): return max(r["peak_hbm_mb"].values())
print(f"{'metric':<22}{'baseline':>14}{'pypto':>14}")
print(f"{'elapsed (s)':<22}{b['elapsed_s']:>14.3f}{p['elapsed_s']:>14.3f}")
print(f"{'throughput (tok/s)':<22}{b['throughput_tok_s']:>14.2f}{p['throughput_tok_s']:>14.2f}")
print(f"{'peak HBM/card (MB)':<22}{peak(b):>14.1f}{peak(p):>14.1f}")
PY

echo ""
echo "========================================"
echo "  Benchmark Complete"
echo "========================================"
