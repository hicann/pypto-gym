#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# msprof kernel-level profiling for Qwen3.6-27B — baseline vs PyPTO backend.
# Collects per-op NPU timing so the fused GatedDeltaRule kernel can be compared
# against the upstream chunk path.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.6-27B bash prof_Qwen3.6-27B.sh
# Optional env: OUTPUT_LENGTH (default 20), DEVICE (default 0).
# Note: a short OUTPUT_LENGTH is used for kernel profiling; this differs from the
# bench/README 性能对比 table (output_length 50) — prof is for per-op traces, not throughput.

set -e

MODEL_ID="Qwen3.6-27B"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Qwen3.6-27B weights directory}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/sample_inputs.txt}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-20}"
DEVICE="${DEVICE:-0}"
ASK="${SCRIPT_DIR}/ask_${MODEL_ID}.py"
PROF_OUTPUT="${SCRIPT_DIR}/msprof_output"
MSPROF="$(which msprof 2>/dev/null || echo 'msprof')"

# sample_inputs.txt is gitignored (*_inputs.txt); regenerate a default if absent.
if [ ! -f "${INPUT_FILE}" ]; then
    printf '请简要介绍一下你自己。\n用三句话解释什么是大语言模型。\n写一首关于秋天的五言绝句。\n' > "${INPUT_FILE}"
fi

BASELINE_CMD="python3 ${ASK} --model-path ${MODEL_PATH} --device ${DEVICE} \
--sentence_file ${INPUT_FILE} --output_length ${OUTPUT_LENGTH}"
PYPTO_CMD="${BASELINE_CMD} --use_pypto"

echo "========================================"
echo "  msprof Kernel-Level Profiling — ${MODEL_ID}"
echo "========================================"

echo ">>> Phase 1: baseline..."
rm -rf "${PROF_OUTPUT}/baseline"
${MSPROF} --output="${PROF_OUTPUT}/baseline" --application="${BASELINE_CMD}" \
    --aic-metrics=ArithmeticUtilization --task-time=on --ai-core=on

echo ">>> Phase 2: PyPTO..."
rm -rf "${PROF_OUTPUT}/pypto"
${MSPROF} --output="${PROF_OUTPUT}/pypto" --application="${PYPTO_CMD}" \
    --aic-metrics=ArithmeticUtilization --task-time=on --ai-core=on

echo ""
echo "Output: ${PROF_OUTPUT}/{baseline,pypto}/device_*/summary/op_statistic_*.csv"
echo "Compare: diff <(sort baseline/.../op_statistic_*.csv) <(sort pypto/.../op_statistic_*.csv)"
