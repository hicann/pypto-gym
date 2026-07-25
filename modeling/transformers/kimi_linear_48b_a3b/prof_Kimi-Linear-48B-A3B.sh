#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Kernel-level profiling for Kimi-Linear-48B-A3B using msprof
# Compares baseline vs PyPTO at the operator level

set -e

MSPROF="$(which msprof 2>/dev/null || echo 'msprof')"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/models/Kimi-Linear-48B-A3B-Instruct}"
INPUT_FILE="${SCRIPT_DIR}/sample_inputs.txt"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-20}"
NUM_NPUS="${NUM_NPUS:-4}"
PROF_OUTPUT="${SCRIPT_DIR}/msprof_output"

ASK_SCRIPT="${SCRIPT_DIR}/ask_Kimi-Linear-48B-A3B.py"
BASELINE_CMD="python3 ${ASK_SCRIPT} --model-path ${MODEL_PATH} --sentence_file ${INPUT_FILE} --output_length ${OUTPUT_LENGTH} --num_npus ${NUM_NPUS}"
PYPTO_CMD="python3 ${ASK_SCRIPT} --model-path ${MODEL_PATH} --sentence_file ${INPUT_FILE} --output_length ${OUTPUT_LENGTH} --num_npus ${NUM_NPUS} --use_pypto"

echo "========================================"
echo "  msprof Kernel-Level Profiling"
echo "  Model: Kimi-Linear-48B-A3B"
echo "========================================"
echo ""
echo "Output tokens per run: ${OUTPUT_LENGTH} (keep small for profiling)"
echo "Output dir: ${PROF_OUTPUT}"
echo ""

# ---- Phase 1: Baseline ----
echo ">>> Phase 1: Profiling Baseline..."
rm -rf "${PROF_OUTPUT}/baseline"
${MSPROF} \
    --output="${PROF_OUTPUT}/baseline" \
    --application="${BASELINE_CMD}" \
    --aic-metrics=ArithmeticUtilization \
    --task-time=on \
    --ai-core=on

echo ""

# ---- Phase 2: PyPTO ----
echo ">>> Phase 2: Profiling PyPTO..."
rm -rf "${PROF_OUTPUT}/pypto"
${MSPROF} \
    --output="${PROF_OUTPUT}/pypto" \
    --application="${PYPTO_CMD}" \
    --aic-metrics=ArithmeticUtilization \
    --task-time=on \
    --ai-core=on

echo ""
echo "========================================"
echo "  Profiling Complete"
echo "========================================"
echo ""
echo "Output directories:"
echo "  Baseline: ${PROF_OUTPUT}/baseline"
echo "  PyPTO:    ${PROF_OUTPUT}/pypto"
echo ""
echo "Key files in each directory:"
echo "  device_*/summary/op_statistic_*.csv   — op timing summary"
echo "  device_*/timeline/*.json              — Chrome trace timeline"
echo "  device_*/aicore_metrics_*.csv         — AI Core utilization"
echo ""
echo "Compare: diff <(sort baseline/summary/op_statistic_*.csv) <(sort pypto/summary/op_statistic_*.csv)"
echo ""

# ---- Quick op-level comparison if files exist ----
echo ""
echo ">>> Op-level time comparison:"
PROF_OUTPUT="${PROF_OUTPUT}" python3 - <<'PYEOF'
import os
import glob
import csv

PROF_OUTPUT = os.environ["PROF_OUTPUT"]


def read_op_summary(dir_label):
    # msprof writes op_statistic_*.csv under PROF_*/mindstudio_profiler_output/;
    # columns: Device_id, OP Type, Core Type, Count, Total Time(us), ...
    pattern = f"{PROF_OUTPUT}/{dir_label}/PROF_*/mindstudio_profiler_output/op_statistic_*.csv"
    files = glob.glob(pattern)
    if not files:
        print(f"  {dir_label}: no op_statistic_*.csv found")
        return {}
    ops = {}
    for path in files:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("OP Type", row.get("Op Name", ""))
                try:
                    time_us = float(row.get("Total Time(us)", row.get("Task Time(us)", 0)))
                except (ValueError, TypeError):
                    continue
                ops[name] = ops.get(name, 0.0) + time_us  # sum across devices
    return ops


baseline = read_op_summary("baseline")
pypto = read_op_summary("pypto")

if baseline and pypto:
    all_ops = sorted(set(baseline) | set(pypto),
                     key=lambda x: baseline.get(x, 0) + pypto.get(x, 0), reverse=True)[:15]
    print(f"{'Operator':<40} {'Baseline(us)':>14} {'PyPTO(us)':>14} {'Diff':>10}")
    print("-" * 80)
    for op in all_ops:
        bv = baseline.get(op, 0)
        pv = pypto.get(op, 0)
        print(f"{op:<40} {bv:>14.1f} {pv:>14.1f} {pv - bv:>+10.1f}")
PYEOF
