#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

# Kernel-level profiling for Phi-3-mini-4k-instruct using msprof
# Compares baseline vs PyPTO at the operator level

set -e

MSPROF="$(which msprof 2>/dev/null || echo 'msprof')"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="/npu/s00454010/models/Phi-3-mini-4k-instruct"
INPUT_FILE="&#36;{SCRIPT_DIR}/sample_inputs.txt"
OUTPUT_LENGTH=20
PROF_OUTPUT="&#36;{SCRIPT_DIR}/msprof_output"

ASK_SCRIPT="&#36;{SCRIPT_DIR}/ask_Phi-3-mini-4k-instruct.py"
BASELINE_CMD="python3 &#36;{ASK_SCRIPT} --model-path &#36;{MODEL_PATH} --sentence_file &#36;{INPUT_FILE} --output_length &#36;{OUTPUT_LENGTH}"
PYPTO_CMD="python3 &#36;{ASK_SCRIPT} --model-path &#36;{MODEL_PATH} --sentence_file &#36;{INPUT_FILE} --output_length &#36;{OUTPUT_LENGTH} --use_pypto"

echo "========================================"
echo "  msprof Kernel-Level Profiling"
echo "  Model: Phi-3-mini-4k-instruct"
echo "========================================"
echo ""
echo "Output tokens per run: &#36;{OUTPUT_LENGTH} (keep small for profiling)"
echo "Output dir: &#36;{PROF_OUTPUT}"
echo ""

# ---- Phase 1: Baseline ----
echo ">>> Phase 1: Profiling Baseline..."
rm -rf "&#36;{PROF_OUTPUT}/baseline"
&#36;{MSPROF} \
    --output="&#36;{PROF_OUTPUT}/baseline" \
    --application="&#36;{BASELINE_CMD}" \
    --aic-metrics=ArithmeticUtilization \
    --task-time=on \
    --ai-core=on

echo ""

# ---- Phase 2: PyPTO ----
echo ">>> Phase 2: Profiling PyPTO..."
rm -rf "&#36;{PROF_OUTPUT}/pypto"
&#36;{MSPROF} \
    --output="&#36;{PROF_OUTPUT}/pypto" \
    --application="&#36;{PYPTO_CMD}" \
    --aic-metrics=ArithmeticUtilization \
    --task-time=on \
    --ai-core=on

echo ""
echo "========================================"
echo "  Profiling Complete"
echo "========================================"
echo ""
echo "Output directories:"
echo "  Baseline: &#36;{PROF_OUTPUT}/baseline"
echo "  PyPTO:    &#36;{PROF_OUTPUT}/pypto"
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
python3 -c "
import glob, csv
from collections import defaultdict

def read_op_summary(dir_label):
    pattern = f'&#36;{PROF_OUTPUT}/{dir_label}/device_*/summary/op_statistic_*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f'  {dir_label}: no op_statistic_*.csv found')
        return {}
    ops = {}
    with open(files[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Op Name', row.get('op_name', ''))
            try:
                time_us = float(row.get('Task Time(us)', row.get('task_time', 0)))
            except (ValueError, TypeError):
                continue
            ops[name] = time_us
    return ops

baseline = read_op_summary('baseline')
pypto = read_op_summary('pypto')

if baseline and pypto:
    all_ops = sorted(set(baseline) | set(pypto), key=lambda x: baseline.get(x,0)+pypto.get(x,0), reverse=True)[:15]
    print(f"{'Operator':<40} {'Baseline(us)':>14} {'PyPTO(us)':>14} {'Diff':>10}")
    print('-' * 80)
    for op in all_ops:
        bv = baseline.get(op, 0)
        pv = pypto.get(op, 0)
        diff = pv - bv
        print(f'{op:<40} {bv:>14.1f} {pv:>14.1f} {diff:>+10.1f}')
"
