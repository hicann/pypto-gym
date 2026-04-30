#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Generate msprof profiling script prof_{model_name}.sh
Runs msprof to collect kernel-level traces for baseline vs PyPTO comparison.

Usage: python3 generate_prof_script.py --model-name "ModelName" --script-dir "/scripts/dir" --model-dir "/model/dir"
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')

parser = argparse.ArgumentParser(description="Generate msprof profiling script")
parser.add_argument("--model-name", required=True, help="Model name (e.g. Qwen2-7B)")
parser.add_argument("--script-dir", required=True, help="Script output directory (absolute path)")
parser.add_argument("--model-dir", required=True, help="Model weight directory (absolute path)")
args = parser.parse_args()

content = f'''#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

# Kernel-level profiling for {args.model_name} using msprof
# Compares baseline vs PyPTO at the operator level

set -e

MSPROF="$(which msprof 2>/dev/null || echo 'msprof')"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="{args.model_dir}"
INPUT_FILE="${{SCRIPT_DIR}}/sample_inputs.txt"
OUTPUT_LENGTH=20
PROF_OUTPUT="${{SCRIPT_DIR}}/msprof_output"

ASK_SCRIPT="${{SCRIPT_DIR}}/ask_{args.model_name}.py"
BASELINE_CMD="python3 ${{ASK_SCRIPT}} --model-path ${{MODEL_PATH}} --sentence_file ${{INPUT_FILE}} --output_length ${{OUTPUT_LENGTH}}"
PYPTO_CMD="python3 ${{ASK_SCRIPT}} --model-path ${{MODEL_PATH}} --sentence_file ${{INPUT_FILE}} --output_length ${{OUTPUT_LENGTH}} --use_pypto"

echo "========================================"
echo "  msprof Kernel-Level Profiling"
echo "  Model: {args.model_name}"
echo "========================================"
echo ""
echo "Output tokens per run: ${{OUTPUT_LENGTH}} (keep small for profiling)"
echo "Output dir: ${{PROF_OUTPUT}}"
echo ""

# ---- Phase 1: Baseline ----
echo ">>> Phase 1: Profiling Baseline..."
rm -rf "${{PROF_OUTPUT}}/baseline"
${{MSPROF}} \\
    --output="${{PROF_OUTPUT}}/baseline" \\
    --application="${{BASELINE_CMD}}" \\
    --aic-metrics=ArithmeticUtilization \\
    --task-time=on \\
    --ai-core=on

echo ""

# ---- Phase 2: PyPTO ----
echo ">>> Phase 2: Profiling PyPTO..."
rm -rf "${{PROF_OUTPUT}}/pypto"
${{MSPROF}} \\
    --output="${{PROF_OUTPUT}}/pypto" \\
    --application="${{PYPTO_CMD}}" \\
    --aic-metrics=ArithmeticUtilization \\
    --task-time=on \\
    --ai-core=on

echo ""
echo "========================================"
echo "  Profiling Complete"
echo "========================================"
echo ""
echo "Output directories:"
echo "  Baseline: ${{PROF_OUTPUT}}/baseline"
echo "  PyPTO:    ${{PROF_OUTPUT}}/pypto"
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
    pattern = f'${{PROF_OUTPUT}}/{{dir_label}}/device_*/summary/op_statistic_*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f'  {{dir_label}}: no op_statistic_*.csv found')
        return {{}}
    ops = {{}}
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
    print(f\"{{'Operator':<40}} {{'Baseline(us)':>14}} {{'PyPTO(us)':>14}} {{'Diff':>10}}\")
    print('-' * 80)
    for op in all_ops:
        bv = baseline.get(op, 0)
        pv = pypto.get(op, 0)
        diff = pv - bv
        print(f'{{op:<40}} {{bv:>14.1f}} {{pv:>14.1f}} {{diff:>+10.1f}}')
"
'''

os.makedirs(args.script_dir, exist_ok=True)
output_file = os.path.join(args.script_dir, f"prof_{args.model_name}.sh")
with open(output_file, "w") as f:
    f.write(content)

os.chmod(output_file, 0o755)

logging.info(f"msprof profiling script generated: {output_file}")
logging.info(f"Usage: bash {output_file}")
