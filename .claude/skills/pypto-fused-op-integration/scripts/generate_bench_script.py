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
Generate benchmark script bench_{model_name}.sh with timing/memory comparison.

Usage: python3 generate_bench_script.py --model-name "ModelName" --script-dir "/scripts/dir" --model-dir "/model/dir"
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')

parser = argparse.ArgumentParser(description="Generate benchmark script with metrics comparison")
parser.add_argument("--model-name", required=True, help="Model name (e.g. Qwen2-7B)")
parser.add_argument("--script-dir", required=True, help="Script output directory (absolute path)")
parser.add_argument("--model-dir", required=True, help="Model weight directory (absolute path)")
args = parser.parse_args()

content = f'''#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

# Benchmark script for {args.model_name}
# Compares baseline vs PyPTO with timing + memory metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ID="{args.model_name}"
MODEL_PATH="{args.model_dir}"
INPUT_FILE="${{SCRIPT_DIR}}/sample_inputs.txt"
OUTPUT_LENGTH=100
BASELINE_REPORT="${{SCRIPT_DIR}}/bench_baseline.json"
PYPTO_REPORT="${{SCRIPT_DIR}}/bench_pypto.json"

echo "========================================"
echo "  {args.model_name} Performance Benchmark"
echo "========================================"
echo ""
echo "Model: ${{MODEL_ID}}"
echo "Model path: ${{MODEL_PATH}}"
echo "Input: ${{INPUT_FILE}}"
echo "Output length: ${{OUTPUT_LENGTH}} tokens"
echo ""

# ---- Phase 1: Baseline ----
echo ">>> Phase 1: Baseline (no PyPTO)"
python3 "${{SCRIPT_DIR}}/ask_{args.model_name}.py" \\
    --model-path "${{MODEL_PATH}}" \\
    --sentence_file "${{INPUT_FILE}}" \\
    --output_length ${{OUTPUT_LENGTH}} \\
    --report-file "${{BASELINE_REPORT}}"

echo ""

# ---- Phase 2: PyPTO ----
echo ">>> Phase 2: PyPTO fused mode"
python3 "${{SCRIPT_DIR}}/ask_{args.model_name}.py" \\
    --model-path "${{MODEL_PATH}}" \\
    --sentence_file "${{INPUT_FILE}}" \\
    --output_length ${{OUTPUT_LENGTH}} \\
    --use_pypto \\
    --report-file "${{PYPTO_REPORT}}"

echo ""
echo "========================================"
echo "  Benchmark Comparison"
echo "========================================"
echo ""

python3 -c "
import json

with open('${{BASELINE_REPORT}}') as f:
    b = json.load(f)
with open('${{PYPTO_REPORT}}') as f:
    p = json.load(f)

def pct(a, b):
    if b == 0: return 'N/A'
    return f\"{{((a-b)/b*100):+.1f}}%\"

print(f\"{{'指标':<24}} {{'Baseline':>12}} {{'PyPTO':>12}} {{'Diff':>10}}")
print('-' * 60)
fields = [
    ('model_load_s',         '模型加载 (s)'),
    ('generate_s',           '推理耗时 (s)'),
    ('generated_tokens',     '生成token数'),
    ('tokens_per_second',    '吞吐 (tokens/s)'),
    ('generate_peak_mem_mb', '峰值显存 (MB)'),
]
for key, label in fields:
    bv = b.get(key, 0)
    pv = p.get(key, 0)
    diff = pct(pv, bv)
    print(f'{{label:<24}} {{bv:>12.2f}} {{pv:>12.2f}} {{diff:>10}}')
print()
print('报告文件:')
print(f'  Baseline: ${{BASELINE_REPORT}}')
print(f'  PyPTO:    ${{PYPTO_REPORT}}')
"
'''

os.makedirs(args.script_dir, exist_ok=True)
output_file = os.path.join(args.script_dir, f"bench_{args.model_name}.sh")
with open(output_file, "w") as f:
    f.write(content)

os.chmod(output_file, 0o755)

logging.info(f"Benchmark script generated: {output_file}")
logging.info(f"Usage: bash {output_file}")
