#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

# Benchmark script for Phi-3-mini-4k-instruct
# Compares baseline vs PyPTO with timing + memory metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ID="Phi-3-mini-4k-instruct"
MODEL_PATH="${PHI3_MODEL_PATH:?请设置环境变量 PHI3_MODEL_PATH，指向模型权重目录}"
INPUT_FILE="&#36;{SCRIPT_DIR}/sample_inputs.txt"
OUTPUT_LENGTH=100
BASELINE_REPORT="&#36;{SCRIPT_DIR}/bench_baseline.json"
PYPTO_REPORT="&#36;{SCRIPT_DIR}/bench_pypto.json"

echo "========================================"
echo "  Phi-3-mini-4k-instruct Performance Benchmark"
echo "========================================"
echo ""
echo "Model: &#36;{MODEL_ID}"
echo "Model path: &#36;{MODEL_PATH}"
echo "Input: &#36;{INPUT_FILE}"
echo "Output length: &#36;{OUTPUT_LENGTH} tokens"
echo ""

# ---- Phase 1: Baseline ----
echo ">>> Phase 1: Baseline (no PyPTO)"
python3 "&#36;{SCRIPT_DIR}/ask_Phi-3-mini-4k-instruct.py" \
    --model-path "&#36;{MODEL_PATH}" \
    --sentence_file "&#36;{INPUT_FILE}" \
    --output_length &#36;{OUTPUT_LENGTH} \
    --report-file "&#36;{BASELINE_REPORT}"

echo ""

# ---- Phase 2: PyPTO ----
echo ">>> Phase 2: PyPTO fused mode"
python3 "&#36;{SCRIPT_DIR}/ask_Phi-3-mini-4k-instruct.py" \
    --model-path "&#36;{MODEL_PATH}" \
    --sentence_file "&#36;{INPUT_FILE}" \
    --output_length &#36;{OUTPUT_LENGTH} \
    --use_pypto \
    --report-file "&#36;{PYPTO_REPORT}"

echo ""
echo "========================================"
echo "  Benchmark Comparison"
echo "========================================"
echo ""

python3 -c "
import json

with open('&#36;{BASELINE_REPORT}') as f:
    b = json.load(f)
with open('&#36;{PYPTO_REPORT}') as f:
    p = json.load(f)

def pct(a, b):
    if b == 0: return 'N/A'
    return f"{((a-b)/b*100):+.1f}%"

print(f"{'指标':<24} {'Baseline':>12} {'PyPTO':>12} {'Diff':>10}")
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
    print(f'{label:<24} {bv:>12.2f} {pv:>12.2f} {diff:>10}')
print()
print('报告文件:')
print(f'  Baseline: &#36;{BASELINE_REPORT}')
print(f'  PyPTO:    ${PYPTO_REPORT}')
"
