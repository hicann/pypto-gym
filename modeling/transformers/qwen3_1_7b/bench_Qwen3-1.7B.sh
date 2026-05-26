#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

# Benchmark script for Qwen3-1.7B
# Compares baseline vs PyPTO with timing + memory metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ID="Qwen3-1.7B"
MODEL_PATH="/mnt/workspace/gitCode/cann/models/Qwen3-1.7B"
INPUT_FILE="${SCRIPT_DIR}/sample_inputs.txt"
OUTPUT_LENGTH=100
BASELINE_REPORT="${SCRIPT_DIR}/bench_baseline.json"
PYPTO_REPORT="${SCRIPT_DIR}/bench_pypto.json"

echo "========================================"
echo "  Qwen3-1.7B Performance Benchmark"
echo "========================================"
echo ""
echo "Model: ${MODEL_ID}"
echo "Model path: ${MODEL_PATH}"
echo "Input: ${INPUT_FILE}"
echo "Output length: ${OUTPUT_LENGTH} tokens"
echo ""

# ---- Phase 1: Baseline ----
echo ">>> Phase 1: Baseline (no PyPTO)"
python3 "${SCRIPT_DIR}/ask_Qwen3-1.7B.py" \
    --model-path "${MODEL_PATH}" \
    --sentence_file "${INPUT_FILE}" \
    --output_length ${OUTPUT_LENGTH} \
    --report-file "${BASELINE_REPORT}"

echo ""

# ---- Phase 2: PyPTO ----
echo ">>> Phase 2: PyPTO fused mode"
python3 "${SCRIPT_DIR}/ask_Qwen3-1.7B.py" \
    --model-path "${MODEL_PATH}" \
    --sentence_file "${INPUT_FILE}" \
    --output_length ${OUTPUT_LENGTH} \
    --use_pypto \
    --report-file "${PYPTO_REPORT}"

echo ""
echo "========================================"
echo "  Benchmark Comparison"
echo "========================================"
echo ""

python3 -c "
import json

with open('${BASELINE_REPORT}') as f:
    b = json.load(f)
with open('${PYPTO_REPORT}') as f:
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
print(f'  Baseline: ${BASELINE_REPORT}')
print(f'  PyPTO:    ${PYPTO_REPORT}')
"
