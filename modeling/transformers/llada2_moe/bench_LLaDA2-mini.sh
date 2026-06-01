#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

# E2E benchmark for LLaDA2.0-mini: baseline vs PyPTO
# Uses block-wise masked diffusion (gen_length/steps/block_length)
# Warmup absorbs JIT compilation; measurement iterations timed separately.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/path/to/LLaDA2.0-mini}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-100}"
DEVICE="${DEVICE:-14}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
BASELINE_REPORT="${SCRIPT_DIR}/bench_baseline.json"
PYPTO_REPORT="${SCRIPT_DIR}/bench_pypto.json"

export TILE_FWK_DEVICE_ID=${DEVICE}
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

echo "========================================"
echo "  LLaDA2.0-mini E2E Benchmark"
echo "========================================"
echo ""
echo "Model path:    ${MODEL_PATH}"
echo "Output length: ${OUTPUT_LENGTH} tokens"
echo "Device:        npu:${DEVICE}"
echo "Warmup:        ${WARMUP} iters"
echo "Measurement:   ${ITERS} iters"
echo ""

# ---- Phase 1: Baseline ----
echo ">>> Phase 1: Baseline (no PyPTO)"
python3 "${SCRIPT_DIR}/bench_LLaDA2-mini.py" \
    --model-path "${MODEL_PATH}" \
    --device ${DEVICE} \
    --output_length ${OUTPUT_LENGTH} \
    --warmup ${WARMUP} \
    --iters ${ITERS} \
    --report-file "${BASELINE_REPORT}"

echo ""

# ---- Phase 2: PyPTO ----
echo ">>> Phase 2: PyPTO fused mode"
python3 "${SCRIPT_DIR}/bench_LLaDA2-mini.py" \
    --model-path "${MODEL_PATH}" \
    --device ${DEVICE} \
    --output_length ${OUTPUT_LENGTH} \
    --warmup ${WARMUP} \
    --iters ${ITERS} \
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

print(f\"{'Metric':<24} {'Baseline':>12} {'PyPTO':>12} {'Speedup':>10}\")
print('-' * 60)
fields = [
    ('time_mean_s',          'Time/iter (s)'),
    ('tps_mean',             'Throughput (tok/s)'),
    ('generate_peak_mem_mb', 'Peak memory (MB)'),
]
for key, label in fields:
    bv = b.get(key, 0)
    pv = p.get(key, 0)
    if key == 'time_mean_s':
        speedup = f'{bv / pv:.1f}x' if pv > 0 else 'N/A'
    elif key == 'tps_mean':
        speedup = f'{pv / bv:.1f}x' if bv > 0 else 'N/A'
    else:
        diff = (pv - bv) / bv * 100 if bv > 0 else 0
        speedup = f'{diff:+.1f}%'
    print(f'{label:<24} {bv:>12.1f} {pv:>12.1f} {speedup:>10}')
print()
print('Reports:')
print(f'  Baseline: ${BASELINE_REPORT}')
print(f'  PyPTO:    ${PYPTO_REPORT}')
"
