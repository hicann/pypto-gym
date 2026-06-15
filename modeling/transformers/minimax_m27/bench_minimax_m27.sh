#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# E2E benchmark for MiniMax M2.7: 8-die pipeline-parallel prefill, eager vs PyPTO (fused grouped
# GEMM MoE). Warmup absorbs JIT compilation; measurement iterations are timed separately.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/path/to/MiniMax-M2.7}"
NPROC="${NPROC:-8}"                 # one process per die
SEQ="${SEQ:-64}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
PROMPT_TEXT="${PROMPT_TEXT:-}"      # optional real text input (sets seq from its tokenized length)
BASELINE_REPORT="${SCRIPT_DIR}/bench_baseline.json"
PYPTO_REPORT="${SCRIPT_DIR}/bench_pypto.json"

export PYPTO_VEC_TILE="${PYPTO_VEC_TILE:-128}"   # vector tile must fit the 910B 192 KB UB
export TE_PARALLEL_COMPILER="${TE_PARALLEL_COMPILER:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

PROMPT_ARG=()
[ -n "${PROMPT_TEXT}" ] && PROMPT_ARG=(--prompt-text "${PROMPT_TEXT}")

echo "========================================"
echo "  MiniMax M2.7 E2E Benchmark (${NPROC} dies, prefill)"
echo "========================================"
echo "Model: ${MODEL_PATH}  |  seq: ${SEQ}  |  iters: ${ITERS}"
echo ""

# ---- Phase 1: eager baseline ----
echo ">>> Phase 1: eager (stock per-expert loop)"
torchrun --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/bench_minimax_m27.py" \
    --moe-impl eager --seq "${SEQ}" --iters "${ITERS}" --warmup "${WARMUP}" \
    "${PROMPT_ARG[@]}" --report-file "${BASELINE_REPORT}"
echo ""

# ---- Phase 2: PyPTO (default) ----
echo ">>> Phase 2: PyPTO fused grouped GEMM"
torchrun --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/bench_minimax_m27.py" \
    --moe-impl pypto --seq "${SEQ}" --iters "${ITERS}" --warmup "${WARMUP}" \
    "${PROMPT_ARG[@]}" --report-file "${PYPTO_REPORT}"
echo ""

echo "========================================"
echo "  Benchmark Comparison"
echo "========================================"
python3 -c "
import json
b = json.load(open('${BASELINE_REPORT}')); p = json.load(open('${PYPTO_REPORT}'))
print(f\"{'Metric':<24} {'eager':>12} {'PyPTO':>12} {'Speedup':>10}\")
print('-' * 60)
for key, label, hi in [('best_ms', 'Forward best (ms)', False), ('prefill_tok_s', 'Throughput (tok/s)', True)]:
    bv, pv = b.get(key, 0), p.get(key, 0)
    sp = (pv / bv if hi else bv / pv) if (bv and pv) else 0
    print(f'{label:<24} {bv:>12.1f} {pv:>12.1f} {sp:>9.1f}x')
"
