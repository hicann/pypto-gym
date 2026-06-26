#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Unified single-die **E2E generation** benchmark driver for MiniMax (M2.7 / M3). Pick the variant
# with the VARIANT env var or the first positional arg (m27 | m3). Each arm is a separate process that
# runs the real model.generate() loop, counts the tokens produced, and reports the AVERAGE over ITERS
# generations (mean +/- std) in three regimes (prefill / decode / both). The two graph arms capture a
# fixed window once and replay it per generated token (decode-generate tok/s).
#
#   eager       : stock per-expert MoE FFN loop          (generate; M3 only -- M2.7 always streams)
#   pypto       : PyPTO fused grouped-GEMM expert FFN     (generate; prefill/decode/both)
#   vecgraph    : NPU-friendly vectorized FFN + NPUGraph  (capture/replay; decode-generate)
#   pyptograph  : PyPTO grouped-GEMM      + NPUGraph      (capture/replay; decode-generate)
#
# The two graph arms are the E2E half of the Kernel/E2E table. The Kernel half (Eager vs PyPTO) is an
# operator microbench kept local-only (WARMUP=5 / ITERS=20); the kernel is covered in-repo by
# tests/ops/minimax_m27/ and tests/ops/minimax_m3/.
#
#   MODEL_PATH=/path/to/MiniMax-M3 VARIANT=m3 DEVICE=7 LAYERS=4 bash bench_minimax.sh
#   MODEL_PATH=/path/to/MiniMax-M2.7 bash bench_minimax.sh m27
#
# M3 layers 0-2 are dense, 3+ are MoE -> LAYERS=4 is the fits-on-die comparison. ARMS selects a subset.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

VARIANT="${1:-${VARIANT:-m27}}"
case "${VARIANT}" in
    m27|m3) ;;
    *) echo "VARIANT must be 'm27' or 'm3' (got '${VARIANT}')" >&2; exit 2 ;;
esac

: "${MODEL_PATH:?set MODEL_PATH to a MiniMax-${VARIANT} checkpoint path}"
DEVICE="${DEVICE:-${TILE_FWK_DEVICE_ID:-0}}"
LAYERS="${LAYERS:-4}"               # fits-on-die slice (M3: >=4 to include a MoE layer); empty => full
NEW_TOKENS="${NEW_TOKENS:-32}"      # tokens to generate per measured run (N)
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-5}"
WINDOW="${WINDOW:-32}"              # fixed captured window for the graph arms
PROMPT="${PROMPT:-Hello.}"

export PYPTO_VEC_TILE="${PYPTO_VEC_TILE:-128}"
export TE_PARALLEL_COMPILER="${TE_PARALLEL_COMPILER:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"
export TILE_FWK_DEVICE_ID="${DEVICE}"

COMMON=(--variant "${VARIANT}" --device "${DEVICE}" --max-new-tokens "${NEW_TOKENS}"
        --warmup "${WARMUP}" --iters "${ITERS}" --prompt "${PROMPT}")
[ -n "${LAYERS}" ] && COMMON+=(--max-layers "${LAYERS}" --no-streaming)

# M2.7 always streams the fused kernel (no stock eager generate arm); M3 has the full four-arm set.
if [ "${VARIANT}" = "m27" ]; then
    DEFAULT_ARMS="pypto,vecgraph,pyptograph"
else
    DEFAULT_ARMS="eager,pypto,vecgraph,pyptograph"
fi
ARMS="${ARMS:-${DEFAULT_ARMS}}"
want() { case ",${ARMS}," in *",$1,"*) return 0 ;; *) return 1 ;; esac ; }

REPORT_DIR="${REPORT_DIR:-${SCRIPT_DIR}}"
R_EAGER="${REPORT_DIR}/bench_latest_eager.json"
R_PYPTO="${REPORT_DIR}/bench_latest_pypto.json"
R_VECG="${REPORT_DIR}/bench_latest_vecgraph.json"
R_PYPTOG="${REPORT_DIR}/bench_latest_pyptograph.json"
rm -f "${R_EAGER}" "${R_PYPTO}" "${R_VECG}" "${R_PYPTOG}"

run() { echo ">>> $1"; shift; python3 "${SCRIPT_DIR}/bench_minimax.py" "${COMMON[@]}" "$@"; echo ""; }

echo "========================================"
echo "  MiniMax-${VARIANT} single-die E2E generation (die ${DEVICE}, layers ${LAYERS:-full})"
echo "========================================"
echo "Model: ${MODEL_PATH}  |  new_tokens: ${NEW_TOKENS}  |  iters: ${ITERS}  |  window: ${WINDOW}"
echo ""

want eager      && run "eager (generate)"            --report-file "${R_EAGER}"
want pypto      && run "pypto (generate)"  --use_pypto --report-file "${R_PYPTO}"
want vecgraph   && run "vec + NPUGraph"     --graph --graph-window "${WINDOW}" --report-file "${R_VECG}"
want pyptograph && run "pypto + NPUGraph"   --graph --graph-window "${WINDOW}" --use_pypto --report-file "${R_PYPTOG}"

echo "========================================"
echo "  E2E generation comparison (tok/s, higher better)"
echo "========================================"
python3 - "${R_EAGER}" "${R_PYPTO}" "${R_VECG}" "${R_PYPTOG}" <<'PY'
import json, os, sys
labels = ["eager", "pypto", "vec+graph", "pypto+graph"]
d = {}
for lab, p in zip(labels, sys.argv[1:5]):
    if p and os.path.exists(p):
        d[lab] = json.load(open(p))

print(f"{'arm':<13} {'prefill':>10} {'decode':>10} {'e2e':>10}   (tok/s)")
print("-" * 50)
for lab in labels:
    r = d.get(lab)
    if not r:
        print(f"{lab:<13} {'(skipped)':>10}")
        continue
    pf = r.get("prefill_tps_mean", float("nan"))
    dc = r.get("decode_tps_mean", float("nan"))
    e2 = r.get("e2e_tps_mean", float("nan"))
    pf_s = "-" if pf != pf else f"{pf:.1f}"
    e2_s = "-" if e2 != e2 else f"{e2:.1f}"
    print(f"{lab:<13} {pf_s:>10} {dc:>10.1f} {e2_s:>10}")

vg = d.get("vec+graph", {}).get("decode_tps_mean")
pg = d.get("pypto+graph", {}).get("decode_tps_mean")
if vg and pg:
    print(f"\nE2E (graph) PyPTO+graph / vec+graph = {pg/vg:.2f}x   "
          f"(vec+graph {vg:.1f} -> pypto+graph {pg:.1f} tok/s)")
print("\nKernel (Eager vs PyPTO) half of the table: operator microbench kept local-only")
print("  (WARMUP=5 / ITERS=20); kernel correctness is covered by tests/ops/minimax_{m27,m3}/")
PY
