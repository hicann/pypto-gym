# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F403, F405, F841
"""Reviewed production closures; host inputs/references are local and independent.

The mixed task retains two vector participants and its complete contributor
set, internal output seeding and all-vector rendezvous.
"""

from ascriptor.a5 import *

ROWS = 9

N = 64

T = 256


@simt(num_threads=T)
def seed_rows_simt(init: GMTensor, out: GMTensor, total: Var, vec_idx: Var, vec_num: Var):
    base = vec_idx * simt_thread_num() + simt_thread_id()
    stride = vec_num * simt_thread_num()
    for i in range(base, total, stride):
        out[i] = init[i]


@simt(num_threads=T)
def atomic_family_simt(x: GMTensor, out: GMTensor, n: Var, vec_idx: Var):
    t = vec_idx * simt_thread_num() + simt_thread_id()
    i = t % n
    v = x[t]
    simt_atomic_add(out[0 * n + i], v)
    simt_atomic_sub(out[1 * n + i], v)
    simt_atomic_max(out[2 * n + i], v)
    simt_atomic_min(out[3 * n + i], v)
    simt_atomic_exch(out[4 * n + i], i)
    simt_atomic_and(out[5 * n + i], v)
    simt_atomic_or(out[6 * n + i], v)
    simt_atomic_xor(out[7 * n + i], v)
    if t < n:
        simt_atomic_cas(out[8 * n + i], i, 1000 + i)  # the row holds i: this one succeeds
    else:
        simt_atomic_cas(out[8 * n + i], -7, 5)  # never matches: the row keeps 1000 + i (or i, if this ran first)


@kernel(mode="mix", block_dim=1)
def simt_atomic_family(x: GM[i32, (1, 512)], init: GM[i32, (ROWS, N)], out: GM[i32, (ROWS, N)]):
    with auto_sync():
        vec_idx = GetVecIdx()
        vec_num = GetVecNum()
        seed_rows_simt(init, out, ROWS * N, vec_idx, vec_num)
        allvec_ready(0, pipe=Pipe.V)  # the seeded rows are visible to every vec block before the atomics
        allvec_wait(0, pipe=Pipe.V)
        atomic_family_simt(x, out, N, vec_idx)
    return out
