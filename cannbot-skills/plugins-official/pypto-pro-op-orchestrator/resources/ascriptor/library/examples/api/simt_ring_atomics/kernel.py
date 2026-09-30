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

Pure-vector teaching units explicitly launch one vector participant. Original
mode=vec bodies, masks, byte footprints and overlapping-write barriers remain.
"""

from ascriptor.a5 import *

N = 64

T = 256

LIMIT = 5


@simt(num_threads=T)
def seed_simt(out: GMTensor, n: Var):
    t = simt_thread_id()
    if t < n:
        out[t] = 0
        out[n + t] = LIMIT


@simt(num_threads=T)
def incdec_simt(out: GMTensor, n: Var):
    t = simt_thread_id()
    i = t % n
    simt_atomic_inc(out[i], LIMIT)
    simt_threadfence()
    simt_atomic_dec(out[n + i], LIMIT)


@kernel(mode="vec", block_dim=1)
def simt_atomic_incdec(dummy: GM[u32, (1, 8)], out: GM[u32, (2, N)]):
    d_ub = Tensor(DT.uint32, [1, 8], Position.UB, name="aid_d")
    with auto_sync():
        d_ub[:, :] <<= dummy[0:1, :]  # satisfy the input-tensor requirement
        seed_simt(out, N)
        incdec_simt(out, N)
    return out
