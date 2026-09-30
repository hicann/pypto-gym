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

# SHA256: 28c3b987d781d74e3550cda51f54ce36c2d9788909939550b866b7f389f69b9a

from ascriptor.a5 import *

N = 64

FROWS = 16

IROWS = 6


@simt(num_threads=N)
def math_family_simt(x: GMTensor, xs: GMTensor, xi: GMTensor, of: GMTensor, oi: GMTensor, n: Var):
    t = simt_thread_id()
    v = x[t]
    of[0 * n + t] = simt_exp(v)
    of[1 * n + t] = simt_exp2(v)
    of[2 * n + t] = simt_log(v + 4.5)
    of[3 * n + t] = simt_log2(v + 4.5)
    of[4 * n + t] = simt_log1p(v + 4.5)
    of[5 * n + t] = simt_sin(v * 37.0)
    of[6 * n + t] = simt_cos(v * 37.0)
    of[7 * n + t] = simt_tanh(v)
    of[8 * n + t] = simt_rsqrt(v + 4.5)
    of[9 * n + t] = simt_rint(v * 10.0)
    of[10 * n + t] = simt_round(v * 10.0)
    of[11 * n + t] = simt_floor(v * 10.0)
    of[12 * n + t] = simt_ceil(v * 10.0)
    of[13 * n + t] = simt_trunc(v * 10.0)
    of[14 * n + t] = simt_fmod(v * 10.0, 1.7)
    of[15 * n + t] = simt_fma(v, 2.0, 1.0)
    s = xs[t]
    b = xi[t]
    oi[0 * n + t] = simt_isnan(s)
    oi[1 * n + t] = simt_isinf(s)
    oi[2 * n + t] = simt_isfinite(s)
    oi[3 * n + t] = simt_popc(b)
    oi[4 * n + t] = simt_mul_hi(b, b)
    oi[5 * n + t] = simt_ffs(b)


@kernel(mode="vec", block_dim=1)
def simt_math_family(x: GM[f32, (1, N)], xs: GM[f32, (1, N)], xi: GM[i32, (1, N)], of: GM[f32, (FROWS, N)],
    oi: GM[i32, (IROWS, N)]):
    with auto_sync():
        math_family_simt(x, xs, xi, of, oi, N)
    return of, oi
