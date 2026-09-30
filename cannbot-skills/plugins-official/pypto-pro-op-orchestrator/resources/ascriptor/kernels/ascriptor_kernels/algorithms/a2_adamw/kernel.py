# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserved A2-family FP32-internal/BF16-state AdamW step."""

import importlib

from ascriptor.a2 import *
from functools import lru_cache

TILE = 2048

def adamw_step(g: GM[bf16, (1, 'N')], p_in: GM[bf16, (1, 'N')], m_in: GM[bf16, (1, 'N')], v_in: GM[bf16, (1, 'N')], p_out: GM[bf16, (1, 'N')], m_out: GM[bf16, (1, 'N')], v_out: GM[bf16, (1, 'N')], decay: f32, beta1: f32, one_minus_beta1: f32, beta2: f32, one_minus_beta2: f32, step_size: f32, inv_bc2_sqrt: f32, eps: f32, numel: i32):
    bf_g_in = DBuff(DT.bfloat16, [1, TILE], Position.UB)
    bf_p_in = DBuff(DT.bfloat16, [1, TILE], Position.UB)
    bf_m_in = DBuff(DT.bfloat16, [1, TILE], Position.UB)
    bf_v_in = DBuff(DT.bfloat16, [1, TILE], Position.UB)
    bf_p_out = DBuff(DT.bfloat16, [1, TILE], Position.UB)
    bf_m_out = DBuff(DT.bfloat16, [1, TILE], Position.UB)
    bf_v_out = DBuff(DT.bfloat16, [1, TILE], Position.UB)
    f_p = Tensor(DT.float, [1, TILE], Position.UB)
    f_g = Tensor(DT.float, [1, TILE], Position.UB)
    f_m = Tensor(DT.float, [1, TILE], Position.UB)
    f_v = Tensor(DT.float, [1, TILE], Position.UB)
    f_t = Tensor(DT.float, [1, TILE], Position.UB)
    n_tiles = CeilDiv(numel, TILE)
    tiles_per_core = CeilDiv(n_tiles, GetVecNum())
    t_begin = Var(tiles_per_core * GetVecIdx())
    t_end = Min(t_begin + tiles_per_core, n_tiles)
    with auto_sync():
        set_mask((1 << 64) - 1, (1 << 64) - 1)
        for ti in range(t_begin, t_end):
            n0 = Var(ti * TILE)
            n_valid = Min(numel - n0, TILE)
            bf_p_in[ti] <<= p_in[0:1, n0:n0 + n_valid]
            bf_g_in[ti] <<= g[0:1, n0:n0 + n_valid]
            bf_m_in[ti] <<= m_in[0:1, n0:n0 + n_valid]
            bf_v_in[ti] <<= v_in[0:1, n0:n0 + n_valid]
            cast(f_p, bf_p_in[ti])
            cast(f_g, bf_g_in[ti])
            cast(f_m, bf_m_in[ti])
            cast(f_v, bf_v_in[ti])
            muls(f_p, f_p, decay)
            muls(f_m, f_m, beta1)
            axpy(f_m, f_g, one_minus_beta1)
            mul(f_g, f_g, f_g)
            muls(f_v, f_v, beta2)
            axpy(f_v, f_g, one_minus_beta2)
            sqrt(f_g, f_v)
            muls(f_g, f_g, inv_bc2_sqrt)
            adds(f_g, f_g, eps)
            div(f_t, f_m, f_g)
            muls(f_t, f_t, step_size)
            sub(f_p, f_p, f_t)
            cast(bf_p_out[ti], f_p)
            cast(bf_m_out[ti], f_m)
            cast(bf_v_out[ti], f_v)
            p_out[0:1, n0:n0 + n_valid] <<= bf_p_out[ti]
            m_out[0:1, n0:n0 + n_valid] <<= bf_m_out[ti]
            v_out[0:1, n0:n0 + n_valid] <<= bf_v_out[ti]
    return (p_out, m_out, v_out)

@lru_cache(maxsize=2)
def kernel_for(device):
    if device not in ("a2", "a3"):
        raise ValueError("AdamW requires the A2/A3 tensor-vector family")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec", block_dim=1)(adamw_step)
