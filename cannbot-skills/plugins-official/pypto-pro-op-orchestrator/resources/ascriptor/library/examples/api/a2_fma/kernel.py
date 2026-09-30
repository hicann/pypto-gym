# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Separate load and accumulator slots preserve A2/A3 FMA reuse and mixed strides."""

import importlib


def make_kernels(device='a2'):
    if device not in ('a2', 'a3'):
        raise ValueError('this family declares a2 or a3')
    api = importlib.import_module('ascriptor.' + device)

    TILE_F32_MAX = 5120

    TILE_F16_MAX = 5120

    TILE_MIX_MAX = 5120

    @api.kernel(mode='vec', block_dim=1)
    def fma_float_kernel(a: api.GM[api.f32, (1, 'n')], b: api.GM[api.f32, (1, 'n')], c: api.GM[api.f32, (1, 'n')], y: api.GM[api.f32, (1, 'n')], n: api.i32, tile_len: api.i32):
        c_ub = api.DBuff(api.DT.float, [1, TILE_F32_MAX], api.Position.UB)
        a_ub = api.DBuff(api.DT.float, [1, TILE_F32_MAX], api.Position.UB)
        b_ub = api.DBuff(api.DT.float, [1, TILE_F32_MAX], api.Position.UB)
        acc = api.DBuff(api.DT.float, [1, TILE_F32_MAX], api.Position.UB)
        n_tiles = api.CeilDiv(n, tile_len)
        tiles_per_core = api.CeilDiv(n_tiles, api.GetVecNum())
        tile_begin = api.Var(tiles_per_core * api.GetVecIdx())
        tile_end = api.Min(tile_begin + tiles_per_core, n_tiles)
        with api.auto_sync():
            for tile_idx in range(tile_begin, tile_end):
                n0 = api.Var(tile_idx * tile_len)
                valid = api.Min(tile_len, n - n0)
                c_s = c_ub[tile_idx]
                a_s = a_ub[tile_idx]
                b_s = b_ub[tile_idx]
                acc_s = acc[tile_idx]
                c_s[0:1, 0:valid] <<= c[0:1, n0:n0 + valid]
                a_s[0:1, 0:valid] <<= a[0:1, n0:n0 + valid]
                b_s[0:1, 0:valid] <<= b[0:1, n0:n0 + valid]
                api.adds(acc_s[0:1, 0:valid], c_s[0:1, 0:valid], 0.0)
                api.muladddst(acc_s[0:1, 0:valid], a_s[0:1, 0:valid], b_s[0:1, 0:valid])
                y[0:1, n0:n0 + valid] <<= acc_s[0:1, 0:valid]
        return y

    @api.kernel(mode='vec', block_dim=1)
    def fma_half_kernel(a: api.GM[api.f16, (1, 'n')], b: api.GM[api.f16, (1, 'n')], c: api.GM[api.f16, (1, 'n')], y: api.GM[api.f16, (1, 'n')], n: api.i32, tile_len: api.i32):
        c_ub = api.DBuff(api.DT.half, [1, TILE_F16_MAX], api.Position.UB)
        a_ub = api.DBuff(api.DT.half, [1, TILE_F16_MAX], api.Position.UB)
        b_ub = api.DBuff(api.DT.half, [1, TILE_F16_MAX], api.Position.UB)
        acc = api.DBuff(api.DT.half, [1, TILE_F16_MAX], api.Position.UB)
        n_tiles = api.CeilDiv(n, tile_len)
        tiles_per_core = api.CeilDiv(n_tiles, api.GetVecNum())
        tile_begin = api.Var(tiles_per_core * api.GetVecIdx())
        tile_end = api.Min(tile_begin + tiles_per_core, n_tiles)
        with api.auto_sync():
            for tile_idx in range(tile_begin, tile_end):
                n0 = api.Var(tile_idx * tile_len)
                valid = api.Min(tile_len, n - n0)
                c_s = c_ub[tile_idx]
                a_s = a_ub[tile_idx]
                b_s = b_ub[tile_idx]
                acc_s = acc[tile_idx]
                c_s[0:1, 0:valid] <<= c[0:1, n0:n0 + valid]
                a_s[0:1, 0:valid] <<= a[0:1, n0:n0 + valid]
                b_s[0:1, 0:valid] <<= b[0:1, n0:n0 + valid]
                api.adds(acc_s[0:1, 0:valid], c_s[0:1, 0:valid], 0.0)
                api.muladddst(acc_s[0:1, 0:valid], a_s[0:1, 0:valid], b_s[0:1, 0:valid])
                y[0:1, n0:n0 + valid] <<= acc_s[0:1, 0:valid]
        return y

    @api.kernel(mode='vec', block_dim=1)
    def fma_mixed_kernel(a: api.GM[api.f16, (1, 'n')], b: api.GM[api.f16, (1, 'n')], c: api.GM[api.f32, (1, 'n')], y: api.GM[api.f32, (1, 'n')], n: api.i32, tile_len: api.i32):
        c_ub = api.DBuff(api.DT.float, [1, TILE_MIX_MAX], api.Position.UB)
        a_ub = api.DBuff(api.DT.half, [1, TILE_MIX_MAX], api.Position.UB)
        b_ub = api.DBuff(api.DT.half, [1, TILE_MIX_MAX], api.Position.UB)
        acc = api.DBuff(api.DT.float, [1, TILE_MIX_MAX], api.Position.UB)
        n_tiles = api.CeilDiv(n, tile_len)
        tiles_per_core = api.CeilDiv(n_tiles, api.GetVecNum())
        tile_begin = api.Var(tiles_per_core * api.GetVecIdx())
        tile_end = api.Min(tile_begin + tiles_per_core, n_tiles)
        with api.auto_sync():
            for tile_idx in range(tile_begin, tile_end):
                n0 = api.Var(tile_idx * tile_len)
                valid = api.Min(tile_len, n - n0)
                c_s = c_ub[tile_idx]
                a_s = a_ub[tile_idx]
                b_s = b_ub[tile_idx]
                acc_s = acc[tile_idx]
                c_s[0:1, 0:valid] <<= c[0:1, n0:n0 + valid]
                a_s[0:1, 0:valid] <<= a[0:1, n0:n0 + valid]
                b_s[0:1, 0:valid] <<= b[0:1, n0:n0 + valid]
                api.adds(acc_s[0:1, 0:valid], c_s[0:1, 0:valid], 0.0)
                api.muladddst(acc_s[0:1, 0:valid], a_s[0:1, 0:valid], b_s[0:1, 0:valid], repeat=api.CeilDiv(valid, 64), dst_blk_stride=1, src1_blk_stride=1, src2_blk_stride=1, dst_rep_stride=8, src1_rep_stride=4, src2_rep_stride=4)
                y[0:1, n0:n0 + valid] <<= acc_s[0:1, 0:valid]
        return y

    return {'f32': fma_float_kernel, 'f16': fma_half_kernel, 'mixed': fma_mixed_kernel}
