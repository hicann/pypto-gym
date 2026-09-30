# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 signed-int4 tail accumulation retains separate 16/8-carrier L1 slots."""

import importlib


def make_kernel(device='a2'):
    if device not in ('a2', 'a3'):
        raise ValueError('this family declares a2 or a3')
    api = importlib.import_module('ascriptor.' + device)

    TILE_M = 64

    TILE_N = 64

    SPLIT_K = 128

    SPLIT_K_CARRIER = SPLIT_K // 8

    TAIL_SAFE_K = 64

    TAIL_SAFE_K_CARRIER = TAIL_SAFE_K // 8

    @api.kernel(mode='cube', block_dim=1)
    def matmul_int4_splitk_kernel(x: api.GM[api.i32, ('M', 'KC')], y: api.GM[api.i32, ('N', 'KC')], z: api.GM[api.i32, ('M', 'N')], packed_x: api.GM[api.i32, ('M', 'KC')], packed_y: api.GM[api.i32, ('N', 'KC')], M: api.i32, N: api.i32, K: api.i32, KC: api.i32):
        word = api.Var(0, api.i32)
        tail_mask = api.Var(0, api.i32)
        if K % 8 != 0:
            tail_mask.set((1 << (4 * (K % 8))) - 1)
        else:
            tail_mask.set(-1)
        # mad_s4 consumes both nibbles of its final byte. Normalize private
        # carriers on the device; the caller's padding and inputs stay intact.
        for row in range(M):
            for col in range(KC):
                word.GetValueFrom(x[row, col:col + 1])
                if col == KC - 1:
                    word.set(api.var_and(word, tail_mask))
                word.SetValueTo(packed_x[row, col:col + 1])
        for row in range(N):
            for col in range(KC):
                word.GetValueFrom(y[row, col:col + 1])
                if col == KC - 1:
                    word.set(api.var_and(word, tail_mask))
                word.SetValueTo(packed_y[row, col:col + 1])
        # The public operation cleans the entire scalar data cache; its GM
        # address operand does not limit the flush to the input allocation.
        api.clean_dcache(x)
        l1x = api.DBuff(api.DT.int, [TILE_M, SPLIT_K_CARRIER], api.Position.L1)
        l1y = api.DBuff(api.DT.int, [TILE_N, SPLIT_K_CARRIER], api.Position.L1)
        l1x_tail = api.DBuff(api.DT.int, [TILE_M, TAIL_SAFE_K_CARRIER], api.Position.L1)
        l1y_tail = api.DBuff(api.DT.int, [TILE_N, TAIL_SAFE_K_CARRIER], api.Position.L1)
        l0c = api.DBuff(api.DT.int, [TILE_M, TILE_N], api.Position.L0C)
        l1_cnt = api.Var(0)
        l0c_cnt = api.Var(0)
        tile_m = api.CeilDiv(M, TILE_M)
        tile_m_per_core = api.CeilDiv(tile_m, api.GetCubeNum())
        tile_m_begin = api.Var(tile_m_per_core * api.GetCubeIdx())
        tile_m_end = api.Min(tile_m_begin + tile_m_per_core, tile_m)
        with api.auto_sync():
            for mt in range(tile_m_begin, tile_m_end):
                m0 = api.Var(mt * TILE_M)
                valid_m = api.Min(TILE_M, M - m0)
                for n0 in range(0, N, TILE_N):
                    valid_n = api.Min(TILE_N, N - n0)
                    for k0 in range(0, K, SPLIT_K):
                        valid_k = api.Min(SPLIT_K, K - k0)
                        k0_carrier = api.Var(k0 // 8)
                        valid_k_carrier = api.Min(SPLIT_K_CARRIER, KC - k0_carrier)
                        valid_k_tail_carrier = api.Min(TAIL_SAFE_K_CARRIER, KC - k0_carrier)
                        if valid_k <= TAIL_SAFE_K:
                            l1x_tail[l1_cnt] <<= packed_x[m0:m0 + valid_m, k0_carrier:k0_carrier + valid_k_tail_carrier]
                            l1y_tail[l1_cnt] <<= packed_y[n0:n0 + valid_n, k0_carrier:k0_carrier + valid_k_tail_carrier]
                            api.matmul(l0c[l0c_cnt], l1x_tail[l1_cnt].reinterpret(api.DT.int4), l1y_tail[l1_cnt].reinterpret(api.DT.int4), k=valid_k, is_init=k0 == 0)
                        else:
                            l1x[l1_cnt] <<= packed_x[m0:m0 + valid_m, k0_carrier:k0_carrier + valid_k_carrier]
                            l1y[l1_cnt] <<= packed_y[n0:n0 + valid_n, k0_carrier:k0_carrier + valid_k_carrier]
                            api.matmul(l0c[l0c_cnt], l1x[l1_cnt].reinterpret(api.DT.int4), l1y[l1_cnt].reinterpret(api.DT.int4), k=valid_k, is_init=k0 == 0)
                        l1_cnt += 1
                    z[m0:m0 + valid_m, n0:n0 + valid_n] <<= l0c[l0c_cnt][0:valid_m, 0:valid_n]
                    l0c_cnt += 1
        return z

    return matmul_int4_splitk_kernel
