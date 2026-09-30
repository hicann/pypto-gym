# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The same dense scale inputs feed a shortcut and an explicit MX path."""

import ascriptor.a5 as api


def make_mx(fmt="e4m3", path="shortcut"):
    if fmt not in ("e4m3", "e5m2", "fp4_mixed") or path not in ("shortcut", "explicit"):
        raise ValueError("unknown MX format/path")
    packed = fmt == "fp4_mixed"
    cols = 64 if packed else 128
    dtype_a = api.DT.fp4_e2m1 if packed else getattr(api.DT, fmt)
    dtype_b = api.DT.fp4_e1m2 if packed else dtype_a
    # Compatibility aliases identify the same physical bytes; the following
    # explicit MX load/compute operations carry scale association.
    l0_dtype = api.u8 if packed else (api.DT.mx_e4m3 if fmt == "e4m3" else api.DT.mx_e5m2)

    @api.kernel(mode="cube", block_dim=1)
    def cube_mx(x: api.GM[api.u8, (16, cols)], y: api.GM[api.u8, (16, cols)], sa: api.GM[api.u8, (16, 4)],
        sb: api.GM[api.u8, (16, 4)], o: api.GM[api.f32, (16, 16)]):
        ac = api.Tensor(api.u8, [16, cols], api.Position.L1)
        bc = api.Tensor(api.u8, [16, cols], api.Position.L1)
        a = ac.reinterpret(dtype_a)
        b = bc.reinterpret(dtype_b)
        scale_a = api.Tensor(api.u8, [16, 4], api.Position.L1)
        scale_b = api.Tensor(api.u8, [16, 4], api.Position.L1)
        product = api.Tensor(api.f32, [16, 16], api.Position.L0C)
        with api.auto_sync():
            ac <<= x
            bc <<= y
            api.gm_to_l1_mx_scale_nd2nz(scale_a, sa)
            api.gm_to_l1_mx_scale_nd2nz(scale_b, sb)
            if path == "shortcut":
                api.matmul_mx(product, a, b, scale_a, scale_b, m=16, n=16, k=128)
            else:
                operand_a = api.Tensor(l0_dtype, [16, cols], api.Position.L0A)
                operand_b = api.Tensor(l0_dtype, [16, cols], api.Position.L0B)
                if packed:
                    l0a = operand_a.reinterpret(dtype_a)
                    l0b = operand_b.reinterpret(dtype_b)
                else:
                    l0a = operand_a
                    l0b = operand_b
                api.l1_to_l0_mx(l0a, a, scale_a, m_dst=16, n_dst=128, m_src=16, n_src=128)
                api.l1_to_l0_mx(l0b, b, scale_b, m_dst=16, n_dst=128, m_src=16, n_src=128)
                api.mmad_mx(product, l0a, l0b, M=16, N=16, K=128)
            o <<= product
        return o

    return cube_mx
