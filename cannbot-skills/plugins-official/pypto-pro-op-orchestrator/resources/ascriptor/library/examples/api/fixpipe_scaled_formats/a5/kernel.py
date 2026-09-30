# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Corrected direct-GM FIX bodies; typed output modes and explicit cube ownership."""

import importlib


def make_kernels(device='a5'):
    if device not in ('a5',):
        raise ValueError('unsupported device for this FIX unit')
    api = importlib.import_module('ascriptor.' + device)
    QF_SCALE, QF_OFFSET = 0.5, 8
    RQ_SCALE, RQ_OFFSET = 0.5, 0
    DQ_SCALE, SC_SCALE = 0.25, 0.5

    @api.kernel(mode='cube', block_dim=1)
    def matmul_scaledcast_bf16(x: api.GM[api.f16, ('M', 'K')], y: api.GM[api.f16, ('N', 'K')], z: api.GM[api.bf16, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """The bf16 output of matmul_scaledcast_fp32 (the old script called one kernel with four output dtypes; D-026)."""
        l1x = api.Tensor(api.DT.half, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.half, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.float, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            z[:, :] <<= l0c.requant(scale=SC_SCALE)
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_scaledcast_f32(x: api.GM[api.f16, ('M', 'K')], y: api.GM[api.f16, ('N', 'K')], z: api.GM[api.f32, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """The f32 output of matmul_scaledcast_fp32 (the old script called one kernel with four output dtypes; D-026)."""
        l1x = api.Tensor(api.DT.half, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.half, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.float, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            z[:, :] <<= l0c.requant(scale=SC_SCALE)
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_scaledcast_e4m3(x: api.GM[api.f16, ('M', 'K')], y: api.GM[api.f16, ('N', 'K')], z: api.GM[api.DT.e4m3, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """The e4m3 output of matmul_scaledcast_fp32 (the old script called one kernel with four output dtypes; D-026)."""
        l1x = api.Tensor(api.DT.half, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.half, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.float, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            z[:, :] <<= l0c.requant(scale=SC_SCALE)
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_dequant_int32_bf16(x: api.GM[api.i8, ('M', 'K')], y: api.GM[api.i8, ('N', 'K')], z: api.GM[api.bf16, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """int8 @ int8 -> int32 accumulate, dequantized to bf16 on store (QS322BF16_PRE)."""
        l1x = api.Tensor(api.DT.int8, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.int8, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.int, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            z[:, :] <<= l0c.requant(scale=SC_SCALE)
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_quant_fp32_hif8(x: api.GM[api.f16, ('M', 'K')], y: api.GM[api.f16, ('N', 'K')], z: api.GM[api.u8, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """half @ half -> fp32 accumulate, scaled-quantized to hif8 on store (QF322HIF8_PRE)."""
        l1x = api.Tensor(api.DT.half, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.half, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.float, [M, N], api.Position.L0C)
        zh = z.reinterpret(api.DT.hif8, name='z_hif8')
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            zh[:, :] <<= l0c.requant(scale=SC_SCALE)
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_quant_fp32_hif8_hybrid(x: api.GM[api.f16, ('M', 'K')], y: api.GM[api.f16, ('N', 'K')], z: api.GM[api.u8, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """half @ half -> fp32 accumulate, scaled-quantized to hif8 (QF322HIF8_PRE_HYBRID)."""
        l1x = api.Tensor(api.DT.half, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.half, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.float, [M, N], api.Position.L0C)
        zh = z.reinterpret(api.DT.hif8, name='z_hif8')
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            zh[:, :] <<= l0c.requant(scale=SC_SCALE, hif8_hybrid=True)
        return z

    return {'scaled_bf16': matmul_scaledcast_bf16, 'scaled_f32': matmul_scaledcast_f32, 'scaled_e4m3': matmul_scaledcast_e4m3, 'deq_bf16': matmul_dequant_int32_bf16, 'hif8_ta': matmul_quant_fp32_hif8, 'hif8_hybrid': matmul_quant_fp32_hif8_hybrid}
