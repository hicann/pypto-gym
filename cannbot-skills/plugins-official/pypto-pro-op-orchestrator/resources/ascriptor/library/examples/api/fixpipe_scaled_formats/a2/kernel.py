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


def make_kernels(device='a2'):
    if device not in ('a2', 'a3'):
        raise ValueError('unsupported device for this FIX unit')
    api = importlib.import_module('ascriptor.' + device)
    QF_SCALE, QF_OFFSET = 0.5, 8
    RQ_SCALE, RQ_OFFSET = 0.5, 0
    DQ_SCALE, SC_SCALE = 0.25, 0.5

    @api.kernel(mode='cube', block_dim=1)
    def matmul_quant_fp32_byte(x: api.GM[api.f16, ('M', 'K')], y: api.GM[api.f16, ('N', 'K')], z: api.GM[api.i8, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """half @ half -> fp32 accumulate, quantized to 8-bit on store (QF322B8_PRE).

        Signedness follows the z dtype: int8 -> signed, uint8 -> unsigned (same kernel).
        """
        l1x = api.Tensor(api.DT.half, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.half, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.float, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            z[:, :] <<= l0c.requant(scale=QF_SCALE, offset=QF_OFFSET)
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_quant_fp32_ubyte(x: api.GM[api.f16, ('M', 'K')], y: api.GM[api.f16, ('N', 'K')], z: api.GM[api.u8, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """The unsigned face of :func:`matmul_quant_fp32_byte` — the old kernel took either byte dtype; the
        typed signature needs one kernel per signedness (the quant path follows the declared z dtype)."""
        l1x = api.Tensor(api.DT.half, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.half, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.float, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            z[:, :] <<= l0c.requant(scale=QF_SCALE, offset=QF_OFFSET)
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_requant_int32_byte(x: api.GM[api.i8, ('M', 'K')], y: api.GM[api.i8, ('N', 'K')], z: api.GM[api.i8, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """int8 @ int8 -> int32 accumulate, requantized to 8-bit on store (REQ8)."""
        l1x = api.Tensor(api.DT.int8, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.int8, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.int, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            z[:, :] <<= l0c.requant(scale=RQ_SCALE, offset=RQ_OFFSET)
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_dequant_int32_to_fp16(x: api.GM[api.i8, ('M', 'K')], y: api.GM[api.i8, ('N', 'K')], z: api.GM[api.f16, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        """int8 @ int8 -> int32 accumulate, dequantized to fp16 on store (DEQF16)."""
        l1x = api.Tensor(api.DT.int8, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.int8, [N, K], api.Position.L1)
        l0c = api.Tensor(api.DT.int, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
            z[:, :] <<= l0c.requant(scale=DQ_SCALE)
        return z

    return {'qf_i8': matmul_quant_fp32_byte, 'qf_u8': matmul_quant_fp32_ubyte, 'rq_i8': matmul_requant_int32_byte, 'deq_f16': matmul_dequant_int32_to_fp16}
