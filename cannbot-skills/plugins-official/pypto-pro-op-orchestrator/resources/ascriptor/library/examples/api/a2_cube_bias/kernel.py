# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 C2 bias paths retain ND staging and original no/N/K split structure."""

import importlib


def make_kernels(device='a2'):
    if device not in ('a2', 'a3'):
        raise ValueError('this source family declares a2 or a3')
    api = importlib.import_module('ascriptor.' + device)
    STATIC_BIAS_N = 64

    @api.kernel(mode='cube', block_dim=1)
    def matmul_float_bias_nosplit(x: api.GM[api.f32, ('M', 'K')], y: api.GM[api.f32, (64, 'K')], bias: api.GM[api.f32, (1, 64)], z: api.GM[api.f32, ('M', 64)], M: api.i32, K: api.i32):
        l1x = api.Tensor(api.DT.float, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.float, [STATIC_BIAS_N, K], api.Position.L1)
        l1b = api.Tensor(api.DT.float, [1, STATIC_BIAS_N], api.Position.L1, layout=api.Layout.ND)
        l0c = api.Tensor(api.DT.float, [M, STATIC_BIAS_N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            l1b <<= bias[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=STATIC_BIAS_N, k=K, is_init=True, bias=l1b)
            z[:, :] <<= l0c
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_float_bias_splitk(x: api.GM[api.f32, ('M', 'K')], y: api.GM[api.f32, (64, 'K')], bias: api.GM[api.f32, (1, 64)], z: api.GM[api.f32, ('M', 64)], M: api.i32, K: api.i32):
        l1x = api.Tensor(api.DT.float, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.float, [STATIC_BIAS_N, K], api.Position.L1)
        l1b = api.Tensor(api.DT.float, [1, STATIC_BIAS_N], api.Position.L1, layout=api.Layout.ND)
        l0c = api.Tensor(api.DT.float, [M, STATIC_BIAS_N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            l1b <<= bias[:, :]
            api.matmul(l0c, l1x, l1y, splitk=16, m=M, n=STATIC_BIAS_N, k=K, is_init=True, bias=l1b)
            z[:, :] <<= l0c
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_float_bias_splitn(x: api.GM[api.f32, ('M', 'K')], y: api.GM[api.f32, ('N', 'K')], bias: api.GM[api.f32, (1, 'N')], z: api.GM[api.f32, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        l1x = api.Tensor(api.DT.float, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.float, [N, K], api.Position.L1)
        l1b = api.Tensor(api.DT.float, [1, N], api.Position.L1, layout=api.Layout.ND)
        l0c = api.Tensor(api.DT.float, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            l1b <<= bias[:, :]
            api.matmul(l0c, l1x, l1y, splitn=32, m=M, n=N, k=K, is_init=True, bias=l1b)
            z[:, :] <<= l0c
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_int8_bias_kernel(x: api.GM[api.i8, ('M', 'K')], y: api.GM[api.i8, (64, 'K')], bias: api.GM[api.i32, (1, 64)], z: api.GM[api.i32, ('M', 64)], M: api.i32, K: api.i32):
        l1x = api.Tensor(api.DT.int8, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.int8, [STATIC_BIAS_N, K], api.Position.L1)
        l1b = api.Tensor(api.DT.int, [1, STATIC_BIAS_N], api.Position.L1, layout=api.Layout.ND)
        l0c = api.Tensor(api.DT.int, [M, STATIC_BIAS_N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            l1b <<= bias[:, :]
            api.matmul(l0c, l1x, l1y, m=M, n=STATIC_BIAS_N, k=K, is_init=True, bias=l1b)
            z[:, :] <<= l0c
        return z

    @api.kernel(mode='cube', block_dim=1)
    def matmul_int8_bias_splitn(x: api.GM[api.i8, ('M', 'K')], y: api.GM[api.i8, ('N', 'K')], bias: api.GM[api.i32, (1, 'N')], z: api.GM[api.i32, ('M', 'N')], M: api.i32, N: api.i32, K: api.i32):
        l1x = api.Tensor(api.DT.int8, [M, K], api.Position.L1)
        l1y = api.Tensor(api.DT.int8, [N, K], api.Position.L1)
        l1b = api.Tensor(api.DT.int, [1, N], api.Position.L1, layout=api.Layout.ND)
        l0c = api.Tensor(api.DT.int, [M, N], api.Position.L0C)
        with api.auto_sync():
            l1x <<= x[:, :]
            l1y <<= y[:, :]
            l1b <<= bias[:, :]
            api.matmul(l0c, l1x, l1y, splitn=32, m=M, n=N, k=K, is_init=True, bias=l1b)
            z[:, :] <<= l0c
        return z

    return {'f32_none': matmul_float_bias_nosplit, 'f32_splitk': matmul_float_bias_splitk,
        'f32_splitn': matmul_float_bias_splitn, 'i8_none': matmul_int8_bias_kernel, 'i8_splitn': matmul_int8_bias_splitn}
