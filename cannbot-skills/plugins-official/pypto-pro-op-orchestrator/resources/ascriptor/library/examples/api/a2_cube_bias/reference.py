# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact dyadic or integer products with one explicit bias row; no DSL import."""

import torch


def make_inputs(case):
    params = case['parameters']
    mode, m, n, k = params['mode'], params['M'], params['N'], params['K']
    generator = torch.Generator().manual_seed(case['seed'])
    integer = mode.startswith('i8')
    x = torch.randint(-4, 5, (m, k), generator=generator)
    y = torch.randint(-4, 5, (n, k), generator=generator)
    bias = torch.arange(1, n + 1).reshape(1, n) * 3
    inputs = {'mode': mode, 'x': x.to(torch.int8) if integer else x.float() / 8,
        'y': y.to(torch.int8) if integer else y.float() / 8,
        'bias': bias.int() if integer else bias.float() / 4}
    validate(inputs)
    return inputs


def validate(inputs):
    mode, x, y, bias = (inputs.get(key) for key in ['mode', 'x', 'y', 'bias'])
    if mode not in ['f32_none', 'f32_splitk', 'f32_splitn', 'i8_none', 'i8_splitn']:
        raise ValueError('unknown source bias mode')
    integer = mode.startswith('i8')
    for value, dtype in [(x, torch.int8 if integer else torch.float32), (y, torch.int8 if integer else torch.float32),
        (bias, torch.int32 if integer else torch.float32)]:
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.ndim != 2 or not value.is_contiguous():
            raise ValueError('contiguous typed input and bias matrices are required')
    m, k = x.shape
    n = y.shape[0]
    if not 16 <= m <= 64 or m % 16 or n not in [32, 64, 96] or not 16 <= k <= 128 or k % (32 if integer else 16):
        raise ValueError('source operands require M multiple16, N multiple32 and aligned compact K')
    if y.shape[1] != k or bias.shape != (1, n) or mode.endswith(('none', 'splitk')) and n != 64:
        raise ValueError('matrix/bias shapes must agree; no-split and split-K declare static N=64')
    if integer:
        if not bool(((x >= -4) & (x <= 4)).all()) or not bool(((y >= -4) & (y <= 4)).all()) or not bool((bias.long().abs() <= 4096).all()):
            raise ValueError('integer values exceed the exact no-overflow sample domain')
    else:
        if any(not bool(torch.isfinite(v).all()) for v in [x, y, bias]):
            raise ValueError('finite dyadic sample inputs are required')
        if any(not bool((v * 8 == (v * 8).trunc()).all()) or not bool((v.abs() <= 32).all()) for v in [x, y]) or not bool((bias.abs() <= 4096).all()) or not bool((bias * 4 == (bias * 4).trunc()).all()):
            raise ValueError('the exact floating sample uses bounded eighth-unit operands and quarter-unit bias')


def product(inputs):
    integer = inputs['mode'].startswith('i8')
    dtype = torch.int64 if integer else torch.float64
    return inputs['x'].to(dtype) @ inputs['y'].to(dtype).T


def reference(inputs):
    validate(inputs)
    return {'o': (product(inputs) + inputs['bias']).to(torch.int32 if inputs['mode'].startswith('i8') else torch.float32)}
