# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent products, scalar field truncation and explicit byte conversion."""

import math
import struct

import torch

M, N, KF, KI = 16, 32, 48, 64
NAMES = ['qf_i8', 'qf_u8', 'rq_i8', 'deq_f16']


def fp19(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0] & 0xFFFFE000
    return struct.unpack('<f', struct.pack('<I', word))[0]


def quant_byte(value, signed, scale, offset):
    rounded = min(255, max(-256, round(value * fp19(scale)))) + offset
    lower, upper = (-128, 127) if signed else (0, 255)
    return min(upper, max(lower, rounded)) & 255


def half_bytes(value):
    try:
        return list(struct.pack('<e', value))
    except OverflowError:
        return list(struct.pack('<e', math.copysign(math.inf, value)))


def make_inputs(case):
    dataset = case['parameters']['dataset']
    if dataset not in ['source', 'boundaries', 'random'] or case.get('block_dim', 1) != 1:
        raise ValueError('declare source, boundaries or random with one cube participant')
    generator = torch.Generator().manual_seed(case['seed'])
    if dataset == 'boundaries':
        xf, yf = torch.zeros((M, KF), dtype=torch.float16), torch.zeros((N, KF), dtype=torch.float16)
        xi, yi = torch.zeros((M, KI), dtype=torch.int8), torch.zeros((N, KI), dtype=torch.int8)
        values = [-1024, -529, -513, -511, -257, -255, -17, -3, -1, 0, 1, 3, 15, 237,
            239, 253, 255, 257, 493, 495, 509, 511, 513, 1024, -5, -7, 5, 7, 9, 11, 13, 17]
        xf[:, 0] = torch.tensor([1 if row % 2 == 0 else -1 for row in range(M)])
        yf[:, 0] = torch.tensor(values)
        xi[:, 0] = torch.tensor([1, 3, 127, -128] * 4, dtype=torch.int8)
        yi[:, 0] = torch.tensor([-128, -127, -3, -1, 0, 1, 3, 127] * 4, dtype=torch.int8)
    else:
        bound = 3 if dataset == 'source' else 8
        # The source seed-0 generator consumes the FP16 pair before the INT8 pair.
        xf = torch.randint(-bound, bound + 1, (M, KF), generator=generator).to(torch.float16)
        yf = torch.randint(-bound, bound + 1, (N, KF), generator=generator).to(torch.float16)
        xi = torch.randint(-bound, bound + 1, (M, KI), generator=generator, dtype=torch.int8)
        yi = torch.randint(-bound, bound + 1, (N, KI), generator=generator, dtype=torch.int8)
    inputs = {'xf': xf, 'yf': yf, 'xi': xi, 'yi': yi}
    validate(inputs)
    return inputs


def validate(inputs):
    fields = {'xf': (torch.float16, (M, KF)), 'yf': (torch.float16, (N, KF)),
        'xi': (torch.int8, (M, KI)), 'yi': (torch.int8, (N, KI))}
    if set(inputs) != set(fields):
        raise ValueError('all four typed operands are required')
    for name, (dtype, shape) in fields.items():
        value = inputs[name]
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.shape != shape or value.device.type != 'cpu' or not value.is_contiguous():
            raise ValueError('require the declared contiguous CPU tensor dtype and shape')
        if dtype == torch.float16 and (not bool(torch.isfinite(value).all()) or not bool((value.abs() <= 1024).all()) or not torch.equal(value, value.round())):
            raise ValueError('FP16 operands must be finite integers in [-1024,1024]')


def reference(inputs):
    validate(inputs)
    floating = inputs['xf'].double() @ inputs['yf'].double().T
    integer = inputs['xi'].long() @ inputs['yi'].long().T
    result = {}
    for name, product, signed, scale, offset in [
        ('qf_i8', floating, True, .5, 8), ('qf_u8', floating, False, .5, 8), ('rq_i8', integer, True, .5, 0)]:
        result[name] = torch.tensor([[quant_byte(value, signed, scale, offset) for value in row]
            for row in product.tolist()], dtype=torch.uint8)
    result['deq_f16'] = torch.tensor([[byte for value in row for byte in half_bytes(value * fp19(.25))]
        for row in integer.tolist()], dtype=torch.uint8)
    return result
