# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent FIX output fields and HiFloat8 bit arithmetic after host products."""

import struct

import torch

from .fp_bits import fp8_code
from .hif8_bits import expected_code

NAMES = ['scaled_bf16', 'scaled_f32', 'scaled_e4m3', 'deq_bf16', 'hif8_ta', 'hif8_hybrid']


def word32(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def bf16_bytes(value):
    word = word32(value)
    rounded = ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF
    return [rounded & 255, rounded >> 8]


def fp19(value):
    return struct.unpack('<f', struct.pack('<I', word32(value) & 0xFFFFE000))[0]


def geometry(case):
    dataset = case['parameters']['dataset']
    if dataset not in ['source', 'boundaries', 'corrected'] or case.get('block_dim', 1) != 1:
        raise ValueError('declare source, boundaries or corrected with one cube participant')
    return (32, 64, 64, 64) if dataset == 'corrected' else (16, 32, 48, 64)


def make_inputs(case):
    m, n, kf, ki = geometry(case)
    generator = torch.Generator().manual_seed(case['seed'])
    dataset = case['parameters']['dataset']
    if dataset == 'boundaries':
        xf, yf = torch.zeros((m, kf), dtype=torch.float16), torch.zeros((n, kf), dtype=torch.float16)
        xf[:, 0] = torch.tensor([1 if row % 2 == 0 else -1 for row in range(m)])
        xf[:, 1] = xf[:, 0] * 2 ** -10
        values = [-512, -256, -128, -32, -24, -20, -16, -1.00390625, -1, -.5009765625,
            -2 ** -10, 0, 2 ** -10, .5, 1, 1.00390625, 1.01171875, 15.5, 16, 20, 24,
            32, 128, 255, 256, 448, 464, 2 ** -22, -2 ** -22, 17, 21, 25]
        yf[:, 0] = torch.tensor([value * 2 for value in values], dtype=torch.float16)
        # Exact FP32 predecessor of 16 after the half-operand product and scale.
        yf[18, 1] = -(2 ** -9)
        hx, hy = xf.clone(), yf.clone()
        xi, yi = torch.zeros((m, ki), dtype=torch.int8), torch.zeros((n, ki), dtype=torch.int8)
        xi[:, 0] = torch.tensor([1, 3, 127, -128] * 4, dtype=torch.int8)
        yi[:, 0] = torch.tensor([-128, -127, -3, -1, 0, 1, 3, 127] * 4, dtype=torch.int8)
    else:
        if dataset == 'corrected':
            # Exact corrected _quant_case seed/shape/FP16 generation for its three
            # scaled output types; no recorded output accompanies that helper.
            xf = (torch.randn((m, kf), generator=generator) * .5).to(torch.float16)
            yf = (torch.randn((n, kf), generator=generator) * .5).to(torch.float16)
        else:
            xf = torch.randint(-3, 4, (m, kf), generator=generator).to(torch.float16)
            yf = torch.randint(-3, 4, (n, kf), generator=generator).to(torch.float16)
        xi = torch.randint(-3, 4, (m, ki), generator=generator, dtype=torch.int8)
        yi = torch.randint(-3, 4, (n, ki), generator=generator, dtype=torch.int8)
        if dataset == 'source':
            hx, hy = xf.clone(), yf.clone()
        else:
            hx = torch.randint(-3, 4, (m, kf), generator=generator).to(torch.float16)
            hy = torch.randint(-3, 4, (n, kf), generator=generator).to(torch.float16)
    inputs = {'xf': xf, 'yf': yf, 'xi': xi, 'yi': yi, 'hx': hx, 'hy': hy, 'geometry': (m, n, kf, ki)}
    validate(inputs)
    return inputs


def validate(inputs):
    if set(inputs) != {'xf', 'yf', 'xi', 'yi', 'hx', 'hy', 'geometry'}:
        raise ValueError('all declared operands and geometry are required')
    if inputs['geometry'] not in [(16, 32, 48, 64), (32, 64, 64, 64)]:
        raise ValueError('only the original and corrected source geometries are declared')
    m, n, kf, ki = inputs['geometry']
    fields = {'xf': (torch.float16, (m, kf)), 'yf': (torch.float16, (n, kf)),
        'hx': (torch.float16, (m, kf)), 'hy': (torch.float16, (n, kf)),
        'xi': (torch.int8, (m, ki)), 'yi': (torch.int8, (n, ki))}
    for name, (dtype, shape) in fields.items():
        value = inputs[name]
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.shape != shape or value.device.type != 'cpu' or not value.is_contiguous():
            raise ValueError('require the declared contiguous CPU tensor dtype and shape')
        if dtype == torch.float16 and (not bool(torch.isfinite(value).all()) or not bool((value.abs() <= 1024).all())):
            raise ValueError('FP16 operands must be finite and bounded by 1024')


def reference(inputs):
    validate(inputs)
    floating = inputs['xf'].double() @ inputs['yf'].double().T
    integer = inputs['xi'].long() @ inputs['yi'].long().T
    hif = inputs['hx'].double() @ inputs['hy'].double().T
    # The L0C accumulation is FP32 before the exact power-of-two FIX scale.
    scaled = [[struct.unpack('<f', struct.pack('<f', value))[0] * fp19(.5) for value in row]
        for row in floating.tolist()]
    hif_words = [[word32(value * fp19(.5)) for value in row] for row in hif.tolist()]
    result = {'scaled_f32': torch.tensor(scaled, dtype=torch.float32)}
    result['scaled_bf16'] = torch.tensor([[byte for value in row for byte in bf16_bytes(value)]
        for row in scaled], dtype=torch.uint8)
    result['scaled_e4m3'] = torch.tensor([[fp8_code(value, 'e4m3') for value in row] for row in scaled], dtype=torch.uint8)
    result['deq_bf16'] = torch.tensor([[byte for value in row for byte in bf16_bytes(value * fp19(.5))]
        for row in integer.tolist()], dtype=torch.uint8)
    for name, mode in [('hif8_ta', 'round'), ('hif8_hybrid', 'hybrid')]:
        result[name] = torch.tensor([[expected_code(word, 32, mode) for word in row] for row in hif_words], dtype=torch.uint8)
    return result
