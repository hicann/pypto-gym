# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generated inputs and independent logical/bit references; imports no DSL."""

import math

import torch

from .fp_bits import LATTICES, fp4_e1m2_code, fp8_code
from .hif8_bits import boundary_words, decode_code, expected_code

FP4_SOURCE = [-3, -1.875, -1.75, -1.625, -1.5, -1.375, -1.25, -1.125, -1,
    -.875, -.75, -.625, -.5, -.375, -.25, -.125, 0, .125, .25, .375, .5, .625,
    .75, .875, 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 3]
MODES = {'uint2_pack', 'uint2_unpack', 'fp4_pack', 'fp8_dual', 'hif8_to_f32', 'hif8_to_f16',
    'f32_to_hif8_ta', 'f32_to_hif8_hybrid', 'f16_to_hif8_ta', 'f16_to_hif8_hybrid'}


def tensor_from_words(words, bits):
    sign = 1 << (bits - 1)
    return torch.tensor([word if word < sign else word - (sign << 1) for word in words],
        dtype=torch.int32 if bits == 32 else torch.int16).view(torch.float32 if bits == 32 else torch.float16)


def fp8_boundaries():
    points = {0.0, 464.0, 61440.0, math.inf}
    for lattice in LATTICES.values():
        points.update(lattice)
        points.update((a + b) / 2 for a, b in zip(lattice, lattice[1:]))
    finite = torch.tensor(sorted(points - {math.inf}), dtype=torch.float32)
    values = torch.cat((torch.nextafter(finite, torch.full_like(finite, -math.inf)), finite,
        torch.nextafter(finite, torch.full_like(finite, math.inf))))
    values = torch.cat((values, -values, torch.tensor([math.inf, -math.inf, math.nan])))
    return torch.cat((values, torch.zeros(-len(values) % 64))).reshape(-1, 64)


def make_inputs(case):
    params = case['parameters']
    mode, dataset = params['mode'], params['dataset']
    generator = torch.Generator().manual_seed(case['seed'])
    inputs = {'mode': mode}
    if mode in ('uint2_pack', 'uint2_unpack'):
        rows = params['rows']
        carriers = (torch.arange(rows * 32, dtype=torch.int32) % 256).to(torch.uint8).reshape(rows, 32)
        if dataset == 'random':
            carriers = torch.randint(0, 256, (rows, 32), generator=generator, dtype=torch.uint8)
        if mode == 'uint2_pack':
            inputs['x'] = torch.tensor([[(int(v) >> (2 * k)) & 3 for v in row for k in range(4)]
                for row in carriers], dtype=torch.bfloat16)
        else:
            inputs['x'] = carriers
            fill = params['padding']
            inputs['padding'] = torch.full_like(carriers, fill) if fill >= 0 else torch.randint(
                0, 256, carriers.shape, generator=generator, dtype=torch.uint8)
    elif mode == 'fp4_pack':
        if dataset == 'finite_bits':
            words = [word for word in range(65536) if (word & 0x7F80) != 0x7F80]
            inputs['x'] = torch.tensor([word if word < 32768 else word - 65536 for word in words],
                dtype=torch.int16).view(torch.bfloat16).reshape(-1, 128)
        else:
            inputs['x'] = torch.tensor(FP4_SOURCE, dtype=torch.bfloat16).repeat(params['rows'] * 4).reshape(-1, 128)
    elif mode == 'fp8_dual':
        if dataset == 'boundaries':
            inputs['x'] = fp8_boundaries()
        else:
            inputs['x'] = (torch.randint(-32, 33, (params['rows'], 64), generator=generator).float() / 16)
    elif mode.startswith('hif8_to'):
        inputs['x'] = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    else:
        bits = 32 if mode.startswith('f32') else 16
        words = list(range(65536)) if dataset == 'all_half_bits' else boundary_words(bits)
        tile = 64 if bits == 32 else 128
        words += [0] * (-len(words) % tile)
        inputs['x'] = tensor_from_words(words, bits)
    validate(inputs)
    return inputs


def validate(inputs):
    mode, x = inputs.get('mode'), inputs.get('x')
    if mode not in MODES or not isinstance(x, torch.Tensor) or not x.is_contiguous():
        raise ValueError('declared mode and a contiguous tensor are required')
    if mode.startswith('hif8_to'):
        dtype, tile, rank = torch.uint8, 64 if mode.endswith('f32') else 128, 1
    elif mode.startswith(('f32_to', 'f16_to')):
        dtype, tile, rank = (torch.float32, 64, 1) if mode.startswith('f32') else (torch.float16, 128, 1)
    else:
        dtype = {'uint2_pack': torch.bfloat16, 'uint2_unpack': torch.uint8,
            'fp4_pack': torch.bfloat16, 'fp8_dual': torch.float32}[mode]
        tile, rank = (32 if mode == 'uint2_unpack' else 64 if mode == 'fp8_dual' else 128), 2
    if x.dtype != dtype or x.ndim != rank or x.numel() == 0 or x.numel() % tile:
        raise ValueError('dtype, rank and complete physical tile must match the selected entry')
    if rank == 2 and (x.shape[1] != tile or not 1 <= x.shape[0] <= 1024):
        raise ValueError('row modes declare 1..1024 complete rows')
    if mode == 'uint2_pack' and not bool(((x >= 0) & (x <= 3) & (x == x.trunc())).all()):
        raise ValueError('UINT2 packing accepts only exact BF16 values 0, 1, 2, 3')
    if mode == 'uint2_unpack':
        padding = inputs.get('padding')
        if not isinstance(padding, torch.Tensor) or padding.dtype != torch.uint8 or padding.shape != x.shape or not padding.is_contiguous():
            raise ValueError('UINT2 physical padding is an explicit contiguous UINT8 row')
    if mode == 'fp4_pack' and not bool(torch.isfinite(x).all()):
        raise ValueError('the preserved FP4 input domain is finite BF16')


def reference(inputs):
    validate(inputs)
    mode, x = inputs['mode'], inputs['x']
    if mode == 'uint2_pack':
        o = torch.tensor([[sum(int(row[i + k]) << (2 * k) for k in range(4))
            for i in range(0, 128, 4)] for row in x.tolist()], dtype=torch.uint8)
    elif mode == 'uint2_unpack':
        o = torch.tensor([[(int(v) >> (2 * k)) & 3 for v in row for k in range(4)]
            for row in x.tolist()], dtype=torch.bfloat16)
    elif mode == 'fp4_pack':
        o = torch.tensor([[fp4_e1m2_code(row[i]) | (fp4_e1m2_code(row[i + 1]) << 4)
            for i in range(0, 128, 2)] for row in x.tolist()], dtype=torch.uint8)
    elif mode == 'fp8_dual':
        o = torch.tensor([[[fp8_code(v, kind) for v in row] for row in x.tolist()]
            for kind in ('e5m2', 'e4m3')], dtype=torch.uint8)
    elif mode.startswith('hif8_to'):
        o = torch.tensor([decode_code(code) for code in x.tolist()],
            dtype=torch.float32 if mode.endswith('f32') else torch.float16)
    else:
        bits = 32 if mode.startswith('f32') else 16
        words = [word & ((1 << bits) - 1) for word in x.view(torch.int32 if bits == 32 else torch.int16).tolist()]
        o = torch.tensor([expected_code(word, bits, 'hybrid' if mode.endswith('hybrid') else 'round')
            for word in words], dtype=torch.uint8)
    return {'o': o}
