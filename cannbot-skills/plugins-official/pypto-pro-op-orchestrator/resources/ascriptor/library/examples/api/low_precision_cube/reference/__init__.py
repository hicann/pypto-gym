# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent format fields and literal FP4 lattices feed high-precision products."""

import math

import torch

from .hif8_bits import POSITIVE, decode_code, expected_code
from .fp_bits import fp8_code

HIF_NAMES = [a + b + '_' + split for a in ['n', 't'] for b in ['n', 't'] for split in ['nosplit', 'splitn', 'splitk']]
FP4_E2M1 = [0., .5, 1., 1.5, 2., 3., 4., 6.]
FP4_E1M2 = [0., .25, .5, .75, 1., 1.25, 1.5, 1.75]


def hif_encode(values):
    words = [word & 0xFFFFFFFF for word in values.contiguous().view(torch.int32).flatten().tolist()]
    return torch.tensor([expected_code(word, 32, 'round') for word in words], dtype=torch.uint8).reshape_as(values)


def e5_decode(codes):
    output = []
    for word in codes.flatten().tolist():
        sign, exponent, fraction = word >> 7, (word & 127) >> 2, word & 3
        value = math.ldexp(fraction, -16) if exponent == 0 else math.ldexp(4 + fraction, exponent - 17)
        output.append(-value if sign else value)
    return torch.tensor(output, dtype=torch.float64).reshape_as(codes)


def fp4_encode(values, table):
    result = []
    for row in values.tolist():
        nibbles = []
        for value in row:
            code = min(range(8), key=lambda index: (abs(table[index] - abs(value)), index & 1))
            nibbles.append(code | (8 if value < 0 else 0))
        result.append([nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2)])
    return torch.tensor(result, dtype=torch.uint8)


def fp4_decode(values, table):
    decoded = []
    for row in values.tolist():
        output = []
        for word in row:
            for nibble in [word & 15, word >> 4]:
                value = table[nibble & 7]
                output.append(-value if nibble & 8 else value)
        decoded.append(output)
    return torch.tensor(decoded, dtype=torch.float64)


def make_inputs(case):
    dataset = case['parameters']['dataset']
    generator = torch.Generator().manual_seed(case['seed'])
    if dataset == 'finite_carriers':
        pool = sorted(POSITIVE.values())
        a = torch.tensor([[pool[(r * 64 + k) % len(pool)] | (128 if r % 2 and pool[(r * 64 + k) % len(pool)] else 0)
            for k in range(64)] for r in range(32)], dtype=torch.uint8)
        b = torch.tensor([[pool[(r * 17 + k * 3) % len(pool)] | (128 if r % 3 and pool[(r * 17 + k * 3) % len(pool)] else 0)
            for k in range(64)] for r in range(64)], dtype=torch.uint8)
        e5a = torch.tensor([[(r * 32 + k) % 124 | (128 if r % 2 else 0) for k in range(32)] for r in range(64)], dtype=torch.uint8)
        e5b = torch.tensor([[(r * 13 + k * 3) % 124 | (128 if r % 3 else 0) for k in range(32)] for r in range(48)], dtype=torch.uint8)
        mx_a = (torch.arange(512) % 256).to(torch.uint8).reshape(16, 32)
        mx_b = ((torch.arange(512) * 7 + 13) % 256).to(torch.uint8).reshape(16, 32)
    else:
        if dataset == 'source':
            a = hif_encode(torch.randn((32, 64), generator=generator) * .75)
            b = hif_encode(torch.randn((64, 64), generator=generator) * .75)
        else:
            a = hif_encode(torch.randint(-16, 17, (32, 64), generator=generator).float() / 8)
            b = hif_encode(torch.randint(-16, 17, (64, 64), generator=generator).float() / 8)
        e5a = torch.tensor([[fp8_code(x, 'e5m2') for x in row] for row in torch.randn((64, 32), generator=generator).tolist()], dtype=torch.uint8)
        e5b = torch.tensor([[fp8_code(x, 'e5m2') for x in row] for row in torch.randn((48, 32), generator=generator).tolist()], dtype=torch.uint8)
        a_values = ((torch.arange(1024).float().reshape(16, 64) * 5 + 3) % 25 - 12) / 12 * 6
        b_values = ((torch.arange(1024).float().reshape(16, 64) * 7 + 1) % 17 - 8) / 8 * 1.75
        mx_a, mx_b = fp4_encode(a_values, FP4_E2M1), fp4_encode(b_values, FP4_E1M2)
    if dataset == 'identity_scales':
        sa = sb = torch.full((16, 2), 127, dtype=torch.uint8)
    else:
        sa = torch.tensor([[126 + (r + group) % 3 for group in range(2)] for r in range(16)], dtype=torch.uint8)
        sb = torch.tensor([[126 + (2 * r + group + 1) % 3 for group in range(2)] for r in range(16)], dtype=torch.uint8)
    inputs = {'hif_a': a, 'hif_at': a.T.contiguous(), 'hif_b': b, 'hif_bt': b.T.contiguous(),
        'e5_a': e5a, 'e5_b': e5b, 'mx_a': mx_a, 'mx_b': mx_b,
        'scale_a': sa.reshape(1, 32).clone(), 'scale_b': sb.reshape(1, 32).clone()}
    validate(inputs)
    return inputs


def validate(inputs):
    shapes = {'hif_a': (32, 64), 'hif_at': (64, 32), 'hif_b': (64, 64), 'hif_bt': (64, 64),
        'e5_a': (64, 32), 'e5_b': (48, 32), 'mx_a': (16, 32), 'mx_b': (16, 32),
        'scale_a': (1, 32), 'scale_b': (1, 32)}
    if set(inputs) != set(shapes):
        raise ValueError('all declared carrier and scale inputs are required')
    for name, shape in shapes.items():
        value = inputs[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.shape != shape or not value.is_contiguous():
            raise ValueError('this unit declares fixed contiguous UINT8 carrier/scale shapes')
        if name.startswith('hif') and bool(((value == 0x80) | (value == 0x6F) | (value == 0xEF)).any()):
            raise ValueError('the numerical HiFloat8 cube domain is finite')
        if name.startswith('e5') and bool(((value & 127) >= 124).any()):
            raise ValueError('the numerical E5M2 cube domain is finite')
        if name.startswith('scale') and not bool(((value >= 126) & (value <= 128)).all()):
            raise ValueError('prepacked MX scales preserve the source exponent domain126..128')
    if not torch.equal(inputs['hif_at'], inputs['hif_a'].T) or not torch.equal(inputs['hif_bt'], inputs['hif_b'].T):
        raise ValueError('transposed physical inputs must denote the same logical operands')


def reference(inputs):
    validate(inputs)
    a = torch.tensor([decode_code(code) for code in inputs['hif_a'].flatten().tolist()], dtype=torch.float64).reshape(32, 64)
    b = torch.tensor([decode_code(code) for code in inputs['hif_b'].flatten().tolist()], dtype=torch.float64).reshape(64, 64)
    hif = (a @ b.T).float()
    e5 = (e5_decode(inputs['e5_a']) @ e5_decode(inputs['e5_b']).T).float()
    mx_a, mx_b = fp4_decode(inputs['mx_a'], FP4_E2M1), fp4_decode(inputs['mx_b'], FP4_E1M2)
    scales = []
    for name in ['scale_a', 'scale_b']:
        block = inputs[name].flatten().tolist()
        scales.append(torch.tensor([[math.ldexp(1., block[2 * r + k // 32] - 127) for k in range(64)]
            for r in range(16)], dtype=torch.float64))
    mx = ((mx_a * scales[0]) @ (mx_b * scales[1]).T).float()
    return {'hif_manual': hif.clone(), **{'hif_' + name: hif.clone() for name in HIF_NAMES}, 'e5m2': e5, 'mxfp4': mx}
