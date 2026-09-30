# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Python integer arithmetic, width wrapping and ordering references."""

import random

import torch

from .plans import PLANS, output_name

MOD = 1 << 64


def signed(value, bits=64):
    value %= 1 << bits
    return value if value < 1 << (bits - 1) else value - (1 << bits)


def tensor(values, rows, dtype):
    if dtype == torch.uint64:
        return torch.tensor([signed(value) for value in values], dtype=torch.int64).reshape(rows, -1).view(torch.uint64)
    return torch.tensor(values, dtype=dtype).reshape(rows, -1)


def make_inputs(case):
    rows, dataset = case['parameters']['rows'], case['parameters']['dataset']
    count = rows * 32
    generator = random.Random(case['seed'])
    if dataset == 'source':
        na = [(i - count // 2) * 262147 for i in range(count)]
        nb = [((i * 7) % 4099 - 2048) * 131101 for i in range(count)]
        ea = [(i - count // 2) * ((1 << 33) + 5) for i in range(count)]
        eb = [((i * 7) % 101 - 50) * ((1 << 37) + 3) for i in range(count)]
        ua = [(i * ((1 << 58) + 1315423911) + 7) % MOD for i in range(count)]
        ub = [(i * ((1 << 47) + 2654435761) + (1 << 63)) % MOD for i in range(count)]
    elif dataset == 'edges':
        values = [-(1 << 28) - 1, -(1 << 24) - 1, -1, 0, 1, (1 << 24) + 1, 12345678, 12345679]
        na = [values[i % len(values)] for i in range(count)]
        nb = [values[(i + 3) % len(values)] for i in range(count)]
        values = [-(1 << 43) - 1, -400000001, -400000000, -1, 0, 500000000, 500000001, (1 << 43) + 1]
        ea = [values[i % len(values)] for i in range(count)]
        eb = [values[(i + 3) % len(values)] for i in range(count)]
        values = [0, 1, (1 << 63) - 1, 1 << 63, (1 << 63) + 1, MOD - 1, MOD - 2, (1 << 40) + 7]
        ua = [values[i % len(values)] for i in range(count)]
        ub = [values[(i + 3) % len(values)] for i in range(count)]
    else:
        na, nb = [[generator.randint(-(1 << 28), 1 << 28) for _ in range(count)] for _ in range(2)]
        ea, eb = [[generator.randint(-(1 << 43), 1 << 43) for _ in range(count)] for _ in range(2)]
        ua, ub = [[generator.getrandbits(64) for _ in range(count)] for _ in range(2)]
    inputs = {name: tensor(values, rows, dtype) for name, values, dtype in [
        ('native_a', na, torch.int64), ('native_b', nb, torch.int64), ('extended_a', ea, torch.int64),
        ('extended_b', eb, torch.int64), ('unsigned_a', ua, torch.uint64), ('unsigned_b', ub, torch.uint64)]}
    for bits, lanes in [(32, 64), (64, 32)]:
        if dataset == 'source':
            values = [signed(i * 0x04010203 - 0x60000000, 32) for i in range(lanes)] if bits == 32 else [i * 0x0102030405 - (1 << 44) for i in range(lanes)]
        elif dataset == 'edges':
            values = [signed(value, bits) for value in [-(1 << (bits - 1)), (1 << (bits - 1)) - 1, -1, 1, 0, 12345, -12345, 1 << (bits - 2)] * (lanes // 8)]
        else:
            values = [signed(generator.getrandbits(bits), bits) for _ in range(lanes)]
        inputs[f'shift{bits}_data'] = tensor(values, 1, torch.int32 if bits == 32 else torch.int64)
        inputs[f'shift{bits}_count'] = tensor([i % 31 for i in range(lanes)], 1, torch.int32 if bits == 32 else torch.int64)
    validate(inputs)
    return inputs


def validate(inputs):
    rows = None
    for name in ['native_a', 'native_b', 'extended_a', 'extended_b', 'unsigned_a', 'unsigned_b']:
        value = inputs.get(name)
        dtype = torch.uint64 if name.startswith('unsigned') else torch.int64
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.ndim != 2 or value.shape[1] != 32 or not value.is_contiguous():
            raise ValueError('complete contiguous 32-lane signed/unsigned rows are required')
        rows = value.shape[0] if rows is None else rows
        if value.shape[0] != rows or not 1 <= rows <= 64:
            raise ValueError('all main inputs share 1..64 rows')
        limit = (1 << 29) if name.startswith('native') else (1 << 47)
        if not name.startswith('unsigned') and any(abs(v) > limit for v in value.flatten().tolist()):
            raise ValueError('signed input exceeds the source family no-overflow domain')
    for bits, lanes in [(32, 64), (64, 32)]:
        for role in ['data', 'count']:
            value = inputs.get(f'shift{bits}_{role}')
            if not isinstance(value, torch.Tensor) or value.dtype != (torch.int32 if bits == 32 else torch.int64) or value.shape != (1, lanes) or not value.is_contiguous():
                raise ValueError('variable shifts preserve one complete row at the matching signed width')
            if role == 'count' and not bool(((value >= 0) & (value <= 30)).all()):
                raise ValueError('per-lane shifts are integral counts in 0..30')


def reference(inputs):
    validate(inputs)
    results = {}
    for module in ['native', 'extended', 'unsigned']:
        a, b = inputs[module + '_a'], inputs[module + '_b']
        av, bv = a.flatten().tolist(), b.flatten().tolist()
        wrap = module == 'unsigned'
        values = {}
        if module != 'extended':
            values.update(add=[x + y for x, y in zip(av, bv)], sub=[x - y for x, y in zip(av, bv)],
                mul=[x * y for x, y in zip(av, bv)], shift_left=[x << 5 for x in av], shift_right=[x >> 5 for x in av],
                square_plus_b=[x * x + y for x, y in zip(av, bv)], select_reg=[x if x > y else y for x, y in zip(av, bv)],
                select_scalar=[x if x > ((1 << 63) if wrap else 12345678) else y for x, y in zip(av, bv)],
                dup=[(1 << 63) + (1 << 40) + 7 if wrap else (1 << 40) + 12345] * len(av))
            values['and'] = [x & y for x, y in zip(av, bv)]
            values['or'] = [x | y for x, y in zip(av, bv)]
            values['xor'] = [x ^ y for x, y in zip(av, bv)]
            values['not'] = [~x for x in av]
            values['abssub'] = [abs(x - y) for x, y in zip(av, bv)]
        else:
            values.update(min=[min(x, y) for x, y in zip(av, bv)], max=[max(x, y) for x, y in zip(av, bv)],
                neg=[-x for x in av], abs=[abs(x) for x in av], adds=[x + 1000000007 for x in av],
                muls=[x * 3 for x in av], maxs=[max(x, 500000000) for x in av], mins=[min(x, -400000000) for x in av],
                copy=av, axpy=[y + x * 5 for x, y in zip(av, bv)])
            for name, reduce in [('cadd', sum), ('cmax', max), ('cmin', min)]:
                results['extended_' + name] = tensor([reduce(row) for row in a.tolist()], a.shape[0], torch.int64)
        for name, sequence in values.items():
            results[module + '_' + name] = tensor([v % MOD for v in sequence] if wrap else sequence,
                a.shape[0], torch.uint64 if wrap else torch.int64)
    for bits in [32, 64]:
        data = inputs[f'shift{bits}_data'].flatten().tolist()
        count = inputs[f'shift{bits}_count'].flatten().tolist()
        for name in ['left', 'right']:
            values = [signed(x << s, bits) if name == 'left' else x >> s for x, s in zip(data, count)]
            results[f'variable_i{bits}_{name}'] = tensor(values, 1, torch.int32 if bits == 32 else torch.int64)
    return results
