# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent two's-complement nibble packing, decoding and integer products."""

import torch


def pack(values, padding):
    rows, logical_k = values.shape
    count = (logical_k + 7) // 8
    output = []
    for row in values.tolist():
        padded = row + [padding] * (-logical_k % 8)
        words = [sum((padded[i + j] & 15) << (4 * j) for j in range(8)) for i in range(0, count * 8, 8)]
        output.append([word if word < 1 << 31 else word - (1 << 32) for word in words])
    return torch.tensor(output, dtype=torch.int32)


def unpack(carriers, logical_k):
    rows = []
    for row in carriers.tolist():
        values = []
        for word in row:
            for shift in range(0, 32, 4):
                nibble = (word >> shift) & 15
                values.append(nibble - 16 if nibble >= 8 else nibble)
        rows.append(values[:logical_k])
    return torch.tensor(rows, dtype=torch.int64)


def make_inputs(case):
    params = case['parameters']
    m, n, k, padding = (params[key] for key in ['M', 'N', 'K', 'padding'])
    generator = torch.Generator().manual_seed(case['seed'])
    a, b = torch.randint(-8, 8, (m, k), generator=generator), torch.randint(-8, 8, (n, k), generator=generator)
    if params.get('pattern') == 'all_nibbles':
        a = ((torch.arange(m * k).reshape(m, k) % 16) - 8)
        b = (((torch.arange(n * k).reshape(n, k) * 7 + 3) % 16) - 8)
    inputs = {'x': pack(a, padding), 'y': pack(b, padding), 'K': k}
    assert torch.equal(unpack(inputs['x'], k), a)
    assert torch.equal(unpack(inputs['y'], k), b)
    validate(inputs)
    return inputs


def validate(inputs):
    k = inputs.get('K')
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 257:
        raise ValueError('logical K is an integer in [1,257]')
    for key in ['x', 'y']:
        value = inputs.get(key)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.int32 or value.ndim != 2 or not value.is_contiguous():
            raise ValueError('public signed-int4 storage is contiguous INT32 carrier rows')
        if not 1 <= value.shape[0] <= 129 or value.shape[1] != (k + 7) // 8:
            raise ValueError('M/N in[1,129] and exact ceil(K/8) carriers are required')


def reference(inputs):
    validate(inputs)
    a, b = unpack(inputs['x'], inputs['K']), unpack(inputs['y'], inputs['K'])
    return {'o': (a @ b.T).int()}
