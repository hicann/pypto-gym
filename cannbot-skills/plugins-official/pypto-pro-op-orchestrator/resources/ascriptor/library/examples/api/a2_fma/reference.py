# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent high-precision real multiply-add of the actual typed inputs."""

import torch

TYPES = {'f32': (torch.float32, torch.float32), 'f16': (torch.float16, torch.float16),
    'mixed': (torch.float16, torch.float32)}


def make_inputs(case):
    n, tile = case['parameters']['n'], case['parameters']['tile_len']
    generator = torch.Generator().manual_seed(case['seed'])
    inputs = {'tile_len': tile}
    for name, (source, accumulator) in TYPES.items():
        for arg, dtype in [('a', source), ('b', source), ('c', accumulator)]:
            if case['parameters']['dataset'] == 'dyadic':
                value = torch.randint(-8, 9, (1, n), generator=generator).to(dtype) / 8
            else:
                value = torch.empty((1, n), dtype=dtype).uniform_(-1, 1, generator=generator)
            inputs[name + '_' + arg] = value
    validate(inputs)
    return inputs


def validate(inputs):
    tile = inputs.get('tile_len')
    if isinstance(tile, bool) or not isinstance(tile, int) or not 64 <= tile <= 5120 or tile % 64:
        raise ValueError('tile_len is an integer multiple of 64 in [64,5120]')
    size = None
    for name, (source, accumulator) in TYPES.items():
        for arg, dtype in [('a', source), ('b', source), ('c', accumulator)]:
            value = inputs.get(name + '_' + arg)
            if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.ndim != 2 or value.shape[0] != 1 or not value.is_contiguous():
                raise ValueError('source/accumulator dtypes and contiguous [1,n] arrays must agree')
            size = value.shape[1] if size is None else size
            if value.shape[1] != size or not 1 <= size <= 40000:
                raise ValueError('all inputs share a positive n at most 40000')
            if not bool(torch.isfinite(value).all()) or not bool((value.abs() <= 2).all()):
                raise ValueError('finite values in [-2,2] define the bounded source reference domain')


def reference(inputs):
    validate(inputs)
    results = {}
    for name, (_, dtype) in TYPES.items():
        a, b, c = [inputs[name + '_' + arg].double() for arg in ['a', 'b', 'c']]
        results[name] = (a * b + c).to(dtype)
    return results
