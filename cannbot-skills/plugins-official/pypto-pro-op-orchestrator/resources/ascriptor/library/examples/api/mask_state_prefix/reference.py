# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expected lanes come from Python bits and exclusive prefix arithmetic."""

import torch


def make_inputs(case):
    params = case['parameters']
    mode = params['mode']
    generator = torch.Generator().manual_seed(case['seed'])
    if mode == 'spr':
        return {'mode': mode, 'count': params['count'], 'pattern': params['pattern'],
            'dummy': torch.randint(-999, 999, (1, 8), generator=generator, dtype=torch.int32)}
    selector = torch.full((1, 64), -3, dtype=torch.int32)
    for index in params['active']:
        selector[0, index] = 2 + index % 5
    values = torch.randint(-100000, 100000, (2, 64), generator=generator, dtype=torch.int32)
    values[1] += 234567  # Two different old destinations must produce the same prefix counts.
    return {'mode': mode, 'selector': selector, 'values': values}


def validate(inputs):
    if inputs['mode'] == 'spr':
        if type(inputs['count']) is not int or not 0 <= inputs['count'] <= 64:
            raise ValueError('SPR count must be an integer in [0,64]')
        if type(inputs['pattern']) is not int or not 0 <= inputs['pattern'] < 1 << 64:
            raise ValueError('SPR pattern must be an unsigned 64-bit integer')
        specs = [('dummy', (1, 8))]
    elif inputs['mode'] == 'prefix':
        specs = [('selector', (1, 64)), ('values', (2, 64))]
    else:
        raise ValueError('unknown mask-state mode')
    for key, shape in specs:
        value = inputs[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.int32 or tuple(value.shape) != shape or not value.is_contiguous():
            raise ValueError(f'{key} must be contiguous int32{shape}')


def reference(inputs):
    validate(inputs)
    if inputs['mode'] == 'spr':
        count = [7 if lane < inputs['count'] else 0 for lane in range(64)]
        pattern = [7 if inputs['pattern'] & (1 << lane) else 0 for lane in range(64)]
        return {'o': torch.tensor([count, pattern, [7] * 64], dtype=torch.int32)}
    total = 0
    prefix = []
    for selector in inputs['selector'][0].tolist():
        prefix.append(total)
        total += selector > 0
    return {'o': torch.tensor([prefix, prefix], dtype=torch.int32)}
