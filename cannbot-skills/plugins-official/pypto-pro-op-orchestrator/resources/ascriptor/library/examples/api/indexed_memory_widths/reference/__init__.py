# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent byte-list addressing preserves payloads without numerical casts."""

import random

import torch

from .plans import PLANS


def make_inputs(case):
    dataset, padding = case['parameters']['dataset'], case['parameters']['padding']
    generator = random.Random(case['seed'])
    inputs = {'buffers': {}}
    for name, src_type, src_count, idx_type, idx_count, _, dst_count, width, active, operation in PLANS:
        size = src_count * width
        raw = [(i * 37 + i // 17 + 131) % 256 for i in range(size)]
        source = torch.tensor(raw, dtype=torch.uint8).view(getattr(torch, src_type)).reshape(1, src_count)
        limit = 16 if operation == 'blocks' else src_count if operation == 'gather' else dst_count
        if dataset == 'reverse':
            indices = list(reversed(range(limit)))[:active]
        elif dataset == 'repeat_gather' and operation in ['gather', 'blocks']:
            indices = [0 if i % 2 == 0 else limit - 1 for i in range(active)]
        else:
            indices = generator.sample(range(limit), active)
        if operation == 'blocks':
            indices = [value * 32 for value in indices]
        indices += [padding] * (idx_count - active)
        inputs['buffers'][name] = {'src': source, 'idx': torch.tensor(indices, dtype=getattr(torch, idx_type)).reshape(1, idx_count)}
    validate(inputs)
    return inputs


def validate(inputs):
    buffers = inputs.get('buffers')
    if not isinstance(buffers, dict) or set(buffers) != {plan[0] for plan in PLANS}:
        raise ValueError('all declared typed movement buffers are required')
    for name, src_type, src_count, idx_type, idx_count, _, dst_count, width, active, operation in PLANS:
        item = buffers[name]
        for key, dtype, count in [('src', src_type, src_count), ('idx', idx_type, idx_count)]:
            value = item.get(key)
            if not isinstance(value, torch.Tensor) or value.dtype != getattr(torch, dtype) or value.shape != (1, count) or not value.is_contiguous():
                raise ValueError('this source family declares one complete typed source/index row')
        indices = item['idx'].flatten().tolist()[:active]
        limit = src_count if operation == 'gather' else dst_count
        if operation == 'blocks':
            if any(i < 0 or i % 32 or i + 32 > src_count * width for i in indices):
                raise ValueError('active GatherB indices are aligned 32-byte offsets within the source')
        elif any(i < 0 or i >= limit for i in indices):
            raise ValueError('active element index is outside the declared source/destination')
        if operation == 'scatter' and len(set(indices)) != active:
            raise ValueError('source scatter cases require unique destinations; duplicate-writer ordering is undeclared')


def reference(inputs):
    validate(inputs)
    output = {}
    for name, _, _, _, _, _, _, width, active, operation in PLANS:
        item = inputs['buffers'][name]
        source = item['src'].view(torch.uint8).flatten().tolist()
        indices = item['idx'].flatten().tolist()[:active]
        result = [0] * 256
        for lane, index in enumerate(indices):
            if operation == 'blocks':
                result[lane * 32:(lane + 1) * 32] = source[index:index + 32]
            elif operation == 'gather':
                if width == 1:
                    result[lane * 2] = source[index]
                    result[lane * 2 + 1] = 0
                else:
                    result[lane * width:(lane + 1) * width] = source[index * width:(index + 1) * width]
            else:
                offset = lane * 2 if width == 1 else lane * width
                result[index * width:(index + 1) * width] = source[offset:offset + width]
        output[name] = torch.tensor(result, dtype=torch.uint8).reshape(1, 256)
    return output
