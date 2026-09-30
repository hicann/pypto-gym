# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent NC1HWC0 indexing in Torch, plus the host-side physical packing the kernel is fed."""

import torch

def geometry(p):
    """Derive the physical extents from the case's logical dimensions: C1 is the rounded
    16-channel block count, pad_hw rounds HW up to 16, and `channels` differs per variant --
    host_pad is handed the full C1*16 channel grid, ub_pad only the C real channels. The two
    static input capacities (3072 and 1920) are the source's own GM signatures."""
    if p['variant'] not in ('host_pad', 'ub_pad'):
        raise ValueError('Unknown channel-padding variant')
    for name in ('B', 'C', 'HW'):
        if type(p[name]) is not int or p[name] <= 0:
            raise ValueError('Layout dimensions must be positive integers')
    b, c, hw = (p[name] for name in ('B', 'C', 'HW'))
    c1, padded_hw = (c + 15) // 16, (hw + 15) // 16 * 16
    if type(p['C1']) is not int or p['C1'] != c1:
        raise ValueError('C1 must equal the rounded16-channel block count')
    input_capacity = 3072 if p['variant'] == 'host_pad' else 1920
    channels = c1 * 16 if p['variant'] == 'host_pad' else c
    if b * channels * padded_hw > input_capacity or b * c1 * hw > 144:
        raise ValueError('Shape exceeds the corrected source allocation')
    return c1, padded_hw, channels, input_capacity


def pack(image, p):
    _, padded_hw, channels, capacity = geometry(p)
    grid = torch.zeros((p['B'], channels, padded_hw), dtype=torch.float16)
    grid[:, :p['C'], :p['HW']] = image
    data = torch.zeros((capacity, 1), dtype=torch.float16)
    data[:grid.numel(), 0] = grid.reshape(-1)
    return data


def make_inputs(case):
    p = dict(case['parameters'])
    geometry(p)
    if case.get('block_dim', 1) != 1:
        raise ValueError('The preserved two-vector partition uses one MIX core group')
    generator = torch.Generator().manual_seed(case['seed'])
    sampling = p.get('sampling', 'rounded_fp32')
    if sampling == 'source_fp16':
        image = torch.randn((p['B'], p['C'], p['HW']), generator=generator, dtype=torch.float16)
    elif sampling == 'rounded_fp32':
        image = torch.randn((p['B'], p['C'], p['HW']), generator=generator).half()
    else:
        raise ValueError('Unknown input sampling scheme')
    return {'image': image, 'data': pack(image, p), 'parameters': p, 'block_dim': 1}


def reference(inputs):
    """Scatter channel c of the logical NCHW image to block c//16, lane c%16 of NC1HWC0, by
    indexing rather than by any transpose primitive. Channels past C are never written, so the
    zeros they keep are the reference's claim about the channel tail -- not an artefact.
    Spatial padding does not appear at all: the result has exactly HW rows per plane."""
    p = inputs['parameters']
    c1, _, _, _ = geometry(p)
    result = torch.zeros((p['B'], c1, p['HW'], 16), dtype=torch.float16)
    for channel in range(p['C']):
        result[:, channel // 16, :, channel % 16] = inputs['image'][:, channel]
    return {'output': result.reshape(-1, 16)}
