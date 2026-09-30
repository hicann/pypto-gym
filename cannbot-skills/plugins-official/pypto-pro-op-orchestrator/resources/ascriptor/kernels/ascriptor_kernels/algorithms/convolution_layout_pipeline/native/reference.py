# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent FP64 NCHW convolution, the host packing the cube's ABI expects, and the padded-M rows the sliding window keeps computing."""

import torch
import torch.nn.functional as F

def geometry(p):
    """Everything the four static bodies bake in, recovered from the case's mode: the filter,
    the pad/stride/dilation descriptor, the M tile, the fixed M_PAD and batch count, and the
    worst-case capacities the L1 tensors are sized against. It also refuses the geometries the
    hardware cannot serve -- notably `(M_PAD rounded up to TILE_M) > align16(HO*WO)`, which is
    the historical A5 large 14x14 / M_PAD 224 defect: that M loop runs off the output plane."""
    mode = p['mode']
    if mode in ('basic', 'bias'):
        capacity = (32, 8, 8, 48)
        kh, pad, stride, dilation, tile, mpad, batches = 3, (1, 1, 1, 1), 1, 1, 32, 64, 1
    elif mode == 'dilation':
        capacity = (16, 12, 12, 32)
        kh, pad, stride, dilation, tile, mpad, batches = 3, (2, 2, 2, 2), 2, 2, 16, 48, 1
    elif mode == 'large':
        capacity = (32, 16, 16, 64)
        kh, pad, stride, dilation, tile, mpad, batches = 4, (1, 2, 1, 2), 1, 1, 32, 224, 2
    else:
        raise ValueError('Unknown preserved native convolution mode')
    for key, cap in zip(('C', 'H', 'W', 'COUT'), capacity):
        if type(p[key]) is not int or not 1 <= p[key] <= cap:
            raise ValueError(f'{key} exceeds the fixed source capacity')
    if p['B'] != batches or p['M_PAD'] != mpad:
        raise ValueError('Batch and physical output plane must match the corrected fixed signature')
    ho = (p['H'] + pad[2] + pad[3] - dilation * (kh - 1) - 1) // stride + 1
    wo = (p['W'] + pad[0] + pad[1] - dilation * (kh - 1) - 1) // stride + 1
    if (ho, wo) != (p['HO'], p['WO']) or ho < 1 or wo < 1 or ho * wo > mpad:
        raise ValueError('Logical output shape disagrees with convolution geometry')
    if ((mpad + tile - 1) // tile) * tile > ((ho * wo + 15) // 16) * 16:
        raise ValueError('The full source M loop leaves the hardware output plane')
    if mode == 'large' and p['COUT'] != 64:
        raise ValueError('The retained two-batch large valuation has a64-row weight signature')
    c1max = (capacity[0] + 15) // 16
    return {'capacity': capacity, 'kh': kh, 'pad': pad, 'stride': stride, 'dilation': dilation,
            'fm_plane': c1max * capacity[1] * capacity[2], 'k_max': c1max * kh * kh * 16,
            'cout_pad': (capacity[3] + 15) // 16 * 16}


def pack(inputs):
    """Logical NCHW/OIHW to the physical ABI the cube reads: NC1HWC0 image planes zero-padded
    to the static per-batch stride, weights reordered Cout, C1, Kh, Kw, C0 and zero-padded to
    the rounded Cout, and the bias widened to the same rounded Cout. Built by indexing, not by
    any kernel."""
    p = inputs['parameters']
    g = geometry(p)
    c1 = (p['C'] + 15) // 16
    padded_x = torch.zeros((p['B'], c1 * 16, p['H'], p['W']), dtype=torch.float16)
    padded_x[:, :p['C']] = inputs['image']
    planes = padded_x.reshape(p['B'], c1, 16, p['H'], p['W']).permute(0, 1, 3, 4, 2).contiguous()
    data = torch.zeros((p['B'], g['fm_plane'], 16), dtype=torch.float16)
    data[:, :c1 * p['H'] * p['W']] = planes.reshape(p['B'], -1, 16)
    padded_w = torch.zeros((p['COUT'], c1 * 16, g['kh'], g['kh']), dtype=torch.float16)
    padded_w[:, :p['C']] = inputs['weights']
    reordered = padded_w.reshape(p['COUT'], c1, 16, g['kh'], g['kh']).permute(0, 1, 3, 4, 2).contiguous()
    weight = torch.zeros((g['cout_pad'], g['k_max']), dtype=torch.float16)
    weight[:p['COUT'], :c1 * g['kh'] * g['kh'] * 16] = reordered.reshape(p['COUT'], -1)
    bias = torch.zeros((1, g['cout_pad']), dtype=torch.float32)
    bias[0, :p['COUT']] = inputs['bias']
    return data.reshape(-1, 16), weight, bias


def make_inputs(case):
    p = dict(case['parameters'])
    g = geometry(p)
    if case.get('block_dim', 1) != 1:
        raise ValueError('These preserved native cases use one cube core')
    generator = torch.Generator().manual_seed(case['seed'])
    xshape = (p['B'], p['C'], p['H'], p['W'])
    wshape = (p['COUT'], p['C'], g['kh'], g['kh'])
    if p['pattern'] == 'source_random':
        x = torch.randn(xshape, generator=generator, dtype=torch.float16)
        w = torch.randn(wshape, generator=generator, dtype=torch.float16)
    elif p['pattern'] == 'integer':
        x = torch.randint(-2, 3, xshape, generator=generator).half()
        w = torch.randint(-2, 3, wshape, generator=generator).half()
    elif p['pattern'] in ('random', 'bias_only'):
        scale = 0.1 if p['mode'] == 'large' else 0.5
        x = (torch.randn(xshape, generator=generator) * scale).half()
        w = (torch.randn(wshape, generator=generator) * scale).half()
        if p['pattern'] == 'bias_only':
            if p['mode'] != 'bias':
                raise ValueError('The bias-only control requires the bias entry')
            x.zero_()
    else:
        raise ValueError('Unknown generated input pattern')
    if p['pattern'] == 'source_random' and p['mode'] == 'bias':
        bias = torch.randn((p['COUT'],), generator=generator, dtype=torch.float32)
    else:
        bias = torch.linspace(-1, 1, p['COUT']) if p['mode'] == 'bias' else torch.zeros(p['COUT'])
    result = {'image': x, 'weights': w, 'bias': bias, 'parameters': p, 'block_dim': 1}
    result['data'], result['packed_weights'], result['packed_bias'] = pack(result)
    return result


def reference(inputs):
    """Two expected tensors. `output` is the logical NCHW result. `physical_nz` is every cell
    the cube actually computes, including the M_PAD rows past HO*WO and the zero-padded
    channels up to the rounded Cout -- see the D-223 note below. Both come from one FP64
    convolution over a deliberately over-padded image, never from the device's tile order."""
    p = inputs['parameters']
    g = geometry(p)
    # D-223: padded M rows keep sliding the window. Extra bottom zeros express
    # the same mathematical windows, independently of the device's load3d code.
    extra_rows = max(0, (p['M_PAD'] + p['WO'] - 1) // p['WO'] - p['HO'])
    left, right, top, bottom = g['pad']
    padded = F.pad(inputs['image'].double(), (left, right, top, bottom + extra_rows * g['stride']))
    computed_n = (p['COUT'] + 15) // 16 * 16
    weight = torch.zeros((computed_n, p['C'], g['kh'], g['kh']), dtype=torch.float64)
    weight[:p['COUT']] = inputs['weights'].double()
    bias = torch.zeros(computed_n, dtype=torch.float64)
    bias[:p['COUT']] = inputs['bias'].double()
    extended = F.conv2d(padded, weight, bias=bias, stride=g['stride'], dilation=g['dilation']).float()
    rows = extended.reshape(p['B'], computed_n, -1)[:, :, :p['M_PAD']]
    physical = torch.empty((p['B'], computed_n // 16 * p['M_PAD'], 16), dtype=torch.float32)
    for channel in range(computed_n):
        physical[:, channel // 16 * p['M_PAD']:(channel // 16 + 1) * p['M_PAD'], channel % 16] = rows[:, channel]
    logical = extended[:, :p['COUT'], :p['HO'], :p['WO']].contiguous()
    return {'output': logical, 'physical_nz': physical.reshape(-1, 16)}
