# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent FP64 NCHW convolution plus a hand-indexed reference for each of the four stages."""

import struct
import torch
import torch.nn.functional as F

def make_inputs(case):
    p = dict(case['parameters'])
    if p.get('shape') != '1x64x32x32_128x64x3x3' or case.get('block_dim', 1) != 1:
        raise ValueError('The corrected source valuation requires the fixed shape and one cube/MIX group')
    generator = torch.Generator().manual_seed(case['seed'])
    sampling = p.get('sampling', 'rounded_fp32')
    if sampling == 'source_fp16':
        image = torch.randn((1, 64, 32, 32), generator=generator, dtype=torch.float16)
        weights = torch.randn((128, 64, 3, 3), generator=generator, dtype=torch.float16)
    elif sampling == 'rounded_fp32':
        image = torch.randn((1, 64, 32, 32), generator=generator).half()
        weights = torch.randn((128, 64, 3, 3), generator=generator).half()
    else:
        raise ValueError('Unknown input sampling scheme')
    if p['pattern'] == 'integer':
        image = torch.randint(-2, 3, image.shape, generator=generator).half()
        weights = torch.randint(-2, 3, weights.shape, generator=generator).half()
    elif p['pattern'] == 'corners':
        image.zero_()
        weights.zero_()
        for c in range(64):
            image[0, c, 0, 0] = (c % 7) - 3
            image[0, c, -1, -1] = (c % 5) - 2
        for out in range(128):
            weights[out, out % 64, :, :] = torch.arange(9).reshape(3, 3).half() - 4
    elif p['pattern'] != 'random':
        raise ValueError('Unknown independent convolution input pattern')
    # Includes both signs and both parities of FP16 midpoint mantissas.
    probes = [1 + 1 / 2048, 1 + 3 / 2048, -1 - 1 / 2048, -1 - 3 / 2048,
              0.0, -0.0, 2**-25, -(2**-25)]
    cast_source = torch.tensor(probes * (2048 * 16 // len(probes)), dtype=torch.float32).reshape(2048, 16)
    return {'image': image, 'weights': weights, 'cast_source': cast_source, 'parameters': p, 'block_dim': 1}


def input_stages(inputs):
    """The two staging outputs built by hand, so stage 1 and stage 2 can be checked on inputs
    that never passed through a kernel. 5HD is [B*C1, HP, W, C0] with the two pad rows above
    and below each plane left zero; the fractal weight order is Cout, C1, Kh, Kw, C0."""
    image = inputs['image']
    data = torch.zeros((4, 36, 32, 16), dtype=torch.float16)
    for c in range(64):
        data[c // 16, 2:34, :, c % 16] = image[0, c]
    weights = torch.empty((128, 4, 9, 16), dtype=torch.float16)
    for c in range(64):
        weights[:, c // 16, :, c % 16] = inputs['weights'][:, c].reshape(128, 9)
    return data.reshape(4608, 16), weights.reshape(128, 576)


def convolution(inputs):
    """One FP64 convolution over the whole logical NCHW input. The kernels reach the same
    numbers through 5HD planes, a fractal weight layout, an FP32 L0C accumulator and a final
    nearest-even cast; nothing but the mathematics is shared."""
    return F.conv2d(inputs['image'].double(), inputs['weights'].double(), padding=2, stride=2, dilation=2)


def pack_nz(nchw):
    """NCHW [128, 256] -> the NZ order the cube writes: Cout1, M, Cout0."""
    return nchw.reshape(8, 16, 256).permute(0, 2, 1).contiguous().reshape(2048, 16)


def reference(inputs):
    return {'output': convolution(inputs).half().reshape(128, 256)}


def reference_stages(inputs):
    """One expected tensor per stage, each derived independently of the others. `cast_nchw` is
    deliberately not the convolution's output: it is the probe tensor from make_inputs rounded
    by the host's own struct.pack('<e'), which is what pins stage 4 to ties-to-even rather than
    to whatever the cube happened to produce."""
    data, weights = input_stages(inputs)
    cast = inputs['cast_source'].reshape(8, 256, 16).permute(0, 2, 1).contiguous().reshape(128, 256)
    # The independent host standard-library binary16 conversion specifies ties-to-even.
    raw = b''.join(struct.pack('<e', value) for value in cast.reshape(-1).tolist())
    rounded = torch.frombuffer(bytearray(raw), dtype=torch.float16).clone().reshape(128, 256)
    return {'data5hd': data, 'weights_fractal': weights,
            'conv_nz': pack_nz(convolution(inputs).float()), 'cast_nchw': rounded}

