# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent FP64 NCHW convolution, narrowed to FP16 only at the end."""

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
    return {'image': image, 'weights': weights, 'parameters': p, 'block_dim': 1}


def reference(inputs):
    """One FP64 convolution over the whole NCHW input, narrowed to FP16 once. The kernel gets
    to the same numbers by a completely different route -- 5HD image planes, a fractal weight
    layout, an FP32 L0C accumulator and a final nearest-even cast -- so nothing but the
    mathematics is shared between the two."""
    conv = F.conv2d(inputs['image'].double(), inputs['weights'].double(),
                    padding=2, stride=2, dilation=2)
    return {'output': conv.half().reshape(128, 256)}
