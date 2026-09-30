# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One 16x16 convolution output tile, with host-owned zero channel padding."""

import ascriptor.a5 as api


def make_conv(stride=1, dilation=1, pad=1):
    if (stride, dilation, pad) not in ((1, 1, 1), (2, 1, 1), (1, 2, 2)):
        raise ValueError("geometry is outside the three declared cases")

    @api.kernel(mode="cube", block_dim=1)
    def cube_conv(x: api.GM[api.f16, (16, 16)], w: api.GM[api.f16, (16, 144)], o: api.GM[api.f32, (16, 16)]):
        fm = api.Tensor(api.f16, [16, 16], api.Position.L1)
        weights = api.Tensor(api.f16, [16, 144], api.Position.L1)
        product = api.Tensor(api.f32, [16, 16], api.Position.L0C)
        config = api.Conv2D(3, 3, stride=(stride, stride), dilation=(dilation, dilation), pad=(pad, pad, pad, pad))
        with api.auto_sync():
            fm <<= x
            weights <<= w
            api.conv2d(product, fm, weights, config, h=4, w=4, c=3, cout=5, m0=0, tile_k=144)
            o <<= product
        return o

    return cube_conv
