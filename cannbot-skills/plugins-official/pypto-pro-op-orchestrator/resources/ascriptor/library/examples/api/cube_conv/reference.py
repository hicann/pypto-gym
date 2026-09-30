# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F


def make_inputs(case):
    g = torch.Generator().manual_seed(case["seed"])
    image = torch.randint(-2, 3, (1, 3, 4, 4), generator=g).half()
    weights = torch.randint(-2, 3, (5, 3, 3, 3), generator=g).half()
    packed_image = torch.zeros(4, 4, 16, dtype=torch.float16)
    packed_image[:, :, :3] = image[0].permute(1, 2, 0)
    packed_weight = torch.zeros(16, 3, 3, 16, dtype=torch.float16)
    packed_weight[:5, :, :, :3] = weights.permute(0, 2, 3, 1)
    return {
        "image": image,
        "weights": weights,
        "packed_image": packed_image.reshape(16, 16),
        "packed_weights": packed_weight.reshape(16, 144),
        "geometry": dict(case["parameters"]),
    }


def reference(inputs):
    p = inputs["geometry"]
    # Hardware load3d continues the raster window past the logical Ho*Wo.
    # A larger *bottom* zero pad expresses those same physical rows through
    # independent convolution math, without reproducing the simulator loader.
    width = (4 + 2 * p["pad"] - p["dilation"] * 2 - 1) // p["stride"] + 1
    rows = (16 + width - 1) // width
    bottom = max(p["pad"], (rows - 1) * p["stride"] - 4 + p["dilation"] * 2 + 1 - p["pad"])
    image = F.pad(inputs["image"].float(), (p["pad"], p["pad"], p["pad"], bottom))
    result = F.conv2d(image, inputs["weights"].float(), stride=p["stride"], dilation=p["dilation"])
    out = torch.zeros(16, 16)
    out[:, :5] = result[0].permute(1, 2, 0).reshape(-1, 5)[:16]
    return {"o": out}
