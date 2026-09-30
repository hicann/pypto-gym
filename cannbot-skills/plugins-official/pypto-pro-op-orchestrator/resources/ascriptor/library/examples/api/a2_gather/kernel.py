# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Element and block gathers use explicit byte-offset tensors."""

import importlib


def make_gather(mode="elements", device="a2"):
    if mode not in ("elements", "blocks") or device not in ("a2", "a3"):
        raise ValueError("declared gather modes are elements/blocks on A2/A3")
    api = importlib.import_module("ascriptor." + device)

    @api.kernel(mode="vec", block_dim=1)
    def a2_gather(x: api.GM[api.f32, (1, 128)], offsets: api.GM[api.u32, (1, 64)],
        o: api.GM[api.f32, (1, 64)]):
        source = api.Tensor(api.f32, [1, 128], api.Position.UB)
        addresses = api.Tensor(api.u32, [1, 64], api.Position.UB)
        output = api.Tensor(api.f32, [1, 64], api.Position.UB)
        with api.auto_sync():
            source <<= x
            addresses <<= offsets
            if mode == "elements":
                api.gather(output, source, addresses, repeat=1, dst_rep_stride=8)
            else:
                api.gather_block(output, source, addresses, repeat=1, dst_blk_stride=1, dst_rep_stride=8)
            o <<= output
        return o

    return a2_gather
