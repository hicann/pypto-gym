# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two counted reduction stages and a two-stage runtime broadcast."""

import importlib


def make_reduce(width=128, device="a2"):
    if width not in (64, 128, 256) or device not in ("a2", "a3"):
        raise ValueError("declared widths are64/128/256 on A2/A3")
    api = importlib.import_module("ascriptor." + device)
    groups = width // 64

    @api.kernel(mode="vec", block_dim=1)
    def a2_reduce(x: api.GM[api.f32, (3, width)], o: api.GM[api.f32, (3, width)],
        sums: api.GM[api.f32, (3, 64)]):
        source = api.Tensor(api.f32, [1, width], api.Position.UB)
        output = api.Tensor(api.f32, [1, width], api.Position.UB)
        partial = api.Tensor(api.f32, [1, 64], api.Position.UB)
        total = api.Tensor(api.f32, [1, 64], api.Position.UB)
        block = api.Tensor(api.f32, [8, 8], api.Position.UB)
        scale = api.Tensor(api.f32, [8, 8], api.Position.UB)
        with api.auto_sync():
            for row in range(3):
                source <<= x[row : row + 1, :]
                api.dup(partial, 0.0)
                api.dup(total, 0.0)
                api.dup(block, 0.0)
                api.dup(scale, 0.0)
                api.cadd(partial, source, repeat=groups, src_blk_stride=1, src_rep_stride=8, count_per_rep=64)
                api.cadd(total[:, :1], partial, repeat=1, count_per_rep=groups)
                api.brcb(block, total, repeat=1, dst_blk_stride=1, dst_rep_stride=8)
                api.brcb(scale, block[:1, :8], repeat=1, dst_blk_stride=1, dst_rep_stride=8)
                for group in range(groups):
                    start = group * 64
                    api.mul(output[:, start : start + 64], source[:, start : start + 64], scale, repeat=1)
                o[row : row + 1, :] <<= output
                sums[row : row + 1, :] <<= total
        return o, sums

    return a2_reduce
