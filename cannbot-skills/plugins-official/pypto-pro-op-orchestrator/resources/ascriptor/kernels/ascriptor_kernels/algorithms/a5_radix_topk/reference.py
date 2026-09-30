# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch numerical top-k over the first count keys."""

import torch

MAXN, MAXK = 4096, 512


def make_inputs(case):
    p = case["parameters"]
    count, k, flavour = p["count"], p["k"], p["flavour"]
    if type(count) is not int or type(k) is not int or not 1 <= count <= MAXN or not 1 <= k <= min(count, MAXK):
        raise ValueError("Require 1 <= k <= min(count,512), 1 <= count <=4096")
    if type(case.get("block_dim", 1)) is not int or case.get("block_dim", 1) != 1:
        raise ValueError("The source algorithm owns one vector core")
    generator = torch.Generator().manual_seed(case["seed"])
    if flavour == "random":
        values = torch.randn(count, generator=generator) * 100.0
    elif flavour == "narrow":
        values = 1.0 + torch.rand(count, generator=generator) * 1e-6
    elif flavour == "ties":
        values = torch.randint(0, 5, (count,), generator=generator).float()
    elif flavour == "signed":
        values = torch.randn(count, generator=generator)
    elif flavour == "all_equal":
        values = torch.full((count,), -3.0)
    elif flavour == "signed_zero":
        values = torch.zeros(count)
        values[::2] = -0.0
    elif flavour == "ascending":
        values = torch.arange(count, dtype=torch.float32) - count // 2
    else:
        raise ValueError("Unknown generated input flavour")
    keys = torch.full((1, MAXN), float("-inf"), dtype=torch.float32)
    keys[0, :count] = values
    return {"keys": keys, "count": count, "k": k, "block_dim": 1}


def reference(inputs):
    return {"values_descending": torch.topk(inputs["keys"][0, :inputs["count"]], k=inputs["k"]).values}
