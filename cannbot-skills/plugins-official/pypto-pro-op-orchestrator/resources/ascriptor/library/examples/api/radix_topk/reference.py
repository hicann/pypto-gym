# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The selected multiset, and the properties a valid selection has. Imports no DSL.

There is no single right answer here: which index is returned for a tie is not part of the contract.
So the reference is the *multiset* of selected values, and the index side is checked by its
properties instead of against a table.
"""

import torch

LANES = 4096      # the complete source tile
SLOTS = 512       # the complete destination tiles
PADDING = float("-inf")


def make_inputs(case):
    p = case["parameters"]
    count, k, dataset = p["count"], p["k"], p["dataset"]
    validate_parameters(count, k)
    generator = torch.Generator().manual_seed(case["seed"])
    src = torch.full((1, LANES), PADDING)
    if dataset == "ties":
        # A small pool, so many lanes hold the same value and several compete for the same rank.
        pool = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        src[0, :count] = pool[torch.randint(0, pool.numel(), (count,), generator=generator)]
    else:
        src[0, :count] = torch.randn(count, generator=generator)
    inputs = {"src": src, "count": count, "k": k}
    validate(inputs)
    return inputs


def validate_parameters(count, k):
    if type(count) is not int or type(k) is not int:
        raise ValueError("count and k are plain ints")
    if not 1 <= count <= LANES or not 1 <= k <= min(count, SLOTS):
        raise ValueError(f"require 1 <= count <= {LANES} and 1 <= k <= min(count, {SLOTS})")


def validate(inputs):
    validate_parameters(inputs["count"], inputs["k"])
    src = inputs["src"]
    if not isinstance(src, torch.Tensor) or src.dtype != torch.float32:
        raise ValueError("the source must be float32")
    if tuple(src.shape) != (1, LANES) or not src.is_contiguous():
        raise ValueError(f"the source must be a contiguous [1, {LANES}] tile")
    live = src[0, :inputs["count"]]
    if not bool(torch.isfinite(live).all()):
        raise ValueError("every live lane must be finite")
    if not bool((src[0, inputs["count"]:] == PADDING).all()):
        raise ValueError("every lane past the count must hold the -inf padding")


def reference(inputs):
    """The k largest live values, in descending order. The device's order is not compared."""
    validate(inputs)
    live = inputs["src"].flatten()[:inputs["count"]]
    return {"values": torch.topk(live, inputs["k"]).values}
