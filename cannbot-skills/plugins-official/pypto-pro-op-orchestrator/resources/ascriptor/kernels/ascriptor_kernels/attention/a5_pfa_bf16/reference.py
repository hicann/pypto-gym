# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent dense MQA mathematics, and the deterministic BF16 inputs it is checked against.

The whole batch attention is computed densely in FP32 with a single rounding to BF16 at the end:
no tiling, no online softmax and no partial merge, so neither sharding's scheduling decisions can
be inherited by the answer. One K/V stream per batch is shared by every query row of that batch."""

import math
import torch

def attention(q, k, v, batches, queries, keys):
    q = q.reshape(batches, queries, 128).float()
    k = k.reshape(batches, keys, 128).float()
    v = v.reshape(batches, keys, 128).float()
    probability = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(128), dim=-1)
    return (probability @ v).reshape(batches * queries, 128).to(torch.bfloat16)


def make_inputs(case):
    """One standard-normal draw per operand, all three in BF16. The case parameters travel on in
    the returned dict because B/MQ/N are kernel arguments as well as tensor geometry."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    return {**p, **{name: torch.randn((p["B"] * p["MQ" if name == "q" else "N"], 128),
                                      generator=generator).bfloat16()
                    for name in ("q", "k", "v")}}


def reference(inputs):
    return {"out": attention(inputs["q"], inputs["k"], inputs["v"],
                             inputs["B"], inputs["MQ"], inputs["N"])}
