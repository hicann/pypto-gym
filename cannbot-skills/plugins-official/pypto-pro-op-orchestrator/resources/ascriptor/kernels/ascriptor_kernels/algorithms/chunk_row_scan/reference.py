# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch sequential FP32 row prefixes with explicit chunk resets."""

import torch

UB_BYTES = 256 * 1024


def domain(parameters, block_dim):
    m, h, chunk = (parameters[name] for name in ("M", "H", "chunk_size"))
    if any(type(v) is not int or v <= 0 for v in (m, h, chunk)):
        raise ValueError("M, H and chunk_size must be positive integers")
    if h >= 64 and h % 64:
        raise ValueError("Wide H must be divisible by64")
    if type(block_dim) is not int or block_dim not in (1, 2, 4):
        raise ValueError("Declare1,2or4 source MIX core groups")
    allocated = 2 * ((chunk * h * 4 + 31) // 32 * 32) + 2 * chunk * 64 * 4
    if allocated > UB_BYTES:
        raise ValueError("All four aligned source UB buffers must fit256KiB")
    total_chunks = (m + chunk - 1) // chunk
    chunks_per_vector = (total_chunks + 2 * block_dim - 1) // (2 * block_dim)
    boundary = chunks_per_vector * chunk
    if boundary < m and boundary * h * 4 % 32:
        raise ValueError("Adjacent vector owners require32-byte-aligned GM output boundaries")
    return m, h, chunk


def make_inputs(case):
    block_dim = case.get("block_dim", 1)
    m, h, chunk = domain(case["parameters"], block_dim)
    generator = torch.Generator().manual_seed(case["seed"])
    x = torch.randn((m, h), generator=generator)
    if case["parameters"].get("pattern") == "cancellation":
        if m != 3 or chunk != 3:
            raise ValueError("The named cancellation case has exactly three rows")
        x[:] = torch.tensor([1e8, 1.0, -1e8])[:, None]
    else:
        x[0, 0] = 1.0
    return {"x": x, "M": m, "H": h, "chunk_size": chunk, "block_dim": block_dim}


def reference(inputs):
    x, chunk = inputs["x"], inputs["chunk_size"]
    out = torch.empty_like(x)
    for start in range(0, x.shape[0], chunk):
        out[start] = x[start]
        for row in range(start + 1, min(start + chunk, x.shape[0])):
            out[row] = x[row] + out[row - 1]
    return {"prefixes": out}
