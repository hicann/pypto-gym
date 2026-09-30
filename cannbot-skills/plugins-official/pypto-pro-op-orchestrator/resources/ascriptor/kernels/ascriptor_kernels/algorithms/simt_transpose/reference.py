# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent FP64 matrix product and transpose, and the deterministic FP32 inputs each case names."""

import torch


def transpose_product(inputs):
    product = inputs["a"].double() @ inputs["b"].double().T
    return {"out": product.T.contiguous().float()}


def make_inputs(case):
    """Deterministic FP32 A[M, K] and B[N, K] for one case. The `integer` distribution draws
    whole numbers in [-4, 4], so the FP32 product is exact and a transposed or misindexed
    output is unmistakably wrong rather than merely close; `zero` is the control that only
    proves anything against a poisoned destination."""
    parameters = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    inputs = {**parameters, "block_dim": case.get("block_dim", 1)}
    for name, rows in (("a", parameters["M"]), ("b", parameters["N"])):
        shape = (rows, parameters["K"])
        if parameters["distribution"] == "zero":
            value = torch.zeros(shape)
        elif parameters["distribution"] == "integer":
            value = torch.randint(-4, 5, shape, generator=generator).float()
        else:
            value = torch.randn(shape, generator=generator)
        inputs[name] = value
    return inputs


def reference(inputs):
    return transpose_product(inputs)
