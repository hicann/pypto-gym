# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent FP64 matrix product and the four named normalization denominators."""

import torch


def normalize(inputs):
    product = inputs["x"].double() @ inputs["y"].double().T
    variant = inputs["variant"]
    if variant.startswith("row_sum"):
        denominator = product.sum(-1, keepdim=True)
    elif variant == "row_l2":
        denominator = product.square().sum(-1, keepdim=True).sqrt()
    else:
        product = product.reshape(inputs["M"], inputs["N"] // 128, 128)
        denominator = product.abs().amax(-1, keepdim=True)
    if (denominator.abs() <= 1e-6).any():
        raise ValueError("Source normalization has no epsilon; every denominator must exceed1e-6 in magnitude")
    return {"out": (product / denominator).reshape(inputs["M"], inputs["N"]).float()}


def make_inputs(case):
    """Deterministic FP16 X[M, K] and Y[N, K] for one case. The row-sum variants divide by a
    plain sum, so their inputs are drawn strictly positive (0.125 + U) -- a signed input would
    let the sum cancel towards zero and the domain has no epsilon to protect it. The L2 and
    absmax variants take signed normal values, where the denominator cannot cancel."""
    parameters = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    inputs = dict(parameters)
    for name, rows in (("x", parameters["M"]), ("y", parameters["N"])):
        shape = (rows, parameters["K"])
        value = (0.125 + torch.rand(shape, generator=generator)
                 if parameters["variant"].startswith("row_sum")
                 else torch.randn(shape, generator=generator))
        inputs[name] = value.half()
    return inputs


def reference(inputs):
    return normalize(inputs)
