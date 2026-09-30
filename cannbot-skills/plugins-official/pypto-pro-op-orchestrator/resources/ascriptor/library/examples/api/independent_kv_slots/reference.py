# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent dense host oracle for the small two-product primitive."""

import torch

CASES = {
    "small_k2v2p2": ("k2v2p2", 64, 3, "float16", 1),
    "small_k1v2p2": ("k1v2p2", 64, 3, "float16", 1),
    "small_sequential": ("sequential", 64, 3, "float16", 1),
    "bf16_idle": ("k1v2p2", 64, 1, "bfloat16", 3),
    "large_k1v2p2": ("k1v2p2", 256, 1, "float16", 1),
    "large_sequential": ("sequential", 256, 1, "float16", 1),
}


def parameters(case_id):
    policy, width, items, dtype, _ = CASES[case_id]
    return {"policy": policy, "m": 16, "n": width, "d": width,
            "items": items, "beats": 5, "dtype": dtype}


def output_seed(parameters):
    items, beats, width = (parameters[name] for name in ("items", "beats", "d"))
    result = torch.full((items, beats, 32, width), float("nan"), dtype=torch.float32)
    # Every untouched guard element has a distinct exactly represented label.
    guard = -(32768 + torch.arange(items * beats * 16 * width).reshape(items, beats, 16, width))
    result[:, :, 16:, :] = guard.float()
    return result


def make_inputs(case):
    if case["id"] not in CASES or case["parameters"] != parameters(case["id"]):
        raise ValueError("Require one exact declared primitive geometry and slot policy")
    if case.get("block_dim", 1) != CASES[case["id"]][4]:
        raise ValueError("The primitive case keeps its declared mixed-core count")
    p = case["parameters"]
    items, beats, width = (p[name] for name in ("items", "beats", "d"))
    dtype = getattr(torch, p["dtype"])
    q = torch.zeros((items, 16, width), dtype=dtype)
    for item in range(items):
        for row in range(16):
            q[item, row, (row * 3 + item * 5) % width] = 1
            q[item, row, (row * 3 + item * 5 + 17) % width] = 1 / 256
    item = torch.arange(items).reshape(items, 1, 1, 1)
    beat = torch.arange(beats).reshape(1, beats, 1, 1)
    row = torch.arange(width).reshape(1, 1, width, 1)
    column = torch.arange(width).reshape(1, 1, 1, width)
    k = (((item * 7 + beat * 11 + row * 5 + column * 3 + case["seed"]) % 33 - 16) / 16).to(dtype)
    v = (((item * 13 + beat * 17 + row * 7 + column * 11 + case["seed"] + 5) % 33 - 16) / 16).to(dtype)
    return {"case_id": case["id"], "parameters": dict(p), "q": q, "k": k, "v": v}


def validate_inputs(inputs, case=None):
    if set(inputs) != {"case_id", "parameters", "q", "k", "v"} or inputs["case_id"] not in CASES:
        raise ValueError("Require exactly independent Q/K/V and the declared host selectors")
    p = parameters(inputs["case_id"])
    if inputs["parameters"] != p or (case is not None and case["parameters"] != p):
        raise ValueError("Input geometry differs from the declared case")
    items, beats, width = (p[name] for name in ("items", "beats", "d"))
    shapes = {"q": (items, 16, width), "k": (items, beats, width, width),
              "v": (items, beats, width, width)}
    dtype = getattr(torch, p["dtype"])
    for name, shape in shapes.items():
        tensor = inputs[name]
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype
                or tensor.device.type != "cpu" or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all()) or bool((tensor.abs() > 1).any())):
            raise ValueError(f"{name} requires finite contiguous b16 CPU values in [-1, 1] and its exact shape")
    if inputs["k"].untyped_storage().data_ptr() == inputs["v"].untyped_storage().data_ptr():
        raise ValueError("Independent K and V must not alias storage")


def reference(inputs):
    validate_inputs(inputs)
    p = inputs["parameters"]
    # This code imports no DSL, simulator, lowering or packing helper. The first
    # FP32 result is rounded to b16 before the independent second dense product.
    score = inputs["q"].double().unsqueeze(1) @ inputs["k"].double().transpose(-1, -2)
    probability = score.float().to(inputs["q"].dtype)
    product = probability.double() @ inputs["v"].double()
    output = output_seed(p)
    output[:, :, :16, :] = product.float()
    return {"o": output}
