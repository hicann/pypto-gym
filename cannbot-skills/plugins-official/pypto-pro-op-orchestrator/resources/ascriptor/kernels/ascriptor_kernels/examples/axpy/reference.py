# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch arithmetic for o = round_f32(round_f32(2*x) + y)."""

import torch


def make_inputs(case: dict) -> dict:
    parameters = case["parameters"]
    if parameters.get("shape") != [1, 64] or parameters.get("dtype") != "float32":
        raise ValueError("AXPY supports exactly contiguous float32 shape [1, 64]")
    generator = torch.Generator(device="cpu").manual_seed(case["seed"])
    x = torch.randn(1, 64, generator=generator, dtype=torch.float32) * 3.0
    y = torch.randn(1, 64, generator=generator, dtype=torch.float32) * 3.0
    pattern = parameters.get("pattern", "random")
    if pattern == "cancellation":
        y = -2.0 * x
    elif pattern == "zeros":
        x.zero_()
        y = torch.arange(-32, 32, dtype=torch.float32).reshape(1, 64)
    elif pattern != "random":
        raise ValueError(f"unknown AXPY input pattern {pattern!r}")
    return {"x": x, "y": y, "o": torch.full_like(x, float("nan"))}


def validate_inputs(inputs: dict, case: dict | None = None) -> None:
    for name in ("x", "y", "o"):
        value = inputs[name]
        if value.device.type != "cpu" or value.dtype != torch.float32 or tuple(value.shape) != (1, 64) or not value.is_contiguous():
            raise ValueError(f"{name} requires contiguous CPU float32 [1, 64]")
    if len({inputs[name].untyped_storage().data_ptr() for name in ("x", "y", "o")}) != 3:
        raise ValueError("AXPY inputs and output must not alias")
    if not torch.isfinite(inputs["x"]).all() or not torch.isfinite(inputs["y"]).all():
        raise ValueError("AXPY inputs must be finite")


def reference(inputs: dict) -> dict:
    validate_inputs(inputs)
    scaled = inputs["x"] * 2.0
    return {"o": scaled + inputs["y"]}


def validate_reference(inputs: dict, outputs: dict, case: dict) -> None:
    if not torch.isfinite(outputs["o"]).all():
        raise ValueError("bounded AXPY inputs must produce finite outputs")
    if case["parameters"].get("pattern") == "cancellation" and torch.count_nonzero(outputs["o"]):
        raise ValueError("cancellation case must produce zero")
