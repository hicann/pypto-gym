# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent FP32 HiFloat8 quantize-dequantize reference and inputs."""

import torch

OVERFLOW_THRESHOLD = 2.0 ** 15 * 1.25
MIN_VALUE = 2.0 ** -23
DML_SCALE = 2.0 ** -22


def edge_values() -> torch.Tensor:
    minimum = torch.tensor(MIN_VALUE, dtype=torch.float32)
    overflow = torch.tensor(OVERFLOW_THRESHOLD, dtype=torch.float32)
    values = [
        0.0, -0.0,
        float(torch.nextafter(minimum, torch.tensor(0.0))),
        MIN_VALUE,
        float(torch.nextafter(minimum, torch.tensor(torch.inf))),
        DML_SCALE,
        1.0625, -1.0625,
        7.5, -7.5, 8.5, -8.5,
        127.5, -127.5, 128.5, -128.5,
        32767.0, -32767.0, 32768.0, -32768.0,
        float(torch.nextafter(overflow, torch.tensor(0.0))),
        OVERFLOW_THRESHOLD,
        float(torch.nextafter(overflow, torch.tensor(torch.inf))),
        -float(torch.nextafter(overflow, torch.tensor(0.0))),
        -OVERFLOW_THRESHOLD,
        -float(torch.nextafter(overflow, torch.tensor(torch.inf))),
        float(torch.finfo(torch.float32).max),
        -float(torch.finfo(torch.float32).max),
        float("inf"), float("-inf"), float("nan"),
    ]
    for exponent in (-22, -16, -15, -8, -7, -4, -3, 0, 3, 4, 7, 8, 15, 16):
        value = 2.0 ** exponent
        values.extend((value, -value, value * 1.5, -value * 1.5))
    return torch.tensor(values, dtype=torch.float32)


def make_values(total: int, seed: int, scale: float, pattern: str) -> torch.Tensor:
    edges = edge_values()
    if pattern == "edge":
        if total != edges.numel():
            raise ValueError("edge case total must equal the complete edge table")
        return edges.reshape(1, total).contiguous()
    if pattern != "source":
        raise ValueError("pattern must be edge or source")
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(total, generator=generator, dtype=torch.float32) * scale
    count = min(total, edges.numel())
    values[:count] = edges[:count]
    return values.reshape(1, total).contiguous()


def quantize_dequantize(x: torch.Tensor) -> torch.Tensor:
    values = x.float()
    absolute = values.abs()
    out = torch.empty_like(values)
    finite = torch.isfinite(values)
    overflow = finite & (absolute >= OVERFLOW_THRESHOLD)
    underflow = finite & (absolute < MIN_VALUE)
    quantized = finite & ~(overflow | underflow)
    out[~finite] = values[~finite]
    out[underflow] = 0.0
    out[overflow] = torch.copysign(torch.full_like(values[overflow], torch.inf), values[overflow])
    if bool(quantized.any()):
        magnitude = absolute[quantized]
        exponent = torch.floor(torch.log2(magnitude))
        exponent = torch.where(exponent == -23.0, torch.full_like(exponent, -22.0), exponent)
        bits = torch.zeros_like(exponent)
        bits = torch.where(exponent.abs() <= 15.0, torch.ones_like(bits), bits)
        bits = torch.where(exponent.abs() <= 7.0, torch.full_like(bits, 2.0), bits)
        bits = torch.where(exponent.abs() <= 3.0, torch.full_like(bits, 3.0), bits)
        step = torch.pow(2.0, exponent - bits)
        rounded = torch.floor(magnitude / step + 0.5) * step
        out[quantized] = torch.copysign(rounded, values[quantized])
    return out.contiguous()


def make_inputs(case):
    """Deterministic input for one case: the edge table first, then scaled normal noise.
    Every `source` case begins with as much of the edge table as fits, so a kernel that
    only handles ordinary magnitudes fails on the very first tile rather than by luck."""
    p = case["parameters"]
    return {**p, "x": make_values(p["total"], case["seed"], float(p["scale"]), p["pattern"])}


def reference(inputs):
    return {"y": quantize_dequantize(inputs["x"])}
