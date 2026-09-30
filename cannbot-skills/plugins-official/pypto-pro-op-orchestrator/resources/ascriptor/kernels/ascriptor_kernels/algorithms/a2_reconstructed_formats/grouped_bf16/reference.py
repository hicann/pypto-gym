# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent reconstructed-value reference vocabulary; vendor copies into each unit.

This module imports no execution codec. BF16 boundaries are constructed from bits;
FP4 thresholds, scale hierarchies and HiF8 exponent classes are specified locally."""

import numpy as np
import torch

def bf16(values, rounding="away"):
    if rounding not in ("away", "even"):
        raise ValueError("BF16 reference rounding must be away or even")
    data = values.detach().cpu().float().contiguous().numpy()
    bits = data.view(np.uint32)
    bias = np.uint32(0x8000) if rounding == "away" else np.uint32(0x7FFF) + ((bits >> 16) & 1)
    with np.errstate(over="ignore"):
        out = ((bits + bias) >> 16).astype(np.uint16)
    nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    out = np.where(nan, out | 0x0040, out).astype(np.uint16)
    return torch.from_numpy(out.copy()).view(torch.bfloat16)


def round_away(values):
    # Compare the represented FP32 input to exact half-integer boundaries;
    # adding0.5 in FP32 would itself round the predecessor of0.5 upward.
    data = values.double()
    return (torch.sign(data) * torch.floor(data.abs() + 0.5)).float()


def e2m1(values):
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    magnitude = levels[torch.bucketize(values.abs().contiguous(), thresholds, right=True)]
    return torch.where(values < 0, -magnitude, magnitude)


def floor_e8m0(values):
    safe = values.double().clamp_min(2.0 ** -127)
    return torch.exp2(torch.floor(torch.log2(safe))).float()


def group_scaled(values, group, signed_integer=False):
    shape = values.shape
    data = values.float().reshape(-1, group)
    maximum = data.abs().amax(-1, keepdim=True)
    reciprocal = torch.tensor(1.0 / (7.0 if signed_integer else 6.0), dtype=torch.float32)
    scale = bf16(maximum * reciprocal).float()
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    normalized = data / safe
    quantized = round_away(normalized).clamp(-8, 7) if signed_integer else e2m1(normalized)
    return (quantized * scale).reshape(shape)


def plain_mxfp4(values):
    shape = values.shape
    data = values.float().reshape(-1, 32)
    scale = floor_e8m0(data.abs().amax(-1, keepdim=True) * 0.25)
    return (e2m1(data / scale) * scale).reshape(shape)


def mbs_mxfp4(values):
    shape = values.shape
    data = values.float().reshape(-1, 128)
    maximum = data.abs().amax(-1, keepdim=True)
    reciprocal = (6.0 / maximum).contiguous()
    # The source's mathematical macro factor is defined by the top8 mantissa
    # bits of an ordinary FP32 reciprocal, including the infinity bit pattern.
    # No empirically fitted division correction table is used here.
    mantissa = (reciprocal.view(torch.int32) & 0x007F8000) >> 15
    factor = 1.0 + mantissa.float() / 256.0
    factor = torch.where(maximum == 0, torch.ones_like(factor), factor)
    normalized = data * factor
    quantized = plain_mxfp4(normalized)
    return (quantized / factor).reshape(shape)


def mxfp8_e5m2(values):
    shape = values.shape
    data = values.float().reshape(-1, 32)
    magnitude = data.abs()
    # Unlike an ideal private exponent, the source clears the exponent field
    # of FP32 subnormals to zero before the group maximum.
    power = torch.exp2(torch.floor(torch.log2(magnitude.double().clamp_min(2.0 ** -126)))).float()
    power = torch.where(magnitude >= 2.0 ** -126, power, torch.zeros_like(power))
    group_power = power.amax(-1, keepdim=True)
    step = torch.maximum(power, group_power * (2.0 ** -29)) * 0.25
    step = step + torch.tensor(5.421011e-20, dtype=torch.float32)
    quantized = round_away(data / step) * step
    limit = group_power * 1.75
    return torch.maximum(torch.minimum(quantized, limit), -limit).reshape(shape)


def hifx(values, bits, *, max_level=3):
    if bits not in (4, 5) or max_level not in (1, 2, 3):
        raise ValueError("HiFX reference supports N4/N5 and explicit levels1..3")
    shape = values.shape
    data = values.float().reshape(-1, 64)
    magnitude = data.abs()
    scale_input = bf16(magnitude.amax(-1, keepdim=True) * 0.142578125, "even").float()
    scale_input = scale_input.clamp(2.0 ** -48, 49152.0)
    power = torch.exp2(torch.floor(torch.log2(scale_input.double()))).float()
    scale = torch.round((scale_input / power) * 4.0) * power * 0.25
    reciprocal = bf16(1.0 / scale, "even").float()
    maximum8 = magnitude.reshape(-1, 8, 8).amax(-1)
    exponent8 = (maximum8 * reciprocal >= 4.0).to(torch.int32) if max_level >= 2 else torch.zeros_like(maximum8, dtype=torch.int32)
    maximum4 = magnitude.reshape(-1, 16, 4).amax(-1)
    adjust8 = torch.exp2(-exponent8.float()).repeat_interleave(2, -1)
    exponent4 = (maximum4 * reciprocal * adjust8 >= 2.0).to(torch.int32) if max_level >= 3 else torch.zeros_like(maximum4, dtype=torch.int32)
    exponent = exponent8.repeat_interleave(8, -1) + exponent4.repeat_interleave(4, -1)
    normalized = bf16(magnitude * reciprocal * torch.exp2(-exponent.float()), "even").float()
    grid = 2.0 ** (bits - 2)
    quantized = torch.floor(normalized * grid + 0.5) / grid
    quantized = quantized.clamp_max(2.0 - 1.0 / grid)
    out = quantized * scale * torch.exp2(exponent.float())
    return torch.where(data < 0, -out, out).reshape(shape)



# variant -> (algorithm, group size, HiFX bit width). This is everything a reference needs to
# know about a variant; which kernel entry it launches is main.py's business.
VARIANTS = {
    "e2m1_g16": ("group_e2m1", 16, None),
    "signed_int4_g16": ("signed_integer", 16, None),
    "e2m1_g32": ("group_e2m1", 32, None),
    "signed_int4_g32": ("signed_integer", 32, None),
    "e2m1_g64": ("group_e2m1", 64, None),
    "signed_int4_g64": ("signed_integer", 64, None),
    "plain_mxfp4": ("mxfp4", 32, None),
    "mbs_mxfp4": ("mbs", 128, None),
    "mxfp8_e5m2": ("mxfp8_e5m2", 32, None),
    "hifx4": ("hifx", 64, 4),
    "hifx5": ("hifx", 64, 5),
}


def make_inputs(case):
    """Deterministic input for one case. The structured patterns are not decoration: each one
    puts an exact scale-defining anchor in every group, so a group whose scale is computed one
    ulp off produces a visibly different grid rather than a slightly different number."""
    p = case["parameters"]
    algorithm, group, _ = VARIANTS[p["variant"]]
    shape, pattern = (p["rows"], p["cols"]), p["pattern"]
    generator = torch.Generator().manual_seed(case["seed"])
    if pattern == "zero":
        value = torch.zeros(shape)
    elif pattern == "tiny":
        value = torch.full(shape, 2.0 ** -126)
    elif pattern == "macro_boundary":
        value = torch.zeros(shape)
        value[0, :128] = 6
        value[1, :128] = 2.0 ** -126
        value[2, :128] = torch.tensor([1.5, 1.75, 2.0, 3.0] * 32)
    elif pattern == "logspace":
        magnitude = torch.logspace(-8, 8, p["rows"] * p["cols"]).reshape(shape)
        value = magnitude * torch.where(torch.arange(magnitude.numel()).reshape(shape) % 2 == 0, 1.0, -1.0)
    elif pattern == "epsilon":
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -1.5, -2.0, -3.0]) * (2.0 ** -64)
        value = levels[torch.arange(p["rows"] * p["cols"]) % len(levels)].reshape(shape)
    elif pattern == "hierarchy":
        levels = torch.tensor([1.0, 1.0, 1.0, 1.0, 7.0, 7.0, 7.0, 7.0,
                               -1.0, -1.0, -1.0, -1.0, -7.0, -7.0, -7.0, -7.0])
        value = levels[torch.arange(p["rows"] * p["cols"]) % len(levels)].reshape(shape)
    elif pattern == "boundaries":
        if algorithm == "signed_integer":
            levels = torch.tensor([0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, -0.5, -1.5, -2.5, -3.5, -4.5, -5.5])
            anchor, factor = 7.0, 2.0
        else:
            levels = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
                                   -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0])
            anchor, factor = 6.0, (2.0 if algorithm == "group_e2m1" else 1.0)
        values = levels[torch.arange(p["rows"] * p["cols"]) % len(levels)].reshape(-1, group)
        values[:, -1] = anchor  # Each group has its own exact scale-defining anchor.
        value = (values * factor).reshape(shape)
    else:
        value = torch.randn(shape, generator=generator) * p["scale"]
    return {**p, "x": value.to(torch.bfloat16).contiguous()}


def reference(inputs):
    algorithm, group, bits = VARIANTS[inputs["variant"]]
    if algorithm in ("group_e2m1", "signed_integer"):
        value = group_scaled(inputs["x"], group, algorithm == "signed_integer")
    elif algorithm == "mxfp4":
        value = plain_mxfp4(inputs["x"])
    elif algorithm == "mbs":
        value = mbs_mxfp4(inputs["x"])
    elif algorithm == "mxfp8_e5m2":
        value = mxfp8_e5m2(inputs["x"])
    else:
        value = hifx(inputs["x"], bits)
    # HiFX narrows every intermediate with ties-to-even, the rest of the family with ties-away;
    # the final store follows whichever the variant used throughout.
    return {"out": bf16(value, "even" if algorithm == "hifx" else "away")}
