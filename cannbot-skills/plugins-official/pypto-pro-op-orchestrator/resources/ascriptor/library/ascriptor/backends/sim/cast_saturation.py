# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Integer-destination conversion: round source values, then clamp or retain low bits."""

import torch


def round_integer(values: torch.Tensor, mode: str) -> torch.Tensor:
    match mode:
        case "to_even" | "rint" | "none" | "CAST_RINT":
            return torch.round(values)
        case "round" | "away" | "CAST_ROUND":
            return torch.sign(values) * torch.floor(values.abs() + 0.5)
        case "floor" | "CAST_FLOOR":
            return torch.floor(values)
        case "ceil" | "CAST_CEIL":
            return torch.ceil(values)
        case _:
            return torch.trunc(values)


def convert_integer(values: torch.Tensor, target: torch.dtype, mode: str, saturate: bool) -> torch.Tensor:
    bounds = torch.iinfo(target)
    if not values.is_floating_point():
        return values.clamp(bounds.min, bounds.max).to(target) if saturate else values.to(target)

    # Work in float64 before ties-away's +0.5. In float32 it can round a value just below
    # a half tie up to the tie, or change an already-integral odd value above 2**23.
    rounded = round_integer(values.double(), mode)
    if saturate:
        # Compare with the EXACT power-of-two upper bound before converting. INT64_MAX is not
        # representable as float64, so clamping a float to it would produce 2**63 and overflow.
        low = rounded < bounds.min
        high = rounded >= bounds.max + 1
        inside = torch.isfinite(rounded) & ~low & ~high
        result = torch.where(inside, rounded, 0.0).to(torch.int64)
        result = torch.where(low, bounds.min, result)
        result = torch.where(high, bounds.max, result)
    else:
        # fmod keeps small negative integers exact; remainder(-1, 2**64) cannot do so in float64.
        residual = torch.fmod(rounded, float(2 ** bounds.bits))
        half = float(2 ** (bounds.bits - 1))
        residual = torch.where(residual >= half, residual - 2 * half, residual)
        residual = torch.where(residual < -half, residual + 2 * half, residual)
        result = torch.where(torch.isfinite(rounded), residual, 0.0).to(torch.int64)
        # Non-finite conversion is independent of CTRL/RS: NaN -> 0, infinities -> extrema.
        # Measured on f32 -> i16/i32/i64; torch's native conversions disagree (D-233).
        result = torch.where(torch.isneginf(rounded), bounds.min, result)
        result = torch.where(torch.isposinf(rounded), bounds.max, result)
    return result.to(target)
