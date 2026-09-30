# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent grouped FP32 format references and deterministic inputs."""

import torch

EXP_MASK = 0x7F800000
MANTISSA_TOP8_MASK = 0x007F8000
E8M0_MIN = 2.0 ** -127
E5M2_PRIVEXP_FLOOR = 2.0 ** -29
E5M2_EPS = 5.421011e-20
E5M2_UPPER = 1.75


def _floor_e8m0(value):
    safe = torch.clamp(value.float(), min=E8M0_MIN).contiguous()
    bits = safe.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return (bits & EXP_MASK).to(torch.int32).view(torch.float32).clamp_min(E8M0_MIN)


def _e2m1(values):
    absolute = values.abs()
    quant = torch.zeros_like(values)
    for threshold, level in ((0.25, 0.5), (0.75, 1.0), (1.25, 1.5), (1.75, 2.0), (2.5, 3.0), (3.5, 4.0), (5.0, 6.0)):
        quant = torch.where(absolute < threshold, quant, torch.full_like(quant, level))
    return torch.copysign(quant, values)


def plain_mxfp4(x):
    rows, cols = x.shape
    grouped = x.reshape(rows, cols // 32, 32).float()
    scale = _floor_e8m0(grouped.abs().amax(-1, keepdim=True) * 0.25)
    return (_e2m1(grouped / scale) * scale).reshape(rows, cols).contiguous()


def _macro_factor(amax):
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    reciprocal = (6.0 / safe.float()).contiguous()
    bits = reciprocal.view(torch.int32)
    top8 = (bits & MANTISSA_TOP8_MASK) >> 15
    factor = 1.0 + top8.float() / 256.0
    return torch.where(amax > 0, factor, torch.ones_like(factor))


def _a2_mbs_div(prediv, factor):
    family10_toward = {8, 10, 78, 124}
    family10_away = {232}
    family15_toward = {8, 29, 46, 91, 102, 104, 108, 122, 140}
    quotient = prediv / factor
    toward = torch.nextafter(quotient, torch.zeros_like(quotient))
    direction = torch.where(quotient >= 0, torch.full_like(quotient, torch.inf), torch.full_like(quotient, -torch.inf))
    away = torch.nextafter(quotient, direction)
    absolute = prediv.abs().contiguous()
    mantissa = absolute.view(torch.int32) & 0x007FFFFF
    nonzero = absolute != 0
    code = torch.round((factor.double() - 1.0) * 256.0).long()
    def member(values):
        result = torch.zeros_like(code, dtype=torch.bool)
        for value in values:
            result |= code == value
        return result
    to_zero = (nonzero & (mantissa == 0) & member(family10_toward)) | (nonzero & (mantissa == 0x00400000) & member(family15_toward))
    from_zero = nonzero & (mantissa == 0) & member(family10_away)
    return torch.where(from_zero, away, torch.where(to_zero, toward, quotient))


def mbs_mxfp4(x):
    rows, cols = x.shape
    macros = x.float().reshape(rows, cols // 128, 128)
    factor = _macro_factor(macros.abs().amax(-1, keepdim=True))
    normalized = (macros * factor).reshape(rows, cols // 128, 4, 32)
    scale = _floor_e8m0(normalized.abs().amax(-1, keepdim=True) * 0.25)
    prediv = (_e2m1(normalized / scale) * scale).reshape(rows, cols // 128, 128)
    return _a2_mbs_div(prediv, factor.expand_as(prediv)).reshape(rows, cols).contiguous()


def mxfp8_e5m2(x):
    rows, cols = x.shape
    grouped = x.float().reshape(rows, cols // 32, 32)
    bits = grouped.abs().contiguous().view(torch.int32) & EXP_MASK
    pexp = bits.view(torch.float32)
    group_exp = pexp.amax(-1, keepdim=True)
    private = torch.maximum(group_exp * E5M2_PRIVEXP_FLOOR, pexp) * 0.25 + E5M2_EPS
    scaled = grouped / private
    rounded = torch.copysign(torch.floor(scaled.abs() + 0.5), scaled) * private
    upper = E5M2_UPPER * group_exp
    return torch.maximum(torch.minimum(rounded, upper), -upper).reshape(rows, cols).contiguous()


def make_values(parameters, seed):
    rows, cols = parameters["rows"], parameters["cols"]
    pattern, scale = parameters["pattern"], float(parameters["scale"])
    if pattern == "zero":
        return torch.zeros((rows, cols), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((rows, cols), generator=generator) * scale
    if pattern == "e2m1_boundaries":
        thresholds = torch.tensor([0.0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
        points = torch.cat((torch.nextafter(thresholds, torch.full_like(thresholds, -torch.inf)), thresholds,
                            torch.nextafter(thresholds, torch.full_like(thresholds, torch.inf)), -thresholds))
        values.zero_()
        for first in range(0, cols, 32):
            values[:, first:first + 32] = 6.0
            count = min(32, points.numel())
            values[:, first:first + count] = points[:count]
    elif pattern == "macro_boundaries":
        values.zero_()
        values[0, 0:128] = 6.0
        values[1, 0:128] = 2.0 ** -126
        values[2, 0:128] = torch.tensor([1.5, 1.75, 2.0, 3.0] * 32)
    elif pattern == "e5m2_epsilon":
        wide = torch.logspace(-20, 20, 32)
        for first in range(0, cols, 32):
            values[:, first:first + 32] = wide
        values[1::2] *= -1
    elif pattern == "factor_reuse":
        for macro in range(cols // 128):
            values[:, macro * 128:(macro + 1) * 128] *= 2.0 ** (macro - 1)
    elif pattern != "source":
        raise ValueError("unknown pattern")
    if pattern in ("source", "factor_reuse") and rows:
        values[0, 0:8] = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 6.0, -6.0])
        if rows > 1 and cols >= 128:
            values[1, 120:128] = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    return values.contiguous()


def reconstruct(inputs):
    return {"plain_mxfp4": plain_mxfp4, "mbs_mxfp4": mbs_mxfp4, "mxfp8_e5m2": mxfp8_e5m2}[inputs["variant"]](inputs["x"])


def make_inputs(case):
    """Deterministic input for one case. Every `source` row 0 starts with the signed zeros
    and the exact E2M1 grid points, so a kernel that mishandles a zero group or a tie is
    wrong on the first group rather than only on unlucky noise."""
    p = case["parameters"]
    return {**p, "x": make_values(p, case["seed"])}


def reference(inputs):
    return {"y": reconstruct(inputs)}
