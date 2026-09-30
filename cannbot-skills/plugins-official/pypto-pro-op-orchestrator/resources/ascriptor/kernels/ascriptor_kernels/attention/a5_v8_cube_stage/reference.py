# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent HiFloat8 value model, physical NZ packing and the Cube bridge mathematics.

Every HiFloat8 value is constructed from exponent/fraction classes rather than by calling the
execution codec, so the reference cannot inherit a codec defect from the thing it checks."""

import functools
import math
import torch

# ----------------------------------------------------------------------------------------------------
# hif8.py
# Independent HiFloat8 value model for attention references.
#
# Construct every value from exponent/fraction classes, rather than calling the
# execution codec. Conversion uses monotone midpoint intervals with ties away
# from zero (TA). Hybrid/SSR conversion is deliberately outside this helper.
# ----------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def value_table():
    values = [None] * 128
    values[0] = 0.0
    for code in range(1, 8):
        values[code] = math.ldexp(1.0, code - 23)
    classes = ((0, 0, 3, 0x08), (1, 1, 3, 0x10), (2, 3, 3, 0x20), (4, 7, 2, 0x40), (8, 15, 1, 0x60))
    for low, high, fraction_bits, prefix in classes:
        for exponent_sign in (1, -1) if low else (1,):
            sign_offset = (high - low + 1) * (1 << fraction_bits) if exponent_sign < 0 else 0
            for magnitude in range(low, high + 1):
                for fraction in range(1 << fraction_bits):
                    code = prefix + sign_offset + (magnitude - low) * (1 << fraction_bits) + fraction
                    value = math.ldexp(1.0 + fraction / (1 << fraction_bits), exponent_sign * magnitude)
                    values[code] = value
    values[0x6F] = math.inf
    assert all(value is not None for value in values)
    signed = values + [-value for value in values]
    signed[0x80] = math.nan
    return torch.tensor(signed, dtype=torch.float32)


@functools.lru_cache(maxsize=1)
def _intervals():
    table = value_table()[:128]
    ordered = sorted((float(value), code) for code, value in enumerate(table) if math.isfinite(value))
    levels, codes = zip(*ordered, strict=True)
    boundaries = [(low + high) / 2 for low, high in zip(levels[:-1], levels[1:], strict=True)]
    boundaries.append(40960.0)
    return torch.tensor(boundaries, dtype=torch.float64), torch.tensor((*codes, 0x6F), dtype=torch.uint8)


def decode(codes):
    if codes.dtype != torch.uint8 or codes.device.type != "cpu":
        raise ValueError("HiFloat8 reference expects CPU uint8 carrier bytes")
    return value_table()[codes.long()]


def encode(values, *, saturate=False, nan_to_zero=False):
    """Float32 TA conversion; both zero signs map to zero, NaN normally to0x80."""
    if values.device.type != "cpu":
        raise ValueError("HiFloat8 reference is CPU only")
    source = values.float()
    boundaries, codes = _intervals()
    magnitude = source.abs().double().contiguous()
    intervals = torch.bucketize(magnitude, boundaries, right=True)
    result = codes[intervals]
    if saturate:
        result = torch.where(result == 0x6F, torch.tensor(0x6E, dtype=torch.uint8), result)
    result = torch.where((source < 0) & (result != 0), result | 0x80, result)
    return torch.where(torch.isnan(source), torch.tensor(0 if nan_to_zero else 0x80, dtype=torch.uint8), result)


def quantize(values):
    return decode(encode(values))

# ----------------------------------------------------------------------------------------------------
# oracle.py
# Independent physical NZ packing and Cube bridge mathematics.
# ----------------------------------------------------------------------------------------------------

def pack(values, padding_byte=0):
    codes = encode(values)
    physical = torch.full((8, 33, 32), padding_byte, dtype=torch.uint8)
    for group in range(4):
        physical[2 * group, :32] = codes[group * 32:(group + 1) * 32, :32]
        physical[2 * group + 1, :32] = codes[group * 32:(group + 1) * 32, 32:]
    return physical.reshape(33, 256)


def unpack(physical):
    fractals = physical.reshape(8, 33, 32)
    return torch.cat([torch.cat((fractals[2 * group, :32], fractals[2 * group + 1, :32]), dim=1) for group in range(4)], dim=0)


def stage(inputs):
    q, k, v = (decode(inputs[name]) for name in ("q", "k", "v"))
    score = k @ q.T
    p = decode(unpack(inputs["pbytes"]))
    p2 = decode(unpack(inputs["pbytes2"]))
    half = p.T @ v
    half2 = p2.T @ v
    return {"score": score, "score2": torch.cat((score[:, :64], score[:, 64:]), dim=0), "score3": score.clone(), "pv": torch.cat((half, half), dim=0), "pv2": torch.cat((half2, half2), dim=0)}


def make_inputs(case):
    """HiFloat8 q/k/v built through the independent encoder, and two physically packed P
    operands. `padding_byte` fills the NZ fractal's dead row 33: a kernel that reads it as
    data produces a different answer for a poisoned pad than for a zero one."""
    generator = torch.Generator().manual_seed(case["seed"])
    padding = case["parameters"]["padding_byte"]
    qkv = {name: encode(torch.randn((128, 128), generator=generator) * 0.6) for name in ("q", "k", "v")}
    return {**qkv,
            "pbytes": pack(torch.rand((128, 64), generator=generator), padding),
            "pbytes2": pack(torch.eye(128, 64) * 16, padding)}


def reference(inputs):
    return stage(inputs)
