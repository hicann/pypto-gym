# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent HiFloat8 value model and dense MQA mathematics.

Every HiFloat8 value is constructed from exponent/fraction classes rather than by calling the
execution codec, so the reference cannot inherit a codec defect from the thing it checks. The
decoded operands then go through one dense FP32 attention with a single rounding to BF16 at the
end: no tiling, no online softmax and no partial merge."""

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
# Independent dense MQA/batch mathematics.
# ----------------------------------------------------------------------------------------------------

def attention(q, k, v, batches, queries, keys):
    q = q.reshape(batches, queries, 128).float()
    k = k.reshape(batches, keys, 128).float()
    v = v.reshape(batches, keys, 128).float()
    probability = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(128), dim=-1)
    return (probability @ v).reshape(batches * queries, 128).to(torch.bfloat16)


def make_inputs(case):
    """One standard-normal draw per operand, each put into the carrier it actually travels in: Q
    and K through the independent HiFloat8 encoder into uint8 bytes, V to BF16. The case
    parameters travel on in the returned dict because B/MQ/N are kernel arguments as well as
    tensor geometry."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    result = dict(p)
    for name in ("q", "k", "v"):
        value = torch.randn((p["B"] * p["MQ" if name == "q" else "N"], 128), generator=generator)
        result[name] = value.bfloat16() if name == "v" else encode(value)
    return result


def reference(inputs):
    return {"out": attention(decode(inputs["q"]), decode(inputs["k"]), inputs["v"],
                             inputs["B"], inputs["MQ"], inputs["N"])}
