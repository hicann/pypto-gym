# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference for the BF16 backward stage, with the two precision models
it depends on: IEEE nearest-ties-away narrowing and an enumerated HiFloat8 value table."""

import functools
import math
import torch

# ----------------------------------------------------------------------------------------------------
# ieee_narrow.py
# Independent IEEE FP32-to-FP16/BF16 nearest, ties-away narrowing.
#
# This independent host reference encodes exponent/fraction/sign bits directly.
# It calls neither the execution codec nor Torch's narrowing conversion. Values
# are interpreted as FP32, matching the source tensor-vector cast input.
# ----------------------------------------------------------------------------------------------------

def narrow_away(values, dtype):
    values=values.float().contiguous()
    if not torch.isfinite(values).all():
        raise ValueError('This generated-attention reference requires finite FP32 cast inputs')
    word=values.view(torch.int32).to(torch.int64)
    sign=(word & 0x80000000)>>16
    magnitude=word & 0x7fffffff
    if dtype==torch.bfloat16:
        encoded=((magnitude+0x8000)>>16)|sign
    elif dtype==torch.float16:
        exponent=magnitude>>23
        # Drop thirteen fraction bits with a half-unit increment: every exact
        # midpoint rounds upward in magnitude, including a carry into infinity.
        normal=((magnitude-(112<<23)+(1<<12))>>13).clamp(0,0x7c00)
        # Binary powers preserve FP32 values exactly in FP64. A half subnormal
        # has quantum2^-24; explicitly round its nonnegative integer mantissa.
        subnormal=torch.floor(values.abs().double().clamp(max=2**-14)*(2**24)+0.5).to(torch.int64)
        encoded=torch.where(exponent>=113,normal,subnormal)|sign
    else:
        raise ValueError('Only FP16/BF16 are in this source narrowing contract')
    return encoded.to(torch.int16).view(dtype)

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
# Independent saved-state preparation and source-precision backward stage.
# ----------------------------------------------------------------------------------------------------

def forward(q, k, v):
    scores = (q.float() @ k.float().transpose(-1, -2)) * (128 ** -0.5)
    maximum = scores.amax(dim=-1)
    unnormalized = torch.exp(scores - maximum.unsqueeze(-1))
    denominator = unnormalized.sum(dim=-1)
    out = (unnormalized / denominator.unsqueeze(-1)) @ v.float()
    return maximum, denominator, out.bfloat16()


def backward(inputs):
    q, k, v, o, grad = (inputs[name].float() for name in ("q", "k", "v", "o", "grad"))
    score = (q @ k.transpose(-1, -2)) * (128 ** -0.5)
    probability = torch.exp(score - inputs["qkmax"].unsqueeze(-1)) / inputs["qksum"].unsqueeze(-1)
    dprobability = grad @ v.transpose(-1, -2)
    delta = (o * grad).sum(dim=-1, keepdim=True)
    dscore = narrow_away(probability * (dprobability - delta) * (128 ** -0.5), torch.bfloat16).float()
    stored_probability = narrow_away(quantize(probability), torch.bfloat16).float()
    return {"gq": narrow_away(dscore @ k, torch.bfloat16), "gk": narrow_away(dscore.transpose(-1, -2) @ q, torch.bfloat16), "gv": narrow_away(stored_probability.transpose(-1, -2) @ grad, torch.bfloat16)}


def make_inputs(case):
    """Q/K/V, the upstream gradient and the saved row statistics for one case.

    The two `state_mode` values are two different claims. `prepared` overwrites O with this
    module's own forward output, so the saved state really is the state a forward pass would
    have left and the gradients are the gradients of that attention. `source_arbitrary` keeps
    the randomly drawn O: it exercises the stage ABI at a saved state no forward produced, and
    its result is not a derivative of anything."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    result = dict(p)
    for name in ("q", "k", "v", "o", "grad"):
        rows = p["S1"] if name in ("q", "o", "grad") else p["S2"]
        result[name] = torch.randn((p["B"], p["H"], rows, 128), generator=generator).bfloat16()
    result["qkmax"], result["qksum"], prepared_o = forward(result["q"], result["k"], result["v"])
    if p["state_mode"] == "prepared":
        result["o"] = prepared_o
    return result


def reference(inputs):
    return backward(inputs)
