# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent HiFloat8 value model, the stage's arithmetic, and the measured native-exp budget.

Every HiFloat8 value is constructed from exponent/fraction classes rather than by calling the
execution codec. `candidates` propagates a one-ULP FP32 interval around a 100-digit exp through
the same ties-away conversion, which is what the one_of comparison in main.py consumes."""

import functools
import math
import struct
from decimal import Decimal, localcontext

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
# Independent V8 probability bytes, online state and PV accumulation.
# ----------------------------------------------------------------------------------------------------

def pack(codes, initial, rows=128):
    physical = initial.clone().reshape(8, 33, 32)
    for key in range(rows):
        group, row = divmod(key, 32)
        physical[2 * group, row] = codes[key, :32]
        physical[2 * group + 1, row] = codes[key, 32:]
    return physical.reshape(33, 256)


def stage(inputs):
    scores = [inputs[name] * (128 ** -0.5) for name in ("s0", "s1", "s2")]
    maximum = torch.cat(scores, dim=0).amax(dim=0, keepdim=True)
    exponent_max = maximum - math.log(16.0)
    probabilities = [torch.exp(score - exponent_max) for score in scores]
    sums = [probabilities[0].sum(dim=0, keepdim=True)]
    sums.append(sums[-1] + probabilities[1].sum(dim=0, keepdim=True))
    sums.append(sums[-1] + probabilities[2][:3].sum(dim=0, keepdim=True))
    state = {f"st{index + 1}": torch.cat((maximum, value), dim=0) for index, value in enumerate(sums)}
    next_maximum = torch.maximum(maximum, torch.cat(scores[:2], dim=0).amax(dim=0, keepdim=True))
    old_weight = torch.exp(maximum - next_maximum)
    state["st4"] = torch.cat((next_maximum, sums[-1] * old_weight, old_weight), dim=0)
    # The source uses an FMA in its accumulator update. Float64 intermediates
    # round once to FP32 for these finite FP32-generated operands.
    accumulator = (inputs["acc_in"].double() * inputs["w_in"].T.double() + inputs["pv_in"].double()).float()
    return {**{f"p{index}": pack(encode(probability), inputs["zp"], 3 if index == 2 else 128) for index, probability in enumerate(probabilities)}, **state, "acc_out": accumulator, "fin": (accumulator / sums[-1].T).bfloat16()}

# ----------------------------------------------------------------------------------------------------
# exp_budget.py
# Independent FP32 exponent budget propagated through HiFloat8 TA conversion.
# ----------------------------------------------------------------------------------------------------

def rounded_exp(arguments):
    """Round a 100-digit exp to nearest FP32, checking adjacent words and ties.

    Finite stage inputs produce finite exponent arguments. Below -110 the
    correctly rounded FP32 result is zero; no Decimal exponent-range extension
    is needed for very negative finite score differences.
    """
    result = []
    with localcontext() as context:
        context.prec = 100
        for x in arguments.flatten().tolist():
            if x < -110:
                result.append(0.)
                continue
            exact = Decimal.from_float(x).exp()
            word = struct.unpack("<I", struct.pack("<f", float(exact)))[0]
            neighbors = [i for i in (word - 1, word, word + 1) if 0 <= i < 0x7F800000]
            def key(i, target=exact):
                value = struct.unpack("<f", struct.pack("<I", i))[0]
                return abs(Decimal.from_float(value) - target), i & 1
            chosen = min(neighbors, key=key)
            result.append(struct.unpack("<f", struct.pack("<I", chosen))[0])
    return torch.tensor(result, dtype=torch.float32).reshape(arguments.shape)


def candidates(inputs, *, ulps=1):
    """Two exact byte candidates per position; padding never changes.

    The one-step FP32 neighbors allow a different code only at a quantization
    boundary. The interval is narrower than a HiFloat8 bin at that magnitude,
    so its two encoded endpoints enumerate all possible codes in that interval.
    Subnormal FP32 values, including flushed zeros, encode to HiFloat8 zero.
    """
    if isinstance(ulps, bool) or ulps != 1:
        raise ValueError("This measured contract supports exactly one FP32 ULP")
    scores = [inputs[name] * (128 ** -0.5) for name in ("s0", "s1", "s2")]
    exponent_max = torch.cat(scores).amax(dim=0, keepdim=True) - math.log(16.)
    result = {}
    for index, score in enumerate(scores):
        rows = 3 if index == 2 else 128
        center = rounded_exp(score[:rows] - exponent_max)
        low = torch.nextafter(center, torch.full_like(center, -math.inf)).clamp_min(0)
        high = torch.nextafter(center, torch.full_like(center, math.inf))
        result[f"p{index}"] = [pack(encode(value), inputs["zp"], rows) for value in (low, high)]
    return result


NATIVE_EXP_ULP_BUDGET = 1  # measured on A5 V8 only; not a universal native-exp guarantee


def make_inputs(case):
    """Scores are scaled by 18 so the exp arguments span the range where a one-ULP FP32
    difference can cross a HiFloat8 quantization boundary. `zp` is the packed-byte padding
    the stage must overwrite in full."""
    generator = torch.Generator().manual_seed(case["seed"])
    padding = case["parameters"]["padding_byte"]
    scores = {name: torch.randn((128, 64), generator=generator) * 18 for name in ("s0", "s1", "s2")}
    return {**scores,
            "acc_in": torch.randn((64, 128), generator=generator),
            "pv_in": torch.randn((64, 128), generator=generator),
            "w_in": torch.rand((1, 64), generator=generator),
            "zp": torch.full((33, 256), padding, dtype=torch.uint8)}


def reference(inputs):
    return stage(inputs)


def reference_candidates(inputs):
    """The admissible HiFloat8 encodings of each P element: the two endpoints of the
    adjacent-FP32 interval around an independently rounded 100-digit exp, but only where that
    interval actually crosses a quantization boundary. Everywhere else there is one candidate."""
    return candidates(inputs, ulps=NATIVE_EXP_ULP_BUDGET)
