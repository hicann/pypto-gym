# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent unscaled-HiFloat8 streamed attention with the source's own reduction orders.

Torch only. Every HiFloat8 value is constructed from exponent/fraction classes rather than by
calling the execution codec; `tile_sum` reproduces each variant's exact addition sequence, so
the comparison does not silently absorb a different reduction order; and the two-owner flash
decoding merge follows the same core partition the kernels use."""

import functools
import math
import torch

# ----------------------------------------------------------------------------------------------------
# hif8.py -- the independent HiFloat8 value model.
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
# oracle.py -- the streamed attention and the source's reduction orders.
# Independent unscaled HiFloat8 streamed attention and source reduction orders.
# ----------------------------------------------------------------------------------------------------

def partitions(batches, queries, keys, cores):
    mt, nt = (queries + 127) // 128, (keys + 127) // 128
    total = batches * mt * nt
    boundaries = [total * c // cores for c in range(cores + 1)]
    interior = [cut for cut in boundaries[1:-1] if cut % nt]
    if len(set(interior)) != len(interior):
        raise ValueError('Located source FD gap: repeated interior cuts from idle cores')
    if len({cut // nt for cut in interior}) != len(interior):
        raise ValueError('Located source FD gap: more than two owners per query tile')
    owners = [[] for _ in range(batches * mt)]
    for start, end in zip(boundaries, boundaries[1:]):
        for tile in range(start // nt, (end + nt - 1) // nt):
            first, stop = max(start, tile * nt), min(end, (tile + 1) * nt)
            if first < stop:
                owners[tile].append((first - tile * nt, stop - tile * nt))
    assert all(1 <= len(group) <= 2 for group in owners)
    return owners


def bf16_away(value):
    words = value.float().contiguous().view(torch.int32).to(torch.int64) & 0xffffffff
    return ((words + 0x8000) >> 16).to(torch.int16).view(torch.bfloat16)


def fused_update(previous, weight, pv):
    """FP32 multiply-add with one output rounding; no execution model dependency."""
    return (previous.double() * weight.double() + pv.double()).float()


def tile_sum(probability, variant):
    """Exact source addition sequence: tail rows, full slab or half-pair tree."""
    count = probability.shape[-1]
    if count < 128:
        result = torch.zeros_like(probability[:, :1])
        for key in range(count):
            result = result + probability[:, key:key + 1]
        return result
    totals = []
    for lane in range(4):
        total = torch.zeros_like(probability[:, :1])
        if variant == 'nz4':
            for key in range(lane * 32, (lane + 1) * 32):
                total = total + probability[:, key:key + 1]
        else:
            for key in range(lane, 64, 4):
                total = total + probability[:, key:key + 1]
                total = total + probability[:, key + 64:key + 65]
        totals.append(total)
    return (totals[0] + totals[1]) + (totals[2] + totals[3])


def physical(inputs, *, ignore_probability_quantization=False, probability_scale=1.0):
    b, mq, n, cores = (inputs[name] for name in ('B', 'MQ', 'N', 'block_dim'))
    q, k, v = (decode(inputs[name]).reshape(b, mq if name == 'q' else n, 128) for name in ('q', 'k', 'v'))
    result = torch.empty((b, mq, 128), dtype=torch.bfloat16)
    for tile, owners in enumerate(partitions(b, mq, n, cores)):
        batch, local = divmod(tile, (mq + 127) // 128)
        rows = slice(local * 128, min((local + 1) * 128, mq))
        query = q[batch, rows]
        partials = []
        for first, stop in owners:
            maximum = torch.full((query.shape[0], 1), -1.0e30)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros_like(query)
            for key in range(first, stop):
                cols = slice(key * 128, min((key + 1) * 128, n))
                score = (query @ k[batch, cols].T) * (128 ** -0.5)
                new_maximum = torch.maximum(maximum, score.amax(dim=-1, keepdim=True))
                weight = torch.exp(maximum - new_maximum)
                probability = torch.exp(score - new_maximum)
                stored = probability if ignore_probability_quantization else quantize(probability * probability_scale)
                denominator = denominator * weight + tile_sum(probability, inputs['variant'])
                pv = stored @ v[batch, cols]
                numerator = pv if key == first else fused_update(numerator, weight, pv)
                maximum = new_maximum
            partials.append((maximum, denominator, numerator))
        if len(partials) == 1:
            result[batch, rows] = (partials[0][2] / partials[0][1]).bfloat16()
        else:
            (m0, s0, a0), (m1, s1, a1) = partials
            weight = torch.exp(m0 - m1)
            result[batch, rows] = bf16_away((a0 * weight + a1) / (s0 * weight + s1))
    return result.reshape(b * mq, 128)


def make_inputs(case):
    """Q/K/V are HiFloat8 carrier bytes, so the generator rounds clamped Gaussian values through
    the same independent encoder the reference decodes with. The clamp to +/-8 keeps every value
    inside the HiFloat8 range the studies declare. `block_dim` is part of the recipe because the
    flat task partition -- and therefore which query tiles get two owners -- depends on it."""
    parameters = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    result = {**parameters, "block_dim": case["block_dim"]}
    for name in ("q", "k", "v"):
        rows = parameters["B"] * parameters["MQ" if name == "q" else "N"]
        result[name] = encode(torch.randn((rows, 128), generator=generator).clamp(-8, 8))
    return result


def reference(inputs):
    return {"out": physical(inputs)}
