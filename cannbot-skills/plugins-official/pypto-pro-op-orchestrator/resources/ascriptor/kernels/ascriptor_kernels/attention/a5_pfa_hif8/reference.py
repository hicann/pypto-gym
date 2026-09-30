# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent HiFloat8 value model and the per-variant flash-decoding arithmetic.

Torch only. Every HiFloat8 value is constructed from exponent/fraction classes rather than by
calling the execution codec, the FixP19 fixpipe scalar is rebuilt with frexp/ldexp at the same
boundary the hardware truncates it, and the FP16 variants replay their exact key reduction
order -- so a difference in any of those shows up in the comparison instead of being absorbed
by it. The partition follows the same core split the kernels use, including the balanced
384-key grouping v8 and v9 need and the two-owner merge."""

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
# oracle.py -- the per-variant arithmetic, the FixP19 scalar and the core partition.
# Independent finite HiFloat8 flash-decoding arithmetic and core partition.
# ----------------------------------------------------------------------------------------------------

def partitions(batches, queries, keys, cores, *, group_size=128, balanced=False):
    query_tiles, key_tiles = (queries + 127) // 128, (keys + group_size - 1) // group_size
    tasks = batches * query_tiles * key_tiles
    def boundary(core):
        return core * (tasks // cores) + min(core, tasks % cores) if balanced else tasks * core // cores
    cuts = [boundary(core) for core in range(1, cores) if boundary(core) % key_tiles]
    if len(cuts) != len(set(cuts)):
        raise ValueError("Located FD gap: zero-work cores duplicate an interior merge boundary")
    if len({cut // key_tiles for cut in cuts}) != len(cuts):
        raise ValueError("FD source supports at most two owners per query tile")
    groups = [[] for _ in range(batches * query_tiles)]
    for core in range(cores):
        first, stop = boundary(core), boundary(core + 1)
        for tile in range(first // key_tiles, (stop + key_tiles - 1) // key_tiles):
            begin, end = max(first, tile * key_tiles), min(stop, (tile + 1) * key_tiles)
            if begin < end:
                groups[tile].append((begin % key_tiles, end - tile * key_tiles))
    return groups


def bf16_away(value):
    """Finite float32 nearest BF16, ties away from zero, using integer bits."""
    bits = value.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = ((bits + 0x8000) >> 16).to(torch.int16)
    return rounded.view(torch.bfloat16)


def half(value):
    return value.half().float()


def fixpipe_scale(value):
    """Positive finite scalar scale with the FixP19 ten-fraction-bit contract."""
    mantissa, exponent = math.frexp(float(torch.tensor(value, dtype=torch.float32)))
    return math.ldexp(math.floor(mantissa * 2048.0), exponent - 11)


def half_tile_sum(probability, order):
    """Source FP16 key reduction, including its pair-lane and four-way tree."""
    padded = torch.zeros((probability.shape[0], 128))
    padded[:, :probability.shape[1]] = probability
    if order == "rows":
        total = torch.zeros((probability.shape[0], 1))
        for key in range(128):
            total = half(total + padded[:, key:key + 1])
        return total
    pairs = padded.reshape(probability.shape[0], 64, 2)
    if order == "pairs":
        total = torch.zeros_like(pairs[:, 0])
        for pair in range(64):
            total = half(total + pairs[:, pair])
    else:
        lanes = []
        for lane in range(4):
            total = torch.zeros_like(pairs[:, 0])
            indices = range(lane * 16, (lane + 1) * 16) if order == "blocks" else range(lane, 64, 4)
            for pair in indices:
                total = half(total + pairs[:, pair])
            lanes.append(total)
        total = half(half(lanes[0] + lanes[1]) + half(lanes[2] + lanes[3]))
    return half(total[:, :1] + total[:, 1:])


def physical(inputs, *, ignore_probability_quantization=False):
    b, mq, n, cores = (inputs[name] for name in ("B", "MQ", "N", "block_dim"))
    q, k, v = (decode(inputs[name]).reshape(b, mq if name == "q" else n, 128) for name in ("q", "k", "v"))
    variant = inputs["variant"]
    grouped = variant in ("v8", "v9")
    half_state = variant == "v9" or variant.startswith(("study_v2", "study_v4"))
    scaled_probability = variant in ("nz", "v8")
    group_size = 384 if grouped else 128
    groups = partitions(b, mq, n, cores, group_size=group_size, balanced=grouped)
    output = torch.empty((b, mq, 128), dtype=torch.bfloat16)
    for tile, owners in enumerate(groups):
        batch, local_tile = divmod(tile, (mq + 127) // 128)
        rows = slice(local_tile * 128, min((local_tile + 1) * 128, mq))
        query = q[batch, rows]
        partials = []
        for first, stop in owners:
            maximum = torch.full((query.shape[0], 1), -60000.0 if half_state else -1.0e30)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros_like(query)
            for key_tile in range(first, stop):
                cols = slice(key_tile * group_size, min((key_tile + 1) * group_size, n))
                in_fixpipe = variant == "v9" or variant.startswith(("study_v1", "study_v2"))
                scale = fixpipe_scale(128 ** -0.5) if in_fixpipe else 128 ** -0.5
                score = (query @ k[batch, cols].T) * scale
                if half_state:
                    if variant.startswith("study_v4"):
                        score = half(half(query @ k[batch, cols].T) * float(torch.tensor(128 ** -0.5).half()))
                    else:
                        score = half(score)
                new_maximum = torch.maximum(maximum, score.amax(dim=-1, keepdim=True))
                weight = half(torch.exp(half(maximum - new_maximum))) if half_state else torch.exp(maximum - new_maximum)
                exponential_max = new_maximum - math.log(16.0) if scaled_probability else new_maximum
                probability = half(torch.exp(half(score - exponential_max))) if half_state else torch.exp(score - exponential_max)
                stored = probability if ignore_probability_quantization else quantize(probability)
                if half_state:
                    denominator = half(denominator * weight)
                    for begin in range(0, probability.shape[1], 128):
                        part = probability[:, begin:begin + 128]
                        if variant == "v9":
                            order = "pairs" if part.shape[1] < 128 or probability.shape[1] <= 128 else "interleaved" if begin == 0 and probability.shape[1] <= 256 else "blocks"
                        else:
                            order = "rows" if part.shape[1] < 128 else "interleaved"
                        denominator = half(denominator + half_tile_sum(part, order))
                else:
                    denominator = denominator * weight + probability.sum(dim=-1, keepdim=True)
                pv = stored @ v[batch, cols]
                if variant == "nd":
                    numerator = numerator * weight + pv
                else:
                    numerator = (numerator.double() * weight.double() + pv.double()).float()
                maximum = new_maximum
            partials.append((maximum, denominator, numerator))
        if len(partials) == 1:
            output[batch, rows] = (partials[0][2] / partials[0][1]).bfloat16()
        else:
            (m0, s0, a0), (m1, s1, a1) = partials
            weight = torch.exp(m0 - m1)
            output[batch, rows] = bf16_away((a0 * weight + a1) / (s0 * weight + s1))
    return output.reshape(b * mq, 128)


def make_inputs(case):
    """Q/K/V are HiFloat8 carrier bytes, so the generator rounds Gaussian values through the
    same independent encoder the reference decodes with -- kernel and oracle start from the
    identical set of representable values. `variant` selects the arithmetic the oracle replays
    and `block_dim` is part of the recipe because the flat task partition, and therefore which
    query tiles get two owners, depends on it."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    rows = {"q": p["B"] * p["MQ"], "k": p["B"] * p["N"], "v": p["B"] * p["N"]}
    return {**p, "block_dim": case["block_dim"],
            **{name: encode(torch.randn((count, 128), generator=generator))
               for name, count in rows.items()}}


def reference(inputs):
    return {"out": physical(inputs)}
