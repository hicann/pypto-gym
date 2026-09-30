# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent HiFloat8 value model, the per-task metadata table, and the streamed attention oracle.

Torch only. Every HiFloat8 value is constructed from exponent/fraction classes rather than by
calling the execution codec, and the attention is recomputed over the same effective core
partition the kernel uses, including the two-owner flash-decoding merge."""

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
# metadata.py
# Generate and independently validate the two per-task integer metadata ABIs.
# ----------------------------------------------------------------------------------------------------

FIELDS = {"meta8": ("mt", "kt", "q_base", "kv_base", "m0", "n0", "valid_m", "valid_n"),
          "meta11": ("mt", "q_base", "kv_base", "m0", "n0", "valid_m", "valid_n", "sm", "qm", "first_k", "last_k")}


def make_metadata(batches, queries, keys, variant):
    query_tiles, key_tiles = (queries + 127) // 128, (keys + 127) // 128
    task = torch.arange(batches * query_tiles * key_tiles, dtype=torch.int64)
    mt, kt = task // key_tiles, task % key_tiles
    batch, local_query = mt // query_tiles, mt % query_tiles
    columns = {"mt": mt, "kt": kt, "q_base": batch * queries, "kv_base": batch * keys,
               "m0": local_query * 128, "n0": kt * 128, "valid_m": (queries - local_query * 128).clamp(max=128),
               "valid_n": (keys - kt * 128).clamp(max=128), "sm": mt % 3, "qm": mt % 2,
               "first_k": (kt == 0).long(), "last_k": (kt == key_tiles - 1).long()}
    return torch.stack([columns[name] for name in FIELDS[variant]], dim=1).to(torch.int32).reshape(1, -1)


# ----------------------------------------------------------------------------------------------------
# oracle.py
# Independent unscaled-HiFloat8 online attention with two-owner FD merging.
# ----------------------------------------------------------------------------------------------------

def partitions(batches, queries, keys, cores):
    query_tiles, key_tiles = (queries + 127) // 128, (keys + 127) // 128
    total = batches * query_tiles * key_tiles
    boundaries = [total * core // cores for core in range(cores + 1)]
    cuts = [boundary for boundary in boundaries[1:-1] if boundary % key_tiles]
    if len(cuts) != len(set(cuts)):
        raise ValueError("Zero-work cores duplicate an interior FD boundary")
    groups = [[] for _ in range(batches * query_tiles)]
    for first, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        for tile in range(first // key_tiles, (stop + key_tiles - 1) // key_tiles):
            begin, end = max(first, tile * key_tiles), min(stop, (tile + 1) * key_tiles)
            if begin < end:
                groups[tile].append((begin % key_tiles, end - tile * key_tiles))
    if any(len(owners) > 2 for owners in groups):
        raise ValueError("Source FD merge supports at most two owners per query tile")
    return groups


def fma_update(accumulator, weight, product):
    return (accumulator.double() * weight.double() + product.double()).float()


def bf16_away(value):
    bits = value.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((bits + 0x8000) >> 16).to(torch.int16).view(torch.bfloat16)


def physical(inputs, *, quantize_probability=True):
    batches, queries, keys, cores = (inputs[name] for name in ("B", "MQ", "N", "block_dim"))
    q, k, v = (decode(inputs[name]).reshape(batches, queries if name == "q" else keys, 128) for name in ("q", "k", "v"))
    output = torch.empty((batches, queries, 128), dtype=torch.bfloat16)
    for tile, owners in enumerate(partitions(batches, queries, keys, cores)):
        batch, local_tile = divmod(tile, (queries + 127) // 128)
        rows = slice(local_tile * 128, min((local_tile + 1) * 128, queries))
        query = q[batch, rows]
        partials = []
        for first, stop in owners:
            maximum = torch.full((query.shape[0], 1), -1.0e30)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros_like(query)
            for key_tile in range(first, stop):
                columns = slice(key_tile * 128, min((key_tile + 1) * 128, keys))
                score = (query @ k[batch, columns].T) * (128 ** -0.5)
                next_maximum = torch.maximum(maximum, score.amax(dim=-1, keepdim=True))
                weight = torch.exp(maximum - next_maximum)
                probability = torch.exp(score - next_maximum)
                denominator = denominator * weight + probability.sum(dim=-1, keepdim=True)
                stored = quantize(probability) if quantize_probability else probability
                numerator = fma_update(numerator, weight, stored @ v[batch, columns])
                maximum = next_maximum
            partials.append((maximum, denominator, numerator))
        if len(partials) == 1:
            output[batch, rows] = (partials[0][2] / partials[0][1]).bfloat16()
        else:
            (m0, s0, a0), (m1, s1, a1) = partials
            weight = torch.exp(m0 - m1)
            output[batch, rows] = bf16_away((a0 * weight + a1) / (s0 * weight + s1))
    return output.reshape(batches * queries, 128)


def make_inputs(case):
    """Q/K/V are HiFloat8 carrier bytes, so the generator rounds Gaussian values through the
    same independent encoder the reference decodes with -- the kernel and the oracle then start
    from the identical set of representable values. `meta` is the per-task int32 table for the
    variant this case names, and `block_dim` is part of the recipe because the two-owner merge
    depends on the effective core partition."""
    values = {**case["parameters"], "block_dim": case["block_dim"]}
    generator = torch.Generator().manual_seed(case["seed"])
    for name in ("q", "k", "v"):
        shape = (values["B"] * values["MQ" if name == "q" else "N"], 128)
        values[name] = encode(torch.randn(shape, generator=generator))
    values["meta"] = make_metadata(*(values[name] for name in ("B", "MQ", "N", "variant")))
    return values


def reference(inputs):
    return {"out": physical(inputs)}
