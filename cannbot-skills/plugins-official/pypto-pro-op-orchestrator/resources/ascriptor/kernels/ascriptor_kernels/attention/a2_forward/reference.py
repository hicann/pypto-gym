# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference for every a2_forward variant, plus the two precision models
the family depends on: IEEE nearest-ties-away narrowing and an enumerated HiFloat8 table.

MODELS carries what each variant's arithmetic and ABI actually are, so one formula serves
all nine: the key-group size the online softmax rescales at, the dtype the probabilities are
stored in, whether the probabilities pass through HiFloat8, whether the mask is causal, and
whether the variant publishes normalized attention or unnormalized partial PV."""

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
# Independent online attention and partial PV formulas.
# ----------------------------------------------------------------------------------------------------

def attention(inputs, model):
    bh, s1, s2 = (inputs[name] for name in ("BH", "S1", "S2"))
    dim = inputs["D"]
    q = inputs["q"].reshape(bh, s1, dim).float()
    if model.get("entry_abi") == "gqa":
        k, v = (inputs[name].reshape(inputs["B"], inputs["HKV"], s2, dim).repeat_interleave(inputs["HQ"] // inputs["HKV"], dim=1).reshape(bh, s2, dim).float() for name in ("k", "v"))
    else:
        k, v = (inputs[name].reshape(bh, s2, dim).float() for name in ("k", "v"))
    maximum = torch.full((bh, s1, 1), -1.0e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q)
    partials = []
    rows = torch.arange(s1).view(1, s1, 1)
    for start in range(0, s2, model["group_size"]):
        stop = min(start + model["group_size"], s2)
        score = (q @ k[:, start:stop].transpose(-1, -2)) * (dim ** -0.5)
        if model.get("mask") == "causal":
            score = score.masked_fill(torch.arange(start, stop).view(1, 1, -1) > rows, -1.0e30)
        elif inputs.get("is_causal", 0):
            score = score.masked_fill(torch.arange(start, stop).view(1, 1, -1) // 32 > rows // 32, -1.0e30)
        next_maximum = torch.maximum(maximum, score.amax(dim=-1, keepdim=True))
        weight = torch.exp(maximum - next_maximum)
        probability = torch.exp(score - next_maximum)
        denominator = denominator * weight + probability.sum(dim=-1, keepdim=True)
        if model["quantize_hif8"]:
            # The pj_* sources store the probabilities through HiFloat8, scaled by 128 so the
            # whole [0, 1] range lands in the part of the format that has fraction bits.
            probability = quantize(probability * 128) / 128
        stored = narrow_away(probability, getattr(torch, model["input_dtype"])).float()
        pv = stored @ v[:, start:stop]
        partials.append(pv)
        numerator = numerator * weight + pv
        maximum = next_maximum
    if model["partial_pv"]:
        return {"pv": torch.stack(partials, dim=1).reshape(-1, dim)}
    output = numerator / denominator
    if model["output_dtype"] != "float32":
        output = narrow_away(output, getattr(torch, model["output_dtype"]))
    result = {"out": output.reshape(bh * s1, dim)}
    if model["stats"]:
        result.update(rowmax=maximum.reshape(-1), rowsum=denominator.reshape(-1))
    return result


# What each variant's arithmetic and ABI actually are. `group_size` is the key span the
# online softmax rescales at, `input_dtype` is the precision the probabilities are stored in
# before PV (not only the precision of Q/K/V), `stats` says whether rowmax/rowsum are
# published, and `partial_pv` marks the one variant that is not attention at all.
MODELS = {
    "dense_fp16": {"group_size": 512, "partial_pv": False, "quantize_hif8": False,
                   "input_dtype": "float16", "output_dtype": "float32", "stats": False,
                   "D": 128, "entry_abi": "plain"},
    "gqa_bf16": {"group_size": 512, "partial_pv": False, "quantize_hif8": False,
                 "input_dtype": "bfloat16", "output_dtype": "float32", "stats": True,
                 "D": 128, "entry_abi": "gqa"},
    "mha_bf16": {"group_size": 512, "partial_pv": False, "quantize_hif8": False,
                 "input_dtype": "bfloat16", "output_dtype": "float32", "stats": True,
                 "D": 128, "entry_abi": "stats"},
    "mha_d256": {"group_size": 512, "partial_pv": False, "quantize_hif8": False,
                 "input_dtype": "bfloat16", "output_dtype": "float32", "stats": False,
                 "D": 256, "entry_abi": "d256"},
    "mha_d256_bf16": {"group_size": 512, "partial_pv": False, "quantize_hif8": False,
                      "input_dtype": "bfloat16", "output_dtype": "bfloat16", "stats": False,
                      "D": 256, "entry_abi": "d256"},
    "pj_bf16_lag2": {"group_size": 128, "partial_pv": False, "quantize_hif8": True,
                     "input_dtype": "bfloat16", "output_dtype": "bfloat16", "stats": True,
                     "D": 128, "entry_abi": "stats"},
    "pj_bf16_causal": {"group_size": 128, "partial_pv": False, "quantize_hif8": True,
                       "input_dtype": "bfloat16", "output_dtype": "bfloat16", "stats": True,
                       "D": 128, "entry_abi": "stats", "mask": "causal"},
    "pj_fp16_lag1": {"group_size": 128, "partial_pv": False, "quantize_hif8": True,
                     "input_dtype": "float16", "output_dtype": "float32", "stats": True,
                     "D": 128, "entry_abi": "stats"},
    "pv_stage": {"group_size": 128, "partial_pv": True, "quantize_hif8": False,
                 "input_dtype": "float16", "output_dtype": "float32", "stats": False,
                 "D": 128, "entry_abi": "plain"},
}


def make_inputs(case):
    """Q/K/V for one case, flattened head-major, at the variant's own input dtype.

    The gqa ABI is the one that changes the shapes: K and V carry HKV heads while Q carries
    HQ, so a kernel that assumed one KV head per query head reads the wrong rows rather than
    the wrong values -- which is why `reference` expands K and V with repeat_interleave
    instead of the kernel's head arithmetic."""
    p = case["parameters"]
    model = MODELS[p["variant"]]
    generator = torch.Generator().manual_seed(case["seed"])
    result = dict(p)
    if model["entry_abi"] == "gqa":
        result["BH"] = p["B"] * p["HQ"]
    for name in ("q", "k", "v"):
        heads = (p["B"] * (p["HQ"] if name == "q" else p["HKV"])
                 if model["entry_abi"] == "gqa" else p["BH"])
        rows = heads * p["S1" if name == "q" else "S2"]
        result[name] = torch.randn((rows, p["D"]), generator=generator).to(
            getattr(torch, model["input_dtype"]))
    return result


def reference(inputs):
    return attention(inputs, MODELS[inputs["variant"]])
