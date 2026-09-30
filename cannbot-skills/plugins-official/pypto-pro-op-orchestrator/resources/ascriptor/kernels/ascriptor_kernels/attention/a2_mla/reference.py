# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference for MLA: head sharing, both score feature groups, and the
same online-softmax tiling the kernel uses, with an independent FP16 narrowing model."""

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
# oracle.py
# Independent head-sharing and512+64-feature online attention.
# ----------------------------------------------------------------------------------------------------

def attention(inputs):
    s, skv = inputs["S"], inputs["SKV"]
    qn = inputs["q_nope"].reshape(8, s, 512).float()
    qr = inputs["q_rope"].reshape(8, s, 64).float()
    kn = inputs["k_nope"].reshape(4, skv, 512).repeat_interleave(2, dim=0).float()
    kr = inputs["k_rope"].reshape(4, skv, 64).repeat_interleave(2, dim=0).float()
    v = inputs["v"].reshape(4, skv, 512).repeat_interleave(2, dim=0).float()
    maximum = torch.full((8, s, 1), -1.0e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(qn)
    for first in range(0, skv, 128):
        stop = min(first + 128, skv)
        scores = (qn @ kn[:, first:stop].transpose(-1, -2) + qr @ kr[:, first:stop].transpose(-1, -2)) * (576 ** -0.5)
        next_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
        weight = torch.exp(maximum - next_maximum)
        probability = torch.exp(scores - next_maximum)
        denominator = denominator * weight + probability.sum(dim=-1, keepdim=True)
        numerator = numerator * weight + narrow_away(probability, torch.float16).float() @ v[:, first:stop]
        maximum = next_maximum
    return (numerator / denominator).reshape(8 * s, 512)


def _head_major(generator, sequence, heads, width):
    """Draw [1, sequence, heads, width] and flatten to the head-major [heads * sequence, width]
    the kernel reads: every head owns one contiguous run of rows."""
    source = torch.randn((1, sequence, heads, width), generator=generator).half()
    return source.permute(0, 2, 1, 3).reshape(heads * sequence, width).contiguous()


def make_inputs(case):
    """Q/K in both feature groups plus V for one case.

    `v_mode` decides whether V is a copy of k_nope or an independently drawn tensor. The
    copy is the source layout, where the same storage serves both roles; the independent
    mode is what catches a kernel that reads V out of the K tile it already had on chip."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    s, skv = p["S"], p["SKV"]
    result = dict(p)
    result["q_nope"] = _head_major(generator, s, 8, 512)
    result["q_rope"] = _head_major(generator, s, 8, 64)
    result["k_nope"] = _head_major(generator, skv, 4, 512)
    result["k_rope"] = _head_major(generator, skv, 4, 64)
    result["v"] = (result["k_nope"].clone() if p["v_mode"] == "k_nope"
                   else _head_major(generator, skv, 4, 512))
    return result


def reference(inputs):
    return {"out": attention(inputs)}
