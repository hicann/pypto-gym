# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent FP32 AdamW arithmetic with BF16 state, and the bit-level narrowing its stores are checked against."""

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
        raise ValueError('This narrowing reference requires finite FP32 cast inputs')
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
# reference.py
# Independent generated BF16-state AdamW arithmetic; no DSL imports.
# ----------------------------------------------------------------------------------------------------

SCALAR_NAMES = ("decay", "beta1", "one_minus_beta1", "beta2", "one_minus_beta2",
                "step_size", "inv_bc2_sqrt", "eps")


def prepare_scalars(hyperparameters):
    hp = hyperparameters
    if set(hp) != {"lr", "beta1", "beta2", "eps", "weight_decay", "step"}:
        raise ValueError("Require the six declared AdamW hyperparameters")
    if type(hp["step"]) is not int or hp["step"] < 1:
        raise ValueError("AdamW step must be a positive integer")
    if not all(type(hp[k]) in (int, float) and math.isfinite(hp[k]) for k in hp if k != "step"):
        raise ValueError("Hyperparameters must be finite real scalars")
    if hp["lr"] <= 0 or hp["eps"] <= 0 or hp["weight_decay"] < 0 or not (0 <= hp["beta1"] < 1 and 0 <= hp["beta2"] < 1):
        raise ValueError("Require lr/eps>0, weight_decay>=0 and beta1/beta2 in [0,1)")
    bc1 = 1.0 - hp["beta1"] ** hp["step"]
    bc2 = 1.0 - hp["beta2"] ** hp["step"]
    raw = (1.0 - hp["lr"] * hp["weight_decay"], hp["beta1"], 1.0 - hp["beta1"],
           hp["beta2"], 1.0 - hp["beta2"], hp["lr"] / bc1, 1.0 / math.sqrt(bc2), hp["eps"])
    narrowed = tuple(float(torch.tensor(value, dtype=torch.float32)) for value in raw)
    if not all(math.isfinite(value) for value in narrowed) or narrowed[-1] <= 0 or narrowed[-2] <= 0:
        raise ValueError("Host-computed FP32 scalars must remain finite with positive epsilon/bias factor")
    return narrowed


def make_inputs(case):
    parameters = case["parameters"]
    sizes = parameters["sizes"]
    if not isinstance(sizes, list) or not sizes or any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("Require a nonempty list of positive tensor element counts")
    if parameters.get("total_numel") != sum(sizes):
        raise ValueError("total_numel must equal the sum of independent tensor sizes")
    hp = {"lr": 3e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
          "weight_decay": 0.01, "step": 10, **parameters.get("hyperparameters", {})}
    scalars = prepare_scalars(hp)
    generator = torch.Generator().manual_seed(case["seed"])
    states = []
    for size in sizes:
        state = {"p": (torch.randn(1, size, generator=generator) * 0.02).bfloat16(),
                 "g": (torch.randn(1, size, generator=generator) * 1e-3).bfloat16(),
                 "m": (torch.randn(1, size, generator=generator) * 1e-3).bfloat16(),
                 "v": (torch.randn(1, size, generator=generator).abs() * 1e-3).bfloat16()}
        # A visible update in every small/tail case makes an unchanged-state
        # negative control meaningful even after BF16 storage rounding.
        state["p"][0, 0], state["g"][0, 0] = 0.0, 0.04
        state["m"][0, 0], state["v"][0, 0] = 0.03, 0.002
        if parameters.get("pattern") == "bf16_midpoints":
            if size != 2 or hp["beta1"] != 0.5:
                raise ValueError("The signed midpoint case requires two elements and beta1=0.5")
            state["p"].zero_()
            state["m"][0] = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
            state["g"][0] = torch.tensor([1.0078125, -1.0078125], dtype=torch.bfloat16)
            state["v"].fill_(0.25)
        states.append(state)
    return {"states": states, "hyperparameters": hp, "scalars": scalars,
            "block_dim": case.get("block_dim", 1)}


def state_reference(state, scalars):
    decay, beta1, one1, beta2, one2, step_size, inv_bias2, epsilon = scalars
    p, g, m, v = (state[name].float() for name in ("p", "g", "m", "v"))
    decayed = p * decay
    m_new = (m * beta1) + (g * one1)
    g_squared = g * g
    v_new = (v * beta2) + (g_squared * one2)
    denominator = (torch.sqrt(v_new) * inv_bias2) + epsilon
    correction = (m_new / denominator) * step_size
    return {"p": narrow_away(decayed - correction, torch.bfloat16),
            "m": narrow_away(m_new, torch.bfloat16), "v": narrow_away(v_new, torch.bfloat16)}


def combine(states, outputs):
    result = {name: torch.cat([out[name].flatten() for out in outputs]) for name in ("p", "m", "v")}
    result["p_update"] = result["p"].float() - torch.cat([state["p"].float().flatten() for state in states])
    return result


def reference(inputs):
    return combine(inputs["states"], [state_reference(state, inputs["scalars"]) for state in inputs["states"]])

