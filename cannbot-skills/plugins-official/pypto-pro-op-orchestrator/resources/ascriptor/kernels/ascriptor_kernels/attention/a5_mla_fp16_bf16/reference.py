# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dense FP64 MLA formula, evaluated independently for each batch and head."""

import math
import torch

def attention(inputs, *, arithmetic_dtype=torch.float64):
    p = inputs["parameters"]
    bnsd = p["layout"] == "BNSD"
    qn, qr, kn, kr, value = (inputs[name].to(arithmetic_dtype)
                            for name in ("q_nope", "q_rope", "k_nope", "k_rope", "v"))
    if not bnsd:
        qn, qr, kn, kr, value = (x.permute(0, 2, 1, 3) for x in (qn, qr, kn, kr, value))
    # Fold heads and queries into M. Broadcasting batched matmul over heads
    # can materialize Nq copies of K, even when its expanded view has zero stride.
    query = torch.cat((qn, qr), dim=-1).reshape(p["B"], p["Nq"] * p["SQ"], p["Dn"] + p["Dr"])
    key = torch.cat((kn, kr), dim=-1)[:, 0]
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(p["Dn"] + p["Dr"])
    score = score.reshape(p["B"], p["Nq"], p["SQ"], p["SKV"])
    if p["is_causal"]:
        qi = torch.arange(p["SQ"]).unsqueeze(-1)
        ki = torch.arange(p["SKV"]).unsqueeze(0)
        score = score.masked_fill(ki > qi + p["SKV"] - p["SQ"], float("-inf"))
    probability = torch.softmax(score, dim=-1).reshape(p["B"], p["Nq"] * p["SQ"], p["SKV"])
    result = torch.matmul(probability, value[:, 0]).reshape(p["B"], p["Nq"], p["SQ"], p["Dn"])
    if not bnsd:
        result = result.permute(0, 2, 1, 3)
    return result.to(getattr(torch, p["dtype"])).contiguous()


NAMES = ("q_nope", "q_rope", "k_nope", "k_rope", "v")


def shapes(p):
    """MLA's KV cache has ONE head: k_nope, k_rope and v carry `1` where the query carries Nq.
    Every query head reads the same key, which is the whole point of the layout."""
    if p["layout"] == "BSND":
        query = (p["B"], p["SQ"], p["Nq"], p["Dn"])
        key = (p["B"], p["SKV"], 1, p["Dn"])
    else:
        query = (p["B"], p["Nq"], p["SQ"], p["Dn"])
        key = (p["B"], 1, p["SKV"], p["Dn"])
    return {"q_nope": query, "q_rope": (*query[:-1], p["Dr"]),
            "k_nope": key, "k_rope": (*key[:-1], p["Dr"]), "v": key}


def make_inputs(case):
    """Values stay inside [-1, 1] so the FP64 softmax cannot overflow and the tolerance means
    what it says. `v` is k_nope's own values in separate storage — that is the absorbed-V form
    MLA uses, not an accident of the generator."""
    p = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    dtype = getattr(torch, p["dtype"])
    data = {}
    for name, shape in shapes(p).items():
        if name == "v":
            data[name] = data["k_nope"].clone()
        elif p.get("input_pattern", "uniform") == "zero":
            data[name] = torch.zeros(shape, dtype=dtype)
        else:
            data[name] = (torch.rand(shape, dtype=torch.float64, generator=generator) * 2 - 1).to(dtype)
    data["parameters"] = p
    return data


def reference(inputs):
    return {"out": attention(inputs)}
