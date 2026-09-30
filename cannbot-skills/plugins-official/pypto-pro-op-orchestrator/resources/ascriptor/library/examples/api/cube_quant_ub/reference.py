# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent product, FixP19, rounding and saturation formulas."""
import struct

import torch

GEOMETRY = {"qf_i8": (16,32,48), "qf_u8": (16,32,48), "rq_i8": (16,32,64),
            "deq_f16": (16,32,64), "scaled_f16": (16,32,48),
            "relu_i8": (16,32,48), "split_i8": (32,64,48)}


def make_inputs(case):
    p = case["parameters"]
    mode = p["mode"]
    if mode not in GEOMETRY or tuple(p[name] for name in ("M","N","K")) != GEOMETRY[mode]:
        raise ValueError("Require a declared quantized UB mode/geometry")
    if case.get("block_dim",1) != 1:
        raise ValueError("This complete-tile example requires one core")
    m,n,k = GEOMETRY[mode]
    dtype = torch.int8 if mode in ("rq_i8","deq_f16") else torch.float16
    generator = torch.Generator().manual_seed(case["seed"])
    x = torch.randint(-3,4,(m,k),generator=generator).to(dtype)
    y = torch.randint(-3,4,(n,k),generator=generator).to(dtype)
    x[0] = 8
    y[0],y[1] = 8,-8
    x[2].zero_()
    y[2].zero_()
    x[2,0],y[2,0] = 1,1
    return {"mode":mode,"x":x,"y":y,"block_dim":1}


def validate_inputs(inputs,case=None):
    mode = inputs["mode"]
    if mode not in GEOMETRY or type(inputs["block_dim"]) is not int or inputs["block_dim"] != 1:
        raise ValueError("Unknown mode or unsupported launch count")
    m,n,k = GEOMETRY[mode]
    dtype = torch.int8 if mode in ("rq_i8","deq_f16") else torch.float16
    for name,shape in (("x",(m,k)),("y",(n,k))):
        value = inputs[name]
        if value.shape != shape or value.dtype != dtype or value.device.type != "cpu" or not value.is_contiguous() or not torch.isfinite(value).all() or not (value.float().abs() <= 8).all() or not torch.equal(value.float(),value.float().round()):
            raise ValueError("Require the declared contiguous bounded integer-valued operands")


def fp19(value):
    bits = struct.unpack("<I",struct.pack("<f",value))[0] & 0xFFFFE000
    return struct.unpack("<f",struct.pack("<I",bits))[0]


def reference(inputs):
    validate_inputs(inputs)
    mode = inputs["mode"]
    product = (inputs["x"].int() @ inputs["y"].int().T).float()
    if mode == "relu_i8":
        product = product.clamp_min(0)
    if mode in ("deq_f16","scaled_f16"):
        scale = 0.25 if mode == "deq_f16" else 0.5
        value = (product * fp19(scale)).half()
    else:
        offset = 0 if mode == "rq_i8" else 8
        rounded = (product * fp19(0.5)).round().clamp(-256,255) + offset
        lo,hi,dtype = (0,255,torch.uint8) if mode == "qf_u8" else (-128,127,torch.int8)
        value = rounded.clamp(lo,hi).to(dtype)
    return {"carriers":value.contiguous().view(torch.uint8).flatten()}
