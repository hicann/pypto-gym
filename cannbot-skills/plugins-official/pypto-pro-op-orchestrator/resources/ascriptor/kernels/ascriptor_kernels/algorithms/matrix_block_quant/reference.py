# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent E5M2 grid, nearest-even encoding, and the FP64 transposed product the payload and scale answer to."""

import math
import numpy as np
import torch

# ----------------------------------------------------------------------------------------------------
# e5m2.py
# Independent finite E5M2 grid and nearest-even byte encoding.
# ----------------------------------------------------------------------------------------------------

def positive_values():
    return np.array([math.ldexp((code & 3) / 4, -14) if code < 4 else math.ldexp(1 + (code & 3) / 4, (code >> 2) - 15) for code in range(124)], dtype=np.float64)


def encode(values):
    data = values.detach().cpu().float().numpy()
    magnitude = np.abs(data).astype(np.float64)
    table = positive_values()
    if not np.isfinite(magnitude).all() or (magnitude > table[-1]).any():
        raise ValueError("This finite quantization reference requires values within the E5M2 finite range")
    high = np.searchsorted(table, magnitude, side="left").clip(0, len(table) - 1)
    low = np.maximum(high - 1, 0)
    lower_distance = magnitude - table[low]
    upper_distance = table[high] - magnitude
    choose_high = (upper_distance < lower_distance) | ((upper_distance == lower_distance) & ((high & 1) == 0))
    code = np.where(choose_high, high, low).astype(np.uint8)
    code |= np.where(np.signbit(data), 128, 0).astype(np.uint8)
    return torch.from_numpy(code.copy())

# ----------------------------------------------------------------------------------------------------
# oracle.py
# Independent transposed product, source FP32 scale, and E5M2 payload bytes.
# ----------------------------------------------------------------------------------------------------

def quantize(inputs):
    product = (inputs["x"].double().T @ inputs["y"].double()).float()
    maximum = product.abs().amax(-1, keepdim=True)
    if (maximum <= 1e-6).any():
        raise ValueError("Source block quantization has no epsilon; every block maximum must exceed1e-6")
    scale = maximum / 224.0
    return {"payload": encode(product / scale), "scale": scale}

# ----------------------------------------------------------------------------------------------------
# inputs
# The deterministic input recipe each case names, and the public reference surface.
# ----------------------------------------------------------------------------------------------------

def make_inputs(case):
    """Deterministic K-major FP16 X and Y for one case. The `midpoints` distribution is not
    random at all: X is 16*I and Y is a table of products divided by 16, so the FP32 product is
    exactly that table, its row maximum is exactly 224, and the scale is exactly 1. Every
    normalized value then lands on an E5M2 midpoint, where the payload byte is decided by the
    nearest-even rule and nothing else."""
    parameters = case["parameters"]
    generator = torch.Generator().manual_seed(case["seed"])
    inputs = dict(parameters)
    if parameters["distribution"] == "midpoints":
        levels = torch.tensor([0, 1.125, 1.375, 1.625, 1.875, 2.25, 2.75, 3.25, 3.75, -1.125, -1.375, -1.625, -1.875, -3.25, 144, 176, 208, -208])
        row = torch.arange(128).reshape(-1, 1)
        column = torch.arange(128).reshape(1, -1)
        product = levels[(row * 3 + column) % len(levels)]
        product[:, -1] = 224
        inputs["x"] = (16 * torch.eye(128)).half()
        inputs["y"] = (product / 16).half()
    else:
        inputs["x"] = torch.randn((parameters["K"], parameters["M"]), generator=generator).half()
        inputs["y"] = torch.randn((parameters["K"], 128), generator=generator).half()
    return inputs


def reference(inputs):
    return quantize(inputs)
