# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent tensor and physical-address references; no DSL imports."""

import hashlib

import torch

NARROW_CASES = {
    "f16_single8": ("float16", 8),
    "bf16_single8": ("bfloat16", 8),
    "f16_dual8": ("float16", 16),
    "bf16_dual8": ("bfloat16", 16),
}
FIX_CASES = {
    "fix_contiguous": "contiguous",
    "fix_pitched_columns": "pitched_columns",
    "fix_independent_ubs": "independent_ubs",
}
SOURCE_POISON = -97.0
STAGING_POISON = -13.0


def narrow_parameters(case_id):
    dtype, live_rows = NARROW_CASES[case_id]
    return {"dtype": dtype, "live_rows": live_rows, "participant_rows": 8,
            "columns": 64, "beats": 5, "ub_pitch": 9, "l1_pitch": 16}


def fix_parameters(case_id):
    mode = FIX_CASES[case_id]
    return {"mode": mode, "input_dtype": "float16", "output_dtype": "float32",
            "cube_rows": 32, "participant_rows": 16, "columns_per_fix": 128,
            "beats": 5, "slots": 3, "ub_row_pitch": 256 if mode == "pitched_columns" else 128}


def fix_output_shape(case_id):
    return (5, 2, 2, 16, 128 if FIX_CASES[case_id] == "contiguous" else 256)


def make_inputs(case):
    g = torch.Generator().manual_seed(case["seed"])
    if case["id"] in FIX_CASES:
        if case["parameters"] != fix_parameters(case["id"]) or case.get("block_dim", 1) != 1:
            raise ValueError("Require the declared FIX geometry and one mixed core")
        x = torch.zeros((5, 32, 16), dtype=torch.float16)
        x[:, :, 0] = (torch.arange(5).reshape(-1, 1) * 8192 + torch.arange(32) * 256
                       + case["seed"] % 17 * 32).half()
        x[:, :, 1] = 1
        y = torch.zeros((2, 128, 16), dtype=torch.float16)
        y[:, :, 0] = 1
        y[:, :, 1] = torch.arange(2).reshape(-1, 1) * 2 + torch.arange(128).half() / 128
        initial = -(torch.arange(16 * 256).float().reshape(16, 256) + 10000)
        return {"case_id": case["id"], "x": x, "y": y, "initial": initial}
    if case["id"] in NARROW_CASES:
        parameters = narrow_parameters(case["id"])
        if case["parameters"] != parameters or case.get("block_dim", 1) != 1:
            raise ValueError("Require the declared narrow geometry and one mixed core")
        dtype = getattr(torch, parameters["dtype"])
        x = torch.full((5, 16, 128), SOURCE_POISON, dtype=dtype)
        beat = torch.arange(5).reshape(5, 1, 1)
        row = torch.arange(16).reshape(1, 16, 1)
        column = torch.arange(64).reshape(1, 1, 64)
        # Dyadic bounded values are exactly representable in both storage formats.
        labels = ((beat * 17 + row * 67 + column * 3 + case["seed"] % 251) % 251 - 125) / 8
        x[:, :, :64] = labels.to(dtype)
        return {"case_id": case["id"], "x": x, "y": torch.eye(64, dtype=dtype),
                "initial": torch.full((9, 128), STAGING_POISON, dtype=dtype)}
    if case["id"] not in ("three_beats", "second_seed") or case["parameters"]:
        raise ValueError("Unknown roundtrip case")
    return {
        "x": torch.randint(-2, 3, (3, 32, 128), generator=g).half(),
        "y": torch.randint(-2, 3, (16, 128), generator=g).half(),
    }


def reference(inputs):
    validate_inputs(inputs)
    if inputs.get("case_id") in FIX_CASES:
        # The oracle uses independent dense products and logical host slices;
        # it does not import FIX descriptors, tile emitters or address helpers.
        products = [inputs["x"].double() @ right.double().T for right in inputs["y"]]
        shape = fix_output_shape(inputs["case_id"])
        if FIX_CASES[inputs["case_id"]] == "contiguous":
            return {"o": torch.stack(products, dim=1).reshape(shape).float()}
        expected = inputs["initial"].expand(shape).clone()
        expected[:, 0, :, :, :128] = products[0].reshape(5, 2, 16, 128).float()
        expected[:, 1, :, :, :128] = products[0].reshape(5, 2, 16, 128).float()
        expected[:, 1, :, :, 128:] = products[1].reshape(5, 2, 16, 128).float()
        return {"o": expected}
    if "case_id" in inputs:
        live_rows = NARROW_CASES[inputs["case_id"]][1]
        published = inputs["x"][:, :, :64].float().clone()
        published[:, live_rows:, :] = 0
        return {"o": published @ inputs["y"].float().T}
    pre = (inputs["x"].float() * 2).half().float()
    return {"o": (pre @ inputs["y"].float().T).abs() + 1}


def validate_inputs(inputs, case=None):
    if inputs.get("case_id") in FIX_CASES:
        if set(inputs) != {"case_id", "x", "y", "initial"}:
            raise ValueError("FIX inputs require three tensors and a declared case ID")
        for name, shape, dtype in (("x", (5, 32, 16), torch.float16),
                                   ("y", (2, 128, 16), torch.float16),
                                   ("initial", (16, 256), torch.float32)):
            value = inputs[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype
                    or value.device.type != "cpu" or not value.is_contiguous() or not bool(torch.isfinite(value).all())):
                raise ValueError(f"{name} requires the declared finite contiguous FIX dtype/shape")
        return
    if "case_id" not in inputs:
        shapes = {"x": (3, 32, 128), "y": (16, 128)}
        dtype = torch.float16
    else:
        if inputs["case_id"] not in NARROW_CASES:
            raise ValueError("Unknown narrow dtype/geometry")
        dtype = getattr(torch, NARROW_CASES[inputs["case_id"]][0])
        shapes = {"x": (5, 16, 128), "y": (64, 64), "initial": (9, 128)}
    for name, shape in shapes.items():
        value = inputs[name]
        if value.dtype != dtype or tuple(value.shape) != shape or value.device.type != "cpu" or not value.is_contiguous() or not torch.isfinite(value).all():
            raise ValueError(f"{name} requires the declared finite contiguous CPU dtype/shape")
    if "case_id" in inputs:
        if not torch.equal(inputs["y"], torch.eye(64, dtype=dtype)):
            raise ValueError("Narrow address readback requires the declared identity matrix")
        if not torch.all(inputs["x"][:, :, 64:] == SOURCE_POISON):
            raise ValueError("The initialized full-load slack must retain its declared poison")
        if not torch.all(inputs["initial"] == STAGING_POISON):
            raise ValueError("The complete staging allocation must start with the declared poison")


def packed_reference(inputs):
    """Scalar host address formula, independent of register/NZ simulator helpers."""
    validate_inputs(inputs)
    live_rows = NARROW_CASES[inputs["case_id"]][1]
    expected = inputs["initial"].repeat(5, 2, 1, 1)
    flat = expected.reshape(5, 2, -1)
    for participant in range(2):
        for row in range(8):
            global_row = participant * 8 + row
            for column in range(64):
                offset = row * 16 + (column // 16) * 9 * 16 + column % 16
                flat[:, participant, offset] = inputs["x"][:, global_row, column] if global_row < live_rows else 0
    return expected


def compare_capture(actual, inputs):
    """Check every captured byte, including the untouched padding and guard columns."""
    expected = packed_reference(inputs)
    if actual.dtype != expected.dtype or actual.shape != expected.shape or not actual.is_contiguous():
        raise ValueError("Packing capture dtype/shape/contiguity differs from its allocation")
    actual_bytes = actual.cpu().view(torch.uint8)
    expected_bytes = expected.view(torch.uint8)
    if not torch.equal(actual_bytes, expected_bytes):
        raise ValueError("Packing/padding/guard byte comparison failed")
    raw = bytes(actual_bytes.flatten().tolist())
    return {"comparison": "bitwise", "passed": True, "shape": list(actual.shape),
            "dtype": str(actual.dtype).removeprefix("torch."), "observed_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "regions": ["four live NZ columns", "one padding row per column", "four guard NZ columns"]}
