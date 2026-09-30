# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run an independent full/stage reference handoff on CPU; this is not a kernel.

Formula: output = (2*x + y)^2; saved_affine retains the pre-square state.
Domain: float32 rows, N > 0, with finite inputs, intermediates and outputs.
The generated cases below exercise that domain; no production path is claimed.
Run this file alone with Torch installed; no sibling or repository imports.
"""

from graphlib import TopologicalSorter
import json

import torch


DAG = {"affine": (), "square": ("affine",)}
CASES = ({"seed": 7101, "n": 1}, {"seed": 7102, "n": 64},
         {"seed": 7103, "n": 65}, {"seed": 7104, "n": 257})


def make_inputs(case: dict) -> dict:
    generator = torch.Generator().manual_seed(case["seed"])
    return {"x": torch.randn(1, case["n"], generator=generator),
            "y": torch.randn(1, case["n"], generator=generator)}


def reference(inputs: dict) -> dict:
    affine = torch.add(inputs["x"] * 2, inputs["y"])
    return {"output": torch.square(affine), "saved_affine": affine.clone()}


def reference_stages(inputs: dict) -> dict:
    affine = inputs["x"] + inputs["x"] + inputs["y"]
    square = affine * affine
    return {"affine": {"saved_affine": affine}, "square": {"output": square}}


def compare_named(actual: dict, expected: dict) -> None:
    if actual.keys() != expected.keys():
        raise AssertionError("Named output set differs from the contract")
    for name, wanted in expected.items():
        observed = actual[name]
        if observed.shape != wanted.shape or observed.dtype != wanted.dtype:
            raise AssertionError(f"Output ABI mismatch: {name}")
        if not bool(torch.isfinite(observed).all()) or not bool(torch.isfinite(wanted).all()):
            raise AssertionError(f"Non-finite value outside the declared domain: {name}")
        torch.testing.assert_close(observed, wanted, atol=0, rtol=0)


def main() -> None:
    order = tuple(TopologicalSorter(DAG).static_order())
    if order != ("affine", "square") or not CASES:
        raise AssertionError("Incomplete reference handoff")
    for case in CASES:
        inputs = make_inputs(case)
        original = {name: value.clone() for name, value in inputs.items()}
        expected = reference(inputs)
        stages = reference_stages(make_inputs(case))
        compare_named(stages["affine"], {"saved_affine": expected["saved_affine"]})
        compare_named(stages["square"], {"output": expected["output"]})
        compare_named({**stages["affine"], **stages["square"]}, expected)
        compare_named(inputs, original)
    print(json.dumps({"stage": "reference", "cases": len(CASES), "leaf_stages": len(DAG),
                      "comparison": "exact", "saved_state": "saved_affine", "torch": torch.__version__}, sort_keys=True))


if __name__ == "__main__":
    main()
