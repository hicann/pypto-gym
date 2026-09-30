# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Legacy tolerance must not erase float64 differences before comparison."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/scriptor-runtime"))
from scriptorlib.runner import _compare


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires torch")
class LegacyPrecisionTests(unittest.TestCase):
    def test_float64_values_are_compared_at_original_precision(self):
        import torch

        contract = {"outputs": [{"name": "y", "dtype": "float64"}],
                    "tolerance": {"atol": 0, "rtol": 0}}
        case = {"output_shapes": {"y": [1]}, "output_dtypes": {"y": "float64"}}
        reference = torch.tensor([1.0], dtype=torch.float64)
        actual = torch.tensor([1.00000001], dtype=torch.float64)
        with self.assertRaises(AssertionError):
            _compare({"y": actual}, {"y": reference}, contract, case)
        self.assertEqual(_compare({"y": reference}, {"y": reference}, contract, case)["y"], 0)

    def test_quantized_integer_tolerance_is_explicit_and_does_not_wrap(self):
        import torch

        contract = {"outputs": [{"name": "y", "dtype": "int8"}],
                    "tolerance": {"atol": 0, "rtol": 0}}
        case = {"output_shapes": {"y": [2]}, "output_dtypes": {"y": "int8"}}
        reference = torch.tensor([-128, 127], dtype=torch.int8)
        adjacent = torch.tensor([-127, 126], dtype=torch.int8)
        with self.assertRaises(AssertionError):
            _compare({"y": adjacent}, {"y": reference}, contract, case)
        contract["tolerance"]["integer_atol"] = 1
        self.assertEqual(_compare({"y": adjacent}, {"y": reference}, contract, case)["y"], 1)
        with self.assertRaises(AssertionError):
            _compare({"y": reference.flip(0)}, {"y": reference}, contract, case)

    def test_integer_tolerance_does_not_change_float_or_boolean_gate(self):
        import torch

        for dtype, reference, actual in (("float32", 0.0, 0.01), ("bool", False, True)):
            contract = {"outputs": [{"name": "y", "dtype": dtype}],
                        "tolerance": {"atol": 0, "rtol": 0, "integer_atol": 1}}
            case = {"output_shapes": {"y": [1]}, "output_dtypes": {"y": dtype}}
            with self.assertRaises(AssertionError):
                _compare({"y": torch.tensor([actual], dtype=getattr(torch, dtype))},
                         {"y": torch.tensor([reference], dtype=getattr(torch, dtype))}, contract, case)
