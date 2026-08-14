#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression tests for the canonical SPEC JSON contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from validate_spec import load_spec_contract, load_spec_contract_text, validate_text


VALID = '''```json machine-contract
{
  "schema_version": 1,
  "op_name": "demo_op",
  "formula": "y = x",
  "supported_dtypes": ["float32"],
  "inputs": [{"name": "x", "shape": ["N", 16], "dtype": "float32", "value_range": [-4, 4]}],
  "outputs": [{"name": "y", "shape": ["N", 16], "dtype": "float32", "value_range": [-4, 4]}],
  "default_params": {},
  "tolerance": {"atol": 0.001, "rtol": 0.002},
  "dynamic_axes_ranges": {"N": [1, 128]},
  "shape_constraints": [],
  "p0_cases": [
    {"name": "small", "params": {}, "input_shapes": {"x": [8, 16]}, "output_shapes": {"y": [8, 16]}},
    {"name": "large", "params": {}, "input_shapes": {"x": [32, 16]}, "output_shapes": {"y": [32, 16]}}
  ]
}
```

## Semantic explanation
y = x
'''


class SpecValidationTests(unittest.TestCase):
    def test_valid_contract_and_compatibility_field(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "SPEC.md"
            path.write_text(VALID, encoding="utf-8")
            contract = load_spec_contract(path)
        self.assertEqual(contract["p0_shapes"], [[8, 16]])
        self.assertEqual(len(contract["p0_cases"]), 2)

    def test_rejects_duplicate_key_and_placeholder(self) -> None:
        duplicate = VALID.replace(
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 1,',
        )
        self.assertIn("duplicate JSON key", validate_text(duplicate)[0])
        self.assertIn("placeholder", validate_text(VALID.replace("demo_op", "{{OP_NAME}}"))[0])

    def test_rejects_schema_and_p0_shape_errors(self) -> None:
        self.assertTrue(validate_text(
            VALID.replace('"supported_dtypes": ["float32"]', '"supported_dtypes": []')
        ))
        bad = VALID.replace(
            '"output_shapes": {"y": [32, 16]}',
            '"output_shapes": {"y": [31, 16]}',
        )
        self.assertIn("does not equal", validate_text(bad)[0])

    def test_requires_auditable_formula(self) -> None:
        missing = VALID.replace('  "formula": "y = x",\n', "")
        self.assertIn("missing fields: formula", validate_text(missing)[0])

        empty = VALID.replace('"formula": "y = x"', '"formula": ""')
        self.assertIn("non-empty string", validate_text(empty)[0])

        non_string = VALID.replace('"formula": "y = x"', '"formula": ["y = x"]')
        self.assertIn("non-empty string", validate_text(non_string)[0])

    def test_rejects_second_contract_and_unknown_fields(self) -> None:
        self.assertIn("exactly one", validate_text(VALID + VALID)[0])
        bad = VALID.replace('"schema_version": 1,', '"schema_version": 1,\n  "extra": true,')
        self.assertIn("unknown fields", validate_text(bad)[0])

    def test_rejects_unknown_dtype_and_dynamic_symbol_drift(self) -> None:
        unknown_dtype = VALID.replace('"float32"', '"banana"')
        self.assertIn("canonical dtypes", validate_text(unknown_dtype)[0])

        missing_range = VALID.replace('"dynamic_axes_ranges": {"N": [1, 128]}',
                                      '"dynamic_axes_ranges": {}')
        self.assertIn("exactly match shape symbols", validate_text(missing_range)[0])

        unused_range = VALID.replace('"dynamic_axes_ranges": {"N": [1, 128]}',
                                     '"dynamic_axes_ranges": {"N": [1, 128], "Z": [1, 8]}')
        self.assertIn("exactly match shape symbols", validate_text(unused_range)[0])

    def test_preserves_mixed_public_tensor_dtypes(self) -> None:
        mixed = VALID.replace(
            '"supported_dtypes": ["float32"]',
            '"supported_dtypes": ["int32", "float32"]',
        ).replace(
            '"name": "x", "shape": ["N", 16], "dtype": "float32"',
            '"name": "x", "shape": ["N", 16], "dtype": "int32"',
        )
        contract = load_spec_contract_text(mixed)
        self.assertEqual(contract["supported_dtypes"], ["int32", "float32"])
        self.assertEqual(contract["inputs"][0]["dtype"], "int32")

    def test_requires_each_shape_symbol_to_have_an_input_anchor(self) -> None:
        unanchored = VALID.replace('["N", 16]', '["2*N", 16]')
        self.assertIn("standalone dimension", validate_text(unanchored)[0])


if __name__ == "__main__":
    unittest.main()
