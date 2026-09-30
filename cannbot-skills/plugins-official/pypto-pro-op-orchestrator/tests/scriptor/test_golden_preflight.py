# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The Scriptor bootstrap must detect CPU Golden ABI mistakes before Pro work."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/scriptor-runtime"))
from scriptorlib.common import ContractError, digest
from scriptorlib.golden_preflight import verify_cpu_golden


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires torch")
class GoldenPreflightTests(unittest.TestCase):
    def test_all_p0_output_dtypes_and_special_inputs_are_checked(self):
        with tempfile.TemporaryDirectory() as raw:
            op = Path(raw)
            (op / "SPEC.md").write_text("fixture SPEC\n")
            source = op / "probe_golden_cpu.py"
            factory = '''import torch
def _make_inputs(device):
    return [
        ("case_01", [torch.ones(2, dtype=torch.float16, device=device)], {}),
        ("case_02", [torch.tensor([float("inf"), 0.0], dtype=torch.bfloat16, device=device)], {}),
    ]
def probe_golden_cpu(x):
    return x.float()SUFFIX
'''
            contract = {"op_name": "probe", "inputs": [{"name": "x", "dtype": "float16"}],
                        "outputs": [{"name": "y", "dtype": "float16"}], "p0_cases": [
                            {"name": "case_01", "params": {}, "input_shapes": {"x": [2]},
                             "output_shapes": {"y": [2]}, "input_dtypes": {"x": "float16"},
                             "output_dtypes": {"y": "float16"}},
                            {"name": "case_02", "params": {}, "input_shapes": {"x": [2]},
                             "output_shapes": {"y": [2]}, "input_dtypes": {"x": "bfloat16"},
                             "output_dtypes": {"y": "bfloat16"},
                             "input_special_values": {"x": ["+inf"]}},
                        ]}
            source.write_text(factory.replace("SUFFIX", ""))
            with self.assertRaisesRegex(ContractError, "case_01.*dtype float32 != float16"):
                verify_cpu_golden(op, contract)
            source.write_text(factory.replace("SUFFIX", ".to(x.dtype)"))
            receipt = verify_cpu_golden(op, contract)
            self.assertEqual(receipt, {"status": "PASS", "cases": 2,
                                       "cpu_sha256": digest(source),
                                       "spec_sha256": digest(op / "SPEC.md")})
            source.write_text(factory.replace("SUFFIX", ".to(x.dtype)").replace('float("inf")', "1.0"))
            with self.assertRaisesRegex(ContractError, r"case_02.*lacks \+inf"):
                verify_cpu_golden(op, contract)
