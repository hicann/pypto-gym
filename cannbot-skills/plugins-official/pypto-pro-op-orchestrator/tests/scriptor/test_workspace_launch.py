# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed workspace policy must preserve initialization and launch contracts."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/scriptor-runtime"))
from scriptorlib.common import ContractError, load_module
from scriptorlib.exporter import LAUNCHER, make_case


class WorkspaceLaunchTests(unittest.TestCase):
    def test_generated_wrapper_uses_no_torch_zeros(self):
        self.assertNotIn("torch.zeros", LAUNCHER)

    def test_make_case_preserves_declared_nonfinite_inputs(self):
        import torch
        contract = {"inputs": [{"name": "x", "dtype": "float32", "value_range": [-2, 2]}],
                    "outputs": [{"name": "y", "dtype": "float32"}]}
        case = {"input_shapes": {"x": [3]}, "output_shapes": {"y": [3]},
                "input_special_values": {"x": ["-inf", "+inf"]}}
        data = {"kernel": object(), "args": [torch.tensor([float("-inf"), 1., float("inf")]),
                torch.empty(3)], "input_indices": {"x": 0}, "output_indices": {"y": 1},
                "block_dim": 1}
        task = types.SimpleNamespace(make_case=lambda _: data)
        self.assertIs(make_case(task, case, contract), data)
        data["args"][0] = torch.tensor([-1., 1., 2.])
        with self.assertRaisesRegex(ContractError, "special values differ"):
            make_case(task, case, contract)
        data["args"][0] = torch.tensor([float("-inf"), float("nan"), float("inf")])
        with self.assertRaisesRegex(ContractError, "special values differ"):
            make_case(task, case, contract)
        data["args"][0] = torch.tensor([float("-inf"), 3., float("inf")])
        with self.assertRaisesRegex(ContractError, "value_range"):
            make_case(task, case, contract)

    def test_make_case_checks_per_case_dtype(self):
        import torch
        contract = {"inputs": [{"name": "x", "dtype": "float16", "value_range": [0, 2]}],
                    "outputs": [{"name": "y", "dtype": "float16"}]}
        case = {"input_shapes": {"x": [2]}, "output_shapes": {"y": [2]},
                "input_dtypes": {"x": "bfloat16"}, "output_dtypes": {"y": "bfloat16"}}
        data = {"kernel": object(), "args": [torch.ones(2, dtype=torch.bfloat16),
                torch.empty(2, dtype=torch.bfloat16)], "input_indices": {"x": 0},
                "output_indices": {"y": 1}, "block_dim": 1}
        task = types.SimpleNamespace(make_case=lambda _: data)
        self.assertIs(make_case(task, case, contract), data)
        data["args"][1] = torch.empty(2, dtype=torch.float16)
        with self.assertRaisesRegex(ContractError, "case dtype differs"):
            make_case(task, case, contract)

    def test_launcher_dispatches_same_shape_by_dtype(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = []
            for name, dtype in (("fp16", "float16"), ("bf16", "bfloat16")):
                folder = root / name
                folder.mkdir()
                (folder / "manifest.json").write_text(json.dumps({"entry": "kernel.py", "kernel": "kernel", "block_dim": 1}))
                (folder / "kernel.py").write_text("class K:\n    def __getitem__(self, grid):\n        return lambda x, y: y.copy_(x)\nkernel = K()\n")
                cases.append({"name": name, "params": {}, "input_shapes": {"x": [2]},
                    "input_dtypes": {"x": dtype}, "output_shapes": {"y": [2]},
                    "output_dtypes": {"y": dtype}, "output_aliases": {},
                    "output_initialization": {"y": "empty"}, "directory": name,
                    "arguments": [{"kind": "input", "name": "x"}, {"kind": "output", "name": "y"}]})
            (root / "export.json").write_text(json.dumps({"input_dtypes": {"x": "float16"},
                "output_dtypes": {"y": "float16"}, "output_names": ["y"], "cases": cases}))
            (root / "launch.py").write_text(LAUNCHER)
            launcher = load_module(root / "launch.py", "dtype_dispatch_test")
            pypto = types.SimpleNamespace(options=lambda **kwargs: contextlib.nullcontext())
            with patch.dict(sys.modules, {"pypto": pypto}):
                for dtype, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
                    x = torch.ones(2, dtype=dtype)
                    y = launcher.launch({"x": x}, {})
                    self.assertEqual(y.dtype, dtype)
                    self.assertIn(name, launcher._CACHE)

    def run_wrapper(self, policy):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "p0").mkdir()
            binding = {"kind": "workspace", "bytes": 16}
            if policy is not None:
                binding["initialization"] = policy
            index = {
                "input_dtypes": {"x": "float32"}, "output_dtypes": {"y": "float32"},
                "output_names": ["y"], "cases": [{"name": "p0", "params": {},
                    "input_shapes": {"x": [2]}, "output_shapes": {"y": [2]},
                    "output_aliases": {}, "output_initialization": {"y": "empty"},
                    "directory": "p0", "arguments": [
                        {"kind": "input", "name": "x"}, {"kind": "output", "name": "y"}, binding]}]}
            (root / "export.json").write_text(json.dumps(index))
            (root / "p0/manifest.json").write_text(json.dumps({"entry": "kernel.py", "kernel": "kernel", "block_dim": 1}))
            (root / "p0/kernel.py").write_text('''
class Kernel:
    def __init__(self):
        self.buffers = []
    def __getitem__(self, grid):
        def invoke(x, y, workspace):
            self.buffers.append(workspace)
            if INITIALIZE:
                workspace.fill_(5)
            y.copy_(x + workspace[0].to(x.dtype))
        return invoke
kernel = Kernel()
'''.replace("INITIALIZE", repr(policy in (None, "empty"))))
            (root / "launch.py").write_text(LAUNCHER)
            launcher = load_module(root / "launch.py", "workspace_policy_test")
            original_empty = torch.empty
            def poison_empty(*args, **kwargs):
                tensor = original_empty(*args, **kwargs)
                if tensor.dtype == torch.uint8:
                    tensor.fill_(165)
                return tensor
            pypto = types.SimpleNamespace(options=lambda **kwargs: contextlib.nullcontext())
            with patch.dict(sys.modules, {"pypto": pypto}), patch.object(torch, "empty", side_effect=poison_empty):
                if policy in (None, "empty"):
                    # A device zero would create an extra kernel in the real wrapper.
                    zeros = patch.object(torch, "zeros", side_effect=AssertionError("extra device clear"))
                else:
                    zeros = contextlib.nullcontext()
                with zeros:
                    for offset in (0, 10):
                        x = torch.tensor([1., 2.]) + offset
                        actual = launcher.launch({"x": x}, {})
                        torch.testing.assert_close(actual, x + 5)
            buffers = launcher._CACHE["p0"].buffers
            self.assertEqual(len(buffers), 2)
            self.assertNotEqual(buffers[0].data_ptr(), buffers[1].data_ptr())

    def test_empty_workspace_is_fresh_and_needs_no_device_clear(self):
        self.run_wrapper("empty")

    def test_default_workspace_is_empty_and_kernel_owned(self):
        self.run_wrapper(None)

    def test_invalid_workspace_policy_is_rejected(self):
        import torch
        case = {"input_shapes": {"x": [1]}, "output_shapes": {"y": [1]}}
        contract = {"inputs": [{"name": "x", "dtype": "float32", "value_range": [0, 1]}],
                    "outputs": [{"name": "y", "dtype": "float32"}]}
        for mode in (None, False, [], "random", "zero"):
            with self.subTest(mode=mode):
                result = {"kernel": object(), "args": [torch.zeros(1), torch.empty(1)],
                          "input_indices": {"x": 0}, "output_indices": {"y": 1},
                          "block_dim": 1, "workspace_initialization": mode}
                task = types.SimpleNamespace(make_case=lambda unused: result)
                with self.assertRaisesRegex(ContractError, "workspace_initialization"):
                    make_case(task, case, contract)

    def test_output_zero_initialization_is_rejected(self):
        import torch
        case = {"input_shapes": {"x": [1]}, "output_shapes": {"y": [1]}}
        contract = {"inputs": [{"name": "x", "dtype": "float32", "value_range": [0, 1]}],
                    "outputs": [{"name": "y", "dtype": "float32"}]}
        result = {"kernel": object(), "args": [torch.zeros(1), torch.empty(1)],
                  "input_indices": {"x": 0}, "output_indices": {"y": 1},
                  "block_dim": 1, "output_initialization": {"y": "zero"}}
        with self.assertRaisesRegex(ContractError, "output_initialization"):
            make_case(types.SimpleNamespace(make_case=lambda unused: result), case, contract)


if __name__ == "__main__":
    unittest.main()
