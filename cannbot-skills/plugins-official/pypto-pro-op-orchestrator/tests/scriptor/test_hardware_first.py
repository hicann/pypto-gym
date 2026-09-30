# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execution-order fixtures only; no accelerator result is claimed."""
import csv
import json
import shutil
import unittest
from unittest.mock import patch

import torch
from test_workflow import OPS, WorkflowFixture
from scriptorlib.common import ContractError, atomic_json, digest, read_json
from scriptorlib import runner
from scriptorlib.workflow import load_report


class HardwareFirstTests(WorkflowFixture, unittest.TestCase):
    def run_check(self, *, blocked=False, wrong_output=False):
        order = []
        data = {"kernel": object(), "args": [torch.ones(1), torch.empty(1)],
                "input_indices": {"x": 0}, "output_indices": {"y": 1}, "block_dim": 1}

        def expected(_golden, _contract, _case, prepared):
            order.append("reference")
            return {"y": prepared["args"][0] + 1}

        def execute(directory, argv, options, **kwargs):
            order.append("hardware")
            if blocked:
                raise runner.EnvironmentUnavailable("FIXTURE: assigned board unavailable")
            (directory / "execution.log").write_text("FIXTURE EXECUTION: no hardware was run\n")
            output = directory / "output"; output.mkdir()
            value = torch.tensor([0.0 if wrong_output else 2.0])
            (output / "y.bin").write_bytes(value.view(torch.uint8).numpy().tobytes())
            atomic_json(output / "metadata.json", {"y": {"dtype": "float32", "shape": [1]}})
            with (directory / "op_summary.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["Op Name", "Task Duration(us)"])
                writer.writeheader()
                writer.writerows({"Op Name": "probe_kernel", "Task Duration(us)": 2} for _ in range(32))
            return 0

        import builtins
        original_import = builtins.__import__

        def forbid_simulator(name, *args, **kwargs):
            if name.startswith(("ascriptor.runtime", "ascriptor.backends.sim")):
                raise AssertionError("hardware-first check imported a simulator execution path")
            return original_import(name, *args, **kwargs)

        with patch.object(runner, "activate"), patch.object(runner, "read_spec", return_value=self.contract), \
             patch.object(runner, "verify_export", return_value={"sync_mode": "manual"}), \
             patch.object(runner, "task_module"), patch.object(runner, "make_case", return_value=data), \
             patch.object(runner, "_expected", side_effect=expected), \
             patch.object(runner, "_delivery_copy"), patch.object(runner, "_execute", side_effect=execute), \
             patch("builtins.__import__", side_effect=forbid_simulator):
            result = runner.check(self.config, self.op, "candidate")
        return result, order, read_json(self.op / result["report"])

    def test_device_executes_before_reference_and_skipped_models_remain_not_run(self):
        result, order, raw = self.run_check()
        self.assertEqual(order, ["hardware", "reference"])
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(raw["execution_policy"], "hardware_first")
        self.assertEqual(raw["sync_mode"], "manual")
        for name in ("functional", "pipesim"):
            self.assertEqual(raw["checks"][name], "NOT_RUN")
            self.assertEqual(raw["diagnostics"][name]["status"], "NOT_RUN")
        for name in ("pypto_hardware", "standalone", "latency"):
            self.assertEqual(raw["checks"][name], "PASS")

    def test_unavailable_hardware_is_blocked_without_a_simulator_fallback(self):
        result, order, raw = self.run_check(blocked=True)
        self.assertEqual(order, ["hardware"])
        self.assertEqual(result["verdict"], "BLOCKED")
        self.assertEqual(raw["checks"]["pypto_hardware"], "BLOCKED")
        self.assertEqual(raw["checks"]["functional"], "NOT_RUN")

    def test_wrong_device_output_fails_without_running_full_shape_models(self):
        result, order, raw = self.run_check(wrong_output=True)
        self.assertEqual(order, ["hardware", "reference"])
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(raw["checks"]["pypto_hardware"], "FAIL")
        self.assertEqual(raw["checks"]["pipesim"], "NOT_RUN")

    def test_sealing_cannot_change_the_execution_policy(self):
        self.act("init")
        path = self.report("candidate")
        report = read_json(self.op / path)
        report["execution_policy"] = "hardware_first"
        atomic_json(self.op / path, report)
        with self.assertRaisesRegex(ContractError, "execution policy"):
            load_report(self.config, self.op, path, "candidate", passing=False)

    def test_scheme_a_uses_installed_engine_and_records_result(self):
        path = self.config / "skills/pypto-pro-op-develop/scripts/precision_compare.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OPS / "pypto-pro-op-develop/scripts/precision_compare.py", path)
        self.contract["tolerance"] = {"policy": "pro_scheme_a"}
        result, _, raw = self.run_check()
        identity = raw["precision_policy"]["engine"]
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(identity["sha256"], digest(self.config / identity["path"]))
        self.assertIn("matched_ratio", raw["cases"]["p0"]["precision"]["y"]["summary"])
        self.assertEqual(self.run_check(wrong_output=True)[0]["verdict"], "FAIL")
