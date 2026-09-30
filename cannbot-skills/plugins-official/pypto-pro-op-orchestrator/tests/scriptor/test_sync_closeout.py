# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Evidence/state fixtures and real source emission; no device execution is claimed."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_workflow import WorkflowFixture
from scriptorlib.common import ContractError, atomic_json, digest, read_json
from scriptorlib.exporter import verify_export, verify_recorded_emissions
from scriptorlib.progress import state_summary, write_final


class SyncCloseoutTests(WorkflowFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.generated = self.op / "generated/kernel.py"
        self.generated.parent.mkdir()
        self.generated.write_text("manual\n")
        self.mode_patch = patch("scriptorlib.exporter.verify_export",
                               side_effect=lambda *a: {"sync_mode": self.generated.read_text().strip()})
        self.mode_patch.start()
        self.addCleanup(self.mode_patch.stop)
        self.advance()
        self.before = self.act("finish_optimization")

    def hardware_report(self, stage="candidate", *, failed=False):
        relative = self.report(stage, verdict="FAIL" if failed else "PASS")
        report = read_json(self.op / relative)
        raw_path = self.op / report["check_report"]
        raw = read_json(raw_path)
        checks = {"functional": "NOT_RUN", "pipesim": "NOT_RUN", "pypto_hardware": "FAIL" if failed else "PASS",
                  "standalone": "PASS", "latency": "PASS"}
        raw.update(execution_policy="hardware_first", sync_mode=self.generated.read_text().strip(),
                   checks=checks, cases={"p0": {"checks": checks}}, fixture_evidence=True)
        atomic_json(raw_path, raw)
        report.update(execution_policy=raw["execution_policy"], sync_mode=raw["sync_mode"],
                      checks={**checks, "semantic_review": "PASS"}, cases=raw["cases"])
        report["evidence"][0]["sha256"] = digest(raw_path)
        atomic_json(self.op / relative, report)
        return relative

    def start_native(self):
        state = self.act("begin_sync_trial")
        self.generated.write_text("auto_mutex\n")
        return state

    def test_passing_native_trial_is_selected_without_spending_optimization_rounds(self):
        self.start_native()
        selected = self.act("complete_sync_trial", report=self.hardware_report())
        self.assertEqual(selected["sync_trial"]["selected_mode"], "auto_mutex")
        self.assertEqual(selected["rounds"], self.before["rounds"])
        self.assertEqual(selected["optimization"], self.before["optimization"])
        self.assertNotEqual(selected["best"]["artifact_hash"], self.before["best"]["artifact_hash"])
        final = self.act("complete_accept", report=self.hardware_report("accept"))
        self.assertEqual(final["current_stage"], "done")
        write_final(self.config, self.op, final)
        self.assertEqual(read_json(self.op / "reports/final/final.json")["synchronization"]["selected_mode"], "auto_mutex")
        self.assertIn("auto_mutex trial: passed", (self.op / "reports/final/final.md").read_text())
        with self.assertRaises(ContractError): self.act("begin_sync_trial")

    def test_failed_native_trial_restores_manual_and_reports_its_failure(self):
        self.start_native()
        selected = self.act("complete_sync_trial", report=self.hardware_report(failed=True))
        self.assertEqual(self.generated.read_text().strip(), "manual")
        self.assertEqual(selected["best"], self.before["best"])
        self.assertEqual(selected["sync_trial"]["status"], "failed")
        final = self.act("complete_accept", report=self.hardware_report("accept"))
        write_final(self.config, self.op, final)
        text = (self.op / "reports/final/final.md").read_text()
        self.assertIn("Synchronization: manual", text)
        self.assertIn("auto_mutex trial: failed", text)
        self.assertIn("Trial evidence:", text)

    def test_emission_failure_or_blocker_requires_logs_and_restores_manual(self):
        self.start_native()
        with self.assertRaises(ContractError):
            self.act("abort_sync_trial", status="blocked", reason="FIXTURE board unavailable")
        path = self.op / "reports/native-export.log"
        path.parent.mkdir(exist_ok=True); path.write_text("FIXTURE: native emission unavailable\n")
        state = self.act("abort_sync_trial", status="blocked", reason="FIXTURE native compiler unavailable",
                         evidence=["reports/native-export.log"])
        self.assertEqual(state["sync_trial"]["selected_mode"], "manual")
        self.assertEqual(self.generated.read_text().strip(), "manual")
        self.assertEqual(state["rounds"], self.before["rounds"])
        path.write_text("changed evidence")
        with self.assertRaisesRegex(ContractError, "diagnostics changed"):
            self.act("complete_accept", report=self.hardware_report("accept"))

    def test_acceptance_requires_a_recorded_trial_for_hardware_first_reports(self):
        with self.assertRaisesRegex(ContractError, "closeout trial"):
            self.act("complete_accept", report=self.hardware_report("accept"))

    def test_trial_cannot_hide_a_dsl_change(self):
        self.start_native()
        with (self.op / "scriptor/task.py").open("a") as stream: stream.write("# different candidate\n")
        with self.assertRaisesRegex(ContractError, "preserve the selected DSL"):
            self.act("complete_sync_trial", report=self.hardware_report())

    def test_trial_nested_execution_evidence_is_checked_at_acceptance(self):
        self.start_native()
        report = self.hardware_report()
        raw_path = self.op / read_json(self.op / report)["check_report"]
        self.act("complete_sync_trial", report=report)
        raw_path.write_text("changed native execution evidence")
        with self.assertRaisesRegex(ContractError, "diagnostics changed"):
            self.act("complete_accept", report=self.hardware_report("accept"))

    def test_summary_exposes_the_closeout_commands_in_order(self):
        (self.op / "generated/export.json").write_text("{}")
        summary = state_summary(self.config, self.project, self.op, self.before)
        self.assertEqual(summary["next_action"], "begin_sync_trial")
        self.assertIn("begin_sync_trial", summary["next_command"])
        # The selection guard compares the actual candidate to the checkpoint.
        (self.op / "generated/export.json").unlink()
        started = self.act("begin_sync_trial")
        (self.op / "generated/export.json").write_text("{}")
        summary = state_summary(self.config, self.project, self.op, started)
        self.assertEqual(summary["next_action"], "complete_sync_trial")
        self.assertIn("complete_sync_trial", summary["next_command"])

    def test_a_manual_report_does_not_qualify_as_the_native_trial(self):
        self.act("begin_sync_trial")
        with self.assertRaisesRegex(ContractError, "auto_mutex hardware report"):
            self.act("complete_sync_trial", report=self.hardware_report())


class RecordedEmissionTests(unittest.TestCase):
    def test_ad_hoc_manual_kernel_blocks_fixed_auto_delivery(self):
        with tempfile.TemporaryDirectory() as raw:
            op = Path(raw)
            stage = op / "reports/runs/develop/opexec/case_01/pypto_stage"
            stage.mkdir(parents=True)
            kernel = stage / "kernel_pypto.py"
            manifest = stage / "manifest.json"
            kernel.write_text("@pl.jit(auto_mutex=False)\ndef kernel(): pass\n")
            atomic_json(manifest, {"entry": kernel.name, "pypto": {"sync_mode": "manual"}})
            with self.assertRaisesRegex(ContractError, "used manual instead of auto_mutex"):
                verify_recorded_emissions(op, "auto_mutex")
            atomic_json(manifest, {"entry": kernel.name, "pypto": {"sync_mode": "auto_mutex"}})
            with self.assertRaisesRegex(ContractError, "decorator differs"):
                verify_recorded_emissions(op, "auto_mutex")
            kernel.write_text("@pl.jit(auto_mutex=True)\ndef kernel(): pass\n")
            verify_recorded_emissions(op, "auto_mutex")


class ExportModeTests(unittest.TestCase):
    def test_auto_default_and_user_authorized_manual_emit_distinct_sources(self):
        from test_installation import installer
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            installer.install(project, sys.executable)
            config = project / ".opencode"
            op = project / "custom/probe"; (op / "scriptor").mkdir(parents=True)
            contract = {"schema_version": 1, "op_name": "probe", "formula": "y = x",
                "supported_dtypes": ["float32"], "inputs": [{"name": "x", "shape": [1,64], "dtype": "float32", "value_range": [0,1]}],
                "outputs": [{"name": "y", "shape": [1,64], "dtype": "float32", "value_range": [0,1]}],
                "default_params": {}, "tolerance": {"atol": 0,"rtol": 0}, "dynamic_axes_ranges": {},
                "shape_constraints": [], "p0_cases": [{"name": "p0", "params": {}, "input_shapes": {"x": [1,64]}, "output_shapes": {"y": [1,64]}}],
                "perf_target": None, "exit_criteria": None}
            (op / "SPEC.md").write_text("```json machine-contract\n" + json.dumps(contract) + "\n```\n")
            (op / "scriptor/task.py").write_text('''import torch
from ascriptor.a5 import GM, f32, Tensor, Position, kernel, auto_sync
@kernel(mode="vec", block_dim=1)
def copy(x: GM[f32, (1,64)], y: GM[f32, (1,64)]):
    buf = Tensor(f32, [1,64], Position.UB)
    with auto_sync():
        buf <<= x
        y <<= buf
    return y

def make_case(case):
    return {"kernel": copy, "args": [torch.ones(1,64), torch.empty(1,64)],
            "input_indices": {"x": 0}, "output_indices": {"y": 1}, "block_dim": 1}
''')
            command = [sys.executable, "-B", str(config / "scriptor/scripts/scriptor.py"), "--config-root", str(config),
                       "export", "--op-dir", str(op)]
            sources = []
            state = op / ".scriptor/state.json"
            before_init = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(before_init.returncode, 0)
            self.assertIn("initialize Scriptor state", before_init.stderr)
            state.parent.mkdir(parents=True)
            atomic_json(state, {"delivery_sync_mode": "auto_mutex", "manual_requested_by_user": False})
            for mode in ("auto_mutex", "manual"):
                if mode == "manual":
                    rejected = subprocess.run(command + ["--sync-mode", "manual"], capture_output=True, text=True)
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn("user-authorized delivery mode", rejected.stderr)
                    atomic_json(state, {"delivery_sync_mode": "manual", "manual_requested_by_user": True})
                result = subprocess.run(command + ([] if mode=="auto_mutex" else ["--sync-mode", mode]), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["sync_mode"], mode)
                index = read_json(op / "generated/export.json")
                self.assertEqual(index["sync_mode"], mode)
                source = (op / "generated/p0/kernel_pypto.py").read_text(); sources.append(source)
                self.assertIn(f"@pl.jit(auto_mutex={mode=='auto_mutex'})", source)
                if mode == "auto_mutex":
                    kernel_file = op / "generated/p0/kernel_pypto.py"
                    kernel_file.write_text(source.replace("@pl.jit(auto_mutex=True)",
                                                          "@pl.jit(auto_mutex=False)"))
                    index["cases"][0]["files"]["kernel_pypto.py"] = digest(kernel_file)
                    atomic_json(op / "generated/export.json", index)
                    with self.assertRaisesRegex(ContractError, "kernel decorator differs"):
                        verify_export(config, op)
            self.assertNotEqual(*sources)
            atomic_json(state, {"delivery_sync_mode": "manual", "manual_requested_by_user": False})
            rejected = subprocess.run(command + ["--sync-mode", "manual"], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("explicit user request", rejected.stderr)
