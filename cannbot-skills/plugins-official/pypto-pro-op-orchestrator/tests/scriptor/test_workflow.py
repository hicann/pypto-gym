# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Behavioral adapter tests. Fixture evidence is not an accelerator acceptance result."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ORCHESTRATOR = Path(__file__).resolve().parents[2]
RUNTIME = ORCHESTRATOR / "scripts/scriptor-runtime"
OPS = ORCHESTRATOR.parents[1] / "ops"
sys.path.insert(0, str(RUNTIME))
from scriptorlib.common import ContractError, atomic_json, digest, load_module, read_json
from scriptorlib.policy import resolve_policy, stop_reason
from scriptorlib.progress import state_summary, write_final
from scriptorlib.workflow import (artifact_hashes, bootstrap_check, bootstrap_restart, _verify_bootstrap,
                                  read_spec, subject, transition)

criteria = load_module(OPS / "pypto-pro-intent-understand/scripts/exit_criteria.py", "criteria_test")


def condition(identifier="v", *, basis="model", aggregation="each"):
    return {"id": identifier, "metric": "vector_pipe_utilization_pct", "operator": ">=", "threshold": 70,
            "unit": "%", "basis": basis, "cases": ["p0", "p1"], "aggregation": aggregation,
            "source": "test_model"}


def observation(case, value, **kwargs):
    return {"metric": "vector_pipe_utilization_pct", "case": case, "value": value,
            "unit": "%", "basis": "model", "source": "test_model", **kwargs}


class CriteriaTests(unittest.TestCase):
    def test_missing_case_is_unknown(self):
        report = criteria.evaluate_exit_criteria(condition(), [observation("p0", 90)])
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertFalse(report["met"])

    def test_incompatible_basis_and_unit_do_not_pass(self):
        for changes in ({"basis": "hardware"}, {"unit": "us"}):
            values = [observation(case, 100, **changes) for case in ("p0", "p1")]
            self.assertEqual(criteria.evaluate_exit_criteria(condition(), values)["status"], "UNKNOWN")

    def test_all_any_and_aggregation(self):
        a, b = condition("a"), condition("b", aggregation="mean")
        values = [observation("p0", 60), observation("p1", 90)]
        self.assertFalse(criteria.evaluate_exit_criteria({"all": [a, b]}, values)["met"])
        self.assertTrue(criteria.evaluate_exit_criteria({"any": [a, b]}, values)["met"])

    def test_empty_duplicate_and_nonfinite_criteria_rejected(self):
        bad = [{"all": []}, {"any": []}, {"all": [condition(), condition()]},
               {**condition(), "threshold": float("nan")}, {**condition(), "cases": ["p0", "p0"]}]
        for tree in bad:
            with self.subTest(tree=tree), self.assertRaises(ValueError):
                criteria.validate_exit_criteria(tree)

    def test_absent_condition_is_not_vacuously_met(self):
        self.assertEqual(criteria.evaluate_exit_criteria(None, []),
                         {"status": "NOT_REQUESTED", "met": False, "conditions": []})

    def test_conflicting_observations_are_unknown(self):
        values = [observation("p0", 90), observation("p0", 40), observation("p1", 90)]
        self.assertFalse(criteria.evaluate_exit_criteria(condition(), values)["met"])


class BootstrapGateTests(unittest.TestCase):
    @staticmethod
    def probe(op, contract):
        return {"status": "PASS", "cases": len(contract["p0_cases"]),
                "cpu_sha256": digest(op / f"{contract['op_name']}_golden_cpu.py"),
                "spec_sha256": digest(op / "SPEC.md")}

    @staticmethod
    def probe_files(op, name):
        (op / "SPEC.md").write_text("fixture spec\n")
        (op / f"{name}_golden_cpu.py").write_text("fixture CPU Golden\n")

    @staticmethod
    def hashes(op, name, marker):
        return {"SPEC.md": digest(op / "SPEC.md"),
                f"{name}_golden_cpu.py": digest(op / f"{name}_golden_cpu.py"),
                "fixture_marker": marker}

    def test_external_case_file_is_not_discovered_or_bound(self):
        with tempfile.TemporaryDirectory() as raw:
            project = Path(raw)
            op = project / "custom/sigmoid"
            op.mkdir(parents=True)
            self.probe_files(op, "sigmoid")
            task = project / "tasks/level1/sigmoid/cases.csv"
            task.parent.mkdir(parents=True)
            task.write_text("unrelated task data\n")
            case = {"name": "p0", "input_shapes": {"x": [2]},
                    "input_dtypes": {"x": "float16"}, "params": {}}
            contract = {"op_name": "sigmoid", "inputs": [{"name": "x", "dtype": "float16"}],
                        "p0_cases": [case]}
            with patch("scriptorlib.workflow.read_spec", return_value=contract), \
                 patch("scriptorlib.workflow._bootstrap_hashes", return_value=self.hashes(op, "sigmoid", "stable")), \
                 patch("scriptorlib.workflow.verify_cpu_golden", side_effect=self.probe):
                receipt = bootstrap_check(project, op)
                self.assertNotIn("task_cases_csv", receipt)
                self.assertNotIn("task_cases_csv", read_json(op / receipt["receipt"]))
                _verify_bootstrap(project, op)
                task.write_text("changed unrelated task data\n")
                _verify_bootstrap(project, op)

    def test_sealed_golden_revisions_require_archived_restart_and_fresh_prototype(self):
        self._exercise_bootstrap_restart()

    def test_rollback_upstream_can_reseal_without_resetting_state(self):
        self._exercise_bootstrap_restart(initialized=True)

    def _exercise_bootstrap_restart(self, initialized=False):
        with tempfile.TemporaryDirectory() as raw:
            op = Path(raw) / "custom/probe"
            op.mkdir(parents=True)
            self.probe_files(op, "probe")
            contract = {"op_name": "probe", "p0_cases": [{"name": "p0"}]}
            with patch("scriptorlib.workflow.read_spec", return_value=contract), \
                 patch("scriptorlib.workflow._bootstrap_hashes", return_value=self.hashes(op, "probe", "old")), \
                 patch("scriptorlib.workflow.verify_cpu_golden", side_effect=self.probe):
                first = bootstrap_check(Path(raw), op)
            (op / "test_probe.py").write_text("old prototype\n")
            (op / "KB_SELECTION.json").write_text('{"old": true}\n')
            (op / "build").mkdir()
            (op / "build/kernel.so").write_text("old build\n")
            (op / "reports/prototype-logs").mkdir(parents=True)
            (op / "reports/prototype-logs/run.log").write_text("old run\n")
            (op / "reports/pro-bootstrap.md").write_text("old handoff\n")
            with self.assertRaisesRegex(ContractError, "concrete reason"):
                bootstrap_restart(Path(raw), op, "")
            if initialized:
                state_path = op / ".scriptor/state.json"
                state_path.write_text('{"current_stage":"implement","rounds_used":7}')
                with self.assertRaisesRegex(ContractError, "rollback_prepare"):
                    bootstrap_restart(Path(raw), op, "correct CPU Golden")
                state_path.write_text('{"current_stage":"upstream","rounds_used":7}')
            result = bootstrap_restart(Path(raw), op, "correct CPU Golden")
            self.assertEqual(result["status"], "RESTART_PENDING")
            self.assertTrue(result["had_prototype"])
            archive = op / result["archive"]
            self.assertEqual(digest(archive / "pro-bootstrap.json"), first["receipt_sha256"])
            self.assertTrue((archive / "test_probe.py").exists())
            self.assertTrue((archive / "build/kernel.so").exists())
            self.assertTrue((archive / "prototype-logs/run.log").exists())
            self.assertTrue((archive / "inputs-before-restart/KB_SELECTION.json").exists())
            self.assertFalse((op / "test_probe.py").exists())
            self.assertFalse((op / ".scriptor/receipts/pro-bootstrap.json").exists())
            self.assertTrue((op / ".scriptor/bootstrap-restart.json").exists())
            with patch("scriptorlib.workflow.read_spec", return_value=contract), \
                 patch("scriptorlib.workflow._bootstrap_hashes", return_value=self.hashes(op, "probe", "new")), \
                 patch("scriptorlib.workflow.verify_cpu_golden", side_effect=self.probe):
                second = bootstrap_check(Path(raw), op)
                self.assertEqual(second["restart"]["reason"], "correct CPU Golden")
                if initialized:
                    self.assertEqual(read_json(state_path), {"current_stage":"upstream","rounds_used":7})
                self.assertFalse((op / ".scriptor/bootstrap-restart.json").exists())
                with self.assertRaisesRegex(ContractError, "fresh Pro prototype"):
                    _verify_bootstrap(Path(raw), op)
                (op / "prototype").mkdir()
                (op / "prototype/test_probe.py").write_text("new prototype\n")
                report = op / "reports/pro-bootstrap.md"
                report.write_text("Bootstrap receipt SHA-256: " + second["receipt_sha256"] + "\n")
                sealed_at = read_json(op / ".scriptor/receipts/pro-bootstrap.json")["completed_at"]
                os.utime(report, (sealed_at + 2, sealed_at + 2))
                _verify_bootstrap(Path(raw), op)
                (archive / "pro-bootstrap.json").write_text("tampered\n")
                with self.assertRaisesRegex(ContractError, "previous bootstrap receipt changed"):
                    _verify_bootstrap(Path(raw), op)

    def test_receipt_is_required_before_prototype_and_bound_to_inputs(self):
        with tempfile.TemporaryDirectory() as raw:
            op = Path(raw) / "custom/probe"
            op.mkdir(parents=True)
            self.probe_files(op, "probe")
            contract = {"op_name": "probe", "p0_cases": [{"name": "p0"}]}
            with patch("scriptorlib.workflow.read_spec", return_value=contract), \
                 patch("scriptorlib.workflow._bootstrap_hashes", return_value=self.hashes(op, "probe", "a")), \
                 patch("scriptorlib.workflow.verify_cpu_golden", side_effect=self.probe):
                with self.assertRaises(ContractError):
                    _verify_bootstrap(Path(raw), op)
                result = bootstrap_check(Path(raw), op)
                self.assertEqual(result["hashes"], self.hashes(op, "probe", "a"))
                _verify_bootstrap(Path(raw), op)
                (op / "probe_golden_cpu.py").write_text("changed CPU Golden\n")
                with self.assertRaisesRegex(ContractError, "CPU Golden P0.*stale"):
                    _verify_bootstrap(Path(raw), op)
                (op / "probe_golden_cpu.py").write_text("fixture CPU Golden\n")
                (op / "prototype").mkdir()
                with self.assertRaisesRegex(ContractError, "precede prototype"):
                    bootstrap_check(Path(raw), op)
            with patch("scriptorlib.workflow._bootstrap_hashes", return_value=self.hashes(op, "probe", "changed")):
                with self.assertRaisesRegex(ContractError, "stale"):
                    _verify_bootstrap(Path(raw), op)


class PolicyTests(unittest.TestCase):
    def test_defaults_preserve_legacy_and_enable_from_pro(self):
        policy = resolve_policy()
        self.assertFalse(policy["enabled"])
        self.assertEqual(policy["max_iterations"], 5)
        policy = resolve_policy(from_pro=True)
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["max_iterations"], 10)
        self.assertTrue(policy["stop_on_criteria"])

    def test_prompt_overrides_project_and_null_is_unlimited(self):
        p = resolve_policy({"enabled": True, "max_iterations": None}, {"max_iterations": 2})
        self.assertIsNone(stop_reason(p, 100000, elapsed=100000, no_improvement=100000))
        self.assertEqual(stop_reason(p, 0, criteria_met=True), "criteria_met")

    def test_iteration_limit_and_explicit_secondary_limit(self):
        p = resolve_policy({"enabled": True})
        self.assertIsNone(stop_reason(p, 4))
        self.assertEqual(stop_reason(p, 5), "iteration_budget_exhausted")
        p = resolve_policy({"max_iterations": None, "time_budget_s": 2})
        self.assertEqual(stop_reason(p, 100, elapsed=3), "time_budget_exhausted")

    def test_invalid_limits_rejected(self):
        with self.assertRaises(ContractError):
            resolve_policy({"stop_on_criteria": "false"})
        for value in (0, -1, True, float("inf"), "5"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                resolve_policy({"max_iterations": value})


class WorkflowFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # Match the CLI's resolved roots, including macOS /var -> /private/var.
        self.project = Path(self.temp.name).resolve()
        self.config = self.project / ".opencode"
        scripts = self.config / "skills/pypto-pro-intent-understand/scripts"
        scripts.mkdir(parents=True)
        for name in ("validate_spec.py", "exit_criteria.py"):
            shutil.copy2(OPS / "pypto-pro-intent-understand/scripts" / name, scripts / name)
        atomic_json(self.config / "scriptor-install.json", {"source_id": "test-fixture-sources", "python": sys.executable})
        self.op = self.project / "custom/probe"
        self.op.mkdir(parents=True)
        self.contract = {"schema_version": 1, "op_name": "probe", "formula": "y = x + 1",
            "supported_dtypes": ["float32"], "inputs": [{"name":"x","shape":[1],"dtype":"float32","value_range":[0,1]}],
            "outputs": [{"name":"y","shape":[1],"dtype":"float32","value_range":[1,2]}],
            "default_params":{},"tolerance":{"atol":0,"rtol":0},"dynamic_axes_ranges":{},"shape_constraints":[],
            "p0_cases":[{"name":"p0","params":{},"input_shapes":{"x":[1]},"output_shapes":{"y":[1]}}],
            "perf_target":None,"exit_criteria":None}
        (self.op / "SPEC.md").write_text("```json machine-contract\n" + json.dumps(self.contract) + "\n```\n")
        for name in ("probe_golden.py", "probe_golden_cpu.py", "test_probe.py"):
            (self.op / name).write_text("# unit-test fixture, not an executable kernel\n")
        (self.op / "scriptor").mkdir()
        (self.op / "scriptor/task.py").write_text("# unit-test fixture DSL identity\n")

    def tearDown(self):
        self.temp.cleanup()

    def act(self, action, **kwargs):
        return transition(self.config, self.project, {"action": action, "opDir": "custom/probe", **kwargs})

    def report(self, stage, *, verdict="PASS", metrics=None, checks_override=None, recommendation="keep"):
        sequence = len(list(self.op.glob("reports/*raw.json")))
        raw_path = f"reports/{sequence}-raw.json"
        review = {"artifact_hash":subject(self.op),"findings":["UNIT TEST FIXTURE: no hardware was executed"]}
        keys = ["spec","golden_cpu","golden_npu"] if stage == "prepare" else ["functional","pipesim","pypto_hardware","standalone","latency"]
        checks = {key:"PASS" for key in keys}
        checks.update(checks_override or {})
        raw = {"schema":"cannbot.scriptor-check/1","stage":stage,"artifact_hash":subject(self.op),
               "checks":checks,"verdict":verdict,"fixture_evidence":True}
        atomic_json(self.op / raw_path, raw)
        report = {"schema":"cannbot.scriptor-report/1","stage":stage,"reviewer":"pypto-pro-scriptor-verifier",
                  "source_id":"test-fixture-sources","artifact_hash":subject(self.op),"artifact_hashes":artifact_hashes(self.op),
                  "verdict":verdict,"checks":{**checks,"semantic_review":"PASS"},"semantic_review":review,
                  "check_report":raw_path,"evidence":[{"path":raw_path,"sha256":digest(self.op / raw_path)}],
                  "metrics":metrics or [],"recommendation":recommendation,"fixture_evidence":True}
        relative = f"reports/{sequence}-sealed.json"
        atomic_json(self.op / relative, report)
        return relative

    def advance(self, optimization=None):
        self.act("init", optimization=optimization or {})
        self.act("complete_prepare", report=self.report("prepare"))
        return self.act("complete_implement", report=self.report("candidate"))


class AutoDeliveryPolicyTests(WorkflowFixture, unittest.TestCase):
    @staticmethod
    def _enter_from_pro_fixture(config, op_dir, state):
        # Only the sync policy is under test; the real bootstrap/GOLDEN gate has separate tests.
        state["frozen"] = {}
        state["current_stage"] = "implement"
        state["stages"] = {"implement": "in_progress", "optimize": "pending", "accept": "pending"}

    def _mode_report(self, stage, mode, *, verdict="PASS"):
        relative = self.report(stage, verdict=verdict)
        report = read_json(self.op / relative)
        raw_path = self.op / report["check_report"]
        raw = read_json(raw_path)
        raw["sync_mode"] = mode
        atomic_json(raw_path, raw)
        report["sync_mode"] = mode
        report["evidence"][0]["sha256"] = digest(raw_path)
        atomic_json(self.op / relative, report)
        return relative

    def test_new_from_pro_defaults_auto_and_accepts_without_manual_trial(self):
        with patch("scriptorlib.workflow._enter_from_pro", self._enter_from_pro_fixture), \
             patch("scriptorlib.exporter.verify_export", return_value={"sync_mode": "auto_mutex"}):
            state = self.act("init", entry_mode="from_pro", optimization={"enabled": False})
            self.assertEqual(state["delivery_sync_mode"], "auto_mutex")
            self.assertFalse(state["manual_requested_by_user"])
            with self.assertRaisesRegex(ContractError, "verifier report sync_mode"):
                self.act("complete_implement", report=self._mode_report("candidate", "manual"))
            with self.assertRaisesRegex(ContractError, "verifier did not pass"):
                self.act("complete_implement", report=self._mode_report("candidate", "auto_mutex", verdict="FAIL"))
            self.assertEqual(self.act("status")["current_stage"], "implement")
            self.act("complete_implement", report=self._mode_report("candidate", "auto_mutex"))
            selected = self.act("finish_optimization")
            (self.op / "generated").mkdir(exist_ok=True)
            (self.op / "generated/export.json").write_text("{}")
            summary = state_summary(self.config, self.project, self.op, selected)
            self.assertEqual(summary["next_action"], "complete_accept")
            (self.op / "generated/export.json").unlink()
            state = self.act("complete_accept", report=self._mode_report("accept", "auto_mutex"))
            self.assertEqual(state["current_stage"], "done")
            self.assertNotIn("sync_trial", state)
            write_final(self.config, self.op, state)
            self.assertEqual(read_json(self.op / "reports/final/final.json")["synchronization"]["selected_mode"], "auto_mutex")

    def test_manual_requires_explicit_user_request_at_init(self):
        with patch("scriptorlib.workflow._enter_from_pro", self._enter_from_pro_fixture):
            with self.assertRaisesRegex(ContractError, "explicit user request"):
                self.act("init", entry_mode="from_pro", delivery_sync_mode="manual")
            state = self.act("init", entry_mode="from_pro", delivery_sync_mode="manual",
                             manual_requested_by_user=True)
            self.assertEqual(state["delivery_sync_mode"], "manual")
            with self.assertRaisesRegex(ContractError, "do not generate a second"):
                self.act("begin_sync_trial")


class StateTests(WorkflowFixture, unittest.TestCase):
    def test_select_earlier_measured_candidate_without_recounting_execution(self):
        self.advance({"enabled": True, "max_iterations": None})
        source = self.op / "scriptor/task.py"
        earlier = source.read_text()
        self.act("begin_round")
        source.write_text("# independently measured alternative\n")
        report = self.report("candidate", recommendation="reject")
        measured = self.act("finish_round", report=report)
        candidate_hash = subject(self.op)
        source.write_text(earlier)
        with self.assertRaisesRegex(ContractError, "stale"):
            self.act("select_best", report=report, reason="earliest candidate within the final tie window")
        source.write_text("# independently measured alternative\n")
        selected = self.act("select_best", report=report, reason="earliest candidate within the final tie window")
        self.assertEqual(selected["best"]["artifact_hash"], candidate_hash)
        self.assertEqual(selected["rounds"], measured["rounds"])
        self.assertFalse(selected["rounds"][0]["accepted"])
        self.assertEqual(selected["history"][-1]["action"], "select_best")

    def test_selection_rejects_unsettled_reports_active_rounds_and_missing_reason(self):
        self.advance({"enabled": True, "max_iterations": None})
        report = self.report("candidate")
        with self.assertRaisesRegex(ContractError, "settled PASS"):
            self.act("select_best", report=report, reason="ranking")
        with self.assertRaisesRegex(ContractError, "ranking reason"):
            self.act("select_best", report=report)
        self.act("begin_round")
        with self.assertRaisesRegex(ContractError, "between optimization rounds"):
            self.act("select_best", report=report, reason="ranking")

    def test_selection_revalidates_original_execution_evidence(self):
        self.advance({"enabled": True, "max_iterations": None})
        self.act("begin_round")
        report = self.report("candidate", recommendation="reject")
        self.act("finish_round", report=report)
        raw = self.op / read_json(self.op / report)["check_report"]
        raw.write_text(raw.read_text() + "\n")
        with self.assertRaisesRegex(ContractError, "evidence changed"):
            self.act("select_best", report=report, reason="ranking")

    def test_skip_then_accept(self):
        self.advance()
        s = self.act("finish_optimization")
        self.assertEqual(s["stages"]["optimize"], "skipped")
        s = self.act("complete_accept", report=self.report("accept"))
        self.assertEqual(s["current_stage"], "done")

    def test_cannot_skip_stage_or_reinitialize(self):
        self.act("init")
        with self.assertRaises(ContractError): self.act("complete_implement", report=self.report("candidate"))
        with self.assertRaises(ContractError): self.act("init")

    def test_explicit_source_migration_preserves_history_budget_and_files_but_requires_new_evidence(self):
        self.act("init", optimization={"enabled": True, "max_iterations": 15})
        old_report = self.report("prepare")
        before = self.act("complete_prepare", report=old_report)
        artifacts = artifact_hashes(self.op)
        atomic_json(self.config / "scriptor-install.json", {"source_id": "new-fixture-sources"})
        with self.assertRaises(ContractError): self.act("status")
        migrated = self.act("migrate_sources", from_source_id="test-fixture-sources",
                            to_source_id="new-fixture-sources", reason="User requested compiler upgrade")
        self.assertEqual(migrated["current_stage"], "prepare")
        self.assertEqual(migrated["stages"]["implement"], "pending")
        self.assertEqual(migrated["optimization"], before["optimization"])
        self.assertEqual(migrated["rounds"], before["rounds"])
        self.assertEqual(migrated["source_migrations"][-1]["previous_state"], before)
        self.assertEqual(artifact_hashes(self.op), artifacts)
        self.assertEqual(self.act("status")["source_id"], "new-fixture-sources")
        with self.assertRaises(ContractError): self.act("complete_prepare", report=old_report)
        with self.assertRaises(ContractError): self.act("complete_implement", report=old_report)

    def test_source_migration_rejects_wrong_identity_or_missing_reason_without_changing_state(self):
        before = self.act("init")
        atomic_json(self.config / "scriptor-install.json", {"source_id": "new-fixture-sources"})
        requests = [{"from_source_id": "wrong", "to_source_id": "new-fixture-sources", "reason": "upgrade"},
                    {"from_source_id": "test-fixture-sources", "to_source_id": "wrong", "reason": "upgrade"},
                    {"from_source_id": "test-fixture-sources", "to_source_id": "new-fixture-sources"}]
        for request in requests:
            with self.subTest(request=request), self.assertRaises(ContractError):
                self.act("migrate_sources", **request)
            self.assertEqual(json.loads((self.op / ".scriptor/state.json").read_text()), before)

    def test_source_migration_cannot_interrupt_an_active_round(self):
        self.advance({"enabled": True})
        before = self.act("begin_round")
        atomic_json(self.config / "scriptor-install.json", {"source_id": "new-fixture-sources"})
        with self.assertRaisesRegex(ContractError, "active optimization round"):
            self.act("migrate_sources", from_source_id="test-fixture-sources",
                     to_source_id="new-fixture-sources", reason="upgrade")
        self.assertEqual(json.loads((self.op / ".scriptor/state.json").read_text()), before)

    def test_finalized_source_migration_clears_current_outcome_and_retains_spent_rounds(self):
        self.advance({"enabled": True, "max_iterations": 1})
        self.act("begin_round")
        self.act("finish_round", report=self.report("candidate", recommendation="reject"))
        self.act("finish_optimization")
        before = self.act("complete_accept", report=self.report("accept"))
        self.assertEqual(before["outcome"], "accepted")
        receipt = self.op / before["final_report_snapshot"]
        receipt_hash = digest(receipt)
        atomic_json(self.config / "scriptor-install.json", {"source_id": "new-fixture-sources"})
        migrated = self.act("migrate_sources", from_source_id="test-fixture-sources",
                            to_source_id="new-fixture-sources", reason="User requested compiler upgrade")
        self.assertEqual(migrated["current_stage"], "prepare")
        self.assertEqual(migrated["optimization"], before["optimization"])
        self.assertEqual(migrated["rounds"], before["rounds"])
        self.assertEqual(len(migrated["rounds"]), 1)
        self.assertEqual(migrated["source_migrations"][-1]["previous_state"], before)
        self.assertEqual(digest(receipt), receipt_hash)
        for key in ("best", "best_report", "final_report", "final_report_sha256", "final_report_snapshot",
                    "completed_at", "stop_reason", "validation_status", "criteria_status", "outcome", "verdict"):
            self.assertNotIn(key, migrated)
        self.assertFalse(migrated["last_criteria"]["met"])

    def test_frozen_input_and_stale_report_rejected(self):
        self.act("init")
        old = self.report("prepare")
        (self.op / "probe_golden_cpu.py").write_text("changed")
        with self.assertRaises(ContractError): self.act("complete_prepare", report=old)
        self.act("complete_prepare", report=self.report("prepare"))
        (self.op / "SPEC.md").write_text("changed")
        with self.assertRaises(ContractError): self.act("complete_implement", report="unused")

    def test_no_hardware_does_not_pass(self):
        self.act("init")
        self.act("complete_prepare", report=self.report("prepare"))
        with self.assertRaises(ContractError):
            self.act("complete_implement", report=self.report("candidate", checks_override={"pypto_hardware":"NOT_RUN"}))

    def test_failed_round_counts_and_best_is_restored(self):
        self.advance({"enabled":True,"max_iterations":1})
        original = (self.op / "scriptor/task.py").read_text()
        self.act("begin_round")
        (self.op / "scriptor/task.py").write_text("broken candidate")
        s = self.act("finish_round", report=self.report("candidate", verdict="FAIL", recommendation="reject"))
        self.assertEqual(len(s["rounds"]), 1)
        with self.assertRaises(ContractError): self.act("begin_round")
        self.act("finish_optimization")
        self.assertEqual((self.op / "scriptor/task.py").read_text(), original)

    def test_unlimited_survives_repeated_rounds_and_resume(self):
        self.advance({"enabled":True,"max_iterations":None})
        for _ in range(7):
            self.act("begin_round")
            self.act("finish_round", report=self.report("candidate", recommendation="reject"))
        self.assertEqual(len(self.act("status")["rounds"]), 7)
        with self.assertRaises(ContractError): self.act("finish_optimization")
        self.assertEqual(self.act("finish_optimization", reason="user_stop")["stop_reason"], "user_stop")

    def test_subagent_and_other_workflow_cannot_write_state(self):
        with self.assertRaises(ContractError):
            transition(self.config,self.project,{"action":"init","opDir":"custom/probe"},actor="worker")
        (self.op / ".orchestrator_state.json").write_text("{}")
        with self.assertRaises(ContractError): self.act("init")

    def test_spec_checks_criteria_scope(self):
        self.contract["exit_criteria"] = condition()
        (self.op / "SPEC.md").write_text("```json machine-contract\n"+json.dumps(self.contract)+"\n```\n")
        with self.assertRaises(ValueError): read_spec(self.config,self.op)

    def test_prompt_budget_update_keeps_consumed_rounds(self):
        self.advance({"enabled":True,"max_iterations":1})
        self.act("begin_round")
        self.act("finish_round",report=self.report("candidate",recommendation="reject"))
        s=self.act("update_optimization",optimization={"max_iterations":None},reason="user requested unlimited")
        self.assertEqual(len(s["rounds"]),1)
        self.act("begin_round")

    def test_completed_delivery_can_enter_a_new_optimization_request(self):
        self.advance()
        self.act("finish_optimization")
        self.act("complete_accept",report=self.report("accept"))
        s=self.act("restart_optimization",reason="user asked to optimize the existing DSL",optimization={"max_iterations":2})
        self.assertEqual(s["current_stage"],"prepare")
        self.assertEqual(s["entry_mode"],"optimize_existing")
        self.assertEqual(s["optimization"]["max_iterations"],2)
        self.assertTrue(s["prior_runs"])
        with self.assertRaises(ContractError): self.act("begin_round")

    def test_met_baseline_exits_in_zero_rounds(self):
        leaf=condition(); leaf["cases"]=["p0"]
        self.contract["exit_criteria"]=leaf
        (self.op/"SPEC.md").write_text("```json machine-contract\n"+json.dumps(self.contract)+"\n```\n")
        self.act("init",optimization={"enabled":True,"max_iterations":None})
        self.act("complete_prepare",report=self.report("prepare"))
        evidence=self.op/"reports/model.json"
        atomic_json(evidence,{"artifact_hash":subject(self.op),"value":80})
        metric=observation("p0",80,artifact_hash=subject(self.op),evidence_file="reports/model.json",evidence_sha256=digest(evidence))
        self.act("complete_implement",report=self.report("candidate",metrics=[metric]))
        with self.assertRaises(ContractError): self.act("begin_round")
        s=self.act("finish_optimization")
        self.assertEqual(s["stop_reason"],"criteria_met")
        self.assertEqual(s["rounds"],[])

    def test_metric_from_other_candidate_is_rejected(self):
        self.act("init")
        self.act("complete_prepare",report=self.report("prepare"))
        with self.assertRaises(ContractError):
            self.act("complete_implement",report=self.report("candidate",metrics=[observation("p0",90,artifact_hash="old")]))

    def test_legacy_target_is_not_silently_ignored(self):
        self.contract["perf_target"]=1
        (self.op/"SPEC.md").write_text("```json machine-contract\n"+json.dumps(self.contract)+"\n```\n")
        with self.assertRaises(ContractError): read_spec(self.config,self.op)


if __name__ == "__main__":
    unittest.main()
