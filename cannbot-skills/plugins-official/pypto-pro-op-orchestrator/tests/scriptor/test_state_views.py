# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute suggested requests and preserve old ledgers using protocol fixtures."""
import json
import shlex
import shutil
import subprocess
import unittest
from unittest.mock import patch

import test_finalization as fixtures
from scriptorlib.common import ContractError, atomic_json, read_json
from scriptorlib.progress import state_summary, write_final
from scriptorlib.workflow import transition
from test_workflow import WorkflowFixture


class StateViewTests(fixtures.FinalizationFixture, unittest.TestCase):
    def execute_hint(self):
        # Exercise quoting and existing required request fields, not just hint text.
        hint = self.summary()
        result = subprocess.run(shlex.split(hint["next_command"]), cwd=self.project,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def canonical_report(self, stage):
        relative = self.report(stage)
        path = self.op / f"reports/sealed/{stage}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.op / relative, path)

    def test_hints_execute_normal_flow_without_new_required_fields(self):
        self.act("init")
        self.canonical_report("prepare")
        self.assertEqual(self.execute_hint()["current_stage"], "implement")
        self.canonical_report("candidate")
        self.assertEqual(self.execute_hint()["current_stage"], "optimize")
        hint = self.summary()
        self.assertEqual(hint["stop_condition"], "not_requested")
        self.assertEqual(set(hint["next_request"]), {"action", "opDir"})
        self.assertEqual(self.execute_hint()["current_stage"], "accept")
        self.canonical_report("accept")
        done = self.execute_hint()
        self.assertEqual(done["current_stage"], "done")
        final = subprocess.run(shlex.split(done["next_command"]), capture_output=True, text=True)
        self.assertEqual(final.returncode, 0, final.stderr)
        self.assertIn("NOT_REQUESTED", final.stdout)

    def test_active_round_precedes_budget_exit_and_exact_limit_stops(self):
        self.advance({"enabled": True, "max_iterations": 1})
        self.assertEqual(self.summary()["next_action"], "begin_round")
        self.execute_hint()
        self.assertEqual(self.summary()["next_action"], "finish_round")
        self.canonical_report("candidate")
        self.execute_hint()
        summary = self.summary()
        self.assertEqual(summary["next_action"], "finish_optimization")
        self.assertEqual(summary["stop_condition"], "iteration_budget_exhausted")
        self.assertEqual(self.execute_hint()["current_stage"], "accept")

    def test_explicit_secondary_exits_and_unlimited_use_the_original_policy(self):
        self.set_goal(self.goal())
        with patch("scriptorlib.workflow.time.time", return_value=10):
            self.advance({"enabled": True, "max_iterations": None, "time_budget_s": 5})
        with patch("scriptorlib.workflow.time.time", return_value=16):
            self.assertEqual(self.summary()["stop_condition"], "time_budget_exhausted")
            self.act("finish_optimization")
            self.assertEqual(self.act("complete_accept", report=self.report("accept"))["outcome"], "target_unmet")
        self.act("restart_optimization", reason="no improvement policy", optimization={"max_iterations": None, "max_no_improvement_rounds": 1})
        self.act("complete_prepare", report=self.report("prepare"))
        self.act("complete_implement", report=self.report("candidate"))
        self.act("begin_round")
        self.act("finish_round", report=self.report("candidate", recommendation="reject"))
        self.assertEqual(self.summary()["stop_condition"], "no_improvement")
        self.act("update_optimization", reason="user removed all limits", optimization={"max_no_improvement_rounds": None})
        self.assertEqual(self.summary()["next_action"], "begin_round")
        self.assertNotIn("stop_condition", self.summary())

    def test_verifier_prerequisite_is_not_fabricated_as_ready_evidence(self):
        self.act("init")
        hint = self.summary()
        self.assertIn("verifier", hint["requires"])
        attempted = subprocess.run(shlex.split(hint["next_command"]), cwd=self.project,
                                    capture_output=True, text=True)
        self.assertNotEqual(attempted.returncode, 0)
        self.assertEqual(self.act("status")["current_stage"], "prepare")

    def test_compact_and_full_status_are_read_only(self):
        self.advance()
        ledger = self.op / ".scriptor/state.json"
        lock = self.op / ".scriptor/state.lock"
        lock.unlink()
        before = ledger.read_bytes()
        compact = self.cli_state("status")
        self.assertEqual(compact.returncode, 0, compact.stderr)
        self.assertNotIn("history", json.loads(compact.stdout))
        self.assertFalse(lock.exists())
        for flag in ("--full", "--full-state"):
            full = self.run_cli("state", flag, "--request-json", '{"action":"status","opDir":"custom/probe"}')
            self.assertEqual(json.loads(full.stdout), json.loads(before))
        full = self.cli_state("status", detail="full")
        self.assertEqual(json.loads(full.stdout), json.loads(before))
        self.assertEqual(ledger.read_bytes(), before)
        self.assertFalse(lock.exists())
        with self.assertRaises(ContractError):
            transition(self.config, self.project, {"action": "status", "opDir": "custom/missing"})
        self.assertFalse((self.project / "custom/missing").exists())

    def test_hints_quote_paths_with_spaces_and_shell_metacharacters(self):
        destination = self.project / "project with spaces; literal"
        destination.mkdir()
        config, op = destination / ".opencode", destination / "custom/probe"
        shutil.copytree(self.config, config)
        shutil.copytree(self.op, op)
        state = transition(config, destination, {"action": "init", "opDir": "custom/probe"})
        hint = state_summary(config, destination, op, state)
        self.assertIn(str(config / "scriptor/scripts/scriptor.py"), shlex.split(hint["next_command"]))
        # Parsing the displayed command keeps this cwd/path intact without invoking a shell.
        request = {"action": "status", "opDir": "custom/probe"}
        argv = shlex.split(hint["next_command"])
        argv[-1] = json.dumps(request)
        result = subprocess.run(argv, cwd=destination,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["current_stage"], "prepare")

    def test_old_active_v1_continues_without_refunding_rounds(self):
        self.advance({"enabled": True, "max_iterations": 3})
        self.act("begin_round")
        old = self.report("candidate", recommendation="reject")
        self.act("finish_round", report=old)
        self.act("begin_round")
        ledger = self.op / ".scriptor/state.json"
        legacy = read_json(ledger)
        for row in legacy["rounds"]:
            row.pop("report_sha256", None)
            row.pop("receipt_key", None)
            row.pop("report_snapshot", None)
        atomic_json(ledger, legacy)
        before = ledger.read_bytes()
        with self.assertRaisesRegex(ContractError, "legacy v1 report already settled"):
            self.act("finish_round", report=old)
        self.assertEqual(ledger.read_bytes(), before)
        result = self.act("finish_round", report=self.report("candidate", recommendation="reject"))
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(result["optimization"], legacy["optimization"])
        self.assertEqual(result["history"][:-1], legacy["history"])
        self.assertEqual(result["best"], legacy["best"])

    def test_old_done_v1_regenerates_without_rewriting_historical_state(self):
        self.advance()
        self.act("finish_optimization")
        self.act("complete_accept", report=self.report("accept"))
        ledger = self.op / ".scriptor/state.json"
        legacy = read_json(ledger)
        for key in ("final_report_sha256", "final_report_snapshot", "completed_at", "validation_status", "criteria_status", "outcome", "verdict"):
            legacy.pop(key)
        atomic_json(ledger, legacy)
        before = ledger.read_bytes()
        self.assertEqual(self.cli_state("status").returncode, 0)
        regenerated = self.run_cli("report", "--op-dir", str(self.op))
        self.assertEqual(regenerated.returncode, 0, regenerated.stderr)
        self.assertEqual(ledger.read_bytes(), before)
        final = read_json(self.op / "reports/final/final.json")
        self.assertEqual(final["report_integrity"], "legacy_v1_current_evidence_revalidated")
        self.assertEqual(final["outcome"], "accepted")
        self.assertEqual(self.summary()["next_action"], "read_final")

    def test_old_or_partial_views_are_not_the_next_read_target(self):
        self.advance()
        self.act("finish_optimization")
        state = self.act("complete_accept", report=self.report("accept"))
        write_final(self.config, self.op, state)
        self.assertEqual(self.summary()["next_action"], "read_final")
        (self.op / "reports/final/final.md").write_text("partial result from an interrupted write")
        self.assertEqual(self.summary()["next_action"], "report")
        write_final(self.config, self.op)
        self.act("restart_optimization", reason="new optimization request")
        self.act("complete_prepare", report=self.report("prepare"))
        self.act("complete_implement", report=self.report("candidate"))
        self.act("finish_optimization", reason="user_stop")
        current = self.act("complete_accept", report=self.report("accept"))
        with patch("scriptorlib.progress._atomic_text", side_effect=OSError("fixture write failure")):
            with self.assertRaises(OSError): write_final(self.config, self.op, current)
        self.assertEqual(self.summary()["next_action"], "report")
        write_final(self.config, self.op)
        self.assertEqual(self.summary()["next_action"], "read_final")

    def test_legacy_report_missing_case_details_remains_explicitly_unknown(self):
        self.advance()
        self.act("finish_optimization")
        # Use the legacy fixture with global checks but no per-case detail.
        relative = WorkflowFixture.report(self, "accept")
        state = self.act("complete_accept", report=relative)
        write_final(self.config, self.op, state)
        final = read_json(self.op / "reports/final/final.json")
        self.assertEqual(final["coverage"]["verified"], 0)
        self.assertEqual(final["coverage"]["per_case_details"], "unavailable_in_legacy_report")
        self.assertEqual(final["cases"][0]["checks"]["pypto_hardware"], "UNKNOWN")
        self.assertIsNone(final["cases"][0]["latency_us"])


if __name__ == "__main__":
    unittest.main()
