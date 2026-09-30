# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Workflow and CLI fixtures only: no accelerator execution is claimed."""
from __future__ import annotations

import copy
import csv
import json
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_workflow as fixtures
from scriptorlib.common import ContractError, atomic_json, digest, read_json
from scriptorlib.progress import state_summary, write_final
from scriptorlib.workflow import evaluate_report, subject


class FinalizationFixture(fixtures.WorkflowFixture):
    def setUp(self):
        super().setUp()
        self.contract["inputs"][0]["shape"] = ["N"]
        self.contract["outputs"][0]["shape"] = ["N"]
        self.contract["dynamic_axes_ranges"] = {"N": [1, 2]}
        self.contract["p0_cases"].append({"name": "p1", "params": {},
            "input_shapes": {"x": [2]}, "output_shapes": {"y": [2]}})
        self.save_contract()
        shutil.copytree(fixtures.RUNTIME, self.config / "scriptor/scripts",
                        ignore=shutil.ignore_patterns("__pycache__"))
        self.cli = [sys.executable, str(self.config / "scriptor/scripts/scriptor.py"),
                    "--config-root", str(self.config)]

    def save_contract(self):
        (self.op / "SPEC.md").write_text("```json machine-contract\n" + json.dumps(self.contract) + "\n```\n")

    def goal(self, threshold=1, **fields):
        return {"id": "latency", "metric": "latency_us", "operator": "<=", "threshold": threshold,
                "unit": "us", "basis": "hardware", "cases": ["p0", "p1"],
                "aggregation": "each", "source": "fixture_profile", **fields}

    def set_goal(self, tree):
        self.contract["exit_criteria"] = tree
        self.save_contract()

    def report(self, stage, *, values=(2, 4), extra_metrics=(), **kwargs):
        relative = super().report(stage, **kwargs)
        report = read_json(self.op / relative)
        raw_path = self.op / report["check_report"]
        raw = read_json(raw_path)
        if stage != "prepare":
            raw["cases"] = {name: {"checks": dict(raw["checks"])} for name in ("p0", "p1")}
            evidence_path = relative.replace("-sealed.json", "-metrics.json")
            measurements = [{"metric": "latency_us", "case": name, "value": value,
                             "unit": "us", "basis": "hardware", "source": "fixture_profile"}
                            for name, value in zip(("p0", "p1"), values)] + list(extra_metrics)
            atomic_json(self.op / evidence_path, {"fixture_evidence": True, "measurements": measurements})
            raw["metrics"] = [{**row, "artifact_hash": subject(self.op), "evidence_file": evidence_path,
                               "evidence_sha256": digest(self.op / evidence_path)} for row in measurements]
            report.update(cases=raw["cases"], metrics=raw["metrics"])
            report["evidence"].append({"path": evidence_path, "sha256": digest(self.op / evidence_path)})
        atomic_json(raw_path, raw)
        report["evidence"][0]["sha256"] = digest(raw_path)
        report["criteria"] = evaluate_report(self.config, self.op, report)
        if stage == "accept" and report["criteria"]["status"] not in {"PASS", "NOT_REQUESTED"}:
            report["verdict"] = "FAIL" if report["criteria"]["status"] == "FAIL" else "BLOCKED"
        atomic_json(self.op / relative, report)
        return relative

    def exhaust(self):
        self.advance({"enabled": True, "max_iterations": 1})
        self.act("begin_round")
        self.act("finish_round", report=self.report("candidate", recommendation="reject"))
        self.act("finish_optimization")

    def run_cli(self, *args):
        return subprocess.run(self.cli + list(args), cwd=self.project, text=True, capture_output=True)

    def cli_state(self, action, **fields):
        return self.run_cli("state", "--request-json", json.dumps({"action": action, "opDir": "custom/probe", **fields}))

    def summary(self):
        return state_summary(self.config, self.project, self.op, self.act("status"))


class FinalizationTests(FinalizationFixture, unittest.TestCase):
    def test_explicit_full_budget_keeps_best_after_target_is_met(self):
        self.set_goal(self.goal(threshold=5))
        state = self.advance({"enabled": True, "max_iterations": 2, "stop_on_criteria": False})
        for _ in range(2):
            with self.assertRaises(ContractError):
                self.act("finish_optimization")
            self.act("begin_round")
            result = self.act("finish_round", report=self.report("candidate", values=(3, 5), recommendation="reject"))
            self.assertEqual(result["best"], state["best"])
        self.assertEqual(self.act("finish_optimization")["stop_reason"], "iteration_budget_exhausted")

    def test_budget_closes_unmet_target_without_changing_verdict(self):
        self.set_goal(self.goal())
        self.exhaust()
        report = self.report("accept")
        before = (self.op / report).read_bytes()
        result = self.cli_state("complete_accept", report=report)
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads(result.stdout)
        self.assertEqual((state["current_stage"], state["validation_status"], state["criteria_status"],
                          state["outcome"], state["verdict"]), ("done", "PASS", "FAIL", "target_unmet", "FAIL"))
        self.assertEqual(state["results"]["status"], "written")
        self.assertEqual(state["next_action"], "read_final")
        self.assertEqual((self.op / report).read_bytes(), before)
        final = read_json(self.op / "reports/final/final.json")
        self.assertEqual(final["schema"], "cannbot.scriptor-final/1")
        self.assertTrue(final["fixture_evidence"])
        self.assertEqual(final["criteria"]["conditions"][0]["values"], [2, 4])
        self.assertEqual(final["coverage"]["verified"], 2)
        with (self.op / "reports/final/final.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({x["case"]: float(x["latency_us"]) for x in rows}, {"p0": 2, "p1": 4})
        markdown = (self.op / "reports/final/final.md").read_text()
        for value in ("p0=2, p1=4", "<= 1 us", "target_unmet", "FIXTURE"):
            self.assertIn(value, markdown)

    def test_unknown_custom_goal_stays_unknown(self):
        self.set_goal(self.goal(metric="roofline_efficiency", basis="model", unit="ratio", source="fixture_roofline"))
        self.exhaust()
        state = self.act("complete_accept", report=self.report("accept"))
        self.assertEqual((state["outcome"], state["criteria_status"], state["verdict"]),
                         ("target_unknown", "UNKNOWN", "BLOCKED"))
        write_final(self.config, self.op, state)
        result = read_json(self.op / "reports/final/final.json")
        self.assertEqual(result["criteria"]["conditions"][0]["missing_or_ambiguous_cases"], ["p0", "p1"])
        self.assertIn("Missing/ambiguous: p0, p1", (self.op / "reports/final/final.md").read_text())

    def test_any_and_mean_keep_the_original_condition_scope(self):
        mean = self.goal(3, id="mean_latency", aggregation="mean")
        self.set_goal({"any": [self.goal(1), mean]})
        self.advance({"enabled": True, "max_iterations": None})
        self.assertEqual(self.summary()["next_action"], "finish_optimization")
        self.act("finish_optimization")
        state = self.act("complete_accept", report=self.report("accept"))
        self.assertEqual(state["outcome"], "accepted")
        self.assertEqual(state["rounds"], [])
        write_final(self.config, self.op, state)
        final = read_json(self.op / "reports/final/final.json")
        self.assertEqual([x["status"] for x in final["criteria"]["conditions"]], ["FAIL", "PASS"])
        self.assertEqual(final["criteria_tree"], self.contract["exit_criteria"])
        self.assertIn("any(latency, mean_latency)", (self.op / "reports/final/final.md").read_text())

    def test_failed_required_checks_are_not_hidden_by_a_goal_outcome(self):
        self.set_goal(self.goal())
        self.exhaust()
        for check in ("functional", "pipesim", "pypto_hardware", "standalone", "latency"):
            with self.subTest(check=check):
                relative = self.report("accept", checks_override={check: "NOT_RUN"})
                before = (self.op / ".scriptor/state.json").read_bytes()
                with self.assertRaisesRegex(ContractError, "required checks"):
                    self.act("complete_accept", report=relative)
                self.assertEqual((self.op / ".scriptor/state.json").read_bytes(), before)

    def test_raw_execution_failure_and_semantic_failure_remain_blocking(self):
        self.set_goal(self.goal())
        self.exhaust()
        with self.assertRaisesRegex(ContractError, "raw execution"):
            self.act("complete_accept", report=self.report("accept", verdict="FAIL"))
        relative = self.report("accept")
        report = read_json(self.op / relative)
        report["checks"]["semantic_review"] = "FAIL"
        atomic_json(self.op / relative, report)
        with self.assertRaisesRegex(ContractError, "required checks"):
            self.act("complete_accept", report=relative)

    def test_inconsistent_case_checks_cannot_finalize_before_view_generation(self):
        self.exhaust()
        relative = self.report("accept")
        original = read_json(self.op / relative)
        for change in ("missing_case", "failed_case", "altered_seal"):
            with self.subTest(change=change):
                report = copy.deepcopy(original)
                raw_path = self.op / report["check_report"]
                raw = read_json(raw_path)
                raw["cases"] = copy.deepcopy(original["cases"])
                if change == "missing_case":
                    raw["cases"].pop("p1")
                elif change == "failed_case":
                    raw["cases"]["p1"]["checks"]["functional"] = "FAIL"
                report["cases"] = copy.deepcopy(raw["cases"])
                if change == "altered_seal":
                    report["cases"]["p0"]["checks"]["functional"] = "FAIL"
                atomic_json(raw_path, raw)
                report["evidence"][0]["sha256"] = digest(raw_path)
                atomic_json(self.op / relative, report)
                with self.assertRaisesRegex(ContractError, "per-case"):
                    self.act("complete_accept", report=relative)
                self.assertEqual(self.act("status")["current_stage"], "accept")

    def test_report_source_and_metric_mutations_still_reject(self):
        self.set_goal(self.goal())
        self.exhaust()
        relative = self.report("accept")
        original = read_json(self.op / relative)
        for field, value in (("reviewer", "worker"), ("source_id", "another-source"), ("artifact_hash", "stale"), ("verdict", "PASS")):
            with self.subTest(field=field):
                atomic_json(self.op / relative, {**original, field: value})
                with self.assertRaises(ContractError): self.act("complete_accept", report=relative)
        changed = copy.deepcopy(original)
        changed["metrics"][0]["artifact_hash"] = "another-candidate"
        atomic_json(self.op / relative, changed)
        with self.assertRaisesRegex(ContractError, "execution metric"):
            self.act("complete_accept", report=relative)
        changed = copy.deepcopy(original)
        changed["metrics"][0].update(pointer="/measurements/0/value", value=changed["metrics"][0]["value"] + 1)
        with self.assertRaisesRegex(ContractError, "JSON pointer"): evaluate_report(self.config, self.op, changed)
        atomic_json(self.op / relative, original)
        evidence = self.op / original["metrics"][0]["evidence_file"]
        evidence.write_text("changed evidence")
        with self.assertRaisesRegex(ContractError, "evidence changed"):
            self.act("complete_accept", report=relative)

    def test_selected_checkpoint_and_frozen_golden_are_required(self):
        self.exhaust()
        (self.op / "scriptor/task.py").write_text("another source with its own fresh report")
        with self.assertRaisesRegex(ContractError, "selected checkpoint"):
            self.act("complete_accept", report=self.report("accept"))
        (self.op / "probe_golden_cpu.py").write_text("changed reference")
        with self.assertRaisesRegex(ContractError, "frozen input"):
            self.act("complete_accept", report=self.report("accept"))

    def test_missing_or_inconsistent_stop_cannot_archive_a_failed_target(self):
        self.set_goal(self.goal())
        self.exhaust()
        relative = self.report("accept")
        state = self.act("status")
        for corrupt in ({**state, "stop_reason": "user_stop"},
                        {**state, "history": [e for e in state["history"] if e["action"] != "finish_optimization"]}):
            atomic_json(self.op / ".scriptor/state.json", corrupt)
            with self.assertRaisesRegex(ContractError, "legal optimization stop"):
                self.act("complete_accept", report=relative)
        legacy = copy.deepcopy(state)
        next(event for event in reversed(legacy["history"]) if event["action"] == "finish_optimization").pop("criteria")
        legacy["last_criteria"] = {"met": True, "status": "PASS"}
        atomic_json(self.op / ".scriptor/state.json", legacy)
        self.assertEqual(self.act("complete_accept", report=relative)["outcome"], "target_unmet")

    def test_user_stop_and_disabled_optimization_can_archive_unmet_target(self):
        self.set_goal(self.goal())
        self.advance({"enabled": True, "max_iterations": None})
        with self.assertRaisesRegex(ContractError, "exit condition"):
            self.act("finish_optimization")
        self.act("finish_optimization", reason="user_stop")
        self.assertEqual(self.act("complete_accept", report=self.report("accept"))["outcome"], "target_unmet")
        self.act("restart_optimization", reason="another request")
        self.act("update_optimization", reason="user disabled optimization", optimization={"enabled": False})
        self.act("complete_prepare", report=self.report("prepare"))
        self.act("complete_implement", report=self.report("candidate"))
        self.act("finish_optimization")
        self.assertEqual(self.act("complete_accept", report=self.report("accept"))["outcome"], "target_unmet")

    def test_duplicate_and_resealed_receipts_do_not_settle_a_new_round(self):
        self.advance({"enabled": True, "max_iterations": 3})
        self.act("begin_round")
        relative = self.report("candidate", recommendation="reject")
        settled = self.act("finish_round", report=relative)
        self.assertIn("receipt_key", settled["rounds"][0])
        before = (self.op / ".scriptor/state.json").read_bytes()
        self.act("finish_round", report=relative)
        self.assertEqual((self.op / ".scriptor/state.json").read_bytes(), before)
        self.act("begin_round")
        active = (self.op / ".scriptor/state.json").read_bytes()
        copied = "reports/copied.json"
        shutil.copy2(self.op / relative, self.op / copied)
        self.assertTrue(self.act("finish_round", report=copied)["round_active"])
        self.assertEqual((self.op / ".scriptor/state.json").read_bytes(), active)
        resealed = read_json(self.op / copied)
        resealed["recommendation"] = "keep"
        atomic_json(self.op / copied, resealed)
        with self.assertRaisesRegex(ContractError, "already settled"):
            self.act("finish_round", report=copied)
        self.assertEqual((self.op / ".scriptor/state.json").read_bytes(), active)
        state = self.act("finish_round", report=self.report("candidate", recommendation="reject"))
        self.assertFalse(state["round_active"])
        self.assertEqual(len(state["rounds"]), 2)

    def test_completed_report_change_prevents_regeneration(self):
        self.advance()
        self.act("finish_optimization")
        relative = self.report("accept")
        state = self.act("complete_accept", report=relative)
        write_final(self.config, self.op, state)
        before = (self.op / "reports/final/final.json").read_bytes()
        report = read_json(self.op / relative)
        report["recommendation"] = "changed after completion"
        atomic_json(self.op / relative, report)
        with self.assertRaisesRegex(ContractError, "completed verifier report changed"):
            write_final(self.config, self.op)
        self.assertEqual((self.op / "reports/final/final.json").read_bytes(), before)

    def test_verifier_can_reuse_a_filename_without_overwriting_best_or_replaying_a_round(self):
        self.advance({"enabled": True, "max_iterations": 3})
        canonical = "reports/sealed/candidate.json"
        (self.op / canonical).parent.mkdir(parents=True)
        self.act("begin_round")
        first = self.report("candidate", recommendation="keep")
        shutil.copy2(self.op / first, self.op / canonical)
        kept = self.act("finish_round", report=canonical)
        best_report = kept["best_report"]
        before = (self.op / best_report).read_bytes()
        self.act("begin_round")
        second = self.report("candidate", recommendation="reject")
        shutil.copy2(self.op / second, self.op / canonical)
        settled = self.act("finish_round", report=canonical)
        self.assertEqual(len(settled["rounds"]), 2)
        self.assertFalse(settled["round_active"])
        self.assertEqual(settled["best_report"], best_report)
        self.assertEqual((self.op / best_report).read_bytes(), before)
        self.act("begin_round")
        replay = self.act("finish_round", report=first)
        self.assertTrue(replay["round_active"])
        self.assertEqual(len(replay["rounds"]), 3)
        changed = read_json(self.op / first)
        changed["recommendation"] = "reject"
        atomic_json(self.op / first, changed)
        with self.assertRaisesRegex(ContractError, "already settled"):
            self.act("finish_round", report=first)

    def test_result_write_failure_keeps_acceptance_and_can_be_regenerated(self):
        self.advance()
        self.act("finish_optimization")
        relative = self.report("accept")
        (self.op / "reports/final").write_text("test fixture prevents creating the result directory")
        result = self.cli_state("complete_accept", report=relative)
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["current_stage"], "done")
        self.assertEqual(parsed["results"]["status"], "unavailable")
        state = (self.op / ".scriptor/state.json").read_bytes()
        (self.op / "reports/final").unlink()
        retry = self.run_cli("report", "--op-dir", str(self.op))
        self.assertEqual(retry.returncode, 0, retry.stderr)
        self.assertEqual((self.op / ".scriptor/state.json").read_bytes(), state)
        self.assertTrue((self.op / "reports/final/final.md").is_file())

    def test_stale_state_snapshot_cannot_publish_results_after_restart(self):
        self.advance()
        self.act("finish_optimization")
        state = self.act("complete_accept", report=self.report("accept"))
        self.act("restart_optimization", reason="new user request")
        with self.assertRaisesRegex(ContractError, "workflow changed"):
            write_final(self.config, self.op, state)
        self.assertFalse((self.op / "reports/final").exists())

    def test_new_request_and_rollback_clear_terminal_outcomes(self):
        self.exhaust()
        self.act("complete_accept", report=self.report("accept"))
        restarted = self.act("restart_optimization", reason="new user request")
        self.assertEqual(restarted["prior_runs"][-1]["outcome"], "accepted")
        for key in ("outcome", "criteria_status", "final_report", "completed_at"):
            self.assertNotIn(key, restarted)
        self.act("complete_prepare", report=self.report("prepare"))
        self.act("complete_implement", report=self.report("candidate"))
        self.act("finish_optimization", reason="user_stop")
        self.act("complete_accept", report=self.report("accept"))
        rolled = self.act("rollback_prepare", reason="user changed the formula")
        self.assertNotIn("outcome", rolled)
        self.assertNotIn("final_report", rolled)
        self.assertIsNone(self.summary()["criteria_status"])

    def test_current_run_time_starts_at_restart_without_inventing_author_time(self):
        self.advance()
        self.act("finish_optimization")
        self.act("complete_accept", report=self.report("accept"))
        with patch("scriptorlib.workflow.time.time", return_value=2000):
            self.act("restart_optimization", reason="new request")
            self.act("complete_prepare", report=self.report("prepare"))
            self.act("complete_implement", report=self.report("candidate"))
            self.act("finish_optimization", reason="user_stop")
        with patch("scriptorlib.workflow.time.time", return_value=2005):
            state = self.act("complete_accept", report=self.report("accept"))
        write_final(self.config, self.op, state)
        times = read_json(self.op / "reports/final/final.json")["time"]
        self.assertEqual(times["workflow_wall_time_s"], 5)
        self.assertIsNone(times["author_wall_time_s"])


if __name__ == "__main__":
    unittest.main()
