# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Final delivery rejects gaps in source, case domain and packaged execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/scriptor-runtime"))
from scriptorlib.common import ContractError, digest
from scriptorlib.delivery import check_delivery, run_delivery_test


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.op = Path(self.temp.name) / "custom/gqa"
        self.delivery = Path(self.temp.name) / "delivery/gqa"
        self.op.mkdir(parents=True)
        self.delivery.mkdir(parents=True)
        for relative, body in {
            "SPEC.md": "spec\n", "scriptor/gqa_dsl.py": "dsl\n",
            "scriptor/task.py": "task\n",
            "generated/case_01/kernel_pypto.py": "kernel\n",
            "generated/case_01/manifest.json": json.dumps({"entry": "kernel_pypto.py"}),
        }.items():
            self.put(relative, body)
        for relative, body in {
            "golden_cpu.py": "import torch\ndef make_inputs(case): return case\ndef reference(inputs): return inputs\n",
            "wrapper.py": "def gqa(*args): pass\n", "DESIGN.md": "final design\n",
            "kernels/case_01.py": "kernel\n",
        }.items():
            self.put_delivery(relative, body)
        self.row = {"name": "case_01", "kernel": "kernels/case_01.py",
                    "input_shapes": {"query": [1, 2]}, "input_dtypes": {"query": "float16"},
                    "output_shapes": {"y": [1, 2]}, "output_dtypes": {"y": "float16"},
                    "params": {"is_causal": False}}
        self.index = {"source_hashes": {relative: digest(self.op / relative)
                                        for relative in ("SPEC.md", "scriptor/gqa_dsl.py", "scriptor/task.py")},
                      "input_names": ["query"],
                      "input_dtypes": self.row["input_dtypes"],
                      "output_dtypes": self.row["output_dtypes"],
                      "cases": [{"name": "case_01", "directory": "case_01",
                                 "input_shapes": self.row["input_shapes"],
                                 "output_shapes": self.row["output_shapes"],
                                 "params": self.row["params"],
                                 "files": {relative: digest(self.op / f"generated/case_01/{relative}")
                                           for relative in ("kernel_pypto.py", "manifest.json")}}]}
        self.put("generated/export.json", json.dumps(self.index))
        self.put_delivery("test.py", "import wrapper\nCASES = " + repr([self.row]) +
                          "\ndef _exercise(): wrapper.gqa()\n")
        self.put_delivery("REPORT.md", "Final export SHA-256: " + digest(self.op / "generated/export.json") + "\n" +
                 "\n".join(f"Final source SHA-256 ({relative}): {expected}"
                           for relative, expected in self.index["source_hashes"].items()
                           if relative.startswith("scriptor/")) + "\n")

    def put(self, relative, body):
        path = self.op / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def put_delivery(self, relative, body):
        path = self.delivery / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def refresh_report(self):
        self.put("generated/export.json", json.dumps(self.index))
        self.put_delivery("REPORT.md", "Final export SHA-256: " + digest(self.op / "generated/export.json") + "\n" +
                          "\n".join(f"Final source SHA-256 ({relative}): {expected}"
                                    for relative, expected in self.index["source_hashes"].items()
                                    if relative.startswith("scriptor/")) + "\n")

    def check(self, **kwargs):
        return check_delivery(self.op, self.delivery, **kwargs)

    def test_separate_package_passes_structure(self):
        result = self.check()
        self.assertEqual(result["status"], "STRUCTURE_PASS")
        self.assertEqual(result["cases"], 1)
        self.assertEqual(result["export_sha256"], digest(self.op / "generated/export.json"))

    def test_isolated_test_owns_and_cleans_jit_work_directory(self):
        packaged_test = '''import argparse, json, os, tempfile, wrapper
from pathlib import Path
CASES = ROWS
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
jit = Path(os.environ.get("ASCEND_WORK_PATH", "build"))
jit.mkdir(parents=True, exist_ok=True)
(jit / "jit.marker").write_text("compiled")
scratch = tempfile.mkdtemp(prefix="packaged-test-")
wrapper.gqa()
Path(args.output).write_text(json.dumps({"status": "PASS", "case_count": 1,
    "cases": [{"name": "case_01", "status": "PASS", "kernel": "kernels/case_01.py"}],
    "jit_dir": str(jit), "scratch": scratch}))
'''.replace("ROWS", repr([self.row]))
        self.put_delivery("test.py", packaged_test)
        result = run_delivery_test(self.op, self.delivery, self.check(), timeout=10)
        self.assertEqual(result["status"], "PASS")
        evidence = self.op / result["evidence_dir"]
        record = json.loads((evidence / "test_results.json").read_text())
        self.assertFalse(Path(record["jit_dir"]).exists())
        self.assertFalse(Path(record["scratch"]).exists())
        self.assertFalse((self.delivery / "build").exists())

    def test_mixed_dtype_cases_match_one_export(self):
        second = dict(self.row, name="case_02", kernel="kernels/case_02.py",
                      input_dtypes={"query": "bfloat16"}, output_dtypes={"y": "bfloat16"})
        for relative in ("kernel_pypto.py", "manifest.json"):
            content = (self.op / "generated/case_01" / relative).read_text()
            self.put("generated/case_02/" + relative, content)
        self.put_delivery("kernels/case_02.py", "kernel\n")
        case = dict(self.index["cases"][0], name="case_02", directory="case_02",
                    input_dtypes=second["input_dtypes"], output_dtypes=second["output_dtypes"])
        self.index["cases"].append(case)
        self.put("generated/export.json", json.dumps(self.index))
        self.put_delivery("test.py", "import wrapper\nCASES = " + repr([self.row, second]) +
                          "\ndef _exercise(): wrapper.gqa()\n")
        self.put_delivery("REPORT.md", "Final export SHA-256: " + digest(self.op / "generated/export.json") + "\n" +
                 "\n".join(f"Final source SHA-256 ({relative}): {expected}"
                           for relative, expected in self.index["source_hashes"].items()
                           if relative.startswith("scriptor/")) + "\n")
        self.assertEqual(self.check()["cases"], 2)

    def test_sidecar_dtype_cannot_fill_unmatched_primary_export(self):
        changed = dict(self.row, input_dtypes={"query": "bfloat16"})
        self.put_delivery("test.py", "import wrapper\nCASES = " + repr([changed]) +
                          "\ndef _exercise(): wrapper.gqa()\n")
        with self.assertRaisesRegex(ContractError, "differs from the selected export"):
            self.check()

    def test_selected_special_values_cannot_be_replaced_with_finite_inputs(self):
        special = {"query": ["-inf", "+inf"]}
        self.row["input_special_values"] = special
        self.index["cases"][0]["input_special_values"] = special
        self.refresh_report()
        self.put_delivery("test.py", "import wrapper\nCASES = " + repr([self.row]) +
                          "\ndef _exercise(): wrapper.gqa()\n")
        structure = self.check()
        self.assertEqual(structure["cases"], 1)

        finite_test = '''import argparse, json, torch, wrapper
from pathlib import Path
CASES = ROWS
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
wrapper.gqa(torch.tensor([[0., 1.]], dtype=torch.float16))
Path(args.output).write_text(json.dumps({"status": "PASS", "case_count": 1,
    "cases": [{"name": "case_01", "status": "PASS", "kernel": "kernels/case_01.py"}]}))
'''.replace("ROWS", repr([self.row]))
        self.put_delivery("wrapper.py", "def gqa(query): return query\n")
        self.put_delivery("test.py", finite_test)
        with self.assertRaisesRegex(ContractError, "original special inputs"):
            run_delivery_test(self.op, self.delivery, self.check(), timeout=10)

        self.row.pop("input_special_values")
        self.put_delivery("test.py", finite_test.replace(repr([dict(self.row, input_special_values=special)]),
                                                         repr([self.row])))
        with self.assertRaisesRegex(ContractError, "special inputs differ from the selected export"):
            self.check()

    def test_stale_report_cannot_claim_a_reverted_candidate(self):
        self.put_delivery("REPORT.md", "Final export SHA-256: " + "0" * 64 + "\n" +
                 "Historical accepted export: " + digest(self.op / "generated/export.json") + "\n")
        with self.assertRaisesRegex(ContractError, "selected final export.json SHA-256"):
            self.check()

    def test_historical_source_hash_does_not_override_final_field(self):
        source_hash = self.index["source_hashes"]["scriptor/gqa_dsl.py"]
        report = (self.delivery / "REPORT.md").read_text()
        report = report.replace(f"Final source SHA-256 (scriptor/gqa_dsl.py): {source_hash}",
                                f"Final source SHA-256 (scriptor/gqa_dsl.py): {'0' * 64}")
        self.put_delivery("REPORT.md", report + f"Historical source: {source_hash}\n")
        with self.assertRaisesRegex(ContractError, "final source SHA-256 fields differ"):
            self.check()

    def test_golden_is_independent_and_exposes_the_reference_contract(self):
        self.put_delivery("golden_cpu.py", "import torch\n")
        with self.assertRaisesRegex(ContractError, "independent reference"):
            self.check()
        self.put_delivery("golden_cpu.py", "import ascriptor\n"
                 "def make_inputs(case): return case\ndef reference(inputs): return inputs\n")
        with self.assertRaisesRegex(ContractError, "must not import ascriptor"):
            self.check()

    def test_generated_kernel_copy_must_match_selected_export(self):
        self.put_delivery("kernels/case_01.py", "different kernel\n")
        with self.assertRaisesRegex(ContractError, "delivery kernel differs"):
            self.check()

    def test_delivery_entry_must_call_packaged_router(self):
        self.put_delivery("test.py", "CASES = " + repr([self.row]) + "\n")
        with self.assertRaisesRegex(ContractError, "public wrapper"):
            self.check()

    def test_fixed_delivery_mode_must_match_completed_workflow(self):
        self.index["sync_mode"] = "auto_mutex"
        self.refresh_report()
        state = {"current_stage": "implement", "delivery_sync_mode": "auto_mutex",
                 "manual_requested_by_user": False}
        self.put(".scriptor/state.json", json.dumps(state))
        with self.assertRaisesRegex(ContractError, "completed acceptance"):
            self.check()
        state["current_stage"] = "done"
        self.put(".scriptor/state.json", json.dumps(state))
        self.assertEqual(self.check()["status"], "STRUCTURE_PASS")
        self.index["sync_mode"] = "manual"
        self.refresh_report()
        with self.assertRaisesRegex(ContractError, "recorded synchronization mode"):
            self.check()
        state["delivery_sync_mode"] = "manual"
        self.put(".scriptor/state.json", json.dumps(state))
        with self.assertRaisesRegex(ContractError, "explicit user request"):
            self.check()
        state["manual_requested_by_user"] = True
        self.put(".scriptor/state.json", json.dumps(state))
        self.assertEqual(self.check()["status"], "STRUCTURE_PASS")

    def test_unused_specialization_and_temporary_artifacts_fail(self):
        self.put_delivery("kernels/case_02.py", "unused\n")
        with self.assertRaisesRegex(ContractError, "exactly the exported sources"):
            self.check()
        (self.delivery / "kernels/case_02.py").unlink()
        (self.delivery / ".tmp").mkdir()
        with self.assertRaisesRegex(ContractError, "work-only artifact"):
            self.check()

    def test_isolated_run_rejects_missing_packaged_helper(self):
        self.put_delivery("test.py", "import wrapper\nCASES = " + repr([self.row]) +
                          "\ndef _exercise(): wrapper.gqa()\nimport missing_helper\n")
        with self.assertRaisesRegex(ContractError, "isolated delivery test failed"):
            run_delivery_test(self.op, self.delivery, self.check(), timeout=10)
        logs = list((self.op / "reports/delivery-check").glob("*/test.stderr"))
        self.assertEqual(len(logs), 1)
        self.assertIn("missing_helper", logs[0].read_text())

    def test_isolated_run_checks_case_to_kernel_receipt(self):
        body = '''import argparse, json
import wrapper
from pathlib import Path
CASES = ROWS
def _exercise(): wrapper.gqa()
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
wrapper.gqa()
Path(args.output).write_text(json.dumps({"status": "PASS", "case_count": 1,
    "cases": [{"name": "case_01", "status": "PASS", "kernel": "kernels/case_01.py"}]}))
'''.replace("ROWS", repr([self.row]))
        self.put_delivery("test.py", body)
        result = run_delivery_test(self.op, self.delivery, self.check(), timeout=10)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["device_test_cases"], 1)
        self.put_delivery("test.py", body.replace("wrapper.gqa()\nPath", "Path"))
        with self.assertRaisesRegex(ContractError, "did not call the packaged wrapper"):
            run_delivery_test(self.op, self.delivery, self.check(), timeout=10)
        self.put_delivery("test.py", body.replace("kernels/case_01.py\"}]", "kernels/other.py\"}]"))
        with self.assertRaisesRegex(ContractError, "routing differs"):
            run_delivery_test(self.op, self.delivery, self.check(), timeout=10)


if __name__ == "__main__":
    unittest.main()
