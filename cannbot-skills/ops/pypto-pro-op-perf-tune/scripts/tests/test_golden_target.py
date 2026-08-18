# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import NamedTuple

SCRIPTS_DIR = Path(__file__).parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
MSPROF_SCRIPT = SCRIPTS_DIR / "msprof_perf_summary.py"
MSPROF_SPEC = importlib.util.spec_from_file_location(
    "msprof_perf_summary_golden_target", MSPROF_SCRIPT
)
MSPROF = importlib.util.module_from_spec(MSPROF_SPEC)
assert MSPROF_SPEC and MSPROF_SPEC.loader
sys.modules[MSPROF_SPEC.name] = MSPROF
MSPROF_SPEC.loader.exec_module(MSPROF)
CONTRACT = importlib.import_module("golden_contract")
BATCH_EV = importlib.import_module("batch_evidence")
CLI = importlib.import_module("evidence_cli")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class _BatchPayload(NamedTuple):
    """批量证据夹具的载荷（source/report/原始行/诊断）。"""
    source: dict
    report: dict
    raw_rows: list
    diagnoses: list


class GoldenDefaultTargetTest(unittest.TestCase):
    @staticmethod
    def _write_golden_collection(root: Path, *, shape="(1, 32)", dtype="torch.float16"):
        manifest = root / "PERFORMANCE_CASES.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "cases": [{"id": "p0", "shape": "[1, 32]", "dtype": "fp16"}],
        }), encoding="utf-8")
        golden_source = root / "sample_golden.py"
        golden_source.write_text("def sample_golden(x):\n    return x\n", encoding="utf-8")
        markdown = root / "GOLDEN_PERF_REPORT.md"
        markdown.write_text(
            "# Golden Performance Report\n\n"
            "- **Iterations**: 2\n\n"
            "## Performance Summary\n\n"
            "| Case | Input Shape | dtype | E2E |\n"
            "| --- | --- | --- | --- |\n"
            "| p0 | (1, 32) | torch.float16 | 20.000000us |\n",
            encoding="utf-8",
        )
        manifest_source = CONTRACT.source_file_record(manifest)
        manifest_source["schema_version"] = 1
        contract = {
            "schema_version": 1,
            "golden_source": CONTRACT.source_file_record(golden_source),
            "case_manifest_source": manifest_source,
            "device_id": 0,
            "iterations": 1,
            "warmup": 3,
            "repeats": 3,
            "seed": 42,
            "timing_scope": "all_golden_npu_kernels",
            "aggregation": "median_of_repeat_totals",
            "cases": [{
                "id": "p0",
                "shape": shape,
                "dtype": dtype,
                "raw_repeat_e2e_us": [9.0, 10.0, 11.0],
                "median_total_us": 10.0,
                "per_iteration_e2e_us": 10.0,
            }],
        }
        (root / "GOLDEN_PERF_REPORT.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
        return manifest, contract

    @staticmethod
    def _args(golden_status: str) -> Namespace:
        return Namespace(
            warmup=1,
            repeats=3,
            seed=42,
            collection_id="target-test",
            op_name="kernel",
            case_arg="--case-id",
            case_env=None,
            quick=False,
            deep_round_dir=None,
            golden_iterations=3 if golden_status != "not_provided" else None,
            performance_cases={
                "type": "case_manifest",
                "golden_diagnostic": {"status": golden_status},
                "golden_contract": (
                    {"path": "contract.json", "protocol": {
                        "device_id": 0, "seed": 42,
                    }}
                    if golden_status == "joined" else None
                ),
            },
        )

    @staticmethod
    def _summary(rows: list[dict], golden_status: str = "joined") -> dict:
        return MSPROF.compute_compare_summary(MSPROF.CompareSummaryInput(
            Path("op"), rows, [], [], [], len(rows),
            GoldenDefaultTargetTest._args(golden_status), 0, "test",
        ))

    def test_all_exact_joined_cases_meet_default_target(self) -> None:
        report = self._summary([
            {"case": "p0", "ref_us": 10.0, "asc_us": 5.0},
            {"case": "p1", "ref_us": 12.0, "asc_us": 12.0},
        ])
        self.assertEqual(
            report["default_target_metric"],
            "golden_per_iteration_e2e_us / pypto_target_kernel_us",
        )
        self.assertEqual(report["default_target_threshold"], 1.0)
        self.assertEqual(
            [case["default_target_ratio"] for case in report["per_case"]],
            [2.0, 1.0],
        )
        self.assertEqual(
            [case["golden_reference_ratio"] for case in report["per_case"]],
            [2.0, 1.0],
        )
        self.assertEqual(report["golden_reference_ratio_stats"]["min"], 1.0)
        self.assertTrue(report["valid_for_target_met"])
        self.assertTrue(report["default_target_met"])
        self.assertEqual(report["default_target_status"], "met")
        self.assertFalse(report["valid_for_optimization_speedup"])

    def test_one_case_below_one_misses_default_target(self) -> None:
        report = self._summary([
            {"case": "p0", "ref_us": 10.0, "asc_us": 5.0},
            {"case": "p1", "ref_us": 8.0, "asc_us": 10.0},
        ])
        self.assertTrue(report["valid_for_target_met"])
        self.assertFalse(report["default_target_met"])
        self.assertEqual(report["default_target_status"], "not_met")

    def test_no_golden_keeps_pypto_valid_but_target_unavailable(self) -> None:
        report = self._summary([
            {"case": "p0", "ref_us": None, "asc_us": 5.0},
            {"case": "p1", "ref_us": None, "asc_us": 6.0},
        ], golden_status="not_provided")
        self.assertEqual(report["n_cases_valid"], 2)
        self.assertEqual(report["n_default_target_cases"], 0)
        self.assertFalse(report["valid_for_target_met"])
        self.assertFalse(report["default_target_met"])
        self.assertEqual(report["default_target_status"], "unavailable")

    def test_nonfinite_or_nonpositive_value_invalidates_target(self) -> None:
        for field, value in (
            ("ref_us", float("nan")),
            ("ref_us", float("inf")),
            ("ref_us", 0.0),
            ("asc_us", float("nan")),
            ("asc_us", float("inf")),
            ("asc_us", 0.0),
        ):
            with self.subTest(field=field, value=value):
                row = {"case": "p0", "ref_us": 10.0, "asc_us": 5.0}
                row[field] = value
                report = self._summary([row])
                self.assertFalse(report["valid_for_target_met"])
                self.assertFalse(report["default_target_met"])
                self.assertEqual(report["default_target_status"], "unavailable")

    def test_exact_join_provenance_and_unique_full_case_set_are_required(self) -> None:
        rows = [
            {"case": "p0", "ref_us": 10.0, "asc_us": 5.0},
            {"case": "p1", "ref_us": 12.0, "asc_us": 6.0},
        ]
        unjoined = self._summary(rows, golden_status="not_provided")
        self.assertFalse(unjoined["valid_for_target_met"])
        duplicate = self._summary([
            {"case": "p0", "ref_us": 10.0, "asc_us": 5.0},
            {"case": "p0", "ref_us": 12.0, "asc_us": 6.0},
        ])
        self.assertFalse(duplicate["valid_for_target_met"])

    def test_shape_and_dtype_canonicalization_rejects_mismatch(self) -> None:
        manifest = ("p0", "[1, 32]", "fp16", None)
        equivalent = ("p0", "(1, 32)", "torch.float16", 10.0)
        self.assertIsNone(CONTRACT.golden_case_metadata_error(manifest, equivalent))
        self.assertIn(
            "shape mismatch",
            CONTRACT.golden_case_metadata_error(
                manifest, ("p0", "(1, 33)", "torch.float16", 10.0)
            ),
        )
        self.assertIn(
            "dtype mismatch",
            CONTRACT.golden_case_metadata_error(
                manifest, ("p0", "(1, 32)", "torch.float32", 10.0)
            ),
        )
        named_manifest = (
            "p0", "{q: [1, 32], k: [1, 64]}", "{q: fp16, k: fp32}", None
        )
        named_golden = (
            "p0", "{q: (1, 32), k: (1, 64)}",
            "{q: torch.float16, k: torch.float32}", 10.0,
        )
        self.assertIsNone(
            CONTRACT.golden_case_metadata_error(named_manifest, named_golden)
        )
        self.assertIn(
            "shape mismatch",
            CONTRACT.golden_case_metadata_error(
                named_manifest,
                ("p0", "{x: (1, 32), k: (1, 64)}", named_golden[2], 10.0),
            ),
        )
        self.assertIn(
            "dtype mismatch",
            CONTRACT.golden_case_metadata_error(
                named_manifest,
                ("p0", named_golden[1], "{x: torch.float16, k: torch.float32}", 10.0),
            ),
        )

    def test_quick_measurement_is_not_final_target_evidence(self) -> None:
        parsed_args = self._args("joined")
        parsed_args.quick = True
        rows = [{"case": "p0", "ref_us": 10.0, "asc_us": 5.0}]
        report = MSPROF.compute_compare_summary(MSPROF.CompareSummaryInput(
            Path("op"), rows, [], [], [], 1, parsed_args, 0, "test",
        ))
        self.assertFalse(report["valid_for_target_met"])
        self.assertEqual(report["default_target_status"], "unavailable")

    def test_golden_and_pypto_must_use_same_physical_device(self) -> None:
        parsed_args = self._args("joined")
        rows = [{"case": "p0", "ref_us": 10.0, "asc_us": 5.0}]
        report = MSPROF.compute_compare_summary(MSPROF.CompareSummaryInput(
            Path("op"), rows, [], [], [], 1, parsed_args, 1, "test",
        ))
        self.assertFalse(report["valid_for_target_met"])
        self.assertEqual(report["default_target_status"], "unavailable")

    def test_json_contract_is_the_canonical_joined_target_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._write_golden_collection(root)
            args = Namespace(case_manifest=str(manifest))
            cases, source, iterations, error = CONTRACT.resolve_performance_cases(
                root, args
            )
            self.assertIsNone(error)
            self.assertEqual(cases, [("p0", "[1, 32]", "fp16", 10.0)])
            self.assertEqual(iterations, 1)
            self.assertEqual(source["golden_diagnostic"]["status"], "joined")
            self.assertEqual(source["golden_contract"]["protocol"]["device_id"], 0)
            self.assertIsNone(CONTRACT.performance_case_source_error(source))

    def test_case_manifest_is_required_even_when_markdown_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_golden_collection(root)
            _, _, _, error = CONTRACT.resolve_performance_cases(
                root, Namespace(case_manifest=None)
            )
            self.assertEqual(error, "--case-manifest is required")

    def test_json_contract_rejects_case_metadata_and_sample_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, contract = self._write_golden_collection(
                root, shape="(1, 33)"
            )
            _, _, _, error = CONTRACT.resolve_performance_cases(
                root, Namespace(case_manifest=str(manifest))
            )
            self.assertIn("shape mismatch", error)

            contract["cases"][0]["shape"] = "(1, 32)"
            contract["cases"][0]["median_total_us"] = 19.0
            (root / "GOLDEN_PERF_REPORT.json").write_text(
                json.dumps(contract), encoding="utf-8"
            )
            _, _, _, error = CONTRACT.resolve_performance_cases(
                root, Namespace(case_manifest=str(manifest))
            )
            self.assertIn("median is not reproducible", error)

    def test_joined_json_does_not_require_the_human_readable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._write_golden_collection(root)
            (root / "GOLDEN_PERF_REPORT.md").unlink()
            cases, source, _, error = CONTRACT.resolve_performance_cases(
                root, Namespace(case_manifest=str(manifest))
            )
            self.assertIsNone(error)
            self.assertEqual(cases[0][3], 10.0)
            self.assertEqual(source["golden_diagnostic"]["status"], "joined")

    def test_slim_golden_contract_protocol_and_raw_sample_corruption_fail_closed(self) -> None:
        corruptions = (
            ("invalid device id", lambda payload: payload.__setitem__("device_id", -1)),
            ("iterations must be one", lambda payload: payload.__setitem__("iterations", 2)),
            ("wrong seed", lambda payload: payload.__setitem__("seed", 41)),
            ("bad repeats", lambda payload: payload.__setitem__("repeats", 4)),
            (
                "nonfinite sample",
                lambda payload: payload["cases"][0]["raw_repeat_e2e_us"].__setitem__(
                    0, float("nan")
                ),
            ),
            (
                "per-iteration mismatch",
                lambda payload: payload["cases"][0].__setitem__(
                    "per_iteration_e2e_us", 9.0
                ),
            ),
        )
        for label, mutate in corruptions:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, payload = self._write_golden_collection(root)
                mutate(payload)
                (root / "GOLDEN_PERF_REPORT.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                _, _, error = CONTRACT.parse_golden_contract(root)
                self.assertIsNotNone(error)

    def test_renderers_use_canonical_ratio_not_compatibility_alias(self) -> None:
        report = self._summary([
            {"case": "p0", "shape": "[1]", "dtype": "fp16",
             "ref_us": 10.0, "asc_us": 5.0},
        ])
        report["per_case"][0]["speedup"] = 999.0
        report["per_case"][0]["repeat_bound_diagnoses"] = [{
            "routing_label": "data_movement_candidate",
            "scalar_route_triggered": False,
            "roofline_terminal_status": "insufficient_evidence",
            "pipeline_overlap_status": "unverified",
        }]
        report["geomean_speedup"] = 999.0
        report["mean_speedup"] = 999.0
        markdown = CLI.report_compare_to_markdown(report)
        text = CLI.report_compare_to_text(report)
        self.assertIn("2.000", markdown)
        self.assertNotIn("999.000", markdown)
        self.assertIn("2.00x", text)
        self.assertNotIn("999.00x", text)
        self.assertIn("data_movement_candidate", markdown)
        self.assertIn("不是最终 Roofline 结论", markdown)
        self.assertIn("bound_route=['data_movement_candidate']", text)
        self.assertIn("diagnostic only", text)

    def test_batch_propagates_operator_target_status(self) -> None:
        met = self._summary([
            {"case": "p0", "ref_us": 10.0, "asc_us": 5.0},
        ])
        missed = self._summary([
            {"case": "p0", "ref_us": 5.0, "asc_us": 10.0},
        ])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch.json"
            BATCH_EV.generate_batch_json_report(
                Namespace(collection_id="batch-test", output_json=str(output)),
                [
                    {"name": "met", "data": met},
                    {"name": "missed", "data": missed},
                ],
                Path(directory),
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(payload["valid_for_target_met"])
        self.assertFalse(payload["default_target_met"])
        self.assertEqual(payload["default_target_status"], "not_met")
        self.assertEqual(
            [operator["default_target_status"] for operator in payload["operators"]],
            ["met", "not_met"],
        )
        self.assertFalse(payload["valid_for_optimization_speedup"])

    def test_batch_evidence_missing_and_corrupt_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fx = self._batch_fixture(directory)
            op_dir, case_dir, report, raw_rows = (
                fx["op_dir"], fx["case_dir"], fx["report"], fx["raw_rows"],
            )
            self.assertIsNone(BATCH_EV.batch_evidence_error(op_dir, report))

            missing_metric = case_dir / "repeat_002" / "op_summary_Memory.csv"
            missing_metric.unlink()
            self.assertIn(
                "missing archived metric",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )
            write_csv(missing_metric, [raw_rows[1]])

            corrupt_metric = case_dir / "repeat_002" / "op_summary_MemoryUB.csv"
            corrupt_metric.write_text("", encoding="utf-8")
            self.assertIn(
                "invalid archived metric schema",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )
            write_csv(corrupt_metric, [raw_rows[1]])

    def test_batch_evidence_wrong_target_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fx = self._batch_fixture(directory)
            op_dir, case_dir, args, report, raw_rows = (
                fx["op_dir"], fx["case_dir"], fx["args"], fx["report"], fx["raw_rows"],
            )
            wrong_target_metric = (
                case_dir / "repeat_002" / "op_summary_ArithmeticUtilization.csv"
            )
            write_csv(wrong_target_metric, [{
                "Op Name": "other_kernel", "Task Duration(us)": "5.0",
            }])
            self.assertIn(
                "expected exactly one row",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )
            write_csv(wrong_target_metric, [raw_rows[1]])

            timing_metric = case_dir / "repeat_002" / "op_summary_PipeUtilization.csv"
            write_csv(timing_metric, [{
                "Op Name": args.op_name, "Task Duration(us)": "99.0",
            }])
            self.assertIn(
                "raw PipeUtilization duration mismatch",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )
            write_csv(timing_metric, [raw_rows[1]])

    def test_batch_evidence_status_and_forged_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fx = self._batch_fixture(directory)
            op_dir, case_dir, args, report, raw_rows, diagnoses = (
                fx["op_dir"], fx["case_dir"], fx["args"], fx["report"],
                fx["raw_rows"], fx["diagnoses"],
            )
            repeat_status_path = case_dir / "repeat_003" / "evidence_status.json"
            repeat_status = json.loads(repeat_status_path.read_text(encoding="utf-8"))
            repeat_status["target_op_name"] = "other_kernel"
            repeat_status_path.write_text(json.dumps(repeat_status), encoding="utf-8")
            self.assertIn(
                "repeat evidence status mismatch",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )
            repeat_status["target_op_name"] = args.op_name
            repeat_status_path.write_text(json.dumps(repeat_status), encoding="utf-8")

            # Even if all three derived copies are forged consistently, batch
            # must rebuild the diagnosis from the seven raw CSVs and reject it.
            canonical_diagnosis = CLI.diagnose_bound_route(raw_rows[2])
            forged = json.loads(json.dumps(diagnoses[2]))
            forged["routing_label"] = "forged"
            measurement_path = case_dir / "measurement.json"
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
            measurement["repeat_bound_diagnoses"][2] = forged
            measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
            report["per_case"][0]["repeat_bound_diagnoses"][2] = forged
            repeat_status["bound_diagnosis"] = forged
            repeat_status_path.write_text(json.dumps(repeat_status), encoding="utf-8")
            self.assertIn(
                "raw bound diagnosis mismatch",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )
            measurement["repeat_bound_diagnoses"][2] = canonical_diagnosis
            measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
            report["per_case"][0]["repeat_bound_diagnoses"][2] = canonical_diagnosis
            repeat_status["bound_diagnosis"] = canonical_diagnosis
            repeat_status_path.write_text(json.dumps(repeat_status), encoding="utf-8")

    def test_batch_evidence_protocol_fields_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fx = self._batch_fixture(directory)
            op_dir, round_dir, report = fx["op_dir"], fx["round_dir"], fx["report"]
            for field, tampered_value in (
                ("device_id", 1), ("seed", 41), ("warmup", 2), ("repeats", 4),
            ):
                with self.subTest(protocol_field=field):
                    original_value = report[field]
                    report[field] = tampered_value
                    self.assertIn(
                        f"collection protocol mismatch: {field}",
                        BATCH_EV.batch_evidence_error(op_dir, report),
                    )
                    report[field] = original_value

            collection_path = round_dir / "collection.json"
            collection = json.loads(collection_path.read_text(encoding="utf-8"))
            report["seed"] = 41
            collection["seed"] = 41
            collection_path.write_text(json.dumps(collection), encoding="utf-8")
            self.assertIn(
                "collection seed must be 42",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )
            report["seed"] = 42
            collection["seed"] = 42
            collection_path.write_text(json.dumps(collection), encoding="utf-8")

    def test_batch_evidence_ratio_and_target_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fx = self._batch_fixture(directory)
            op_dir, report = fx["op_dir"], fx["report"]
            report["per_case"][0]["golden_reference_ratio"] = 999.0
            self.assertIn(
                "golden_reference_ratio mismatch",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )
            report["per_case"][0]["golden_reference_ratio"] = 2.0
            report["default_target_met"] = False
            self.assertIn(
                "target decision mismatch",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )
            report["default_target_met"] = True

    def test_batch_evidence_samples_and_golden_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fx = self._batch_fixture(directory)
            op_dir, case_dir, round_dir, source, report = (
                fx["op_dir"], fx["case_dir"], fx["round_dir"], fx["source"], fx["report"],
            )
            measurement_path = case_dir / "measurement.json"
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
            measurement["duration_samples_us"] = [5.0]
            measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
            self.assertIn(
                "duration samples are invalid",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )

            measurement["duration_samples_us"] = [4.0, 5.0, 6.0]
            measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
            golden_path = op_dir / "GOLDEN_PERF_REPORT.json"
            golden_payload = json.loads(golden_path.read_text(encoding="utf-8"))
            extra_case = dict(golden_payload["cases"][0])
            extra_case["id"] = "extra"
            golden_payload["cases"].append(extra_case)
            golden_path.write_text(json.dumps(golden_payload), encoding="utf-8")
            protocol = source["golden_contract"]["protocol"]
            source["golden_contract"] = CONTRACT.source_file_record(golden_path)
            source["golden_contract"]["protocol"] = protocol
            report["performance_cases"] = source
            collection_path = round_dir / "collection.json"
            collection = json.loads(collection_path.read_text(encoding="utf-8"))
            collection["performance_cases"] = source
            collection_path.write_text(json.dumps(collection), encoding="utf-8")
            self.assertIn(
                "case ids do not exactly match",
                BATCH_EV.batch_evidence_error(op_dir, report),
            )

    def _fixture_args_source(self, op_dir: Path):
        manifest_path, _ = self._write_golden_collection(op_dir)
        cases, source, iterations, error = CONTRACT.resolve_performance_cases(
            op_dir, Namespace(case_manifest=str(manifest_path))
        )
        self.assertIsNone(error)
        args = self._args("joined")
        args.performance_cases = source
        args.golden_iterations = iterations
        return args, source, iterations, cases

    def _fixture_report(self, args, op_dir, case_dir, repeat_durations, golden_us):
        rows = [{
            "case": "p0", "shape": "[1, 32]", "dtype": "fp16",
            "ref_us": golden_us,
            "asc_us": 5.0,
            "duration_samples_us": repeat_durations,
            "selected_op_names": [args.op_name] * 3,
        }]
        report = MSPROF.compute_compare_summary(MSPROF.CompareSummaryInput(
            op_dir, rows, [], [], [], 1, args, 0, "test",
        ))
        raw_rows = [
            {"Op Name": args.op_name, "Task Duration(us)": str(duration)}
            for duration in repeat_durations
        ]
        diagnoses = [CLI.diagnose_bound_route(row) for row in raw_rows]
        report["per_case"][0]["repeat_bound_diagnoses"] = diagnoses
        report["per_case"][0]["timing_source_metric"] = "PipeUtilization"
        report["per_case"][0]["timing_source_files"] = [
            str(case_dir / f"repeat_{index:03d}" / "op_summary_PipeUtilization.csv")
            for index in range(1, 4)
        ]
        return report, raw_rows, diagnoses

    def _fixture_files(self, case_dir, args, payload: _BatchPayload):
        repeat_durations = [4.0, 5.0, 6.0]
        (case_dir / "measurement.json").write_text(json.dumps({
            "collection_id": args.collection_id,
            "target_op_name": args.op_name,
            "case": "p0",
            "duration_samples_us": repeat_durations,
            "selected_op_names": [args.op_name] * 3,
            "aggregate_us": 5.0,
            "aggregation_method": "trimmed_mean",
            "repeat_bound_diagnoses": payload.diagnoses,
            "repeat_evidence_dirs": [
                str(case_dir / f"repeat_{index:03d}") for index in range(1, 4)
            ],
            "timing_source_metric": "PipeUtilization",
            "timing_source_files": payload.report["per_case"][0]["timing_source_files"],
        }), encoding="utf-8")
        for repeat_index in range(1, 4):
            repeat_dir = case_dir / f"repeat_{repeat_index:03d}"
            repeat_dir.mkdir()
            (repeat_dir / "evidence_status.json").write_text(json.dumps({
                "collection_id": args.collection_id,
                "case": "p0",
                "target_op_name": args.op_name,
                "seven_metric_status": "complete",
                "bound_diagnosis": payload.diagnoses[repeat_index - 1],
            }), encoding="utf-8")
            for metric in MSPROF.METRICS:
                write_csv(
                    repeat_dir / f"op_summary_{metric}.csv",
                    [payload.raw_rows[repeat_index - 1]],
                )
        (case_dir.parent / "collection.json").write_text(json.dumps({
            "status": "complete",
            "mode": "compare",
            "collection_id": args.collection_id,
            "target_op_name": args.op_name,
            "device_id": payload.report["device_id"],
            "seed": payload.report["seed"],
            "warmup": payload.report["warmup"],
            "repeats": payload.report["repeats"],
            "performance_cases": payload.source,
            "expected_cases": ["p0"],
            "completed_cases": 1,
        }), encoding="utf-8")

    def _batch_fixture(self, directory: str):
        """构造一套完整有效的批量证据目录，返回各关键路径与对象。"""
        op_dir = Path(directory) / "op"
        op_dir.mkdir()
        args, source, iterations, cases = self._fixture_args_source(op_dir)
        round_dir = op_dir / "docs" / "perf" / "round_001"
        case_dir = round_dir / f"case_{MSPROF.safe_case_dir_name('p0')}"
        case_dir.mkdir(parents=True)
        args.deep_round_dir = str(round_dir)
        report, raw_rows, diagnoses = self._fixture_report(
            args, op_dir, case_dir, [4.0, 5.0, 6.0], cases[0][3]
        )
        self._fixture_files(
            case_dir, args, _BatchPayload(source, report, raw_rows, diagnoses),
        )
        return {
            "op_dir": op_dir, "case_dir": case_dir, "round_dir": round_dir,
            "args": args, "source": source, "report": report,
            "raw_rows": raw_rows, "diagnoses": diagnoses,
        }


if __name__ == "__main__":
    unittest.main()
