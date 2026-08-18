# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "collect_golden_reference.py"
SPEC = importlib.util.spec_from_file_location("collect_golden_reference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeNpu:
    def __init__(self, events=None):
        self.events = events

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 8

    def set_device(self, device_id):
        if self.events is not None:
            self.events.append(f"set_device:{device_id}")

    def manual_seed_all(self, seed):
        if self.events is not None:
            self.events.append(f"npu_seed:{seed}")

    def synchronize(self):
        if self.events is not None:
            self.events.append("sync")


class FakeTorch:
    def __init__(self, events=None):
        self.npu = FakeNpu(events)
        self.events = events

    @staticmethod
    def device(value):
        return value

    def manual_seed(self, seed):
        if self.events is not None:
            self.events.append(f"torch_seed:{seed}")


class FakeTorchNpu:
    def __init__(self, events=None):
        self.npu = FakeNpu(events)


def write_manifest(path: Path, case_ids=("p0",)) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "cases": [
            {"id": case_id, "shape": "[1, 32]", "dtype": "fp16"}
            for case_id in case_ids
        ],
    }), encoding="utf-8")


class GoldenReferenceCollectorTest(unittest.TestCase):
    def test_anonymous_single_case_inherits_the_only_manifest_id(self):
        for result in (([1], {}), [1]):
            with self.subTest(result=result):
                self.assertEqual(
                    MODULE.normalize_factory_cases(result, ("p0",)),
                    {"p0": ((1,), {})},
                )
        with self.assertRaisesRegex(ValueError, "multi-case"):
            MODULE.normalize_factory_cases(([1], {}), ("p0", "p1"))

    def test_physical_device_mask_is_removed_before_torch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            golden = Path(directory) / "golden.py"
            golden.write_text(
                "def target(x): return x\n"
                "def factory(device): return ([1], {})\n",
                encoding="utf-8",
            )
            config = MODULE.Config(
                golden, "target", "factory", Path("manifest.json"),
                Path(directory), 3, 0, 1, 42,
            )
            fake_torch = FakeTorch()
            fake_torch_npu = FakeTorchNpu()

            def imports(name):
                self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", os.environ)
                self.assertEqual(os.environ["TILE_FWK_DEVICE_ID"], "3")
                return fake_torch if name == "torch" else fake_torch_npu

            with mock.patch.dict(
                os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "7"}, clear=False,
            ), mock.patch.object(MODULE, "import_module", side_effect=imports):
                runtime = MODULE.load_runtime(config, golden.resolve(), ("p0",))
            self.assertEqual(runtime.device_id, 3)

    def test_kernel_sum_includes_every_positive_row_without_marker_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel_details.csv"
            path.write_text(
                "Device_id,Type,Duration(us)\n"
                "2,ReduceMax,100.0\n"
                "2,TensorMove,4.0\n"
                "2,Mul,3.5\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.sum_kernel_durations(path, 2), 107.5)
            path.write_text(
                "Device_id,Type,Duration(us)\n2,Mul,0\n", encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                MODULE.sum_kernel_durations(path, 2)
            path.write_text(
                "Device_id,Type,Duration(us)\n1,Mul,3\n", encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "physical NPU 2"):
                MODULE.sum_kernel_durations(path, 2)

    def test_each_input_preparation_restarts_the_frozen_seed(self):
        events = []
        runtime = MODULE.Runtime(
            FakeTorch(events), FakeTorchNpu(events), lambda value: value,
            lambda _: ([1], {}), "npu:0", 0, 42, ("p0",),
        )
        MODULE.prepare_invocation(runtime, "p0")
        MODULE.prepare_invocation(runtime, "p0")
        self.assertEqual(events.count("torch_seed:42"), 2)
        self.assertEqual(events.count("npu_seed:42"), 2)

    def test_formal_region_receives_an_already_prepared_callable(self):
        events = []

        class Profiler:
            def __init__(self):
                self.events = events

            def __enter__(self):
                self.events.append("enter")

            @staticmethod
            def __exit__(*_):
                events.append("exit")

            @staticmethod
            def step():
                events.append("step")

        runtime = MODULE.Runtime(
            FakeTorch(events), FakeTorchNpu(events), lambda: None, lambda _: None,
            "npu:2", 2, 42, ("p0",),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            MODULE, "_create_profiler",
            side_effect=lambda *_: events.append("create") or Profiler(),
        ), mock.patch.object(
            MODULE, "_kernel_details_path", return_value=Path("details.csv"),
        ), mock.patch.object(
            MODULE, "sum_kernel_durations", return_value=7.0,
        ):
            events.append("prepare")
            duration = MODULE.profile_invocation(
                runtime, lambda: events.append("invoke"), Path(directory) / "repeat",
            )
        self.assertEqual(duration, 7.0)
        self.assertEqual(events, [
            "prepare", "sync", "create", "enter", "invoke", "sync", "step", "exit",
        ])

    def test_warmup_and_repeats_prepare_fresh_invocations(self):
        events = []
        runtime = MODULE.Runtime(
            FakeTorch(events), FakeTorchNpu(events), lambda: None, lambda _: None,
            "npu:0", 0, 42, ("p0",),
        )
        durations = iter((3.0, 1.0, 2.0))

        def prepare(*_):
            events.append("prepare")
            return lambda: events.append("invoke")

        def profile(_, invocation, __):
            events.append("profile")
            invocation()
            return next(durations)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            MODULE, "prepare_invocation", side_effect=prepare,
        ), mock.patch.object(
            MODULE, "profile_invocation", side_effect=profile,
        ):
            record = MODULE.collect_case(
                runtime, MODULE.ManifestCase("p0", "[1]", "fp16"),
                Path(directory), warmup=2, repeats=3,
            )
        self.assertEqual(events.count("prepare"), 5)
        self.assertEqual(events.count("profile"), 3)
        self.assertEqual(record["raw_repeat_e2e_us"], [3.0, 1.0, 2.0])
        self.assertEqual(record["median_total_us"], 2.0)
        self.assertEqual(record["per_iteration_e2e_us"], 2.0)

    def test_collect_writes_slim_contract_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden = root / "sample_golden.py"
            golden.write_text(
                "def sample_golden(x):\n    return x\n"
                "def _make_inputs(device):\n"
                "    return [('p0', [1], {})]\n",
                encoding="utf-8",
            )
            manifest = root / "PERFORMANCE_CASES.json"
            write_manifest(manifest)
            config = MODULE.Config(
                golden, "sample_golden", "_make_inputs", manifest,
                root, 2, 1, 3, 42,
            )
            durations = iter((9.0, 11.0, 10.0))
            fake_torch = FakeTorch()
            fake_torch_npu = FakeTorchNpu()

            def imports(name):
                return fake_torch if name == "torch" else fake_torch_npu

            with mock.patch.object(MODULE, "import_module", side_effect=imports), \
                    mock.patch.object(
                        MODULE, "profile_invocation",
                        side_effect=lambda *_: next(durations),
                    ):
                payload = MODULE.collect(config)

            self.assertEqual(payload["iterations"], 1)
            self.assertEqual(payload["device_id"], 2)
            self.assertNotIn("device", payload)
            self.assertEqual(payload["cases"][0]["median_total_us"], 10.0)
            self.assertNotIn("raw_repeat_cross_check_us", payload["cases"][0])
            self.assertEqual(len(payload["golden_source"]["sha256"]), 64)
            self.assertEqual(payload["case_manifest_source"]["schema_version"], 1)
            self.assertEqual(
                json.loads((root / "GOLDEN_PERF_REPORT.json").read_text(encoding="utf-8")),
                payload,
            )
            self.assertIn(
                "Performance Summary",
                (root / "GOLDEN_PERF_REPORT.md").read_text(encoding="utf-8"),
            )

    def test_case_mismatch_and_source_mutation_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden = root / "sample_golden.py"
            manifest = root / "PERFORMANCE_CASES.json"
            write_manifest(manifest)
            fake_torch = FakeTorch()
            fake_torch_npu = FakeTorchNpu()

            def imports(name):
                return fake_torch if name == "torch" else fake_torch_npu

            config = MODULE.Config(
                golden, "sample_golden", "_make_inputs", manifest,
                root, 0, 0, 1, 42,
            )
            golden.write_text(
                "def sample_golden(x): return x\n"
                "def _make_inputs(device): return [('other', [1], {})]\n",
                encoding="utf-8",
            )
            with mock.patch.object(MODULE, "import_module", side_effect=imports):
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    MODULE.collect(config)

            golden.write_text(
                "def sample_golden(x): return x\n"
                "def _make_inputs(device): return [('p0', [1], {})]\n",
                encoding="utf-8",
            )

            def mutate_source(*_):
                golden.write_text("# changed\n", encoding="utf-8")
                return 1.0

            with mock.patch.object(MODULE, "import_module", side_effect=imports), \
                    mock.patch.object(
                        MODULE, "profile_invocation", side_effect=mutate_source,
                    ):
                with self.assertRaisesRegex(RuntimeError, "Golden source changed"):
                    MODULE.collect(config)
            self.assertFalse((root / "GOLDEN_PERF_REPORT.json").exists())
            self.assertFalse((root / "GOLDEN_PERF_REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
