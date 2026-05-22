#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device pool preflight and pool-mode case_loader probe behavior."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

from benchmark import case_loader, device_pool


def test_normalize_pool_devices_dedupes_and_preserves_order() -> None:
    assert device_pool.normalize_pool_devices([1, "1", 2, " 2 "]) == ["1", "2"]


def test_normalize_pool_devices_rejects_empty() -> None:
    try:
        device_pool.normalize_pool_devices([])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_normalize_pool_devices_rejects_blank_entry() -> None:
    try:
        device_pool.normalize_pool_devices(["0", "  ", "1"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_quick_test_failure_includes_device_id_and_summary(monkeypatch) -> None:
    class Proc:
        returncode = 1
        stdout = '{"ok": false, "error": "RuntimeError: simulated npu fault"}'
        stderr = ""

    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=Proc()))
    err = device_pool.run_softmax_quick_test_subprocess(
        "7", timeout_sec=5, python_executable="/fake/python"
    )
    assert err is not None
    assert err.device_id == "7"
    assert "RuntimeError" in err.error_summary
    assert "simulated npu fault" in err.error_summary


def test_collect_preflight_aggregates_multiple_failures(monkeypatch) -> None:

    def fake_run(device_id, **kwargs):
        did = str(device_id).strip()
        if did == "0":
            return None
        return device_pool.DeviceQuickFailure(device_id=did, error_summary=f"bad {did}")

    monkeypatch.setattr(device_pool, "run_softmax_quick_test_subprocess", fake_run)
    fails = device_pool.collect_softmax_preflight_failures(["0", "1", "2"])
    assert [f.device_id for f in fails] == ["1", "2"]
    report = device_pool.format_preflight_failure_report(fails)
    assert "device_id='1'" in report
    assert "bad 1" in report


def test_load_case_explicit_probe_skips_list_idle(tmp_path: Path, monkeypatch) -> None:
    def boom():
        raise AssertionError("_list_idle_chip_ids must not be called for explicit probe device")

    monkeypatch.setattr(case_loader, "_list_idle_chip_ids", boom)

    def fake_run_probe(case_path, timeout_sec, probe_outputs, chip_id=None):
        if not probe_outputs:
            return [case_loader.TensorSpec(name="x0", shape=[2, 3], dtype="float32")], [], "[]"
        assert chip_id == "3"
        return [], [case_loader.TensorSpec(name="y0", shape=[2, 3], dtype="float32")], "[]"

    monkeypatch.setattr(case_loader, "_run_probe_subprocess", fake_run_probe)
    case_file = tmp_path / "p.py"
    case_file.write_text(
        textwrap.dedent(
            """
            class Model:
                def forward(self, x):
                    return x

                def __call__(self, x):
                    return self.forward(x)

            class FakeTensor:
                shape = (2, 3)
                dtype = "float32"

            def get_inputs():
                return [FakeTensor()]

            def get_init_inputs():
                return []
            """
        ).strip(),
        encoding="utf-8",
    )

    case = case_loader.load_case(
        case_file,
        case_id="pool_probe",
        output_probe_device_id="3",
        allow_find_free=False,
    )
    assert case.outputs[0].shape == [2, 3]


def test_allow_find_free_false_skips_idle_and_no_npu_smi_find_free(
    tmp_path: Path, monkeypatch,
) -> None:
    def boom():
        raise AssertionError("_list_idle_chip_ids must not run when allow_find_free=False")

    monkeypatch.setattr(case_loader, "_list_idle_chip_ids", boom)

    subprocess_calls: list[list[str]] = []

    real_run = subprocess.run

    def tracing_run(cmd, *args, **kwargs):
        flat = [str(x) for x in cmd]
        subprocess_calls.append(flat)
        if any("find-free" in part for part in flat):
            raise AssertionError("npu-smi find-free must not be invoked")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(case_loader.subprocess, "run", tracing_run)

    case_file = tmp_path / "q.py"
    case_file.write_text(
        textwrap.dedent(
            """
            class Model:
                def forward(self, x):
                    return x

                def __call__(self, x):
                    return self.forward(x)

            class FakeTensor:
                shape = (2, 3)
                dtype = "float32"

            def get_inputs():
                return [FakeTensor()]

            def get_init_inputs():
                return []
            """
        ).strip(),
        encoding="utf-8",
    )

    case = case_loader.load_case(
        case_file,
        case_id="no_find_free",
        allow_find_free=False,
    )
    assert case.inputs
    assert case.outputs == []
    assert not any("find-free" in arg for call in subprocess_calls for arg in call)
