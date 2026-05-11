#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO runner: device_mode=pool initial prompt contract."""

from __future__ import annotations

from pathlib import Path

from benchmark import pypto_runner
from benchmark.pypto_runner import render_prompt, run_pypto_workflow


def test_pool_prompt_includes_fixed_device_and_forbidden_rules() -> None:
    prompt = render_prompt(
        "Foo",
        "custom/Foo",
        device_mode="pool",
        pool_device_id=7,
    )
    assert "device_mode=pool" in prompt
    assert "**7**" in prompt or "设备 **7**" in prompt
    assert "TILE_FWK_DEVICE_ID=7" in prompt
    assert "find-free" in prompt
    assert "换卡" in prompt
    assert "不得" in prompt and "TILE_FWK_DEVICE_ID" in prompt


def test_normal_prompt_omits_pool_device_section() -> None:
    prompt = render_prompt("Foo", "custom/Foo", device_mode="normal")
    assert "device_mode=pool" not in prompt
    assert "find-free" not in prompt


def test_pool_mode_without_pool_device_id_skips_extra_section() -> None:
    prompt = render_prompt("Foo", "custom/Foo", device_mode="pool", pool_device_id=None)
    assert "find-free" not in prompt


def test_run_pypto_workflow_pool_writes_log_prompt_excerpt(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    op_dir = repo_root / "custom" / "Foo"
    op_dir.mkdir(parents=True)
    (op_dir / "SPEC.md").write_text("# spec\n", encoding="utf-8")
    opencode = tmp_path / "opencode"
    opencode.write_text("#!/bin/sh\n", encoding="utf-8")
    opencode.chmod(0o755)

    class FakePopen:
        stdout = None
        pid = 12345
        returncode = 0

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(pypto_runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        pypto_runner,
        "export_session_from_log",
        lambda *args, **kwargs: pypto_runner.OpencodeExportResult(status="skipped"),
    )

    log_path = tmp_path / "pypto_run.log"
    run_pypto_workflow(
        op_name="Foo",
        pypto_repo_root=repo_root,
        opencode_bin=str(opencode),
        log_file=log_path,
        incomplete_workflow_retry=0,
        skip_if_done=False,
        device_id=3,
        device_mode="pool",
        pref_round=4,
    )

    text = log_path.read_text(encoding="utf-8")
    assert "initial prompt 设备约束摘录" in text
    assert "TILE_FWK_DEVICE_ID=3" in text
    assert "find-free" in text
    assert "stage7 性能优化轮次严格控制在4轮" in text
