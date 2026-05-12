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
"""Tests for PyPTO workflow completion classification."""

from __future__ import annotations

import json
import threading

from benchmark import pypto_runner
from benchmark.pypto_runner import (
    PyptoRunResult,
    PyptoRunStatus,
    _attempt_log_file,
    _attempt_logs,
    _build_incomplete_retry_attempt,
    _incomplete_retry_gap_sec,
    _state_all_stages_completed,
    _state_has_failed_stage,
    _state_incomplete_without_failure,
    render_prompt,
    run_pypto_workflow,
)
from benchmark.verifier import script_builder


def test_state_stage_completion_helpers() -> None:
    assert _state_has_failed_stage({"stage_status": {"1": "completed", "2": "failed"}})
    assert not _state_all_stages_completed({"stage_status": {"1": "completed", "2": "failed"}})
    assert not _state_all_stages_completed({"stage_status": {"1": "completed", "2": "completed"}})
    assert _state_all_stages_completed(
        {"stage_status": {str(i): "completed" for i in range(1, 8)}}
    )
    assert _state_incomplete_without_failure(
        {"stage_status": {"1": "completed", "2": "in_progress"}}
    )
    assert not _state_incomplete_without_failure(
        {"stage_status": {"1": "completed", "2": "failed"}}
    )
    assert _state_incomplete_without_failure(None)


def test_attempt_log_file_suffix(tmp_path) -> None:
    log_file = tmp_path / "pypto_run.log"
    assert _attempt_log_file(log_file, 1) == log_file
    assert _attempt_log_file(log_file, 2) == tmp_path / "pypto_run.attempt2.log"
    assert _attempt_logs(log_file) == [log_file]
    assert _attempt_logs(None) == []


def test_incomplete_retry_gate_uses_session_tree_last_update() -> None:
    export = pypto_runner.OpencodeExportResult(
        session_id="ses_parent",
        session_updated_at_ms=1_000,
        tree_updated_at_ms=3_000,
    )

    assert _incomplete_retry_gap_sec(export, 3_603_000) == 3600.0
    attempt = _build_incomplete_retry_attempt(
        attempt_index=1,
        timed_out=True,
        returncode=-15,
        session_export=export,
        finished_at_ms=3_602_000,
        threshold_sec=3600,
        retry_allowed_by_count=True,
    )
    assert attempt["decision"] == "skip_gap_below_threshold"

    attempt = _build_incomplete_retry_attempt(
        attempt_index=1,
        timed_out=True,
        returncode=-15,
        session_export=export,
        finished_at_ms=3_603_000,
        threshold_sec=3600,
        retry_allowed_by_count=True,
    )
    assert attempt["decision"] == "retry"


def test_incomplete_retry_gate_skips_unknown_last_update() -> None:
    attempt = _build_incomplete_retry_attempt(
        attempt_index=1,
        timed_out=True,
        returncode=-15,
        session_export=pypto_runner.OpencodeExportResult(session_id="ses_parent"),
        finished_at_ms=3_603_000,
        threshold_sec=3600,
        retry_allowed_by_count=True,
    )
    assert attempt["decision"] == "skip_unknown_last_update"
    assert attempt["gap_sec"] is None


def test_incomplete_retry_gate_retries_nontimeout_abnormal_exit() -> None:
    export = pypto_runner.OpencodeExportResult(
        session_id="ses_parent",
        session_updated_at_ms=1_000,
        tree_updated_at_ms=3_000,
    )

    attempt = _build_incomplete_retry_attempt(
        attempt_index=1,
        timed_out=False,
        returncode=137,
        session_export=export,
        finished_at_ms=4_000,
        threshold_sec=3600,
        retry_allowed_by_count=True,
    )

    assert attempt["decision"] == "retry"
    assert attempt["trigger"] == "opencode_abnormal_exit"
    assert attempt["gap_sec"] == 1.0


def test_run_pypto_workflow_retries_nontimeout_abnormal_exit(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    op_dir = repo_root / "custom" / "Foo"
    op_dir.mkdir(parents=True)
    (op_dir / "REQUIRE.md").write_text("# require\n", encoding="utf-8")
    opencode = tmp_path / "opencode"
    opencode.write_text("#!/bin/sh\n", encoding="utf-8")
    opencode.chmod(0o755)

    attempts = {"count": 0}

    class FakePopen:
        stdout = None
        pid = 12345

        def __init__(self, *args, **kwargs):
            attempts["count"] += 1
            self.attempt = attempts["count"]
            self.returncode = 42 if self.attempt == 1 else 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.attempt == 1:
                state = {"stage_status": {"1": "completed", "2": "in_progress"}}
                (op_dir / ".orchestrator_state.json").write_text(
                    json.dumps(state),
                    encoding="utf-8",
                )
            else:
                for name in (
                    "SPEC.md",
                    "Foo_impl.py",
                    "Foo_golden.py",
                    "test_Foo.py",
                    "Foo_pypto_impl.py",
                ):
                    (op_dir / name).write_text("# generated\n", encoding="utf-8")
                state = {"stage_status": {str(i): "completed" for i in range(1, 8)}}
                (op_dir / ".orchestrator_state.json").write_text(
                    json.dumps(state),
                    encoding="utf-8",
                )
            return self.returncode

    def fake_export_session_from_log(*args, **kwargs):
        return pypto_runner.OpencodeExportResult(
            session_id=f"ses_{attempts['count']}",
            status="skipped",
        )

    monkeypatch.setattr(pypto_runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        pypto_runner,
        "export_session_from_log",
        fake_export_session_from_log,
    )

    result = run_pypto_workflow(
        op_name="Foo",
        pypto_repo_root=repo_root,
        opencode_bin=str(opencode),
        log_file=tmp_path / "pypto_run.log",
        incomplete_workflow_retry=1,
        incomplete_workflow_retry_min_gap_sec=3600,
        skip_if_done=False,
    )

    assert attempts["count"] == 2
    assert result.status == PyptoRunStatus.SUCCESS
    assert result.retry_count == 1
    assert result.incomplete_retry_attempts[0]["decision"] == "retry"
    assert result.incomplete_retry_attempts[0]["trigger"] == "opencode_abnormal_exit"
    assert "opencode 非 timeout 异常退出 code=42" in result.message


def test_run_result_serializes_retry_metadata(tmp_path) -> None:
    log_file = tmp_path / "pypto_run.attempt2.log"
    result = PyptoRunResult(
        op_name="Foo",
        status=PyptoRunStatus.SUCCESS,
        workdir=tmp_path / "Foo",
        log_file=log_file,
        attempt_log_files=[tmp_path / "pypto_run.log", log_file],
        retry_count=1,
        incomplete_retry_attempts=[{"decision": "retry"}],
    )
    data = result.to_dict()
    assert data["retry_count"] == 1
    assert data["incomplete_retry_attempts"] == [{"decision": "retry"}]
    assert data["attempt_log_files"] == [
        str(tmp_path / "pypto_run.log"),
        str(log_file),
    ]


def test_run_pypto_workflow_rejects_stale_spec_without_state(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    op_dir = repo_root / "custom" / "Foo"
    op_dir.mkdir(parents=True)
    (op_dir / "REQUIRE.md").write_text("# require\n", encoding="utf-8")
    (op_dir / "SPEC.md").write_text("# stale legacy spec\n", encoding="utf-8")
    opencode = tmp_path / "opencode"
    opencode.write_text("#!/bin/sh\n", encoding="utf-8")
    opencode.chmod(0o755)

    def fail_popen(*args, **kwargs):
        raise AssertionError("stale SPEC.md must fail before launching opencode")

    monkeypatch.setattr(pypto_runner.subprocess, "Popen", fail_popen)

    result = run_pypto_workflow(
        op_name="Foo",
        pypto_repo_root=repo_root,
        opencode_bin=str(opencode),
        incomplete_workflow_retry=0,
        skip_if_done=False,
    )

    assert result.status == PyptoRunStatus.ARTIFACT_MISSING
    assert "无状态工作目录中已存在 SPEC.md" in result.message
    assert "Stage 1 基于 REQUIRE.md 生成" in result.message


def test_run_pypto_workflow_interrupt_kills_without_export_or_retry(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    op_dir = repo_root / "custom" / "Foo"
    op_dir.mkdir(parents=True)
    (op_dir / "REQUIRE.md").write_text("# require\n", encoding="utf-8")
    opencode = tmp_path / "opencode"
    opencode.write_text("#!/bin/sh\n", encoding="utf-8")
    opencode.chmod(0o755)

    stop_event = threading.Event()
    killed = {"value": False}

    class FakePopen:
        stdout = None
        pid = 12345
        returncode = None

        def __init__(self, *args, **kwargs):
            stop_event.set()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

    def fake_kill_process_group(proc):
        killed["value"] = True
        proc.returncode = -15

    def fail_export(*args, **kwargs):
        raise AssertionError("interrupted workflow must not export opencode session")

    monkeypatch.setattr(pypto_runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(pypto_runner, "_kill_process_group", fake_kill_process_group)
    monkeypatch.setattr(pypto_runner, "export_session_from_log", fail_export)

    result = run_pypto_workflow(
        op_name="Foo",
        pypto_repo_root=repo_root,
        opencode_bin=str(opencode),
        log_file=tmp_path / "pypto_run.log",
        stop_event=stop_event,
        incomplete_workflow_retry=3,
        skip_if_done=False,
    )

    assert killed["value"]
    assert result.status == PyptoRunStatus.SUBPROCESS_ERROR
    assert "中断信号" in result.message
    assert result.retry_count == 0


def test_run_pypto_workflow_uses_noninteractive_permissions(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    op_dir = repo_root / "custom" / "Foo"
    op_dir.mkdir(parents=True)
    (op_dir / "REQUIRE.md").write_text("# require\n", encoding="utf-8")
    opencode = tmp_path / "opencode"
    opencode.write_text("#!/bin/sh\n", encoding="utf-8")
    opencode.chmod(0o755)
    seen = {}

    class FakePopen:
        stdout = None
        pid = 12345
        returncode = 0

        def __init__(self, cmd, *args, **kwargs):
            seen["cmd"] = cmd

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(pypto_runner.subprocess, "Popen", FakePopen)
    def fake_export_session_from_log(*args, **kwargs):
        seen["export_kwargs"] = kwargs
        return pypto_runner.OpencodeExportResult(status="skipped")

    monkeypatch.setattr(pypto_runner, "export_session_from_log", fake_export_session_from_log)

    result = run_pypto_workflow(
        op_name="Foo",
        pypto_repo_root=repo_root,
        opencode_bin=str(opencode),
        log_file=tmp_path / "pypto_run.log",
        incomplete_workflow_retry=0,
        skip_if_done=False,
    )

    assert "--dangerously-skip-permissions" in seen["cmd"]
    assert seen["export_kwargs"]["output_dir"] == tmp_path / "pypto_sessions" / "attempt_01"
    assert seen["export_kwargs"]["output_file"] == (
        tmp_path / "pypto_sessions" / "attempt_01" / "root_full.md"
    )
    assert result.status == PyptoRunStatus.ARTIFACT_MISSING


def test_run_pypto_workflow_registers_and_unregisters_process(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    op_dir = repo_root / "custom" / "Foo"
    op_dir.mkdir(parents=True)
    (op_dir / "REQUIRE.md").write_text("# require\n", encoding="utf-8")
    opencode = tmp_path / "opencode"
    opencode.write_text("#!/bin/sh\n", encoding="utf-8")
    opencode.chmod(0o755)
    events = []

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
    monkeypatch.setattr(pypto_runner, "register", lambda proc: events.append(("register", proc)))
    monkeypatch.setattr(pypto_runner, "unregister", lambda proc: events.append(("unregister", proc)))
    monkeypatch.setattr(
        pypto_runner,
        "export_session_from_log",
        lambda *args, **kwargs: pypto_runner.OpencodeExportResult(status="skipped"),
    )

    result = run_pypto_workflow(
        op_name="Foo",
        pypto_repo_root=repo_root,
        opencode_bin=str(opencode),
        log_file=tmp_path / "pypto_run.log",
        incomplete_workflow_retry=0,
        skip_if_done=False,
    )

    assert result.status == PyptoRunStatus.ARTIFACT_MISSING
    assert [name for name, _proc in events] == ["register", "unregister"]
    assert events[0][1] is events[1][1]


def test_modelnew_prompt_requires_nn_module_and_to() -> None:
    prompt = render_prompt(
        "Foo",
        "custom/level1/Foo",
        task_desc_rel="custom/level1/Foo/task_desc.py",
        init_args_repr="[]",
        model_init_source="def __init__(self): pass",
        forward_source="def forward(self, x): return x",
    )

    assert "ModelNew` 必须继承 `torch.nn.Module`" in prompt
    assert "assert isinstance(model, nn.Module)" in prompt
    assert "assert hasattr(model, \"to\")" in prompt
    assert "ModelNew.state_dict().keys()" in prompt
    assert "task_desc.Model.state_dict().keys()" in prompt
    assert "assert task_keys == impl_keys" in prompt
    assert "state_dict shape mismatch" in prompt
    assert "state_dict dtype mismatch" in prompt
    assert "self.gemm.weight" in prompt
    assert "torch.Size([])" in prompt
    assert "torch.Size([1])" in prompt
    assert "no_marker" in prompt
    assert "state_transition" in prompt
    assert "failed" in prompt
    assert "禁止清理 `/tmp/*`, `~/.cache/*`, `/home/*/.cache/*`" in prompt
    assert "不要请求 external_directory 权限" in prompt
    assert "stage7 性能优化轮次严格控制在3轮" in prompt
    assert "REQUIRE.md" in prompt
    assert "不是 PyPTO 标准工件" in prompt
    assert "SPEC.md (由 Stage 1 基于 REQUIRE.md 生成)" in prompt


def test_render_prompt_accepts_custom_pref_round() -> None:
    prompt = render_prompt("Foo", "custom/Foo", pref_round=5)

    assert "stage7 性能优化轮次严格控制在5轮" in prompt


def test_verifier_script_checks_state_dict_structure() -> None:
    script = script_builder.build_verify_script(
        op_name="Foo",
        framework_filename="Foo_torch.py",
        device_id=0,
    )

    assert "ModelNew state_dict keys must match task_desc.Model" in script
    assert "missing={missing}, unexpected={unexpected}" in script
    assert "[state_dict] shape mismatch" in script
    assert "[state_dict] dtype mismatch" in script
    assert "load_state_dict(src_state, strict=True)" in script
    assert "strict=False" not in script
    assert "falling back to manual copy" not in script
