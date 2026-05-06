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
"""CLI --repo-root validation for pypto_runner."""

from __future__ import annotations

import sys

import pytest

from benchmark import pypto_runner


def test_cli_rejects_nonexistent_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    missing = tmp_path / "nonexistent"
    monkeypatch.setattr(
        sys,
        "argv",
        ["pypto_runner", "SomeOp", "--repo-root", str(missing)],
    )
    assert pypto_runner._main_cli() == 2
    err = capsys.readouterr().err
    assert "不存在" in err or "不是目录" in err
    assert "download_pypto.sh" not in err


def test_cli_rejects_repo_without_opencode_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    fake = tmp_path / "fake_dir"
    fake.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["pypto_runner", "SomeOp", "--repo-root", str(fake)],
    )
    assert pypto_runner._main_cli() == 2
    err = capsys.readouterr().err
    assert ".opencode" in err
    assert "download_pypto.sh" not in err


def test_cli_default_repo_root_missing_shows_download_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    bench_dir = tmp_path / "benchmark"
    bench_dir.mkdir(parents=True)
    (bench_dir / "pypto_runner.py").write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(pypto_runner, "__file__", str(bench_dir / "pypto_runner.py"))
    monkeypatch.setattr(sys, "argv", ["pypto_runner", "SomeOp"])
    assert pypto_runner._main_cli() == 2
    err = capsys.readouterr().err
    assert "download_pypto.sh" in err


def test_cli_accepts_repo_with_opencode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    (repo / ".opencode").mkdir(parents=True)

    calls: list[tuple[str, Path]] = []

    def fake_run(*_a, pypto_repo_root: Path, **_kw):  # type: ignore[no-untyped-def]
        calls.append(("run", pypto_repo_root.resolve()))
        return pypto_runner.PyptoRunResult(
            op_name="Foo",
            status=pypto_runner.PyptoRunStatus.SKIPPED,
            workdir=pypto_repo_root / "custom" / "Foo",
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pypto_runner",
            "Foo",
            "--repo-root",
            str(repo),
        ],
    )
    monkeypatch.setattr(pypto_runner, "run_pypto_workflow", fake_run)

    assert pypto_runner._main_cli() == 0
    assert calls and calls[0][1] == repo.resolve()
