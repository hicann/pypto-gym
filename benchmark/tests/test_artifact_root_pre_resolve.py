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

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmark import run_kernelbench
from benchmark.run_kernelbench import BENCHMARK_ARTIFACT_ROOT_ENV, _build_artifact_root, pre_resolve_state_dir


def test_build_artifact_root_respects_env_after_yaml_root(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed_run"
    out_yaml = {"root_dir": str(fixed)}
    p, name = _build_artifact_root(out_yaml)
    assert p == fixed.resolve()
    assert name == fixed.name


def test_build_artifact_root_env_skips_uuid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_root = tmp_path / "from_env"
    monkeypatch.setenv(BENCHMARK_ARTIFACT_ROOT_ENV, str(env_root))
    out_yaml: dict = {"base_dir": str(tmp_path / "ignored_base")}
    p, name = _build_artifact_root(out_yaml)
    assert p == env_root.resolve()
    assert name == env_root.name


def test_pre_resolve_injects_env_when_no_root_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BENCHMARK_ARTIFACT_ROOT_ENV, raising=False)
    cfg = tmp_path / "cfg.yaml"
    base = tmp_path / "runs"
    cfg.write_text(
        f"dry_run: true\n"
        f"cases: \"level1=1\"\n"
        f"output:\n"
        f"  base_dir: {base.as_posix()}\n",
        encoding="utf-8",
    )
    sd = pre_resolve_state_dir(cfg)
    assert sd.name == "state"
    injected = os.environ.get(BENCHMARK_ARTIFACT_ROOT_ENV, "")
    assert injected
    assert sd == Path(injected).resolve() / "state"
    artifact, _ = _build_artifact_root({"base_dir": str(base)})
    assert artifact.resolve() == Path(injected).resolve()
    # pre_resolve_state_dir 直接写 os.environ; monkeypatch 不会自动撤销, 需避免波及其他用例.
    monkeypatch.delenv(BENCHMARK_ARTIFACT_ROOT_ENV, raising=False)


def test_preflight_background_run_rejects_missing_repo_root(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    missing_repo = tmp_path / "not_a_pypto_checkout"
    cfg.write_text(
        f'cases: "level1=1"\npypto:\n  repo_root: "{missing_repo.as_posix()}"\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="不存在"):
        run_kernelbench.preflight_background_run(cfg)


def _minimal_pypto_stub(parent: Path) -> Path:
    stub = parent / "stub_pypto"
    (stub / ".opencode").mkdir(parents=True)
    return stub


def test_preflight_background_run_rejects_missing_kernelbench_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BENCHMARK_ARTIFACT_ROOT_ENV, raising=False)
    stub = _minimal_pypto_stub(tmp_path)
    runs = tmp_path / "runs"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f'cases: "level1=1"\n'
        f'pypto:\n  repo_root: "{stub.as_posix()}"\n'
        f'output:\n  base_dir: "{runs.as_posix()}"\n'
        f'verifier:\n  backend: cpu\n',
        encoding="utf-8",
    )
    pre_resolve_state_dir(cfg)
    try:
        with pytest.raises(SystemExit, match="level dir"):
            run_kernelbench.preflight_background_run(cfg)
    finally:
        monkeypatch.delenv(BENCHMARK_ARTIFACT_ROOT_ENV, raising=False)
