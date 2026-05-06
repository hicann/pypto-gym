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
"""run_one_case: VerifierResult.failure_category → CaseRunRecord.failure_category."""

from __future__ import annotations

import asyncio
from pathlib import Path

from benchmark import run_kernelbench
from benchmark.case_loader import CaseSpec
from benchmark.run_kernelbench import _build_cfg, run_one_case
from benchmark.verifier_runner import VerifierResult, VerifierStatus


def test_run_one_case_record_failure_category_matches_verifier(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "pypto_repo"
    repo.mkdir()
    (repo / ".opencode").mkdir()
    cfg_dict = {
        "output": {"root_dir": str(tmp_path / "run_root")},
        "pypto": {
            "skip_pypto_gen": True,
            "repo_root": str(repo),
            "workdir_root": "custom",
        },
    }
    cfg = _build_cfg(cfg_dict)
    cfg.arch_by_device = {0: "ascend910b4"}

    case_path = tmp_path / "level1" / "19_ReLU.py"
    case_path.parent.mkdir(parents=True)
    case_path.touch()

    fake_case = CaseSpec(
        op_name="relu",
        case_id="19_ReLU",
        source_file=str(case_path),
        task_desc="# stub",
        level="level1",
    )

    def fake_load(p: Path, case_id=None) -> CaseSpec:
        return fake_case

    monkeypatch.setattr(run_kernelbench.case_loader, "load_case", fake_load)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_spec", lambda *_a, **_k: None)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_task_desc", lambda *_a, **_k: None)

    async def fake_run_verifier(**_kwargs):
        return VerifierResult(
            op_name="relu",
            status=VerifierStatus.FAILED,
            correctness=False,
            message="x",
            failure_category="semantic_cheat",
        )

    monkeypatch.setattr(run_kernelbench, "run_verifier", fake_run_verifier)

    async def _run():
        sem = asyncio.Semaphore(2)
        return await run_one_case(case_path, 0, cfg, sem)

    record = asyncio.run(_run())
    assert record.failure_category == "semantic_cheat"


def test_run_one_case_verifier_exception_sets_system_error(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "pypto_repo"
    repo.mkdir()
    (repo / ".opencode").mkdir()
    cfg_dict = {
        "output": {"root_dir": str(tmp_path / "run_root")},
        "pypto": {
            "skip_pypto_gen": True,
            "repo_root": str(repo),
            "workdir_root": "custom",
        },
    }
    cfg = _build_cfg(cfg_dict)
    cfg.arch_by_device = {0: "ascend910b4"}

    case_path = tmp_path / "level1" / "19_ReLU.py"
    case_path.parent.mkdir(parents=True)
    case_path.touch()

    fake_case = CaseSpec(
        op_name="relu",
        case_id="19_ReLU",
        source_file=str(case_path),
        task_desc="# stub",
        level="level1",
    )

    def fake_load(p: Path, case_id=None) -> CaseSpec:
        return fake_case

    monkeypatch.setattr(run_kernelbench.case_loader, "load_case", fake_load)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_spec", lambda *_a, **_k: None)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_task_desc", lambda *_a, **_k: None)

    async def fake_run_verifier(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_kernelbench, "run_verifier", fake_run_verifier)

    async def _run():
        sem = asyncio.Semaphore(2)
        return await run_one_case(case_path, 0, cfg, sem)

    record = asyncio.run(_run())
    assert record.failure_category == "system_error"
    assert record.verifier_status == VerifierStatus.ERROR.value
