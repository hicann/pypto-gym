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
"""run_batch / run_one_case: device_mode=pool acquire-release and normal scheduling."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchmark import device_pool, run_kernelbench
from benchmark.case_loader import CaseSpec
from benchmark.run_kernelbench import _build_cfg, run_batch, run_one_case
from benchmark.verifier_runner import VerifierResult, VerifierStatus


def _stub_repo(tmp_path: Path) -> Path:
    root = tmp_path / "pypto_repo"
    root.mkdir()
    (root / ".opencode").mkdir()
    return root


class _FakeDevicePool:
    """Duck-typed pool for tests (same acquire/release contract as DevicePool)."""

    def __init__(self, devices: list[int]) -> None:
        self._q: asyncio.Queue[int] = asyncio.Queue()
        for d in devices:
            self._q.put_nowait(int(d))
        self.events: list[tuple[str, str, int]] = []

    async def acquire(self, *, case_id: str, phase: str) -> int:
        d = await self._q.get()
        self.events.append(("acquire", phase, d))
        return d

    def release(self, device_id: int, *, case_id: str, phase: str) -> None:
        self._q.put_nowait(int(device_id))
        self.events.append(("release", phase, device_id))


def test_pool_skip_pypto_gen_acquires_prepare_and_verifier_only(monkeypatch, tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    cfg_dict = {
        "device_mode": "pool",
        "output": {"root_dir": str(tmp_path / "run_root")},
        "pypto": {
            "skip_pypto_gen": True,
            "repo_root": str(repo),
            "workdir_root": "custom",
        },
    }
    cfg = _build_cfg(cfg_dict)
    cfg.arch_by_device = {0: "ascend910b4", 3: "ascend910b4"}

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

    def fake_load(p: Path, **_kwargs) -> CaseSpec:
        assert _kwargs.get("allow_find_free") is False
        assert str(_kwargs.get("output_probe_device_id")) in {"0", "3"}
        return fake_case

    async def fake_run_verifier(**_kwargs) -> VerifierResult:
        return VerifierResult(op_name="relu", status=VerifierStatus.SUCCESS, correctness=True)

    monkeypatch.setattr(run_kernelbench.case_loader, "load_case", fake_load)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_require", lambda *_a, **_k: None)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_task_desc", lambda *_a, **_k: None)
    monkeypatch.setattr(run_kernelbench, "run_verifier", fake_run_verifier)

    pool = _FakeDevicePool([0, 3])

    async def _run() -> None:
        sem = asyncio.Semaphore(2)
        await run_one_case(case_path, 0, cfg, sem, device_pool=pool)

    asyncio.run(_run())

    phases = [e[1] for e in pool.events if e[0] == "acquire"]
    assert phases == ["prepare", "verifier"]
    rel_phases = [e[1] for e in pool.events if e[0] == "release"]
    assert rel_phases == ["prepare", "verifier"]


def test_pool_load_case_error_still_releases_prepare(monkeypatch, tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    cfg = _build_cfg({
        "device_mode": "pool",
        "output": {"root_dir": str(tmp_path / "run_root")},
        "pypto": {"repo_root": str(repo), "workdir_root": "custom"},
    })

    def fake_load(_p: Path, **_kwargs) -> CaseSpec:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(run_kernelbench.case_loader, "load_case", fake_load)

    pool = _FakeDevicePool([1])

    case_path = tmp_path / "level1" / "x.py"
    case_path.parent.mkdir(parents=True)

    async def _run() -> None:
        sem = asyncio.Semaphore(2)
        with pytest.raises(RuntimeError, match="probe failed"):
            await run_one_case(
                case_path,
                0,
                cfg,
                sem,
                device_pool=pool,
            )

    asyncio.run(_run())
    assert pool.events == [("acquire", "prepare", 1), ("release", "prepare", 1)]


def test_pool_pypto_verifier_acquire_order(monkeypatch, tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    cfg = _build_cfg({
        "device_mode": "pool",
        "output": {"root_dir": str(tmp_path / "run_root")},
        "pypto": {"repo_root": str(repo), "workdir_root": "custom", "skip_pypto_gen": False},
    })
    cfg.arch_by_device = {1: "ascend910b4", 2: "ascend910b4", 5: "ascend910b4"}

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

    monkeypatch.setattr(run_kernelbench.case_loader, "load_case", lambda *_a, **_k: fake_case)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_require", lambda *_a, **_k: None)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_task_desc", lambda *_a, **_k: None)
    monkeypatch.setattr(
        run_kernelbench,
        "run_pypto_workflow",
        lambda **_k: run_kernelbench.PyptoRunResult(
            op_name="relu",
            status=run_kernelbench.PyptoRunStatus.SUCCESS,
            workdir=Path("."),
        ),
    )

    async def ok_verifier(**_kwargs) -> VerifierResult:
        return VerifierResult(op_name="relu", status=VerifierStatus.SUCCESS, correctness=True)

    monkeypatch.setattr(run_kernelbench, "run_verifier", ok_verifier)

    pool = _FakeDevicePool([1, 2, 5])

    async def _run() -> None:
        await run_one_case(case_path, 0, cfg, asyncio.Semaphore(2), device_pool=pool)

    asyncio.run(_run())

    assert [e for e in pool.events if e[0] == "acquire"] == [
        ("acquire", "prepare", 1),
        ("acquire", "pypto", 2),
        ("acquire", "verifier", 5),
    ]
    assert [e for e in pool.events if e[0] == "release"] == [
        ("release", "prepare", 1),
        ("release", "pypto", 2),
        ("release", "verifier", 5),
    ]


@pytest.mark.parametrize("content,expect_pypto_acquire", [("ok", True), ("skip", False)])
def test_pool_caplog_acquire_release_lines(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    content: str,
    expect_pypto_acquire: bool,
) -> None:
    """Pool mode logs case/phase/device for acquire and release (skip_pypto_gen skips pypto lease)."""
    import logging

    repo = _stub_repo(tmp_path)
    skip = content == "skip"
    cfg = _build_cfg({
        "device_mode": "pool",
        "output": {"root_dir": str(tmp_path / "run_root")},
        "pypto": {
            "repo_root": str(repo),
            "workdir_root": "custom",
            "skip_pypto_gen": skip,
        },
    })
    cfg.arch_by_device = {0: "ascend910b4"}

    case_path = tmp_path / "level1" / "21_Foo.py"
    case_path.parent.mkdir(parents=True)
    case_path.touch()
    fake_case = CaseSpec(
        op_name="foo",
        case_id="21_Foo",
        source_file=str(case_path),
        task_desc="# stub",
        level="level1",
    )
    monkeypatch.setattr(run_kernelbench.case_loader, "load_case", lambda *_a, **_k: fake_case)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_require", lambda *_a, **_k: None)
    monkeypatch.setattr(run_kernelbench.case_loader, "write_task_desc", lambda *_a, **_k: None)
    if not skip:
        monkeypatch.setattr(
            run_kernelbench,
            "run_pypto_workflow",
            lambda **_k: run_kernelbench.PyptoRunResult(
                op_name="foo",
                status=run_kernelbench.PyptoRunStatus.SUCCESS,
                workdir=Path("."),
            ),
        )
    monkeypatch.setattr(
        run_kernelbench,
        "run_verifier",
        lambda **_k: VerifierResult(op_name="foo", status=VerifierStatus.SUCCESS, correctness=True),
    )

    real_pool = device_pool.DevicePool([0], log=logging.getLogger("benchmark"))

    async def _run() -> None:
        await run_one_case(case_path, 0, cfg, asyncio.Semaphore(2), device_pool=real_pool)

    caplog.set_level(logging.INFO, logger="benchmark")
    asyncio.run(_run())

    text = caplog.text
    assert "pool acquire phase=prepare device=0" in text
    assert "pool release phase=prepare device=0" in text
    assert "pool acquire phase=verifier device=0" in text
    assert "pool release phase=verifier device=0" in text
    assert ("pool acquire phase=pypto device=0" in text) == expect_pypto_acquire


def test_normal_run_batch_round_robin_and_no_pool(monkeypatch, tmp_path: Path) -> None:
    stub = _stub_repo(tmp_path)
    cfg = _build_cfg({
        "output": {"root_dir": str(tmp_path / "run_root")},
        "pypto": {"repo_root": str(stub)},
    })
    assert cfg.device_mode == "normal"

    paths = [
        tmp_path / "level1" / "1_A.py",
        tmp_path / "level1" / "2_B.py",
        tmp_path / "level1" / "3_C.py",
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# t\n", encoding="utf-8")

    seen: list[tuple[str, int, object]] = []

    async def capture(
        case_path: Path,
        device_id: int,
        cfg_inner: object,
        semaphore: object,
        *,
        device_pool: object = None,
    ) -> object:
        seen.append((case_path.stem, device_id, device_pool))
        return run_kernelbench.CaseRunRecord(
            op_name=case_path.stem,
            case_id=case_path.stem,
            source_file=str(case_path),
            overall_status="success",
        )

    monkeypatch.setattr(run_kernelbench, "run_one_case", capture)

    async def _run():
        return await run_batch(paths, [0, 1], 2, cfg)

    asyncio.run(_run())
    assert seen == [
        ("1_A", 0, None),
        ("2_B", 1, None),
        ("3_C", 0, None),
    ]


def test_pool_run_batch_injects_device_pool(monkeypatch, tmp_path: Path) -> None:
    stub = _stub_repo(tmp_path)
    cfg = _build_cfg({
        "device_mode": "pool",
        "output": {"root_dir": str(tmp_path / "run_root")},
        "pypto": {"repo_root": str(stub)},
    })

    p = tmp_path / "level1" / "1_A.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#", encoding="utf-8")

    got: dict[str, object] = {}

    async def capture(
        case_path: Path,
        device_id: int,
        cfg_inner: object,
        semaphore: object,
        *,
        device_pool: object = None,
    ) -> object:
        got["pool"] = device_pool
        return run_kernelbench.CaseRunRecord(
            op_name="1_A",
            case_id="1_A",
            source_file=str(case_path),
            overall_status="success",
        )

    monkeypatch.setattr(run_kernelbench, "run_one_case", capture)
    asyncio.run(run_batch([p], [0], 1, cfg))
    assert isinstance(got["pool"], device_pool.DevicePool)
