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
"""Device pool helpers: normalization and NPU softmax preflight (PyPTO runtime smoke test).

Each configured device runs in an isolated subprocess with ``TILE_FWK_DEVICE_ID`` set,
exercising ``torch`` + ``torch_npu`` + ``pypto`` import and a tiny ``softmax`` on NPU.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union


@dataclass(frozen=True)
class DeviceQuickFailure:
    """One device failed pool preflight; ``error_summary`` is a short stderr/exit reason."""

    device_id: str
    error_summary: str


_SOFTMAX_PROBE_TEMPLATE = r"""
import os, json, sys
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
try:
    import torch
    import torch_npu  # noqa: F401
    import pypto  # noqa: F401
    torch.npu.set_device(device_id)
    x = torch.randn(32, 128, device="npu:%d" % device_id, dtype=torch.float32)
    y = torch.nn.functional.softmax(x, dim=-1)
    assert y.shape == x.shape
    torch.npu.synchronize()
except Exception as e:
    msg = "%s: %s" % (type(e).__name__, str(e))
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(1)
print(json.dumps({"ok": True, "error": ""}))
"""


def normalize_pool_devices(device_ids: Sequence[Union[str, int]]) -> List[str]:
    """Deduplicate pool devices (stable order), strip, require non-empty list."""
    if not device_ids:
        raise ValueError("pool devices 列表不能为空")
    seen: set[str] = set()
    out: List[str] = []
    for raw in device_ids:
        key = str(raw).strip()
        if not key:
            raise ValueError("pool device id 不能为空字符串")
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def run_softmax_quick_test_subprocess(
    device_id: Union[str, int],
    *,
    timeout_sec: int = 120,
    python_executable: str = sys.executable,
) -> DeviceQuickFailure | None:
    """Run softmax quick test in a child process; return ``DeviceQuickFailure`` or ``None`` if ok."""
    did = str(device_id).strip()
    env = os.environ.copy()
    env["TILE_FWK_DEVICE_ID"] = did
    try:
        proc = subprocess.run(
            [python_executable, "-c", _SOFTMAX_PROBE_TEMPLATE],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return DeviceQuickFailure(
            device_id=did,
            error_summary=f"timeout after {timeout_sec}s",
        )
    summary = ""
    if proc.stdout.strip():
        try:
            line = proc.stdout.strip().splitlines()[-1]
            payload = json.loads(line)
            if payload.get("ok"):
                return None
            summary = str(payload.get("error", "")).strip()
        except (ValueError, json.JSONDecodeError):
            summary = proc.stdout.strip()[:500]
    if not summary:
        tail_err = (proc.stderr or "").strip()[:500]
        tail_out = (proc.stdout or "").strip()[:500]
        summary = tail_err or tail_out or f"exit code {proc.returncode}"
    return DeviceQuickFailure(device_id=did, error_summary=summary)


def collect_softmax_preflight_failures(
    device_ids: Iterable[Union[str, int]],
    *,
    timeout_sec: int = 120,
    python_executable: str = sys.executable,
) -> List[DeviceQuickFailure]:
    """Normalize ``device_ids`` and run one quick test subprocess per device; return failures only."""
    normalized = normalize_pool_devices(list(device_ids))
    failures: List[DeviceQuickFailure] = []
    for did in normalized:
        err = run_softmax_quick_test_subprocess(
            did, timeout_sec=timeout_sec, python_executable=python_executable
        )
        if err is not None:
            failures.append(err)
    return failures


def format_preflight_failure_report(failures: Sequence[DeviceQuickFailure]) -> str:
    lines = ["pool 设备预检 (softmax quick test) 失败:"]
    for f in failures:
        lines.append(f"  - device_id={f.device_id!r}: {f.error_summary}")
    return "\n".join(lines)


class DevicePool:
    """Async FIFO pool of device ids for phase-scoped acquire/release (benchmark kernel runs)."""

    def __init__(self, device_ids: Sequence[Union[str, int]], *, log: Optional[logging.Logger] = None) -> None:
        normalized = normalize_pool_devices(list(device_ids))
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        for d in normalized:
            self._queue.put_nowait(int(d))
        self._log = log

    async def acquire(self, *, case_id: str, phase: str) -> int:
        device_id = await self._queue.get()
        if self._log is not None:
            self._log.info(
                "[%s] pool acquire phase=%s device=%s",
                case_id,
                phase,
                device_id,
            )
        return device_id

    def release(self, device_id: int, *, case_id: str, phase: str) -> None:
        self._queue.put_nowait(int(device_id))
        if self._log is not None:
            self._log.info(
                "[%s] pool release phase=%s device=%s",
                case_id,
                phase,
                device_id,
            )
