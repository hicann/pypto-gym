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
"""Process-group registry for subprocesses spawned by benchmark."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
from typing import Set


_LOCK = threading.Lock()
_PROCS: Set[subprocess.Popen] = set()


def register(proc: subprocess.Popen) -> None:
    with _LOCK:
        _PROCS.add(proc)


def unregister(proc: subprocess.Popen) -> None:
    with _LOCK:
        _PROCS.discard(proc)


def terminate_process_group(proc: subprocess.Popen, grace_sec: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    def _killpg(sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except OSError:
            pass

    _killpg(signal.SIGTERM)
    try:
        proc.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    _killpg(signal.SIGKILL)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def cleanup_registered_process_groups(grace_sec: float = 10.0) -> None:
    with _LOCK:
        procs = list(_PROCS)
    for proc in procs:
        terminate_process_group(proc, grace_sec=grace_sec)
        unregister(proc)


atexit.register(cleanup_registered_process_groups)
