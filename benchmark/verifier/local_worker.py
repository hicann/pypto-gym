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
"""单机 ascend 子进程执行器.

只做两件事:
    1. ``acquire_device`` / ``release_device`` 维护 NPU 卡的简单互斥
       (asyncio.Queue, 一卡一个 token).
    2. ``run_script`` 用 ``asyncio.create_subprocess_exec`` 跑一个 python 文件,
       带 timeout + 进程组 kill, 返回 ``(success, combined_log)``.

性能数据由 ``KernelVerifier`` 生成的 profile 脚本自己用 swimlane / device-time
算时间, 通过 stdout 上的 ``PROFILE_RESULT_GEN_US`` / ``BASE_TIME_US`` 标记行回传,
本 Worker 不参与解析.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import List, Tuple

logger = logging.getLogger(__name__)


class LocalWorker:
    """单机 NPU 子进程执行器.

    Attributes:
        device_ids: 注册时声明的 NPU 卡列表.
        backend:    必须为 ``"ascend"`` (其它 backend 在 manager 层就被拒).
    """

    def __init__(self, device_ids: List[int], backend: str = "ascend") -> None:
        if not device_ids:
            raise ValueError("LocalWorker 至少需要 1 个 device_id")
        if backend != "ascend":
            raise ValueError(
                f"LocalWorker 仅支持 backend='ascend', 实际: {backend!r}"
            )
        self.device_ids: List[int] = list(device_ids)
        self.backend: str = backend
        # 用 asyncio.Queue 做 token 池 — 一张卡 1 个 token, acquire/release 配对.
        # 不在 __init__ 里初始化 Queue (外面可能没 running loop), 而是 lazy-init.
        self._device_queue: "asyncio.Queue[int]" | None = None
        self._queue_init_lock = asyncio.Lock()

    @staticmethod
    async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            await asyncio.sleep(1)
            if process.returncode is None:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning("kill process group failed: %s", e)
            try:
                process.kill()
            except Exception:
                logger.debug("process.kill after killpg failure", exc_info=True)

    async def acquire_device(self) -> int:
        """阻塞直到有空闲 NPU 卡, 返回该 device_id."""
        q = await self._ensure_queue()
        device_id = await q.get()
        return device_id

    async def release_device(self, device_id: int) -> None:
        """归还 device_id 到池.

        若 device_id 不在 device_ids 里, 仅记录 warning, 不抛异常 (避免在
        ``finally`` 块里二次抛错掩盖原始异常).
        """
        if device_id not in self.device_ids:
            logger.warning(
                f"release_device({device_id}) 不在注册的 device_ids={self.device_ids} 中, 忽略"
            )
            return
        q = await self._ensure_queue()
        q.put_nowait(device_id)

    async def run_script(
        self,
        script_name: str,
        cwd: str,
        timeout: int,
        env_overrides: dict | None = None,
    ) -> Tuple[bool, str]:
        """子进程跑 ``python <script_name>``, 返回 (success, combined_stdout+stderr).

        - ``cwd`` 即脚本所在目录, 也作为子进程的工作目录 (因此脚本里 import
          同目录其它 .py 不需要 sys.path hack).
        - 超时时优先 SIGTERM 整个进程组, 1s 后还活着再 SIGKILL — 避免 PyPTO
          autotune 子线程残留占着 NPU.
        - PYTHONUNBUFFERED=1 强制无缓冲, 便于 verifier 即时看到进度.
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if env_overrides:
            env.update(env_overrides)

        preexec_fn = os.setsid if hasattr(os, "setsid") else None
        cmd = [sys.executable, script_name]
        logger.info(f"[{cwd}] running: {' '.join(cmd)} (timeout={timeout}s)")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=preexec_fn,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            await self._kill_process_group(process)
            return False, f"[TIMEOUT] script {script_name} exceeded {timeout}s"

        log = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
        success = process.returncode == 0
        return success, log

    async def _ensure_queue(self) -> "asyncio.Queue[int]":
        if self._device_queue is None:
            async with self._queue_init_lock:
                if self._device_queue is None:
                    q: asyncio.Queue[int] = asyncio.Queue()
                    for d in self.device_ids:
                        q.put_nowait(d)
                    self._device_queue = q
        return self._device_queue
