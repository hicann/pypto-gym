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
"""WorkerManager — 只支持 LocalWorker, 单进程内单例.

公共 API:
    - ``register_local_worker(device_ids, backend, arch)``
    - ``get_worker_manager()``
    - ``manager.has_worker(backend, arch)``
    - ``manager.select(backend, arch)``
    - ``manager.release(worker)``
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .local_worker import LocalWorker

logger = logging.getLogger(__name__)


@dataclass
class _WorkerSlot:
    worker: LocalWorker
    backend: str
    arch: str
    capacity: int = 1
    load: int = 0
    tags: Set[str] = field(default_factory=set)


class WorkerManager:
    """单机 worker 注册表 / 选路器.

    设计要点:
      - 仅 LocalWorker, 不做远端发现.
      - ``select`` 仅维护 load 计数做"最低负载优先"路由,
        实际 NPU 互斥由 ``LocalWorker.device_pool`` (asyncio.Semaphore) 兜底.
      - 不会主动卸载 worker; 同进程内重复注册同 backend/arch 会追加新 slot,
        调用方 (run_verifier 里的 ``has_worker`` 检测) 自行避免重复.
    """

    def __init__(self) -> None:
        self._slots: List[_WorkerSlot] = []
        self._lock = asyncio.Lock()

    async def register(
        self,
        worker: LocalWorker,
        backend: str,
        arch: str,
        tags: Optional[Set[str]] = None,
        capacity: int = 1,
    ) -> None:
        async with self._lock:
            slot = _WorkerSlot(
                worker=worker,
                backend=backend,
                arch=arch,
                capacity=max(1, capacity),
                tags=tags or set(),
            )
            self._slots.append(slot)
            logger.info(
                f"Registered LocalWorker: backend={backend}, arch={arch}, capacity={slot.capacity}"
            )

    async def has_worker(
        self,
        backend: str,
        arch: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        async with self._lock:
            for s in self._slots:
                if s.backend != backend:
                    continue
                if arch and s.arch != arch:
                    continue
                if tags and not tags.issubset(s.tags):
                    continue
                return True
            return False

    async def select(
        self,
        backend: str,
        arch: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> Optional[LocalWorker]:
        async with self._lock:
            cands = []
            for s in self._slots:
                if s.backend != backend:
                    continue
                if arch and s.arch != arch:
                    continue
                if tags and not tags.issubset(s.tags):
                    continue
                cands.append(s)
            if not cands:
                return None
            best = min(cands, key=lambda s: s.load / s.capacity)
            best.load += 1
            return best.worker

    async def release(self, worker: LocalWorker) -> None:
        async with self._lock:
            for s in self._slots:
                if s.worker is worker:
                    s.load = max(0, s.load - 1)
                    return
            logger.warning("Released unknown worker (not in registry)")


_GLOBAL_MANAGER: Optional[WorkerManager] = None


def get_worker_manager() -> WorkerManager:
    """返回进程级 WorkerManager 单例 (lazy 初始化)."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = WorkerManager()
    return _GLOBAL_MANAGER


async def register_local_worker(
    device_ids: List[int],
    backend: str,
    arch: str,
    tags: Optional[Set[str]] = None,
) -> None:
    """便捷函数: 创建 LocalWorker 并注册到全局 WorkerManager.

    Args:
        device_ids: NPU 设备号列表, 例如 ``[0]`` 或 ``[0, 1, 2, 3]``.
        backend: 必须为 ``"ascend"``.
        arch: 硬件架构, 例如 ``"ascend910b4"``.
        tags: 可选标签集合 (本精简版不消费, 但保留签名).

    Raises:
        ValueError: backend != "ascend" 或 device_ids 为空.
    """
    if backend != "ascend":
        raise ValueError(
            f"register_local_worker 仅支持 backend='ascend', 实际: {backend!r}"
        )
    if not device_ids:
        raise ValueError("register_local_worker 需要至少 1 个 device_id")

    worker = LocalWorker(device_ids=list(device_ids), backend=backend)
    manager = get_worker_manager()
    await manager.register(
        worker,
        backend=backend,
        arch=arch,
        tags=tags,
        capacity=len(device_ids),
    )
    logger.info(f"Registered LocalWorker: backend={backend}, arch={arch}, devices={device_ids}")
