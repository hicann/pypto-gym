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
"""benchmark.verifier — KernelBench 桥接层内置 KernelVerifier.

支持的组合 (其它输入直接 ``ValueError``):
    - dsl       = ``"pypto"``
    - backend   = ``"ascend"``
    - framework = ``"torch"``
    - bench     = ``"kernelbench"``
    - worker    = 本子包提供的 ``LocalWorker``

公共入口:
    - ``KernelVerifier``        — ``run`` / ``run_profile``
    - ``register_local_worker`` — 注册并发现 LocalWorker
    - ``get_worker_manager``    — 获取全局 WorkerManager 单例
    - ``load_config``           — 返回桥接层默认配置 dict
"""

__all__ = [
    "KernelVerifier",
    "LocalWorker",
    "WorkerManager",
    "get_worker_manager",
    "load_config",
    "register_local_worker",
]

from .config import load_config
from .kernel_verifier import KernelVerifier
from .local_worker import LocalWorker
from .manager import (
    WorkerManager,
    get_worker_manager,
    register_local_worker,
)
