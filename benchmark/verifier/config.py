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
"""桥接层 verifier 默认配置 + ``load_config`` 入口.

返回纯 dict, 不读 yaml. 桥接层只会用到 ``log_dir`` / ``verify_timeout`` /
``pypto_run_mode`` / ``profile_settings`` 这一小撮字段, 调用方一般会再覆盖.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULTS: Dict[str, Any] = {
    # KernelVerifier 把每次 run 的工件写到 ``log_dir/<op>/Iteration<task>_Step<step>_verify``.
    # 默认运行后清理这些临时工件; 调试时设置 keep_artifacts=True 保留.
    "log_dir": "~/pypto_bench_logs",
    "keep_artifacts": False,

    # 单次 verify 子进程超时(秒). 默认 15 分钟, 避免较慢算子在 autotune / profile
    # 路径上过早超时. 桥接层仍可通过 CLI --timeout / verify_timeout 覆盖.
    "verify_timeout": 900,

    # PyPTO 运行模式: 0=NPU (默认), 1=CPU SIM. 桥接层只跑 NPU.
    "pypto_run_mode": 0,

    # Profile 默认参数. PyPTO 走 swimlane 路径, run_times 仅决定 base 端循环次数;
    # warmup_times 同理. swimlane 聚合本身不重复采样.
    "profile_settings": {
        "warmup_times": 5,
        "run_times": 50,
    },
}


def load_config(
    dsl: str = "pypto",
    config_path: Optional[str] = None,
    backend: Optional[str] = None,
    workflow: Optional[str] = "coder_only",
) -> Dict[str, Any]:
    """返回桥接层 verifier 的默认配置.

    Args:
        dsl: 必须为 ``"pypto"``; 其它值抛 ``ValueError``.
        config_path: 兼容签名, 当前忽略 (yaml 已删).
        backend: 必须为 ``"ascend"``; 其它值抛 ``ValueError``.
        workflow: 兼容签名, 当前忽略.

    Returns:
        dict: 包含 ``log_dir`` / ``verify_timeout`` / ``pypto_run_mode``
              / ``profile_settings`` / ``keep_artifacts`` 的浅拷贝. ``log_dir`` 已根据
              默认 ``~/pypto_bench_logs`` 展开为 ``<root>/Task_<rand>``.

    Raises:
        ValueError: dsl 不是 "pypto" 或 backend 不是 "ascend".
    """
    if dsl and dsl != "pypto":
        raise ValueError(
            f"verifier 仅支持 dsl='pypto', 实际: {dsl!r}."
        )
    if backend and backend != "ascend":
        raise ValueError(
            f"verifier 仅支持 backend='ascend', 实际: {backend!r}. "
            f"PyPTO 自身只支持 NPU."
        )

    config: Dict[str, Any] = {
        "log_dir": _DEFAULTS["log_dir"],
        "keep_artifacts": _DEFAULTS["keep_artifacts"],
        "verify_timeout": _DEFAULTS["verify_timeout"],
        "pypto_run_mode": _DEFAULTS["pypto_run_mode"],
        "profile_settings": dict(_DEFAULTS["profile_settings"]),
    }

    root_dir = os.path.expanduser(config["log_dir"])
    config["log_dir"] = str(Path(root_dir) / f"Task_{uuid.uuid4().hex}")
    return config
