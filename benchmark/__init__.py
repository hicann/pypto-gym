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
"""benchmark — KernelBench 桥接层.

本子包提供"输入 KernelBench 用例 → pypto 7 阶段 agent 工作流生成算子
→ 内置 KernelVerifier 精度验证 + 性能测试"的端到端批处理脚手架.
全部代码自包含在本目录, 无外部源码依赖 (除 pypto 本身与 opencode CLI).

公共入口:
    - case_loader.load_case / case_loader.write_require
    - pypto_runner.run_pypto_workflow
    - verifier_runner.run_verifier
    - python -m benchmark run --config configs/xxx.yaml (CLI)
    - python -m benchmark monitor <state_dir> (CLI)
"""

__all__ = [
    "case_loader",
    "pypto_runner",
    "verifier_runner",
    "report",
    "monitor",
]
