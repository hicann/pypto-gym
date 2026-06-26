#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""为 tests/ops/qwen3_6_27b 下的测试提供 npu_device fixture。

在模块顶层 import torch_npu，使得 collection 阶段（先于 fixture）
import kernel 文件触发 `@pypto.frontend.jit` 装饰器调用 `torch.npu.is_available()`
时 `torch.npu` 已经可用。
"""
import torch  # noqa: F401
import torch_npu  # noqa: F401
import pytest


@pytest.fixture
def npu_device(device):
    """`npu:N` device string with `torch.npu.set_device` already called."""
    return device
