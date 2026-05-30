"""为 tests/ops/qwen3_6_27b 下的测试提供 npu_device fixture。

在模块顶层 import torch_npu，使得 collection 阶段（先于 fixture）
import kernel 文件触发 `@pypto.frontend.jit` 装饰器调用 `torch.npu.is_available()`
时 `torch.npu` 已经可用。
"""
import os

import torch  # noqa: F401
import torch_npu  # noqa: F401
import pytest


@pytest.fixture
def npu_device(device):
    """`npu:N` device string with `torch.npu.set_device` already called."""
    torch.npu.set_device(device)
    os.environ["TILE_FWK_DEVICE_ID"] = str(device)
    return f"npu:{device}"
