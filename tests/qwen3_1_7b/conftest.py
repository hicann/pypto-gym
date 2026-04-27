#!/usr/bin/env python3
# coding: utf-8
"""为 tests/qwen3_1_7b 下的测试注入源算子目录到 sys.path，并提供 npu_device fixture。

测试文件采用 sibling import（如 `from qwen3_decode_attn import ...`），
需要把对应的源算子目录加入 sys.path 才能解析。
"""
import os
import sys
from pathlib import Path

import pytest

_SRC_OP_DIR = (
    Path(__file__).resolve().parents[2]
    / "src" / "pypto_gym" / "ops" / "qwen3_1_7b"
)
if str(_SRC_OP_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_OP_DIR))


@pytest.fixture
def npu_device(device):
    """`npu:N` device string with `torch.npu.set_device` already called."""
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(device)
    os.environ["TILE_FWK_DEVICE_ID"] = str(device)
    return f"npu:{device}"
