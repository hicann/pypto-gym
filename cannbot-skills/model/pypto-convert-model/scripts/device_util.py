# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Auto-detect the best available accelerator (CUDA/ROCm/MPS/XPU/NPU) at runtime.

Used by the test/comparison stage so the PyTorch reference and TorchScript reload
run on the user's accelerator without hardcoding 'cuda'.
"""
from __future__ import annotations

import os
from importlib import import_module
from typing import Optional

import torch

_CACHED: tuple[torch.device, str] | None = None


def _has_npu() -> bool:
    if hasattr(torch, "npu"):
        try:
            return bool(torch.npu.is_available())  # type: ignore[attr-defined]
        except Exception:
            return False
    try:
        import_module("torch_npu")
        return bool(torch.npu.is_available())  # type: ignore[attr-defined]
    except Exception:
        return False


def _try_pick_npu() -> Optional[tuple[torch.device, str]]:
    if _has_npu():
        return (torch.device("npu"), "npu")
    return None


def _try_pick_cuda() -> Optional[tuple[torch.device, str]]:
    if torch.cuda.is_available():
        return (torch.device("cuda"), "cuda")
    return None


def _try_pick_xpu() -> Optional[tuple[torch.device, str]]:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return (torch.device("xpu"), "xpu")
    return None


def _try_pick_mps() -> Optional[tuple[torch.device, str]]:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return (torch.device("mps"), "mps")
    return None


def _try_pick_cpu() -> tuple[torch.device, str]:
    return (torch.device("cpu"), "cpu")


_DEVICE_PICKERS = [
    ("npu", _try_pick_npu),
    ("cuda", _try_pick_cuda),
    ("xpu", _try_pick_xpu),
    ("mps", _try_pick_mps),
    ("cpu", _try_pick_cpu),
]


def _pick_device_by_key(key: str) -> Optional[tuple[torch.device, str]]:
    for picker_key, picker_fn in _DEVICE_PICKERS:
        if picker_key == key:
            return picker_fn()
    return None


def _pick_device_auto() -> tuple[torch.device, str]:
    for _key, picker_fn in _DEVICE_PICKERS:
        result = picker_fn()
        if result is not None:
            return result
    return (torch.device("cpu"), "cpu")


def pick_device() -> tuple[torch.device, str]:
    """Return (device, label). Order: NPU > CUDA/ROCm > XPU(Intel) > MPS(Apple) > CPU.

    The environment variable CONVERT_EXP_DEVICE may force one of:
        npu, cuda, xpu, mps, cpu
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    forced = os.environ.get("CONVERT_EXP_DEVICE", "").strip().lower()

    if forced:
        result = _pick_device_by_key(forced)
        if result is None:
            supported = ", ".join(key for key, _ in _DEVICE_PICKERS)
            raise RuntimeError(
                f"CONVERT_EXP_DEVICE={forced!r} is unknown or unavailable; "
                f"choose an available device from: {supported}"
            )
    else:
        result = _pick_device_auto()

    _CACHED = result
    return _CACHED


def to_device(obj, device: torch.device):
    """Recursively move tensors / containers to a device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    return obj


def onnxruntime_providers() -> list:
    """Return ORT execution providers in priority order, filtered to what's installed."""
    ort = import_module("onnxruntime")
    available = set(ort.get_available_providers())
    preferred = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "CANNExecutionProvider",      # Ascend NPU
        "QNNExecutionProvider",       # Qualcomm NPU
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "OpenVINOExecutionProvider",
        "XnnpackExecutionProvider",
        "CPUExecutionProvider",
    ]
    chosen = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen
