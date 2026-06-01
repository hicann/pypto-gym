"""Auto-detect the best available accelerator (CUDA/ROCm/MPS/XPU/NPU) at runtime.

Used by the test/comparison stage so the PyTorch reference and TorchScript reload
run on the user's accelerator without hardcoding 'cuda'.
"""
from __future__ import annotations

import os
import torch


_CACHED: tuple[torch.device, str] | None = None


def _has_npu() -> bool:
    if hasattr(torch, "npu"):
        try:
            return bool(torch.npu.is_available())  # type: ignore[attr-defined]
        except Exception:
            return False
    try:
        import torch_npu  # noqa: F401
        return bool(torch.npu.is_available())  # type: ignore[attr-defined]
    except Exception:
        return False


def pick_device() -> tuple[torch.device, str]:
    """Return (device, label). Order: NPU > CUDA/ROCm > XPU(Intel) > MPS(Apple) > CPU.

    The environment variable CONVERT_EXP_DEVICE may force one of:
        npu, cuda, xpu, mps, cpu
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    forced = os.environ.get("CONVERT_EXP_DEVICE", "").strip().lower()

    if forced == "npu" or (not forced and _has_npu()):
        if _has_npu():
            _CACHED = (torch.device("npu"), "npu")
            return _CACHED

    if forced == "cuda" or (not forced and torch.cuda.is_available()):
        if torch.cuda.is_available():
            _CACHED = (torch.device("cuda"), "cuda")
            return _CACHED

    if forced == "xpu" or (not forced and hasattr(torch, "xpu") and torch.xpu.is_available()):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            _CACHED = (torch.device("xpu"), "xpu")
            return _CACHED

    if forced == "mps" or (
        not forced
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        if torch.backends.mps.is_available():
            _CACHED = (torch.device("mps"), "mps")
            return _CACHED

    _CACHED = (torch.device("cpu"), "cpu")
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
    import onnxruntime as ort
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
