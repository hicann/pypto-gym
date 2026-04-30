#!/usr/bin/env python3
# coding: utf-8
"""
RMSNorm wrapper - 适配层

调用说明：
  modeling_qwen3.py → sys.modules.get("pto_kernels") → rms_norm_wrapper

替换示例：
  self.weight * hidden_states * torch.rsqrt(variance + eps)  →  rms_norm_wrapper(hidden_states, self.weight, eps)
"""

import torch
import sys


def _get_pto_kernels():
    return sys.modules.get("pto_kernels")


def _pto_available():
    pto = _get_pto_kernels()
    return pto is not None and getattr(pto, "USE_PTO_RMS_NORM", False)


PTO_AVAILABLE = None


def rms_norm_wrapper(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not (_pto_available() or PTO_AVAILABLE):
        from .rms_norm_golden import rms_norm_golden as rms_norm_func
        return rms_norm_func(hidden_states, weight, eps)

    if hidden_states.dim() == 4:
        from .rms_norm_golden import rms_norm_golden as rms_norm_func
        return rms_norm_func(hidden_states, weight, eps)

    if hidden_states.dim() != 3:
        from .rms_norm_golden import rms_norm_golden as rms_norm_func
        return rms_norm_func(hidden_states, weight, eps)

    from .rms_norm_impl import rms_norm_impl as rms_norm_func
    return rms_norm_func(hidden_states, weight, eps)


__all__ = ["rms_norm_wrapper", "PTO_AVAILABLE"]