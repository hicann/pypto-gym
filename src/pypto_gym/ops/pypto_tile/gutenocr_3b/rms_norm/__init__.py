#!/usr/bin/env python3
# coding: utf-8
"""
RMSNorm PyPTO算子模块
提供BF16优化实现（单算子+60%，端到端固化开销抵消）

⚠️ 不推荐启用（固化开销抵消优化）
"""

from .rms_norm_impl import rms_norm_pto_native, rms_norm_bf16_fallback

__all__ = ["rms_norm_pto_native", "rms_norm_bf16_fallback"]