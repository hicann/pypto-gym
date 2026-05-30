#!/usr/bin/env python3
# coding: utf-8
"""
SwiGLU MLP PyPTO算子模块
提供融合算子实现（gate_proj + up_proj + down_proj融合）

✅ 所有batch推荐启用（唯一稳定有效算子）
"""

from .swiglu_mlp_impl import swiglu_mlp_fused, swiglu_mlp_fused_static

__all__ = ["swiglu_mlp_fused", "swiglu_mlp_fused_static"]