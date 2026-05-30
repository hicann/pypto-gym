#!/usr/bin/env python3
# coding: utf-8
"""
MRoPE PyPTO算子模块
提供Multimodal Rotary Position Embedding实现

⚠️ 仅Batch≤4推荐启用（固化开销问题）
"""

from .mrope_impl import mrope_pto_correct, mrope_torch_fallback

__all__ = ["mrope_pto_correct", "mrope_torch_fallback"]