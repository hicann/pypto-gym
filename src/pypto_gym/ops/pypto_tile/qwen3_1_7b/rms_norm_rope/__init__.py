#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.

"""
Qwen3-1.7B RoPE PyPTO Kernel - 实际集成版本

部分融合算子: Q/K per-head RMSNorm + RoPE

输入: [S, N, D] - q_proj/k_proj 输出
输出: [S, N, D] - 经过 RMSNorm + RoPE 的 Q/K
"""

from .rope_impl import qwen3_qk_rope_q, qwen3_qk_rope_k

__all__ = ['qwen3_qk_rope_q', 'qwen3_qk_rope_k']