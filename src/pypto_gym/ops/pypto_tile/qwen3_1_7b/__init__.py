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

"""
Qwen3-1.7B PyPTO 融合算子库 - 实际集成版本

实际集成的算子：
- RoPE (部分融合): q_norm + k_norm + RoPE

融合范围：
- 部分融合: q_proj/k_proj [B,S,N,D] -> [q_norm + k_norm + RoPE] -> Q/K [B,N,S,D]
- q_proj/k_proj/v_proj: 在 PyTorch 中完成
- q_norm/k_norm: 在部分融合 kernel 中完成（USE_PTO_ROPE=True）
"""

USE_PTO_ROPE = False

from .rope.rope_impl import qwen3_qk_rope_q, qwen3_qk_rope_k

__all__ = [
    'USE_PTO_ROPE',
    'qwen3_qk_rope_q',
    'qwen3_qk_rope_k',
]