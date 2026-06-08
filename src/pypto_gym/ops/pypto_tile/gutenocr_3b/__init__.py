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
GutenOCR-3B PyPTO 融合算子库

实际集成的算子：
- SwiGLU MLP (融合版): gate_proj + up_proj + down_proj 三合一融合 kernel
- MRoPE (Multimodal Rotary Position Embedding): temporal + height + width 三维位置编码
- RMSNorm (BF16 优化版): pypto.rms_norm 融合实现

融合范围：
- SwiGLU MLP: gate_proj(x) + SiLU + up_proj(x) + element-wise mul + down_proj(hidden)
- MRoPE: concat(temporal, height, width) + cos/sin + rotation + concat output
- RMSNorm: mean(x^2) + rsqrt(mean+eps) + x * rsqrt * weight

推荐配置：
- SwiGLU MLP: 所有 batch 推荐启用 (唯一稳定有效算子, 端到端 +3%~+16%)
- MRoPE: 仅 Batch <= 4 推荐启用 (固化开销问题)
- RMSNorm: 不推荐启用 (固化开销抵消优化)
"""

USE_PTO_SWIGLU_MLP = False
USE_PTO_MROPE = False
USE_PTO_RMS_NORM = False

from .swiglu_mlp import swiglu_mlp_fused, swiglu_mlp_fused_static
from .mrope import mrope_pto_correct, mrope_torch_fallback
from .rms_norm import rms_norm_pto_native, rms_norm_bf16_fallback

__all__ = [
    'USE_PTO_SWIGLU_MLP',
    'USE_PTO_MROPE',
    'USE_PTO_RMS_NORM',
    'swiglu_mlp_fused',
    'swiglu_mlp_fused_static',
    'mrope_pto_correct',
    'mrope_torch_fallback',
    'rms_norm_pto_native',
    'rms_norm_bf16_fallback',
]
