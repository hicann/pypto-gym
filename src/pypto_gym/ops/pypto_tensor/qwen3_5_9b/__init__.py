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
Qwen3.5-9B PyPTO 融合算子库

实际集成的算子：
- Gated Delta Rule (chunk 融合版): 融合 chunk 级别的 Gated Delta Rule kernel，用于 prefill 阶段替代 FLA/torch 原始实现

融合范围：
- chunk_gated_delta_rule 内部的 l2norm + pre_attn + inverse_pto + recurrent_state_attn_all 等子算子的融合

应用场景：
- Qwen3.5-9B 模型的 prefill 推理加速，通过 sys.modules 注入 qwen3_5_9b_pto_kernels 模块

注入方式：
- sys.modules["qwen3_5_9b_pto_kernels"] 提供 USE_PTO_GATED_DELTA_RULE 开关和 gated_delta_rule_wrapper 函数
"""

from .gated_delta_rule.gated_delta_rule_impl import gated_delta_rule_wrapper, gated_delta_rule_pypto

USE_PTO_GATED_DELTA_RULE = False

__all__ = [
    'USE_PTO_GATED_DELTA_RULE',
    'gated_delta_rule_wrapper',
    'gated_delta_rule_pypto',
]
