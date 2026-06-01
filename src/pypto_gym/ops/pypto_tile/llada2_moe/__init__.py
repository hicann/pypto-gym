#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
"""
LLaDA2 MoE PyPTO fused kernel adapter.

Switch design: per-operator granularity for incremental validation.
"""

USE_PTO_EXPERT_FFN = False

from .llada2_expert_ffn_impl import llada2_expert_ffn as expert_ffn
from .llada2_moe_grouped_gemm_impl import llada2_moe_grouped_gemm as grouped_gemm
