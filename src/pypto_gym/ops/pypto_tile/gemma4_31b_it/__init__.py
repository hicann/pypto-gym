#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
"""
Gemma-4-31B-it PyPTO fused kernel adapter.

Switch design: per-operator granularity for incremental validation.
"""

USE_PTO_SOFTMAX = False
USE_PTO_GQA = False

from .attn_softmax.attn_softmax_impl import attn_softmax_wrapper as attn_softmax
from .gqa_decode_attn.gqa_decode_attn_impl import gqa_decode_attn_wrapper as gqa_decode_attn
