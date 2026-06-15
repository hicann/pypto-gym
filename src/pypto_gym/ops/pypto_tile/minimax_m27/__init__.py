#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MiniMax M2.7 MoE PyPTO kernel adapter.

A self-contained grouped-GEMM kernel for MiniMax M2.7 (H=3072, I=1536, E=256, top-8
sigmoid routing). The algorithm follows llada2_moe's grouped GEMM (all experts in one
kernel call), but the kernel here is **its own implementation, 910B-adapted** — the
per-tile SwiGLU/cast vector width is capped to fit the 910B's 192 KB UB (PYPTO_VEC_TILE);
see minimax_m27_grouped_gemm_impl.py.

The modeling layer reads USE_PTO_GROUPED_GEMM to decide whether to route the MoE FFN
through the fused kernel (the single switch; only grouped GEMM is fused — routing/gate
stay on the host block). It is opt-in (default False, like llada2_moe / gemma4_31b_it);
the inference / benchmark entry scripts enable it.
"""

from .minimax_m27_grouped_gemm_impl import (
    minimax_m27_moe_grouped_gemm as grouped_gemm,
    convert_minimax_weights,
    MoeDims,
)

USE_PTO_GROUPED_GEMM = False     # opt-in; the runner / ask script enables it
