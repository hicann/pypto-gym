# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure-torch KDA goldens for the Kimi-Linear-48B-A3B kda_chunk kernel.

Two reference implementations of Kimi Delta Attention (per-channel log-gate
delta rule), both in FLA layout (q,k,g [B,T,H,K]; v [B,T,H,V]; beta [B,T,H];
state [B,H,V,K]):

* ``_naive_recurrent_kda`` — naive per-timestep recurrent loop (ground truth).
* ``vec_chunk_kda`` — vectorized chunked-torch (subchunk=16), the same blocked
  delta-rule algorithm the PyPTO chunk kernel fuses. Matches the naive op to
  <= 6e-5.
"""
__all__ = ["_CHUNK", "_naive_recurrent_kda", "vec_chunk_kda"]

import sys
from pathlib import Path

# The canonical implementations live in the transformers compat module; import
# them here to avoid duplicating the bodies (golden re-exports the same symbols).
_COMPAT_DIR = (Path(__file__).resolve().parents[3]
               / "src" / "pypto_gym" / "transformers" / "kimi_linear_48b_a3b")
sys.path.insert(0, str(_COMPAT_DIR))

from kimi_fla_compat import _CHUNK, _naive_recurrent_kda, vec_chunk_kda
