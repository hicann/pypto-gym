# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KDA (Kimi Delta Attention) PyPTO kernel package.

Re-exports the chunk (prefill) wrapper that replaces the upstream
``fla.ops.kda.chunk_kda`` entry point in ``KimiDeltaAttention.forward``. See
``README.md`` for the algorithm, the subchunk=16 numerical-stability rationale,
and the dispatch toggles.

Also re-exports the torch.library-registered capturable entry
``kda_chunk_pypto`` (``torch.ops.pypto.kda_chunk_kimi``) used for
torch.compile / aclgraph; the eager default path still uses the wrapper.
"""
__all__ = ["kda_chunk_wrapper", "kda_chunk_pypto"]

from .kda_chunk_impl import kda_chunk_wrapper, kda_chunk_pypto
