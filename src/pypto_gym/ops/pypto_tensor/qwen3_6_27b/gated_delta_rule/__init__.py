# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""gated_delta_rule PyPTO kernel package (Qwen3.6-27B).

Re-exports the fused chunk gated-delta-rule wrapper that replaces
``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` in ``Qwen3_5GatedDeltaNet``
(prefill). See ``README.md`` for the algorithm and constraints.
"""
__all__ = ["gated_delta_rule_wrapper", "gated_delta_rule_pypto"]

from .gated_delta_rule_impl import gated_delta_rule_wrapper, gated_delta_rule_pypto
