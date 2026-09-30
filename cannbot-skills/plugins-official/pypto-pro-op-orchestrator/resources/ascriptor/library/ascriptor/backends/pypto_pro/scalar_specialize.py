# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare an export-local, typed scalar graph before native division expansion."""

from ...passes import PassManager
from ...passes.scalar_simplify import PASS_DEF


def specialize(printer):
    count = printer.block_dim
    facts = {}
    if count:
        vec_count = count if printer.mode == "vec" else count * 2
        sub_count = 1 if printer.mode == "vec" else 2
        facts = {"core.cube_idx": (0, count - 1), "core.vec_idx": (0, vec_count - 1),
                 "core.sub_block_idx": (0, sub_count - 1), "core.cube_num": (count, count),
                 "core.vec_num": (vec_count, vec_count)}
    return PassManager((PASS_DEF,), options={"scalar_bindings": printer.bindings,
                                           "scalar_core_ranges": facts}).run(printer.module)
