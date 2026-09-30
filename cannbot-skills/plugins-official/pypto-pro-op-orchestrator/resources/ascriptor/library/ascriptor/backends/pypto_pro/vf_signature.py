# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select the positional interface actually read by an emitted VF body."""

from __future__ import annotations

import ast


def used_parameters(params: list[str], body: list[str]) -> frozenset[int]:
    """Keep syntactic reads conservatively, including nested control flow.

    Tile writes read the destination's name too. Never deduplicate aliases or
    infer liveness from string matches, mutex IDs or physical addresses.
    """
    tree = ast.parse("\n".join(body))
    reads = {node.id for node in ast.walk(tree)
             if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return frozenset(i for i, name in enumerate(params) if name in reads)
