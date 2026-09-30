# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The ``sim`` backend: an interpreter for Lowered IR and the semantic reference (M4).

It executes the same lowered module every code-generating backend consumes — sides already
split, event ids assigned, addresses allocated — so it never re-derives what a pass decided.
The per-core process model, shared memory store, hazard checker, cycle model and trace export
are ported from the old simulator; the old bridge has no counterpart here.
"""
