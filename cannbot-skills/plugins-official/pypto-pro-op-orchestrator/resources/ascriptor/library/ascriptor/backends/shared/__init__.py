# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Analysis shared between backends that need compile-time quantities out of the Lowered IR.

Both the ``pypto_pro`` and ``pto_isa`` backends target stacks whose tile shapes are compile-time
(a `pl` tile type, a ``pto::Tile`` template argument), so both must fold a kernel's scalar
arithmetic against one valuation. This package holds the parts that are backend-independent.
"""

from .fold_plan import paren, plan_folds, unparen
from .scalar_fold import ScalarFolder

__all__ = ["ScalarFolder", "paren", "plan_folds", "unparen"]
