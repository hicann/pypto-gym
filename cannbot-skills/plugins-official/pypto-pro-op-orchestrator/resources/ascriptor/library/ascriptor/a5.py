# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The a5 (Ascend 950) facade: ``from ascriptor.a5 import *``.

Importing binds names only; it never mutates process-wide state. Kernels decorated here are
compiled for device ``950`` (RFC-0002 §2).
"""

from .composites.topk import radix_topk
from .frontend.dsl import *  # noqa: F401,F403
from .frontend.dsl import __all__ as _dsl_all
from .frontend.dsl import make_decorators as _make_decorators

DEVICE = "950"
kernel, vf, simt = _make_decorators(DEVICE)

__all__ = [*_dsl_all, "DEVICE", "kernel", "vf", "simt", "radix_topk"]
