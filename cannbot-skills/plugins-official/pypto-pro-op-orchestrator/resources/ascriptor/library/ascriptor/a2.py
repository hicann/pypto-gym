# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The a2 (Ascend 910B, c220) facade: ``from ascriptor.a2 import *`` (RFC-0008).

Importing binds names only; it never mutates process-wide state. Kernels decorated here are compiled for
device ``b3`` (the old repository's default). The vector core is programmed with the tensor-vector
instructions of :mod:`.frontend.dsl_vec` — ``cast`` / ``dup`` / ``gather`` / ``compare`` / ``select`` /
``muladddst`` are the UB instructions here, not the a5 register forms — and there is no ``@vf`` / ``@simt``.
"""

from .frontend.dsl import *  # noqa: F401,F403
from .frontend.dsl import __all__ as _dsl_all
from .frontend.dsl import make_decorators as _make_decorators
from .frontend.dsl_vec import *  # noqa: F401,F403
from .frontend.dsl_vec import __all__ as _vec_all

DEVICE = "b3"
kernel, _vf, _simt = _make_decorators(DEVICE)

__all__ = sorted({*_dsl_all, *_vec_all, "DEVICE", "kernel"} - {"vf", "simt"})
