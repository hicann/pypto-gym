# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The A3 facade with the A2 tensor-vector vocabulary (RFC-0008, RFC-0012).

``from ascriptor.a3 import *`` binds kernels to the A3 profile. The shared A2
instruction forms are retained; A5 register VF and SIMT decorators are absent.
Importing a facade never changes process-wide device state.
"""

from .frontend.dsl import *  # noqa: F401,F403
from .frontend.dsl import __all__ as _dsl_all
from .frontend.dsl import make_decorators as _make_decorators
from .frontend.dsl_vec import *  # noqa: F401,F403
from .frontend.dsl_vec import __all__ as _vec_all

DEVICE = "a3"
kernel, _vf, _simt = _make_decorators(DEVICE)

__all__ = sorted({*_dsl_all, *_vec_all, "DEVICE", "kernel"} - {"vf", "simt"})
