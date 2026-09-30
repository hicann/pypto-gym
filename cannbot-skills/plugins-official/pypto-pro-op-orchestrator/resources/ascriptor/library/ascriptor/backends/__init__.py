# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Backends consume Lowered IR and produce artifacts.

Built-in backends (each lands with its milestone): ``sim`` (semantic reference, M4),
``cce`` (primary — one op, one intrinsic, M5), ``pto_isa`` (PTO's tile-level virtual ISA, M6),
``pypto_pro`` (wrapper-level, thin, M7). Third-party backends register through the
``ascriptor.backends`` entry-point group; see :mod:`ascriptor.backends.base`.
"""

from .base import Artifacts, Backend, Capabilities, ResourceLimits, discover, get

__all__ = ["Artifacts", "Backend", "Capabilities", "ResourceLimits", "discover", "get"]
