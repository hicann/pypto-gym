# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 CTRL saturation bits (RFC-0007, D-233); values are raw bits, not logical SAT modes."""

SAT_BITS = {"float": 48, "float8": 50, "int": 53, "cast": 59, "global": 60}

# The interpreter's deterministic lane start, not a measured launch state: A5 CTRL bits have no fixed launch
# value (RFC-0007 §3, I013), so the interpreter warns when a kernel uses a bit it has not written.
SAT_DEFAULTS = {"float": False, "float8": False, "int": False, "cast": True, "global": True}

# Argument-free register conversions which discard high bits independently of CTRL.
TRUNCATING_CASTS = frozenset({("i64", "i32")})  # (source, destination)

# The ordinary integer destinations whose float or narrowing conversions take saturation from CTRL.
CTRL_CAST_DESTINATIONS = frozenset({"i8", "u8", "i16", "u16", "i32", "u32", "i64"})


def cast_uses_ctrl(src, dst) -> bool:
    """Whether bit 60 (and bit 59 while it holds 1) selects a `vf.cast`'s saturation (RFC-0007 §3)."""
    narrowing = src.kind in ("int", "uint", "bool") and dst.bits < src.bits
    return (dst.name in CTRL_CAST_DESTINATIONS and (src.name, dst.name) not in TRUNCATING_CASTS
            and (src.kind == "float" or narrowing))
