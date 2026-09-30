# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""vf.mask_update counters, kept out of emit.py for its size bound."""
from ...ir import Value
from ...ir.types import CellType, MaskType


def mask_update(printer, op) -> None:
    """`plt_bN(count, POST_UPDATE)` decrements its counter in place. A cell is that counter (RFC-0001 §6.6); a
    parameter or other value is immutable (§5.2), so plt decrements a local copy of it (I040)."""
    from .emit import CceGap  # emit.py imports this module

    dst, cnt = op.operands[0], op.attrs.get("cnt")
    if not isinstance(cnt, Value):
        raise CceGap(op, "mask_update needs a scalar counter, not a literal (plt decrements it in place)")
    t = dst.type
    if not isinstance(t, MaskType):
        raise CceGap(op, "mask_update destination must be a mask register")
    fn = "plt_2xvl_b64" if t.width == 64 and t.n == 1 else f"plt_b{t.width // t.n}"
    count = f"(uint32_t&){printer.name(cnt)}"
    if not isinstance(cnt.type, CellType):
        count = printer.name(Value(f"{cnt.name}_count{op.id}", cnt.type))
        printer.emit(f"uint32_t {count} = (uint32_t){printer.name(cnt)};", op)
    printer.emit(f"{printer.name(dst)} = {fn}({count}, POST_UPDATE);", op)
