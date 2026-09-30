# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stride-1 block loads whose predicate may split a block (I036), kept out of emit.py for its size bound."""
from ...ir import Literal, Value
from ...ir.types import MaskType


def whole_blocks(printer, op) -> bool:
    """True when the predicate of a block load keeps whole 32-byte blocks active or inactive, where A5's block test
    and the IR's lanes agree: no mask, or a mask whose every write is a `vf.mask` pattern or a `vf.mask_update`
    count that folds to a multiple of the lanes per block."""
    from ...ir.ops._dsl import REGISTRY

    mask = op.attrs.get("mask")
    if not isinstance(mask, Value):
        return True
    if not isinstance(mask.type, MaskType):
        return False
    per = 256 // mask.type.width  # a lane holds width / 8 of a block's 32 predicate bits
    declared = False
    for o in printer.fn.body.walk():
        if any(r.name == mask.name for r in o.results):
            init = str(getattr(o.attrs.get("init", "all"), "name", o.attrs.get("init", "all")))
            vl = init[2:] if init.startswith("vl") and init[2:].isdigit() else None
            if o.opcode != "vf.mask" or not (init in ("all", "none", "h", "q") or vl and int(vl) % per == 0):
                return False
            declared = True
        spec = REGISTRY.find(o.opcode)
        access = [x.access for x in spec.operands] if spec is not None else []
        for i, x in enumerate(o.operands):
            if not isinstance(x, Value) or x.name != mask.name or i < len(access) and access[i] == "read":
                continue
            count = printer.env.fold(o.attrs.get("cnt")) if o.opcode == "vf.mask_update" else None
            if count is None or count < 0 or count % per:
                return False
    return declared


def lane_load(printer, op) -> bool:
    """Print a `vf.load` whose predicate may split a block and return True; False keeps the block copy.

    A5's vsldb reads a block whole when any of its lanes is active, at stride 1 too (I035, I036), and the IR reads
    lanes at stride 1 (RFC-0001). Stride 1 therefore prints the contiguous load and zeroes the other lanes, the
    vlds + vand pair the cce backend prints."""
    from .emit import PyptoGap  # emit.py imports this module

    if whole_blocks(printer, op):
        return False
    blk = op.attrs.get("blk_stride", 1)
    stride = blk.value if isinstance(blk, Literal) else blk
    if not isinstance(stride, int):
        raise PyptoGap(op, "vf.load with a runtime block stride and a predicate that may cover part of a block: the "
                           "IR reads lanes at stride 1 and whole blocks otherwise, and pypto prints one form", owner="ours")
    if stride != 1:
        return False
    dst, src = op.operands
    base, offset = printer.tile_ref(op, src, op.attrs.get("offset", 0))
    mask, reg = printer.mask_of(op, dst), printer.name(dst)
    printer.emit(f"{reg} = vf.load_align({base}, {offset})")
    printer.emit(f"{reg} = vf.and_({reg}, {reg}, {mask})")
    return True
