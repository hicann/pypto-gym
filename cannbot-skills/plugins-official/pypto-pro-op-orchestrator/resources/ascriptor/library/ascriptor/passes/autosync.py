# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``autosync``: the region-owned same-side synchronisation of one device family.

The pass itself only marks which operations an ``auto_sync`` region owns and hands them to the
planner the device family selects, of which there are exactly two and no option between them:

* family ``a5`` — physical-slot mutexes, marked here and lowered by :mod:`local_mutex`
  (RFC-0005, "A5 local buffer mutex policy").
* family ``a2`` — A2 and A3 — paired ``ready`` / ``valid`` slot sessions planned by
  :mod:`session_sync` (RFC-0005 §5).

The event planner these replaced is gone; ``check_balance`` (now :mod:`balance`) and the cross-side
checks (now :mod:`crosssync`) outlived it, because they judge hand-written protocols too.
"""

from __future__ import annotations

from dataclasses import replace

from ..ir import Block, Module, Op
from . import local_mutex
from .balance import check_balance  # re-exported: the public synchronisation diagnostic
from .manager import Pass, PassContext, PassError

__all__ = ["PASS_DEF", "check_balance", "run"]


def _inline_regions(block: Block) -> tuple[Block, set[int]]:
    owned: set[int] = set()

    def walk(current: Block, active: bool) -> Block:
        output: list[Op] = []
        for op in current.ops:
            if op.opcode == "region.autosync":
                output.extend(walk(op.regions[0], True).ops)
                continue
            if active and op.id is not None:
                owned.add(op.id)
            if op.regions:
                op = replace(op, regions=tuple(walk(region, active) for region in op.regions))
            output.append(op)
        return Block(tuple(output))

    return walk(block, False), owned


def run(module: Module, ctx: PassContext) -> Module:
    if ctx.device.family != "a5":
        from . import session_sync  # family a2: paired ready/valid slot sessions (RFC-0005 §5)

        return session_sync.run(module, ctx)
    if ctx.option("autosync_gm", False):
        raise PassError("autosync", "A5 slot mutexes cover on-chip allocations; use explicit events for GM dependencies")
    functions = []
    for fn in module.functions:
        if fn.kind not in ("kernel", "func"):
            functions.append(fn)
            continue
        body, owned = _inline_regions(fn.body)
        body = local_mutex.mark(module, body, owned)
        functions.append(replace(fn, body=body))
        ctx.explain.note(f"@{fn.name}: {len(owned)} region operations use A5 physical-slot mutex planning",
                         kind="mutex-policy", mode=0)
    # the marker local_mutex obeys: this route planned the module, so its slot mutexes are owed
    return Module(module.name, {**module.attrs, "a5_slot_mutex": True}, tuple(functions))


PASS_DEF = Pass("autosync", run, doc="same-side ownership of autosync regions: A5 slot mutexes, A2 family slot sessions",
                establishes=("3",))
