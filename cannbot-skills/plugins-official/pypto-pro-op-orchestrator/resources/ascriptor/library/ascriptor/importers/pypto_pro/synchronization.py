# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit source synchronization; never allocate IDs or plan locks."""

from ...ir import Block, Ident, Value
from ...ir.types import ScalarType, dtype

PIPES = ("MTE1", "MTE2", "MTE3", "M", "V", "S", "FIX")


def local_mutex(ctx, node, args):
    o = ctx.o
    attrs = ctx.attrs(node, {"pipe", "max_mutex_id", "mutex_ids", "mutex_id_owner_indices"})
    pipe = attrs.get("pipe")
    o.need(type(pipe) is int and 0 <= pipe < len(PIPES), node, "Invalid local mutex pipe")
    pipe = PIPES[pipe]
    allowed = {"cube": {"S", "M", "MTE1", "MTE2", "FIX"}, "vec": {"S", "V", "MTE2", "MTE3"}}
    o.need(pipe in allowed[o.side], node, "Local mutex pipe does not belong to the active side")
    count = attrs.get("max_mutex_id")
    o.need(args and type(count) is int and 1 <= count <= 32, node, "Invalid mutex candidate-count metadata")
    ids = [ctx.snapshot(arg, node) for arg in args]
    for ident in ids:
        o.need(type(ident) is int and 0 <= ident < 32 or isinstance(ident, Value)
               and isinstance(ident.type, ScalarType) and ident.type.dtype.name in {"i32", "i64"},
               node, "Local mutex ID requires an integer scalar in 0..31")
    candidates = attrs.get("mutex_ids")
    if candidates is not None:
        o.need(isinstance(candidates, list) and candidates and all(type(v) is int and 0 <= v < 32 for v in candidates)
               and len(set(candidates)) == len(candidates), node, "Invalid mutex candidate IDs")
        o.need(all(type(v) is not int or v in candidates for v in ids), node, "Mutex ID contradicts its candidates")
    owners = attrs.get("mutex_id_owner_indices", list(range(len(ids))))
    o.need(isinstance(owners, list) and len(owners) == len(ids)
           and all(type(owner) is int and owner >= 0 for owner in owners), node, "Invalid mutex owner metadata")
    # Native codegen deliberately omits comparisons within one owner. Admit
    # that omission only when it is independently justified by these values.
    for i, ident in enumerate(ids):
        for j in range(i):
            if owners[i] == owners[j]:
                o.need(type(ident) is int and type(ids[j]) is int and ident != ids[j], node,
                       "Same-owner mutex IDs require proof of distinct values")
    opcode = "sync.local_mutex_get" if node["fields"]["name"] == "system.mutex_lock_dyn" else "sync.local_mutex_release"
    for i, ident in enumerate(ids):
        condition = None
        for j in range(i):
            if owners[i] == owners[j]:
                continue
            unequal = ctx.emit("scalar.cmp", node, (ident, ids[j]), typ=ScalarType(dtype("b1")), attrs={"pred": Ident("ne")})
            condition = unequal if condition is None else ctx.emit("scalar.and", node, (condition, unequal), typ=ScalarType(dtype("b1")))
        target = ctx if condition is None else ctx.child()
        target.emit(opcode, node, attrs={"id": ident, "pipe": Ident(pipe), "side": Ident(o.side), "mode": 0})
        if condition is not None:
            ctx.emit("cf.if", node, (condition,), regions=(Block(tuple(target.ops)), Block(())))
    o.record(node, len(o.emitted))
    o.ledger[node["id"]]["source_attributes"] = attrs
