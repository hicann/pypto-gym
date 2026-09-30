# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paired declarations and explicit participant/flag mappings."""
import hashlib
import json

from ...ir import Ident
from ...ir.types import ScalarType, dtype
from .synchronization import PIPES

SIDE_PIPES = {"cube": {"S", "M", "MTE1", "MTE2", "FIX"}, "vec": {"S", "V", "MTE2", "MTE3"}}


def declaration_index(owner):
    keys, sides = {}, {}
    if not owner.mixed:
        return keys, sides
    for graph in owner.graphs:
        variables = {}
        seen = set()
        for row in graph["metadata"]["declarations"]:
            var = owner.node(row["value"])
            expected = hashlib.sha256(json.dumps([row["location"], var["fields"]["name"]], sort_keys=True).encode()).hexdigest()
            owner.need(expected == row["key"] and expected not in seen, var, "Inconsistent or duplicate storage declaration key")
            seen.add(expected)
            variables[var["id"]] = expected
        for node in graph["nodes"]:
            if node["kind"] != "AssignStmt":
                continue
            var = owner.node(node["fields"]["var"])
            value = owner.node(node["fields"]["value"])
            if value["kind"] != "Call" or value["fields"]["name"] != "block.make_tile":
                continue
            owner.need(var["id"] in variables, node, "Missing storage declaration identity")
            key = variables[var["id"]]
            keys[value["id"]] = key
            sides.setdefault(key, set()).add(graph["side"])
    return keys, sides


def storage_name(owner, node, typ, addr, size, valid):
    if not owner.mixed:
        return None
    key = owner.declaration_keys.get(node["id"])
    owner.need(key is not None, node, "Unlinked mixed-side storage declaration")
    home = "vec" if typ.space == "ub" else "cube"
    owner.need(home in owner.declaration_sides[key], node, "Foreign descriptor lacks an owner-side declaration")
    signature = {"key": key, "name": "pro_storage_" + key, "type": str(typ),
                 "addr": addr, "size": size, "valid": list(valid), "sides": sorted(owner.declaration_sides[key])}
    previous = owner.storage_links.setdefault(key, signature)
    owner.need(previous == signature, node, "Paired storage declarations disagree in physical descriptor")
    return signature["name"]


def pipe(ctx, node, value):
    ctx.o.need(type(value) is int and 0 <= value < len(PIPES), node, "Invalid source pipe")
    result = PIPES[value]
    ctx.o.need(result in SIDE_PIPES[ctx.o.side], node, "Source pipe does not belong to the active side")
    return Ident(result)


def source_flag(ctx, node, args):
    o, name = ctx.o, node["fields"]["name"]
    if "cross_core" in name:
        attrs = ctx.attrs(node, {"pipe", "event_id", "sync_mode"})
        o.need(not name.endswith("_dyn") and not args, node, "Dynamic cross-core IDs require trace and protocol legalization")
        mode = attrs.get("sync_mode")
        o.need(type(mode) is int and mode in (0, 1, 2), node, "Only INTER_BLOCK, INTER_SUBBLOCK and INTRA_BLOCK flags are admitted")
        ident = attrs.get("event_id")
        o.need(type(ident) is int and 0 <= ident <= o.device.crosscore_id_max, node, "Cross-core flag ID is out of range")
        setting = name == "system.set_cross_core"
        if mode == 2:  # One AIC and both AIVs of its core.
            o.need(o.mixed, node, "Cross-core flags require both active source sides")
            action = ("cube_ready" if setting else "wait_vec") if o.side == "cube" else ("vec_ready" if setting else "wait_cube")
        elif mode == 0:  # Every core of the source side in the launch.
            action = ("allcube_" if o.side == "cube" else "allvec_") + ("ready" if setting else "wait")
        else:  # The AIVs of one core; AIV-only launches have one AIV per core.
            o.need(o.mixed and o.side == "vec", node, "Sub-block collectives are admitted only among AIVs of mixed launches")
            action = "intracore_allvec_" + ("ready" if setting else "wait")
        issue = pipe(ctx, node, attrs.get("pipe"))
        # These sets print as ffts_cross_core_sync; on an AIV, native and imported CCE builds (p7-dcci) both fail
        # the A5 compiler's check that its first parameter is 1 or 4..5 (V, MTE2, MTE3).
        o.need(not (setting and mode != 2 and o.side == "vec" and issue.name == "S"), node,
               "A5 compiles AIV cross-core sets of modes 0 and 1 only on PIPE_V, PIPE_MTE2 or PIPE_MTE3")
        ctx.emit("sync.crosscore." + action, node, attrs={"flag_id": ident, "pipe": issue})
    else:
        attrs = ctx.attrs(node, {"set_pipe", "wait_pipe", "event_id"})
        o.need(name in {"system.sync_src_dyn", "system.sync_dst_dyn"} and len(args) == 1, node,
               "Expected the pinned raw-flag expression form")
        ident = ctx.snapshot(args[0], node)
        opcode = "sync.set_flag" if name == "system.sync_src_dyn" else "sync.wait_flag"
        ctx.emit(opcode, node, attrs={"event_id": ident, "src": pipe(ctx, node, attrs.get("set_pipe")),
                                    "dst": pipe(ctx, node, attrs.get("wait_pipe"))})


# (module mode, side, Pro call) -> launch query. Pro's vector-side get_block_num() is the AIC count
# in mixed launches; AIV-only launches have one sub-block per core (GetVecNum() then equals it).
CORE_QUERIES = {("mix", "vec", "get_subblock_idx"): "core.sub_block_idx", ("mix", "vec", "get_block_idx"): "core.vec_idx",
                ("mix", "vec", "get_block_num"): "core.cube_num", ("mix", "cube", "get_block_idx"): "core.cube_idx",
                ("mix", "cube", "get_block_num"): "core.cube_num", ("cube", "cube", "get_block_idx"): "core.cube_idx",
                ("cube", "cube", "get_block_num"): "core.cube_num", ("vec", "vec", "get_block_idx"): "core.vec_idx",
                ("vec", "vec", "get_block_num"): "core.vec_num"}

FLAG_TARGETS = {"system.set_cross_core": ("sync.crosscore.cube_ready", "sync.crosscore.vec_ready", "sync.crosscore.allcube_ready",
                                         "sync.crosscore.allvec_ready", "sync.crosscore.intracore_allvec_ready"),
                "system.wait_cross_core": ("sync.crosscore.wait_vec", "sync.crosscore.wait_cube", "sync.crosscore.allcube_wait",
                                          "sync.crosscore.allvec_wait", "sync.crosscore.intracore_allvec_wait")}
TARGETS = {"get_block_idx": ("core.cube_idx", "core.vec_idx"), "get_block_num": ("core.cube_num", "core.vec_num"),
           "get_subblock_idx": ("core.sub_block_idx",),
           "get_subblock_num": ("scalar.const", "core.vec_num", "core.cube_num", "scalar.div")}


def launch_bounds(o, mode, opcode):
    """The exported block_dim B bounds the launch (RFC-0015): B blocks, and S AIVs per AIC in mixed launches."""
    blocks = o.document["abi"]["block_dim"]
    per_cube = o.device.vec_cores // o.device.cube_cores
    if opcode == "core.sub_block_idx":
        return 0, per_cube - 1
    vectors = blocks * per_cube if mode == "mix" and opcode == "core.vec_idx" else blocks
    return (1, blocks) if opcode.endswith("num") else (0, vectors - 1)


def core_query(ctx, node, args):
    ctx.attrs(node)
    o, name = ctx.o, node["fields"]["name"]
    mode = "mix" if o.mixed else o.side
    o.need(not args, node, "Core queries take no arguments")
    source_type = o.scalar_type(node["fields"]["type"])
    o.need(source_type.dtype.name in {"i32", "i64"}, node, "Unadmitted core-query result type")
    if name == "get_subblock_num":
        # Pro prints 1 on AIC and get_subblockdim() on AIV: one sub-block per AIV-only core, else GetVecNum()/GetCubeNum().
        if mode != "mix" or o.side == "cube":
            value, bounds = ctx.emit("scalar.const", node, typ=source_type, attrs={"value": 1}), (1, 1)
        else:
            typ = ScalarType(dtype("i32"))
            counts = ctx.emit("core.vec_num", node, typ=typ), ctx.emit("core.cube_num", node, typ=typ)
            value, bounds = ctx.emit("scalar.div", node, counts, typ=typ), (1, 2**31 - 1)
            value = ctx.emit("scalar.cast", node, (value,), typ=source_type) if source_type != typ else value
    else:
        opcode = CORE_QUERIES.get((mode, o.side, name))
        o.need(opcode is not None, node, "Core query needs an explicit participant/launch mapping")
        value = ctx.emit(opcode, node, typ=ScalarType(dtype("i32")))
        if source_type != value.type:
            value = ctx.emit("scalar.cast", node, (value,), typ=source_type)
        bounds = launch_bounds(o, mode, opcode)
    ctx.bounds[value.name] = bounds
    ctx.nonnegative.add(value.name)
    return value
