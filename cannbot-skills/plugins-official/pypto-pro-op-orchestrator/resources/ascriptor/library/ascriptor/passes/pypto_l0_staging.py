# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Give compact origin reloads and their MMAD readers one typed L0 view.

Allocation dimensions describe capacity. The DMA descriptor describes the
layout of the newly loaded data. Existing matrix slices keep their own pitch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..ir import REGISTRY, Block, Literal, Module, Rewriter, Value
from ..ir.scalar_range import ScalarRanges
from ..ir.types import CellType, MemType
from .manager import Pass, PassContext, PassError
from .util import Defs, Emit, function_names, view_of

PASS = "pypto_l0_staging"
SPACES = {"l0a", "l0b"}


@dataclass(frozen=True)
class Loaded:
    value: Value
    compact: bool


def _accesses(op, write=False):
    if op.opcode == "cf.call":
        return op.attrs.get("write" if write else "read", ())
    spec = REGISTRY.get(op.opcode)
    return [value for index, value in enumerate(op.operands)
            if spec.operands and spec.operands[min(index, len(spec.operands) - 1)].access
            in (("write", "readwrite") if write else ("read", "readwrite"))]


def run(module: Module, ctx: PassContext) -> Module:
    rw = Rewriter(module, PASS)

    def function(fn):
        if fn.kind != "func" or str(fn.attrs.get("side")) != "cube":
            return fn
        defs = Defs(replace(module, functions=(fn,)))
        names = function_names(fn)
        ranges = ScalarRanges(fn)

        def origin(value):
            definition = defs.op(value)
            return definition is not None and definition.opcode in ("mem.alloc", "mem.get_buf")

        def scalar_key(value):
            if isinstance(value, Literal):
                return value.value
            if isinstance(value, int) or value is None:
                return value
            if not isinstance(value, Value) or isinstance(value.type, CellType):
                return None
            definition = defs.op(value)
            if definition and definition.opcode in {
                    "scalar.add", "scalar.sub", "scalar.mul", "scalar.mod", "scalar.and", "scalar.cast"}:
                args = tuple(scalar_key(v) for v in definition.operands)
                if all(v is not None for v in args):
                    return definition.opcode, args, repr(sorted(definition.attrs.items()))
            return "ssa", value.name

        def location(value):
            if not isinstance(value, Value) or not isinstance(value.type, MemType) or value.type.space not in SPACES:
                return None
            view = view_of(value, defs)
            slot = scalar_key(view.slot)
            # A Cell cannot be re-evaluated to identify a captured slot. Only
            # the exact get_buf handle can establish identity in that case.
            if view.slot is not None and slot is None:
                current = value
                while (definition := defs.op(current)) is not None and definition.opcode != "mem.get_buf":
                    current = definition.operands[0]
                slot = "handle", current.name
            if isinstance(slot, int):
                slot %= view.root.type.slots
            return view.root.name, slot

        def geometry(op):
            if op.opcode != "dma.l1_to_l0" or "m_copy" in op.attrs:
                return None
            dst = op.operands[0]
            if not origin(dst) or dst.type.dtype.name not in ("f16", "bf16", "f32"):
                return None
            dims = op.attrs.get("m_dst"), op.attrs.get("n_dst")
            if not all(type(v) is int and v > 0 for v in (*dims, *dst.type.dims)):
                return None
            if op.attrs.get("src_is_transpose"):
                dims = dims[::-1]
            c0 = 256 // dst.type.dtype.bits
            shape = (-(-dims[0] // 16) * 16, -(-dims[1] // c0) * c0)
            if any(want > capacity for want, capacity in zip(shape, dst.type.dims, strict=True)):
                return None
            return shape

        roots = {location(op.operands[0])[0] for op in fn.walk()
                 if (shape := geometry(op)) is not None and shape != op.operands[0].type.dims}
        if not roots:
            return fn

        def written(block):
            return {place[0] for op in block.walk() for value in _accesses(op, write=True)
                    if (place := location(value)) is not None and place[0] in roots}

        def invalidate(state, changed):
            return {**state, **{root: {"unknown": None} for root in changed}}

        def fail(op, message):
            raise PassError(PASS, f"{message} at #{op.id} {op.loc}")

        def binding(value, op, state):
            place = location(value)
            if place is None or place[0] not in roots:
                return None
            root, slot = place
            entries = state.get(root, {})
            if slot in entries:
                loaded = entries[slot]
            elif any(v is None or v.compact for v in entries.values()):
                loaded = None
            else:
                return None
            if loaded is None:
                fail(op, f"cannot prove the reaching L0 reload layout for %{value.name}")
            if loaded.compact and not origin(value):
                fail(op, f"a view of reloaded L0 storage %{value.name} needs an explicit layout proof")
            return loaded.value if loaded.compact else None

        def walk(block, incoming):
            state, output = dict(incoming), []
            for op in block.ops:
                if op.regions:
                    changed = set().union(*(written(region) for region in op.regions))
                    initial = state if op.opcode == "cf.if" else invalidate(state, changed)
                    regions = tuple(walk(region, initial) for region in op.regions)
                    output.append(replace(op, regions=regions))
                    state = invalidate(state, changed)
                    continue
                if op.opcode == "dma.l1_to_l0" and (place := location(op.operands[0])) is not None and place[0] in roots:
                    root, slot = place
                    dst, shape = op.operands[0], geometry(op)
                    loaded = None
                    if shape is not None:
                        compact = shape != dst.type.dims
                        if compact:
                            value = Value(Emit(rw, fn, op, names).fresh(dst.name + "_loaded").name,
                                          replace(dst.type, dims=shape))
                            output.append(rw.make("mem.reinterpret", (dst,), results=(value,),
                                                  attrs={"tile": list(shape)}, from_ops=(op,),
                                                  note="typed layout of this compact origin reload"))
                            op = rw.rewritten(op, "load into its typed physical layout",
                                              operands=(value, *op.operands[1:]))
                            loaded = Loaded(value, True)
                            ctx.explain.note(f"%{dst.name} reload uses {shape}", op=op.id, kind="l0-staging")
                        else:
                            loaded = Loaded(dst, False)
                    # A dynamic selection may overwrite any earlier slot. Two
                    # distinct constant slots can retain independent layouts.
                    previous = state.get(root, {})
                    state[root] = {key: old if isinstance(slot, int) and isinstance(key, int) and slot != key else None
                                   for key, old in previous.items()}
                    state[root][slot] = loaded
                elif op.opcode == "cube.mmad":
                    operands = list(op.operands)
                    for index in (1, 2):
                        alias = binding(operands[index], op, state)
                        if alias is not None:
                            axes = ("M", "K") if index == 1 else ("N", "K")
                            for axis, capacity in zip(axes, alias.type.dims, strict=True):
                                extent = ranges.bounds(op.attrs.get(axis))
                                if extent is None or extent[0] < 0 or extent[1] > capacity:
                                    fail(op, f"MMAD {axis} has no proven bound within the compact L0 reload ({capacity})")
                            operands[index] = alias
                    if tuple(operands) != op.operands:
                        op = rw.rewritten(op, "consume the reaching compact L0 load", operands=tuple(operands))
                else:
                    for value in _accesses(op):
                        if binding(value, op, state) is not None:
                            fail(op, f"unsupported consumer of compact L0 reload: {op.opcode}")
                    changed = {place[0] for value in _accesses(op, write=True)
                               if (place := location(value)) is not None and place[0] in roots}
                    state = invalidate(state, changed)
                output.append(op)
            return Block(tuple(output))

        return replace(fn, body=walk(fn.body, {}))

    functions = tuple(function(fn) for fn in module.functions)
    return replace(module, attrs={**module.attrs, "next_id": rw._next_id}, functions=functions)


PASS_DEF = Pass(PASS, run, accepts="lowered/1", produces="lowered/1",
                doc="typed compact L0 reloads with conservative structured reaching-load tracking")
