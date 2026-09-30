# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Materialize otherwise unwritten register inputs for PyPTO's assignment API.

The model initializes fresh registers to zero. CCE permits a declaration with
no defining instruction, while PyPTO requires a producer before every use.
Only declarations whose value can be read before a definite write get a
zero broadcast. A normal load/compute-defined register incurs no extra op.

A ``vf.reinterpret`` reads nothing where it stands: the registry gives its
source no access, and the PyPTO printer materialises the view's ``bit_cast``
at the view's first use. What counts is that use, which reads the register
the view belongs to. Treating the declaration itself as a read seeded every
register whose view was declared ahead of its producer, and for an FP4
register that seed is a ``vbr`` broadcast the vendor compiler does not have
(A5-UP-046), so the whole kernel stopped compiling.
"""

from __future__ import annotations

from ..ir import REGISTRY, Module, Rewriter, Value
from ..ir.types import RegType
from .manager import Pass, PassContext

PASS = "pypto_register_init"


def _needed_for_function(function) -> set[int]:
    """Find declarations read before a definite write in one VF function."""
    needed: set[int] = set()
    declarations = {op.results[0].name: op for op in function.body.walk() if op.opcode == "vf.reg"}
    views = {op.results[0].name: op.operands[0].name for op in function.body.walk()
             if op.opcode == "vf.reinterpret" and isinstance(op.operands[0], Value)}

    def register(name):
        while name in views:
            name = views[name]
        return name

    def scan(block, incoming):
        written = set(incoming)
        for op in block.ops:
            if op.opcode in ("vf.reg", "vf.reinterpret"):
                continue
            spec = REGISTRY.find(op.opcode)
            writes = set()
            for index, value in enumerate(op.operands):
                if not isinstance(value, Value) or not isinstance(value.type, RegType):
                    continue
                access = spec.operands[index].access if spec and index < len(spec.operands) else "read"
                merging = str(op.attrs.get("merge", "")) == "merging"
                name = register(value.name)
                if access != "write" or merging:
                    if name in declarations and name not in written:
                        needed.add(declarations[name].id)
                if access in ("write", "readwrite"):
                    writes.add(name)
            if op.regions:
                ends = [scan(region, written) for region in op.regions]
                if op.opcode == "cf.if" and len(ends) == 2:
                    written |= ends[0] & ends[1]
                # A loop may have zero iterations, and a one-arm if may skip.
            written |= writes
        return written

    scan(function.body, {p.name for p in function.params})
    return needed


def run(module: Module, ctx: PassContext) -> Module:
    needed: set[int] = set()
    for function in module.functions:
        if function.kind == "vf":
            needed.update(_needed_for_function(function))
    rw = Rewriter(module, PASS)

    def rewrite(op):
        if op.opcode == "vf.reg" and op.id in needed:
            reg = op.results[0]
            note = "Materialize the model's zero seed for a register read before its first definite write"
            ctx.explain.note(note, op=op.id, register=reg.name)
            seed = rw.make("vf.dup", (reg, 0), from_ops=(op,), note=note)
            return [op, seed]
        return None

    return rw.rewrite(rewrite)


PASS_DEF = Pass(PASS, run, accepts="lowered/1", produces="lowered/1",
                doc="give PyPTO's implicitly declared VF registers a producer before their first read")
