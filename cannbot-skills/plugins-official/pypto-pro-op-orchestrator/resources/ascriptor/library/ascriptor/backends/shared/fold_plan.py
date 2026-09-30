# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Which scalar temporaries a printer may fold into their single use, and how to parenthesise one.

A backend that prints expressions rather than one statement per IR op needs the same three answers:
which single-use pure scalar values can move to their use, whether the expression still means the same
there, and how to bracket it once inlined. The rules are the IR's, not any one target's, so they live
here; ``printed`` and ``auto`` let a caller say what its own text does with a value.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ...ir import Block, Function, Op, Value
from ...ir.types import ScalarType

FOLD_OPS_EXCLUDED = {"scalar.set", "scalar.store", "scalar.cell"}
CORE_QUERIES = {"core.cube_idx", "core.cube_num", "core.vec_idx", "core.vec_num", "core.sub_block_idx",
                "simt.thread_id", "simt.thread_num", "simt.blk_idx", "simt.blk_num"}
MEM_READS = {"scalar.load", "simt.load"}
PURE_DECLS = {"mem.slice", "mem.get_buf", "mem.reinterpret", "mem.reshape", "mem.view", "mem.alloc", "mem.workspace"}


def foldable_op(op: Op) -> bool:
    return (op.opcode.startswith("scalar.") and op.opcode not in FOLD_OPS_EXCLUDED) or op.opcode in CORE_QUERIES or op.opcode in MEM_READS


def pure_decl(op: Op) -> bool:
    return foldable_op(op) or op.opcode in PURE_DECLS


def refs(op: Op) -> set[str]:
    """Value names an op reads directly (operands and attribute values, not its regions)."""
    names = {v.name for v in op.operands if isinstance(v, Value)}
    names.update(v.name for v in op.attr_values())
    return names


def named_after_its_opcode(op: Op, r: Value) -> bool:
    """The frontend named the result after the opcode (``add``, ``mul.4``, ``ceil_div.1``): nothing named it."""
    return re.fullmatch(re.escape(op.opcode.split(".", 1)[1].replace(".", "_")) + r"(?:[._]\d+)?", r.name) is not None


def plan_folds(fn: Function, *, printed: Callable[[Op], bool] | None = None,
               auto: Callable[[Op, Value], bool] | None = None) -> dict[str, int | None]:
    """The scalar temporaries of ``fn`` a printer folds into their single use.

    A candidate is a scalar result of a pure scalar op (``scalar.*`` but set / store / cell, the core queries) used
    exactly once, by a later op of the same block — either the source never named it (``auto``) or the use is the
    very next op and a ``scalar.set`` (``x = x + 1``). A loop's bounds are never folded (they would be re-evaluated
    per iteration). Between the definition and the use nothing may change what the expression reads: no control
    flow, no ``scalar.set`` / ``scalar.cell``; an expression that reads memory (a ``scalar.load`` or one folded into
    it) allows only pure declarations in between, a plain scalar expression is transparent to DMAs, syncs and stores.

    ``printed(op)`` says whether an op produces a line of the caller's own text: a use by an op that prints nothing
    does not tie the value down. ``auto(op, result)`` decides "the source never named this"; the default reads the
    result's name, which is what CCE has always done.
    """
    printed = printed or (lambda op: True)
    auto = auto or named_after_its_opcode
    uses: dict[str, int] = {}
    for op in fn.body.walk():
        if not printed(op):
            continue
        for n in refs(op):
            uses[n] = uses.get(n, 0) + 1
    out: dict[str, int | None] = {}
    reads_mem: dict[str, bool] = {}

    def plan(block: Block) -> None:
        ops = block.ops
        for i, op in enumerate(ops):
            for region in op.regions:
                plan(region)
            if not foldable_op(op) or len(op.results) != 1:
                continue
            r = op.results[0]
            if not isinstance(r.type, ScalarType) or uses.get(r.name, 0) != 1:
                continue
            mem = op.opcode in MEM_READS or any(reads_mem.get(v.name, False) for v in op.operands if isinstance(v, Value))
            is_auto = auto(op, r)
            for j in range(i + 1, len(ops)):
                later = ops[j]
                if r.name in refs(later):
                    if not printed(later):
                        continue  # a line the caller does not print cannot be the use
                    if later.opcode != "cf.for" and (is_auto or (later.opcode == "scalar.set" and j == i + 1)):
                        out[r.name] = later.id
                        reads_mem[r.name] = mem
                    break
                if later.opcode.startswith("cf.") or later.opcode in ("scalar.set", "scalar.cell"):
                    break
                if mem and not pure_decl(later):
                    break
                if any(r.name in refs(x) for x in later.walk()):
                    break  # used inside a region of a later op: stays a local

    plan(fn.body)
    return out


def unparen(expr: str) -> str:
    """Drop one pair of outer parentheses when they enclose the whole expression."""
    if expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for i, ch in enumerate(expr):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(expr) - 1:
                return expr
        return expr[1:-1]
    return expr


def paren(expr: str) -> str:
    """Wrap a folded expression unless it is atomic (a name, a literal, one call or cast)."""
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == " " and depth == 0:
            return f"({expr})"
    return expr
