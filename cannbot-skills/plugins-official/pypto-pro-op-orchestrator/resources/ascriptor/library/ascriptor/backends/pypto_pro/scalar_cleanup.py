# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Liveness and adjacent-use cleanup of typed kernel and VF emission records.

Scalar assignments must be tagged to be removed or inlined. Python AST supplies
executable uses and scope, allowing empty pure controls to be removed as well.
Tile/memory/sync statements remain observable.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict

from ...ir import Literal, Value
from ...ir.scalar_math import limits
from ...ir.types import CellType, ScalarType
from ...passes.scalar_simplify import PURE


class ScalarLine(str):
    def __new__(cls, text, op):
        result = super().__new__(cls, text)
        result.op = op
        return result


def eligible(op):
    division = False
    if op.opcode in {"scalar.div", "scalar.mod"} and len(op.operands) == 2 and op.results:
        divisor = op.operands[1]
        typ = op.results[0].type
        division = (isinstance(divisor, Literal) and type(divisor.value) is int
                    and isinstance(typ, ScalarType) and typ.dtype.is_integer
                    and 0 < divisor.value <= limits(typ.dtype)[1])
    if op.opcode not in PURE | {"scalar.cell", "scalar.set"} and not division:
        return False
    value = op.operands[0] if op.opcode == "scalar.set" else op.results[0] if op.results else None
    def integer(operand):
        if isinstance(operand, Literal):
            operand = operand.value
        return isinstance(operand, (int, bool)) or (
            isinstance(operand, Value) and isinstance(operand.type, (ScalarType, CellType))
            and operand.type.dtype.is_integer)
    return (value is not None and isinstance(value.type, (ScalarType, CellType))
            and value.type.dtype.is_integer and all(integer(x) for x in op.operands)
            and (op.opcode != "scalar.cell" or integer(op.attrs.get("init", 0))))


def _reads(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _pure_test(node):
    for child in ast.walk(node):
        if isinstance(child, (ast.Subscript, ast.NamedExpr, ast.Await, ast.Yield, ast.Lambda)):
            return False
        if isinstance(child, ast.Call) and not (
            isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "pl" and child.func.attr in {"range", "min", "max"}
        ):
            return False
    return True


def cleanup(lines, *, enabled=True):
    report = {"enabled": enabled, "removed": [], "inlined": [], "empty_controls": 0}
    if not enabled:
        return list(map(str, lines)), report
    text = "\n".join(lines) + "\n"
    tree = ast.parse(text)
    records, cursor = {}, 1
    for line in lines:
        if isinstance(line, ScalarLine):
            records[cursor] = line
        cursor += line.count("\n") + 1
    sites = {node.lineno: node for node in ast.walk(tree)
             if isinstance(node, ast.Assign) and node.lineno in records
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    # Never infer purity for untagged generated assignments or calls.
    definitions = defaultdict(list)
    for line, node in sites.items():
        definitions[node.targets[0].id].append(line)
    removed, empty = set(), set()

    def roots(block):
        result = set()
        for node in block:
            if node.lineno in sites or node.lineno in empty:
                continue
            if isinstance(node, ast.If):
                result.update(_reads(node.test))
                result.update(roots(node.body) | roots(node.orelse))
            elif isinstance(node, ast.For):
                result.update(_reads(node.iter))
                result.update(roots(node.body) | roots(node.orelse))
            else:
                result.update(_reads(node))
        return result

    def prune_controls(block):
        changed = False
        meaningful = False
        for node in block:
            if node.lineno in removed or node.lineno in empty or isinstance(node, ast.Pass):
                continue
            if isinstance(node, (ast.If, ast.For)):
                body, first = prune_controls(node.body)
                other, second = prune_controls(node.orelse)
                changed |= first or second
                test = node.test if isinstance(node, ast.If) else node.iter
                if not body and not other and _pure_test(test):
                    empty.add(node.lineno)
                    changed = True
                    continue
            meaningful = True
        return meaningful, changed

    while True:
        live, pending = set(), list(roots(tree.body))
        while pending:
            name = pending.pop()
            if name in live:
                continue
            live.add(name)
            for line in definitions.get(name, ()):
                pending.extend(_reads(sites[line].value) - live)
        removed = {line for line, node in sites.items() if node.targets[0].id not in live}
        _, changed = prune_controls(tree.body)
        if not changed:
            break

    for line in sorted(removed):
        node, record = sites[line], records[line]
        report["removed"].append({"op": record.op.id, "name": node.targets[0].id,
                                  "reason": "no executable scalar consumer after specialization"})
    report["empty_controls"] = len(empty)
    erased = set(removed)
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For)) and node.lineno in empty:
            erased.update(range(node.lineno, node.end_lineno + 1))

    def kept_reads(block):
        result = Counter()
        for node in block:
            if node.lineno in erased:
                continue
            if isinstance(node, (ast.If, ast.For)):
                parts = [node.test if isinstance(node, ast.If) else node.iter]
                result.update(kept_reads(node.body))
                result.update(kept_reads(node.orelse))
            else:
                parts = [node]
            for part in parts:
                result.update(n.id for n in ast.walk(part) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
        return result

    uses = kept_reads(tree.body)
    replacements = {}

    def inline(block):
        surviving = [node for node in block if node.lineno not in erased]
        for index, node in enumerate(surviving[:-1]):
            if node.lineno not in sites:
                continue
            op = records[node.lineno].op
            if op.opcode in {"scalar.cast", "scalar.const", "scalar.cell", "scalar.set"} or not eligible(op) or not op.results:
                continue
            name = node.targets[0].id
            if uses[name] != 1 or len(definitions[name]) != 1:
                continue
            following = surviving[index + 1]
            target = None
            if isinstance(following, ast.Assign) and following.lineno in records:
                sink = records[following.lineno].op
                # Keep cross-Cell copies materialized: native loop-carry cleanup
                # can otherwise turn them into references to a newer Cell value.
                inplace = (op.opcode in {"scalar.add", "scalar.sub"} and isinstance(node.value, ast.BinOp)
                           and isinstance(node.value.left, ast.Name)
                           and node.value.left.id == following.targets[0].id)
                initialize = (sink.opcode == "scalar.cell" and sink.results[0].type.dtype == op.results[0].type.dtype
                              and all(not isinstance(x, Value) or isinstance(x.type, ScalarType) for x in op.operands))
                update = (inplace and sink.opcode == "scalar.set"
                          and sink.operands[0].type.dtype == op.results[0].type.dtype)
                if ((initialize or update)
                        and isinstance(following.value, ast.Name) and following.value.id == name):
                    target = following.value
            elif (isinstance(following, ast.If) and op.results[0].type.dtype.name == "b1"
                  and isinstance(following.test, ast.Name) and following.test.id == name):
                target = following.test
            # Adjacency in a Block excludes intervening writes, calls and control.
            if target is not None and target.lineno == target.end_lineno:
                expression = ast.get_source_segment(text, node.value)
                replacements[target.lineno] = (target.col_offset, target.end_col_offset, f"({expression})")
                erased.add(node.lineno)
                report["inlined"].append({"op": op.id, "name": name, "reason": "adjacent same-type single use"})
        for node in surviving:
            if isinstance(node, (ast.If, ast.For)):
                inline(node.body)
                inline(node.orelse)

    inline(tree.body)
    output = []
    source_lines = text.splitlines()
    # A retained conditional may lose every scalar in one arm. Preserve valid
    # Python with an explicit pass while its other arm or test remains observable.
    needs_pass = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.For)) or node.lineno in erased:
            continue
        for block in (node.body, node.orelse):
            if block and all(child.lineno in erased for child in block):
                needs_pass[block[-1].end_lineno] = " " * block[0].col_offset + "pass"
    for line, original in enumerate(source_lines, 1):
        if line in erased:
            if line in needs_pass:
                output.append(needs_pass[line])
            continue
        if line in replacements:
            start, end, value = replacements[line]
            encoded = original.encode("utf-8")
            original = (encoded[:start] + value.encode("utf-8") + encoded[end:]).decode("utf-8")
        output.append(original)
    ast.parse("\n".join(output) + "\n")
    return output, report
