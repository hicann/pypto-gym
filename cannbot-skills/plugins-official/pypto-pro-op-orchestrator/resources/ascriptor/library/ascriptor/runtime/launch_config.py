# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dependency-free evaluation of the declared launch geometry (RFC-0002 §3.4) and of an imported kernel's exported
launch (RFC-0015)."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any


def resolve_block_dim(value: int | str | None, bindings: Mapping[str, Any]) -> int | None:
    if value is None:
        return None

    def integer(x: Any) -> int:
        if type(x) is not int:
            raise ValueError("block_dim expressions require integer scalars")
        return x

    def evaluate(node: ast.AST) -> int:
        match node:
            case ast.Constant(value=x):
                return integer(x)
            case ast.Name(id=name):
                if name not in bindings:
                    raise ValueError(f"block_dim expression has no binding for {name!r}")
                return integer(bindings[name])
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return -evaluate(operand)
            case ast.BinOp(left=left, op=op, right=right):
                a, b = evaluate(left), evaluate(right)
                match op:
                    case ast.Add(): return a + b
                    case ast.Sub(): return a - b
                    case ast.Mult(): return a * b
                    case ast.FloorDiv(): return a // b
            case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in {"min", "max", "ceil_div"}:
                numbers = [evaluate(arg) for arg in args]
                if name == "ceil_div" and len(numbers) == 2:
                    return (numbers[0] + numbers[1] - 1) // numbers[1]
                if name in {"min", "max"} and numbers:
                    return (min if name == "min" else max)(numbers)
        raise ValueError("unsupported block_dim expression; use integer arithmetic, min, max or ceil_div")

    try:
        result = integer(value) if not isinstance(value, str) else evaluate(ast.parse(value, mode="eval").body)
    except (SyntaxError, ZeroDivisionError) as error:
        raise ValueError("invalid block_dim arithmetic expression") from error
    if result <= 0:
        raise ValueError("block_dim must be positive")
    return result


class BlockDimError(ValueError):
    """A launch asked an imported kernel for a block_dim other than the one it was exported with."""


def exported_block_dim(module: Any) -> int | None:
    """The block_dim an imported kernel was exported with (RFC-0015), or None for any other module.

    It bounds the kernel's launch identity queries in the import's range proofs, so a launch may use no other."""
    attrs = getattr(module, "attrs", None) or {}
    if "import_producer" not in attrs:
        return None
    return int(attrs["meta"]["block_dim"])


def require_block_dim(fixed: int | None, requested: Any, entry: str, *, kernel: str, source: str | None) -> Any:
    """``requested`` for a kernel without an exported block_dim; else the exported one, refusing any other."""
    if fixed is None:
        return requested
    if requested is None:
        return fixed
    if type(requested) is not int or requested != fixed:
        where = f"{source}: " if source else ""
        raise BlockDimError(f"{where}imported kernel {kernel!r} was exported with block_dim {fixed}, which bounds its launch "
                            f"identity queries (RFC-0015); {entry} was asked for block_dim {requested!r}")
    return fixed


def launch_block_dim(module: Any, requested: Any, entry: str, location: Mapping[str, Any] | None = None) -> Any:
    """The block_dim ``entry`` launches ``module`` with; see ``require_block_dim``. ``location`` is the kernel
    definition when the caller has it, else the module's source file names the kernel."""
    source = f"{location['file']}:{location['line']}:{location['column']}" if location else None
    attrs = getattr(module, "attrs", None) or {}
    kernel = str(attrs.get("meta", {}).get("kernel", getattr(module, "name", "?")))
    return require_block_dim(exported_block_dim(module), requested, entry, kernel=kernel,
                             source=source or attrs.get("source"))


def exported_artifacts(module: Any, entry: str, block_dim: Any, emit: Any) -> Any:
    """``emit(block_dim)``'s artifacts for ``module`` once ``entry`` has checked the launch; an imported kernel's
    manifest also records its exported block_dim and source, which ``project.host_source`` enforces."""
    fixed = exported_block_dim(module)
    artifacts = emit(launch_block_dim(module, block_dim, entry))
    if fixed is None:
        return artifacts
    meta = {**artifacts.metadata, "exported_block_dim": fixed, "exported_from": module.attrs.get("source")}
    files = {**artifacts.files, "manifest.json": (json.dumps(meta, indent=1) + "\n").encode()}
    return replace(artifacts, files=files, metadata=meta)
