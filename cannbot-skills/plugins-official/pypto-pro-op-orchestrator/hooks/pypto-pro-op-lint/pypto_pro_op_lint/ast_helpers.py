#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Small AST helpers shared by source-code lint checks."""

from __future__ import annotations

import ast
from collections.abc import Iterator


def parse_python_source(source: str, filename: str) -> tuple[ast.Module | None, str]:
    """Parse Python source and return a stable, user-facing syntax error."""
    try:
        return ast.parse(source, filename=filename), ""
    except SyntaxError as error:
        location = f"line {error.lineno}"
        if error.offset:
            location += f", column {error.offset}"
        return None, f"Python syntax error ({location}): {error.msg}"


def load_python_ast(abs_path: str, filename: str) -> tuple[ast.Module | None, str]:
    """Read and parse a Python file with one stable error contract."""
    try:
        with open(abs_path, "r", encoding="utf-8") as source_file:
            source = source_file.read()
    except OSError:
        return None, "File is empty or unreadable"
    if not source:
        return None, "File is empty or unreadable"
    return parse_python_source(source, filename)


def dotted_name(node: ast.AST) -> str:
    """Return the dotted spelling of a Name/Attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = dotted_name(node.value)
        return f"{owner}.{node.attr}" if owner else ""
    return ""


def decorator_name(node: ast.AST) -> str:
    """Return a decorator's dotted name, ignoring its call arguments."""
    target = node.func if isinstance(node, ast.Call) else node
    return dotted_name(target)


def iter_decorator_names(tree: ast.Module) -> Iterator[str]:
    """Yield dotted decorator names from all functions in a module."""
    function_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, function_types):
            for decorator in node.decorator_list:
                yield decorator_name(decorator)


def count_decorators(tree: ast.Module, expected_name: str) -> int:
    """Count function decorators with an exact dotted name."""
    count = 0
    for name in iter_decorator_names(tree):
        if name == expected_name:
            count += 1
    return count


def has_import_alias(tree: ast.Module, module: str, alias: str) -> bool:
    """Return whether an exact ``import module as alias`` exists."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for imported in node.names:
            if imported.name == module and imported.asname == alias:
                return True
    return False


def _iter_plain_import_names(node: ast.Import) -> Iterator[str]:
    for alias in node.names:
        yield alias.name


def _iter_from_import_names(node: ast.ImportFrom) -> Iterator[str]:
    module = node.module or ""
    if module:
        yield module
    for alias in node.names:
        if alias.name != "*":
            yield f"{module}.{alias.name}" if module else alias.name


def iter_imported_names(tree: ast.Module) -> Iterator[str]:
    """Yield fully qualified candidates introduced by static or dynamic imports.

    ``from modules import test_demo_module1`` yields both ``modules`` and
    ``modules.test_demo_module1`` so package-level and imported-name rules can
    be expressed without source-text regular expressions. Literal calls to
    ``__import__`` and ``importlib.import_module`` are included as well.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from _iter_plain_import_names(node)
        elif isinstance(node, ast.ImportFrom):
            yield from _iter_from_import_names(node)
    yield from _iter_dynamic_import_names(tree)


def _iter_dynamic_import_names(tree: ast.Module) -> Iterator[str]:
    """Yield literal module names loaded through Python's dynamic import APIs."""
    importlib_aliases = {"importlib"}
    import_module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "importlib":
                    importlib_aliases.add(imported.asname or imported.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for imported in node.names:
                if imported.name == "import_module":
                    import_module_aliases.add(imported.asname or imported.name)

    dynamic_call_names = {"__import__", *import_module_aliases}
    dynamic_call_names.update(
        f"{alias}.import_module" for alias in importlib_aliases
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if dotted_name(node.func) not in dynamic_call_names:
            continue
        module_arg = node.args[0]
        if isinstance(module_arg, ast.Constant) and isinstance(module_arg.value, str):
            yield module_arg.value


def imports_module(tree: ast.Module, module: str) -> bool:
    """Return whether the module itself or one of its submodules is imported."""
    prefix = f"{module}."
    return any(name == module or name.startswith(prefix) for name in iter_imported_names(tree))


def is_framework_import(name: str) -> bool:
    """Return whether an import points at pypto or pypto_pro."""
    roots = ("pypto", "pypto_pro")
    return any(name == root or name.startswith(f"{root}.") for root in roots)


def has_module_component(name: str, prefix: str) -> bool:
    """Return whether a dotted import contains a component with a prefix."""
    return any(part.startswith(prefix) for part in name.split("."))
