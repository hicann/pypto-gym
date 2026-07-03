# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from __future__ import annotations

import ast

from ..ast_helpers import _get_jit_functions
from ..core import CheckContext, Finding, register
from ..utils import _impl_files_to_scan


def _is_dot_t_call(node: ast.AST) -> bool:
    """判定无参数的 `.t()` 调用（拆分布尔条件以满足 G.CTL.03）。"""
    if not isinstance(node, ast.Call) or node.args:
        return False
    return isinstance(node.func, ast.Attribute) and node.func.attr == "t"


@register("OL15")
def check_ol15(ctx: CheckContext) -> Finding:
    """golden 文件须为纯 torch 规范化实现：禁止 import pypto，禁止 `.T` / `.t()`（须用 torch.transpose）。"""
    golden_file = f"{ctx.op_name}_golden.py"
    tree = ctx.parse_file(golden_file)
    if tree is None:
        return ctx.make_finding("OL15", "SKIP", f"{golden_file} 不存在或无法解析")
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pypto" or alias.name.startswith("pypto."):
                    return ctx.make_finding("OL15", "FAIL",
                        "golden 文件禁止 import pypto",
                        file=golden_file, line=node.lineno)
        if (isinstance(node, ast.ImportFrom)
                and node.module and node.module.startswith("pypto")):
            return ctx.make_finding("OL15", "FAIL",
                "golden 文件禁止 from pypto import ...",
                file=golden_file, line=node.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "T":
            return ctx.make_finding("OL15", "FAIL",
                "golden 文件禁止 `.T`，请用 torch.transpose(t, d0, d1) 并注记 `# pypto: b_trans=True`",
                file=golden_file, line=node.lineno)
        if _is_dot_t_call(node):
            return ctx.make_finding("OL15", "FAIL",
                "golden 文件禁止 `.t()`，请用 torch.transpose(t, d0, d1)",
                file=golden_file, line=node.lineno)
    return ctx.make_finding("OL15", "PASS",
        "golden 文件未导入 pypto 且无 `.T` / `.t()`", file=golden_file)


@register("OL16")
def check_ol16(ctx: CheckContext) -> Finding:
    """impl 文件不应导入 golden 模块。

    覆盖范围：顶层集成 impl + modules/<op>_module*_impl.py。
    """
    impl_files = _impl_files_to_scan(ctx)
    if not impl_files:
        return ctx.make_finding("OL16", "SKIP", "无 impl 文件可供检查")
    golden_module = f"{ctx.op_name}_golden"
    parsed_any = False
    for impl_file in impl_files:
        tree = ctx.parse_file(impl_file)
        if tree is None:
            continue
        parsed_any = True
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module == golden_module:
                return ctx.make_finding(
                    "OL16",
                    "FAIL",
                    f"{impl_file} 不应导入 {golden_module}",
                    file=impl_file,
                    line=node.lineno,
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == golden_module:
                        return ctx.make_finding(
                            "OL16",
                            "FAIL",
                            f"{impl_file} 不应导入 {golden_module}",
                            file=impl_file,
                            line=node.lineno,
                        )
    if not parsed_any:
        return ctx.make_finding("OL16", "SKIP", "无 impl 文件可解析")
    return ctx.make_finding(
        "OL16", "PASS", f"所有 impl 文件均未导入 golden（共 {len(impl_files)} 个）"
    )


@register("OL17")
def check_ol17(ctx: CheckContext) -> Finding:
    """test 文件不应包含 kernel 实现代码"""
    test_file = f"test_{ctx.op_name}.py"
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL17", "SKIP", f"{test_file} 不存在或无法解析")
    aliases = ctx.pypto_aliases(test_file)
    jit_funcs = _get_jit_functions(tree, aliases)
    if jit_funcs:
        func = jit_funcs[0]
        return ctx.make_finding("OL17", "FAIL",
            f"test 文件包含 @pypto.frontend.jit 装饰的函数: {func.name}",
            file=test_file, line=func.lineno)
    return ctx.make_finding("OL17", "PASS",
        "test 文件未包含 kernel 实现", file=test_file)


@register("OL18")
def check_ol18(ctx: CheckContext) -> Finding:
    """test 文件必须从 impl 和 golden 分别导入"""
    test_file = f"test_{ctx.op_name}.py"
    tree = ctx.parse_file(test_file)
    if tree is None:
        return ctx.make_finding("OL18", "SKIP", f"{test_file} 不存在或无法解析")
    impl_module = f"{ctx.op_name}_impl"
    golden_module = f"{ctx.op_name}_golden"
    has_impl = False
    has_golden = False

    def _matches_module(module_name: str | None, expected: str) -> bool:
        if module_name is None:
            return False
        return module_name == expected or module_name.endswith(f".{expected}")

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _matches_module(node.module, impl_module):
                has_impl = True
            if _matches_module(node.module, golden_module):
                has_golden = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _matches_module(alias.name, impl_module):
                    has_impl = True
                if _matches_module(alias.name, golden_module):
                    has_golden = True
    missing = []
    if not has_impl:
        missing.append(impl_module)
    if not has_golden:
        missing.append(golden_module)
    if missing:
        return ctx.make_finding("OL18", "FAIL",
            f"test 文件缺少导入: {', '.join(missing)}", file=test_file)
    return ctx.make_finding("OL18", "PASS",
        "test 文件正确导入了 impl 和 golden", file=test_file)
