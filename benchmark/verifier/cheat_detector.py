#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO 算子产物的机械层反作弊检测.

仅做**确定性**检查 — AST 基础结构 + 字符串匹配, 不依赖 LLM.
语义层的隐性作弊判定 (jit 函数体是 pass-through, forward 走 if-branch
绕开 pypto, 等等) 由 ``.opencode/skills/pypto-kernel-validate`` 让 LLM
亲自审阅, 与本检测互补不替代.

机械检查项:
    - ``import_pypto``       源码中存在 ``import pypto`` 或 ``from pypto``.
    - ``has_jit``            源码中至少有一个 ``@pypto.jit`` (装饰器或函数式调用).
    - ``forbidden_patterns`` 注释/字符串里出现 "for testing", "TODO use pypto",
                             "fallback to torch", "workaround" 等可疑文本时
                             标记为 SUSPICIOUS (不直接判 CHEAT, 让 LLM 复核).

裁定层级:
    - ``pass``       —— 检查通过.
    - ``suspicious`` —— 触发软警告, 需要 LLM 进一步审阅.
    - ``cheat``      —— 触发硬铁证, 直接判作弊.
      注意: multi-kernel 的唯一真相源是 runtime profile 的 ``CHEAT_MULTI_KERNEL``.

CLI 用法:
    python -m benchmark.verifier.cheat_detector \\
        <op_dir> [--op-name <name>] [--json-out <file>]

输出 JSON schema:
    {
      "op_name": "...",
      "op_dir": "...",
      "verdict": "pass | suspicious | cheat",
      "checks": [
        {"name": "...", "status": "pass|fail|warn|skip",
         "level": "info|warn|fatal", "detail": "..."},
        ...
      ],
      "summary": "<人类可读总结>"
    }
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 数据模型
# ────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: str           # "pass" | "fail" | "warn" | "skip"
    level: str            # "info" | "warn" | "fatal"
    detail: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheatReport:
    op_name: str
    op_dir: str
    verdict: str          # "pass" | "suspicious" | "cheat"
    checks: List[CheckResult] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ────────────────────────────────────────────────────────────
# 静态文本分析
# ────────────────────────────────────────────────────────────

_FORBIDDEN_TEXT_PATTERNS = [
    re.compile(r"for\s+testing\s+only", re.IGNORECASE),
    re.compile(r"TODO\s*[:\-]?\s*use\s+pypto", re.IGNORECASE),
    re.compile(r"fallback\s+to\s+torch", re.IGNORECASE),
    re.compile(r"workaround", re.IGNORECASE),
    re.compile(r"bypass\s+pypto", re.IGNORECASE),
]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _is_pypto_frontend_jit_attr(node: ast.Attribute) -> bool:
    """``pypto.frontend.jit`` 属性链."""
    val = node.value
    return (
        node.attr == "jit"
        and isinstance(val, ast.Attribute)
        and val.attr == "frontend"
        and isinstance(val.value, ast.Name)
        and val.value.id == "pypto"
    )


def _is_pypto_jit_attr(node: ast.AST) -> bool:
    """判断 AST 节点是否引用 ``pypto.jit`` / ``pypto.frontend.jit``."""
    if isinstance(node, ast.Attribute) and node.attr == "jit":
        if isinstance(node.value, ast.Name) and node.value.id == "pypto":
            return True
        if _is_pypto_frontend_jit_attr(node):
            return True
    if isinstance(node, ast.Name) and node.id == "jit":
        # `from pypto import jit` 后裸用; best-effort 接受.
        return True
    return False


def _has_jit_usage(tree: ast.Module) -> bool:
    """Return whether source uses PyPTO jit in decorator or function-call form."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in node.decorator_list:
                target = deco.func if isinstance(deco, ast.Call) else deco
                if _is_pypto_jit_attr(target):
                    return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_pypto_jit_attr(node.func):
            return True
    return False


def _has_pypto_import(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "pypto" or a.name.startswith("pypto."):
                    return True
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "pypto" or mod.startswith("pypto."):
                return True
    return False


def _scan_forbidden_text(src: str) -> List[str]:
    hits: List[str] = []
    for pat in _FORBIDDEN_TEXT_PATTERNS:
        m = pat.search(src)
        if m:
            hits.append(m.group(0))
    return hits


# ────────────────────────────────────────────────────────────
# 主检查逻辑
# ────────────────────────────────────────────────────────────

def _candidate_source_files(op_dir: Path, op_name: str) -> List[Path]:
    """收集 op_dir 下要扫描的 PyPTO 实现源码文件.

    优先 ``{op}_impl.py`` (核心算子), ``{op}_pypto_impl.py`` (ModelNew 包装).
    遇到 fallback 场景 (只有 _impl.py) 也兼容.
    """
    candidates = [
        op_dir / f"{op_name}_impl.py",
        op_dir / f"{op_name}_pypto_impl.py",
    ]
    return [p for p in candidates if p.exists()]


def _parse_module(path: Path) -> Optional[ast.Module]:
    src = _read_text(path)
    if not src:
        return None
    try:
        return ast.parse(src, filename=str(path))
    except SyntaxError:
        return None


def detect_cheats(op_dir: Path, op_name: str) -> CheatReport:
    """对单个算子产物目录执行机械层反作弊检测."""
    op_dir = op_dir.resolve()
    report = CheatReport(op_name=op_name, op_dir=str(op_dir), verdict="pass")

    sources = _candidate_source_files(op_dir, op_name)
    if not sources:
        report.checks.append(CheckResult(
            name="sources_present",
            status="fail",
            level="fatal",
            detail=(
                f"No source files found in {op_dir}; "
                f"expected at least one of {op_name}_impl.py / {op_name}_pypto_impl.py."
            ),
        ))
        report.verdict = "cheat"
        report.summary = "无源码可分析."
        return report
    report.checks.append(CheckResult(
        name="sources_present",
        status="pass",
        level="info",
        detail=f"Scanning: {', '.join(p.name for p in sources)}.",
    ))

    has_pypto_import = False
    has_jit = False
    forbidden_hits: List[str] = []

    for path in sources:
        src = _read_text(path)
        forbidden_hits.extend(_scan_forbidden_text(src))
        tree = _parse_module(path)
        if tree is None:
            report.checks.append(CheckResult(
                name=f"parse:{path.name}",
                status="fail",
                level="fatal",
                detail=f"SyntaxError parsing {path}; cannot analyze.",
            ))
            report.verdict = "cheat"
            continue
        has_pypto_import = has_pypto_import or _has_pypto_import(tree)
        has_jit = has_jit or _has_jit_usage(tree)

    if has_pypto_import:
        report.checks.append(CheckResult(
            name="import_pypto",
            status="pass",
            level="info",
            detail="源码中存在 import pypto / from pypto.",
        ))
    else:
        report.checks.append(CheckResult(
            name="import_pypto",
            status="fail",
            level="fatal",
            detail=(
                "源码中找不到 import pypto / from pypto — 这意味着算子可能"
                "完全没用 PyPTO, 而是纯 torch 实现. CHEAT."
            ),
        ))
        report.verdict = "cheat"

    if not has_jit:
        report.checks.append(CheckResult(
            name="has_jit",
            status="fail",
            level="fatal",
            detail=(
                "源码中未发现任何 @pypto.jit / pypto.jit(...) 调用 — "
                "算子没有走 PyPTO jit 编译路径. CHEAT."
            ),
        ))
        report.verdict = "cheat"
    else:
        report.checks.append(CheckResult(
            name="has_jit",
            status="pass",
            level="info",
            detail="源码中存在 @pypto.jit / pypto.jit(...) 调用.",
        ))

    if forbidden_hits:
        report.checks.append(CheckResult(
            name="forbidden_text_patterns",
            status="warn",
            level="warn",
            detail=(
                f"源码中出现可疑字符串: {forbidden_hits!r}. "
                f"请 LLM 复核, 可能为占位 / fallback / workaround 实现."
            ),
            extra={"hits": forbidden_hits},
        ))
        if report.verdict == "pass":
            report.verdict = "suspicious"
    else:
        report.checks.append(CheckResult(
            name="forbidden_text_patterns",
            status="pass",
            level="info",
            detail="源码中未出现 testing-only / fallback / workaround 等可疑字符串.",
        ))

    fatal_fails = [c for c in report.checks if c.status == "fail" and c.level == "fatal"]
    warns = [c for c in report.checks if c.status == "warn"]
    if fatal_fails:
        report.summary = (
            f"CHEAT: {len(fatal_fails)} 项硬铁证不通过 — "
            f"{', '.join(c.name for c in fatal_fails)}."
        )
    elif warns:
        report.summary = (
            f"SUSPICIOUS: {len(warns)} 项软警告需要 LLM 语义层复核 — "
            f"{', '.join(c.name for c in warns)}."
        )
    else:
        report.summary = "PASS: 机械层未发现作弊迹象 (语义层仍需 LLM 复核)."

    return report


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmark.verifier.cheat_detector",
        description="PyPTO 算子产物机械层反作弊检测.",
    )
    p.add_argument("op_dir", type=Path,
                   help="算子产物目录 (含 {op}_impl.py / {op}_pypto_impl.py).")
    p.add_argument("--op-name", default=None,
                   help="算子名; 缺省取 op_dir 的 basename.")
    p.add_argument("--json-out", type=Path, default=None,
                   help="JSON 报告输出路径; 缺省打印到 stdout.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    op_dir: Path = args.op_dir.resolve()
    op_name: str = args.op_name or op_dir.name

    if not op_dir.is_dir():
        logger.error("[cheat-detector] op_dir 不存在或非目录: %s", op_dir)
        return 2

    report = detect_cheats(op_dir, op_name)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
        logger.info("[cheat-detector] 报告已写入: %s", args.json_out)
    sys.stdout.write(payload + "\n")
    return {"pass": 0, "suspicious": 0, "cheat": 1}.get(report.verdict, 0)


if __name__ == "__main__":
    sys.exit(main())
