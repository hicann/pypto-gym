#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Generate the verification test file (per-module Step C / L0 E2E).

The per-module tests (`modules/test_<op>_module<suffix_k>.py`) and the E2E test
(`test_<op>.py`) are ~90% boilerplate: the detailed_tensor_compare path
bootstrap, naming-convention imports, and the mandatory `_l0` / `_l1` test
functions. This script emits the whole file from the operator name, the scope
(a phase suffix or the integrated E2E), and SPEC.md front matter, so
`templates/test_template.py.tmpl` no longer needs hand-filling every phase.

Inputs are built inline from SPEC `p0_shapes` (l1) and a small shape (l0); the
primary-input names come from the golden signature. For multi-input ops with
heterogeneous shapes, adjust the generated `make_case_inputs` body (it is
flagged with an ADJUST comment).

Lint invariants honored (rules.json D4): OL19 (detailed_tensor_compare),
OL20 (set_device from TILE_FWK_DEVICE_ID), OL21 (_l0 + _l1), OL22 (manual_seed),
OL42 (no hard-coded sim).

Usage::

    # per-module test for phase k=2 (suffix "12"):
    python gen_module_test.py --op relu --suffix 12 --spec custom/relu/SPEC.md \
        > custom/relu/modules/test_relu_module12.py

    # integrated E2E test:
    python gen_module_test.py --op relu --e2e --spec custom/relu/SPEC.md \
        > custom/relu/test_relu.py

    python gen_module_test.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger("gen_module_test")

_DTYPE_MAP = {
    "float32": "torch.float32", "fp32": "torch.float32",
    "float16": "torch.float16", "fp16": "torch.float16",
    "bfloat16": "torch.bfloat16", "bf16": "torch.bfloat16",
    "int32": "torch.int32", "int64": "torch.int64", "bool": "torch.bool",
}

_IMPORTS = '''from importlib import import_module
import os
import sys

import torch
import_module("torch_npu")  # Register torch.npu on supported installations.'''

_BOOTSTRAP = '''# detailed_tensor_compare path bootstrap (self-contained — no PYTHONPATH needed).
_test_dir = os.path.dirname(os.path.abspath(__file__))
_current = _test_dir
_candidate = None
for _ in range(8):
    _candidate = os.path.join(_current, ".opencode", "skills", "pypto-op-verify", "scripts")
    if os.path.isdir(_candidate):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break
    _parent = os.path.dirname(_current)
    if _parent == _current:
        _candidate = None
        break
    _current = _parent
if _candidate is None or not os.path.isdir(_candidate):
    raise ImportError("Could not locate detailed_tensor_compare under .opencode/skills/pypto-op-verify/scripts")
del _test_dir, _current, _candidate

_compare_module = __import__(
    "detailed_tensor_compare",
    fromlist=["detailed_tensor_compare", "tensor_leaf_pairs"],
)
detailed_tensor_compare = _compare_module.detailed_tensor_compare
tensor_leaf_pairs = _compare_module.tensor_leaf_pairs
del _compare_module
'''


def _parse_front_matter(content: str) -> dict:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}
    try:
        import yaml  # type: ignore  # noqa: PLC0415 -- optional CLI dependency
    except ImportError as exc:
        raise RuntimeError("PyYAML not available; cannot read SPEC front matter.") from exc
    return yaml.safe_load("\n".join(lines[1:end])) or {}


def _golden_params(golden: Path, golden_fn: str) -> list[str]:
    try:
        tree = ast.parse(golden.read_text(encoding="utf-8"))
    except OSError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == golden_fn:
            return [a.arg for a in node.args.args if a.arg != "self"]
    return []


def _shape_literals(p0_shapes: list) -> tuple[list[int], list[int]]:
    """Return (l1_shape, l0_shape). l0 caps each dim at 16 for a fast smoke."""
    first = None
    for item in p0_shapes or []:
        if isinstance(item, (list, tuple)) and item and all(isinstance(x, int) for x in item):
            first = list(item)
            break
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
            first = list(item[0])
            break
    if first is None:
        first = [16, 16]
    l0 = [min(16, d) for d in first]
    return first, l0


@dataclass
class TestSpec:
    """Grouped inputs for build_test (G.FNM.03: name-wrap correlated args)."""
    op: str
    scope: str
    params: list[str]
    dtype: str
    l1_shape: list[int]
    l0_shape: list[int]
    atol: float
    rtol: float
    is_e2e: bool


def _target_names(spec: TestSpec) -> tuple[str, str, str, str, str]:
    if spec.is_e2e:
        op = spec.op
        impl_mod, impl_fn = f"{op}_impl", f"{op}_wrapper"
        gold_mod, gold_fn = f"{op}_golden", f"{op}_golden"
        label = op
    else:
        op, scope = spec.op, spec.scope
        impl_mod, impl_fn = f"{op}_module{scope}_impl", f"{op}_module{scope}_wrapper"
        gold_mod, gold_fn = f"{op}_module{scope}_golden", f"{op}_module{scope}_golden"
        label = f"module{scope}"
    return impl_mod, impl_fn, gold_mod, gold_fn, label


def _header_lines(impl_mod, impl_fn, gold_mod, gold_fn, label):
    return [
        f'"""Auto-generated by gen_module_test.py — verification test for {label}.',
        "Edit make_case_inputs only if this op needs heterogeneous per-input shapes.",
        '"""',
        _IMPORTS,
        f"from {gold_mod} import {gold_fn}",
        f"from {impl_mod} import {impl_fn}",
        "",
        _BOOTSTRAP,
        '_DEVICE = torch.device(f"npu:{int(os.environ.get(\'TILE_FWK_DEVICE_ID\', \'0\'))}")',
        "",
        "",
        "def make_case_inputs(shape, seed=42):",
        '    """Build primary inputs on the NPU. ADJUST per-input shapes if heterogeneous."""',
        "    torch.npu.set_device(int(os.environ.get('TILE_FWK_DEVICE_ID', '0')))",
        "    torch.manual_seed(seed)",
        "    return {",
    ]


def _input_lines(names, torch_dtype):
    if torch_dtype in ("torch.int32", "torch.int64"):
        expression = f"torch.randint(-3, 4, shape, dtype={torch_dtype}, device=_DEVICE)"
    elif torch_dtype == "torch.bool":
        expression = "torch.rand(shape, device=_DEVICE) > 0.5"
    else:
        expression = f"torch.randn(shape, dtype={torch_dtype}, device=_DEVICE)"
    return [f"        \"{name}\": {expression}," for name in names]


def _runner_lines(impl_fn, gold_fn, label, atol, rtol):
    return [
        "    }",
        "",
        "",
        "def _run(shape):",
        "    inputs = make_case_inputs(shape)",
        f"    impl_out = {impl_fn}(*inputs.values())",
        f"    gold_out = {gold_fn}(*inputs.values())",
        f'    pairs = tensor_leaf_pairs(impl_out, gold_out, "{label}_out")',
        "    for output_name, actual, expected in pairs:",
        f"        result = detailed_tensor_compare(actual, expected, atol={atol}, rtol={rtol}, name=output_name)",
        '        assert result["all_close"], f"precision mismatch at {output_name}"',
        "",
        "",
    ]


def _test_case_lines(spec, label):
    lines = [
        f"def test_{label}_l0():",
        '    """L0 — small shapes, fast smoke."""',
        f"    _run({spec.l0_shape})",
        "",
        "",
        f"def test_{label}_l1():",
        '    """L1 — P0 shapes from SPEC.md."""',
        f"    _run({spec.l1_shape})",
        "",
    ]
    if spec.is_e2e:
        lines += [
            "",
            'if __name__ == "__main__":',
            f"    test_{label}_l0()",
            f"    test_{label}_l1()",
            "    print('[PRECISION_PASS]')",
            "",
        ]
    return lines


def build_test(spec: TestSpec) -> str:
    impl_mod, impl_fn, gold_mod, gold_fn, label = _target_names(spec)
    names = spec.params or ["x"]
    torch_dtype = _DTYPE_MAP.get(spec.dtype.lower(), "torch.float32")
    lines = _header_lines(impl_mod, impl_fn, gold_mod, gold_fn, label)
    lines += _input_lines(names, torch_dtype)
    lines += _runner_lines(impl_fn, gold_fn, label, spec.atol, spec.rtol)
    lines += _test_case_lines(spec, label)
    return "\n".join(lines)


def _gen_from_args(op: str, scope: str, is_e2e: bool, spec: Path | None,
                   golden: Path | None) -> str:
    meta = _parse_front_matter(spec.read_text(encoding="utf-8")) if spec else {}
    dtypes = meta.get("supported_dtypes") or ["float32"]
    dtype = dtypes[0] if isinstance(dtypes, list) and dtypes else "float32"
    l1_shape, l0_shape = _shape_literals(meta.get("p0_shapes", []))
    tol = meta.get("tolerance")
    atol = rtol = 1e-3
    if isinstance(tol, dict):
        atol = tol.get("atol", 1e-3)
        rtol = tol.get("rtol", 1e-3)
    gold_fn = f"{op}_golden" if is_e2e else f"{op}_module{scope}_golden"
    params = _golden_params(golden, gold_fn) if golden else []
    return build_test(TestSpec(op, scope, params, dtype, l1_shape, l0_shape,
                               atol, rtol, is_e2e))


def _self_test() -> int:
    txt = build_test(TestSpec("relu", "12", ["x"], "float16", [1024, 128],
                              [16, 16], 1e-3, 1e-3, is_e2e=False))
    ok = ("from relu_module12_impl import relu_module12_wrapper" in txt
          and "def test_module12_l0" in txt and "def test_module12_l1" in txt
          and "tensor_leaf_pairs" in txt and "torch.float16" in txt
          and "TILE_FWK_DEVICE_ID" in txt)
    compile_ok = True
    try:
        compile(txt, "<gen>", "exec")
    except SyntaxError as exc:
        compile_ok = False
        _LOGGER.info("FAIL syntax: %s", exc)
    e2e = build_test(TestSpec("relu", "", ["x"], "float32", [1024, 128],
                              [16, 16], 1e-3, 1e-3, is_e2e=True))
    e2e_ok = ("from relu_impl import relu_wrapper" in e2e
              and "[PRECISION_PASS]" in e2e and "__main__" in e2e)
    try:
        compile(e2e, "<gen-e2e>", "exec")
    except SyntaxError as exc:
        e2e_ok = False
        _LOGGER.info("FAIL e2e syntax: %s", exc)
    passed = ok and compile_ok and e2e_ok
    _LOGGER.info("%s self-test: module_test=%s e2e=%s", "PASS" if passed else "FAIL", ok, e2e_ok)
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op")
    ap.add_argument("--suffix", default="", help="cumulative phase suffix, e.g. 12 (omit for --e2e)")
    ap.add_argument("--e2e", action="store_true", help="generate the integrated test_<op>.py")
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--golden", type=Path, help="golden file for primary-input names (optional)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.self_test:
        return _self_test()
    if not args.op:
        ap.error("--op required (or --self-test)")
    if not args.e2e and not args.suffix:
        ap.error("--suffix required for per-module test (or pass --e2e)")
    _LOGGER.info(_gen_from_args(args.op, args.suffix, args.e2e, args.spec, args.golden))
    return 0


if __name__ == "__main__":
    sys.exit(main())
