#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Generate a verifier-owned test file (verifier.md Step C / L0 E2E).

The per-module tests (`modules/test_<op>_module<suffix_k>.py`) and the E2E test
(`test_<op>.py`) are ~90% boilerplate: the detailed_tensor_compare path
bootstrap, naming-convention imports, and the mandatory `_l0` / `_l1` test
functions. This script emits the whole file from the operator name, the scope
(a phase suffix or the integrated E2E), and SPEC.md front matter, so the
verifier no longer hand-fills `templates/test_template.py` every phase.

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

_BOOTSTRAP = '''import os
import sys

# detailed_tensor_compare path bootstrap (self-contained — no PYTHONPATH needed).
_test_dir = os.path.dirname(os.path.abspath(__file__))
_current = _test_dir
_candidate = None
for _ in range(8):
    _candidate = os.path.join(_current, ".agents", "skills", "pypto-op-verify", "scripts")
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
    raise ImportError("Could not locate detailed_tensor_compare under .agents/skills/pypto-op-verify/scripts")
del _test_dir, _current, _candidate

import torch
import torch_npu  # noqa: F401  required for NPU device init

from detailed_tensor_compare import detailed_tensor_compare
'''


def _parse_front_matter(content: str) -> dict:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}
    try:
        import yaml  # type: ignore
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


def build_test(spec: TestSpec) -> str:
    op, scope, params, dtype = spec.op, spec.scope, spec.params, spec.dtype
    l1_shape, l0_shape = spec.l1_shape, spec.l0_shape
    atol, rtol, is_e2e = spec.atol, spec.rtol, spec.is_e2e
    if is_e2e:
        impl_mod, impl_fn = f"{op}_impl", f"{op}_wrapper"
        gold_mod, gold_fn = f"{op}_golden", f"{op}_golden"
        label = op
    else:
        impl_mod, impl_fn = f"{op}_module{scope}_impl", f"{op}_module{scope}_wrapper"
        gold_mod, gold_fn = f"{op}_module{scope}_golden", f"{op}_module{scope}_golden"
        label = f"module{scope}"
    names = params or ["x"]
    torch_dtype = _DTYPE_MAP.get(dtype.lower(), "torch.float32")

    out = [
        f'"""Auto-generated by gen_module_test.py — verifier-owned test for {label}.',
        "Edit make_case_inputs only if this op needs heterogeneous per-input shapes.",
        '"""',
        _BOOTSTRAP,
        f"from {impl_mod} import {impl_fn}",
        f"from {gold_mod} import {gold_fn}",
        "",
        '_DEVICE = torch.device(f"npu:{int(os.environ.get(\'TILE_FWK_DEVICE_ID\', \'0\'))}")',
        "",
        "",
        "def make_case_inputs(shape, seed=42):",
        '    """Build primary inputs on the NPU. ADJUST per-input shapes if heterogeneous."""',
        "    torch.npu.set_device(int(os.environ.get('TILE_FWK_DEVICE_ID', '0')))",
        "    torch.manual_seed(seed)",
        "    return {",
    ]
    for n in names:
        out.append(f"        \"{n}\": torch.randn(shape, dtype={torch_dtype}, device=_DEVICE),")
    out += [
        "    }",
        "",
        "",
        "def _run(shape):",
        "    inputs = make_case_inputs(shape)",
        f"    impl_out = {impl_fn}(*inputs.values())",
        f"    gold_out = {gold_fn}(*inputs.values())",
        "    impl_out = impl_out if isinstance(impl_out, (tuple, list)) else (impl_out,)",
        "    gold_out = gold_out if isinstance(gold_out, (tuple, list)) else (gold_out,)",
        "    for i, (a, b) in enumerate(zip(impl_out, gold_out)):",
        f"        detailed_tensor_compare(a, b, atol={atol}, rtol={rtol}, name=f\"{label}_out{{i}}\")",
        "",
        "",
        f"def test_{label}_l0():",
        '    """L0 — small shapes, fast smoke."""',
        f"    _run({l0_shape})",
        "",
        "",
        f"def test_{label}_l1():",
        '    """L1 — P0 shapes from SPEC.md."""',
        f"    _run({l1_shape})",
        "",
    ]
    if is_e2e:
        out += [
            "",
            'if __name__ == "__main__":',
            f"    test_{label}_l0()",
            f"    test_{label}_l1()",
            "    print('[PRECISION_PASS]')",
            "",
        ]
    return "\n".join(out)


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
          and "detailed_tensor_compare" in txt and "torch.float16" in txt
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
