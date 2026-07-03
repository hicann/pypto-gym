#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Validate custom/<op>/eval/module_interfaces.yaml wiring (verifier Step A.5.1).

Codifies the six validity rules the verifier previously checked by hand
(.opencode/agents/pypto-op-verifier.md Step A.5.1):

  1. Every inputs[*].source == "primary" name exists in primary_inputs.
  2. Every inputs[*].source == "module_j" has j < current module id, and the
     referenced name exists in module_j.outputs (no forward/self reference).
  3. Every final_outputs[*].source == "module_j" has j <= N, and the
     referenced name exists in module_j.outputs.
  4. No two outputs share the same (module_id, name) key.
  5. Shape expressions parse using only + - * // and name/int tokens.
  6. dtype strings are from the allowed vocabulary.

On FAIL the verifier appends a rejection block to MEMORY.md and stops so the
orchestrator can re-dispatch architect/designer.

Assumed schema (the designer's emitter and this validator must agree):

    primary_inputs:
      - {name: x, shape: "[B, T]", dtype: float32}
    modules:
      - id: 1
        inputs:  [{name: x, source: primary}]
        outputs: [{name: h1, shape: "[B, T]", dtype: float32}]
      - id: 2
        inputs:  [{name: h1, source: module_1}]
        outputs: [{name: y, shape: "[B, T]", dtype: float32}]
    final_outputs:
      - {name: y, source: module_2}

Usage::

    python validate_yaml.py custom/<op>/eval/module_interfaces.yaml [--json]
    python validate_yaml.py --self-test

Exit code: 0 if valid, 1 if any rule fails (or on load/parse error).
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path

# Emit on stdout with a bare (message-only) format so JSON / summary output
# stays machine-parseable for the caller (verifier reads stdout).
_LOGGER = logging.getLogger("validate_yaml")

DTYPE_VOCAB = {"float32", "float16", "bfloat16", "int32", "int64", "bool", "int"}
_MODULE_RE = re.compile(r"^module_(\d+)$")
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.USub, ast.UAdd, ast.Constant,
)


def _shape_dims(shape) -> list[str]:
    """Normalize a shape field (string '[B, T]' or list) into dim expressions."""
    if isinstance(shape, (list, tuple)):
        return [str(d) for d in shape]
    s = str(shape).strip().strip("[]")
    return [d.strip() for d in s.split(",") if d.strip()]


def _dim_parses(expr: str) -> bool:
    """True if a dim expression uses only + - * // and name/int tokens."""
    if expr.lstrip("-").isdigit():
        return True
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, int):
            return False
    return True


def validate(spec: dict) -> list[str]:
    """Return a list of violation strings; empty list means valid."""
    errors: list[str] = []
    primary = {p.get("name") for p in spec.get("primary_inputs", []) if isinstance(p, dict)}
    modules = spec.get("modules", []) or []
    n = len(modules)

    # index module outputs by id
    outputs_by_id: dict[int, set] = {}
    seen_keys: set = set()
    for m in modules:
        mid = m.get("id")
        names = set()
        for o in m.get("outputs", []) or []:
            name = o.get("name")
            names.add(name)
            key = (mid, name)
            if key in seen_keys:  # rule 4
                errors.append(f"rule4: duplicate output key {key}")
            seen_keys.add(key)
            # rule 5: shape expressions
            for dim in _shape_dims(o.get("shape", "")):
                if not _dim_parses(dim):
                    errors.append(f"rule5: module {mid} output '{name}' shape dim '{dim}' not parseable")
            # rule 6: dtype vocab
            dt = o.get("dtype")
            if dt is not None and dt not in DTYPE_VOCAB:
                errors.append(f"rule6: module {mid} output '{name}' dtype '{dt}' not in {sorted(DTYPE_VOCAB)}")
        outputs_by_id[mid] = names

    # rule 1 & 2: module input wiring
    for m in modules:
        mid = m.get("id")
        for inp in m.get("inputs", []) or []:
            src = inp.get("source")
            name = inp.get("name")
            if src == "primary":
                if name not in primary:  # rule 1
                    errors.append(f"rule1: module {mid} input '{name}' source=primary not in primary_inputs")
                continue
            mt = _MODULE_RE.match(str(src or ""))
            if not mt:
                errors.append(f"rule2: module {mid} input '{name}' has invalid source '{src}'")
                continue
            j = int(mt.group(1))
            if not (isinstance(mid, int) and j < mid):  # rule 2: forward/self ref
                errors.append(f"rule2: module {mid} input '{name}' references module_{j} (must be < {mid})")
            elif name not in outputs_by_id.get(j, set()):
                errors.append(f"rule2: module {mid} input '{name}' not produced by module_{j}")

    # rule 3: final_outputs wiring
    for fo in spec.get("final_outputs", []) or []:
        name = fo.get("name")
        src = fo.get("source")
        mt = _MODULE_RE.match(str(src or ""))
        if not mt:
            errors.append(f"rule3: final_output '{name}' has invalid source '{src}'")
            continue
        j = int(mt.group(1))
        if j > n:
            errors.append(f"rule3: final_output '{name}' references module_{j} > N={n}")
        elif name not in outputs_by_id.get(j, set()):
            errors.append(f"rule3: final_output '{name}' not produced by module_{j}")

    return errors


def _load(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML not available; cannot read YAML file.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


_VALID = {
    "primary_inputs": [{"name": "x", "shape": "[B, T]", "dtype": "float32"}],
    "modules": [
        {"id": 1, "inputs": [{"name": "x", "source": "primary"}],
         "outputs": [{"name": "h1", "shape": "[B, T]", "dtype": "float32"}]},
        {"id": 2, "inputs": [{"name": "h1", "source": "module_1"}],
         "outputs": [{"name": "y", "shape": "[B, T]", "dtype": "float32"}]},
    ],
    "final_outputs": [{"name": "y", "source": "module_2"}],
}


def _self_test() -> int:
    import copy
    cases: list[tuple[str, dict, str]] = []
    cases.append(("valid", _VALID, ""))  # expect no error

    c1 = copy.deepcopy(_VALID)
    c1["modules"][0]["inputs"][0]["name"] = "zzz"
    cases.append(("rule1_unknown_primary", c1, "rule1"))
    c2 = copy.deepcopy(_VALID)
    c2["modules"][0]["inputs"][0] = {"name": "y", "source": "module_2"}
    cases.append(("rule2_forward_ref", c2, "rule2"))
    c3 = copy.deepcopy(_VALID)
    c3["final_outputs"][0]["source"] = "module_9"
    cases.append(("rule3_out_of_range", c3, "rule3"))
    c4 = copy.deepcopy(_VALID)
    c4["modules"][1]["outputs"].append({"name": "y", "shape": "[B, T]", "dtype": "float32"})
    c4["modules"][1]["id"] = 2  # two (2,'y') keys
    c4["modules"][1]["outputs"][0]["name"] = "y"
    cases.append(("rule4_dup_key", c4, "rule4"))
    c5 = copy.deepcopy(_VALID)
    c5["modules"][0]["outputs"][0]["shape"] = "[B ** 2, T]"
    cases.append(("rule5_bad_shape", c5, "rule5"))
    c6 = copy.deepcopy(_VALID)
    c6["modules"][0]["outputs"][0]["dtype"] = "float8"
    cases.append(("rule6_bad_dtype", c6, "rule6"))

    bad = 0
    for name, spec, want in cases:
        errs = validate(spec)
        if want == "":
            ok = not errs
        else:
            ok = any(e.startswith(want) for e in errs)
        bad += 0 if ok else 1
        _LOGGER.info("%s %s: %s", "PASS" if ok else "FAIL", name,
                     errs if errs else "no errors")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", nargs="?", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if args.self_test:
        return _self_test()
    if not args.yaml:
        ap.error("yaml path required (or --self-test)")
    spec = _load(args.yaml)
    errors = validate(spec or {})
    if args.json:
        _LOGGER.info(json.dumps(
            {"status": "PASS" if not errors else "FAIL", "violations": errors}, indent=2))
    elif errors:
        _LOGGER.info("FAIL:")
        for e in errors:
            _LOGGER.info("  - %s", e)
    else:
        _LOGGER.info("PASS")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
