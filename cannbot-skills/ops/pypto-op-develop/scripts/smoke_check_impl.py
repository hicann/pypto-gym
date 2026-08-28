#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-side smoke check for a pypto impl file.

Runs ONLY what runs on the host, so coder can self-check before
``submit_for_verify`` without occupying a device:

  1. AST parse            — syntax errors;
  2. module import        — import-time errors;
  3. frontend trace       — for every ``@pypto.frontend.jit`` kernel, build
     CPU example inputs from the cached signature and call
     ``JitCallableWrapper.compile()`` (the same host-side trace the
     JitCallableWrapper runs before any codegen). This is where trace-time
     errors such as F00001 (RecordIfBranch) / F00002 (Not concrete value) /
     F00003 (nested def) surface, within seconds.

Mechanism: ``pypto.pypto_impl.Reset()`` + ``wrapper.compile(torch_tensors,
tensor_defs)`` — no kernel codegen, no NPU, no launch.

Usage::

    python smoke_check_impl.py <impl.py> [--shapes '<json>']

``--shapes`` JSON (only needed when tensor shapes cannot be auto-inferred,
e.g. dynamic/symbolic dims)::

    {"<kernel_name>": [{"shape": [64], "dtype": "fp32"}, ...],
     "*":             [{"shape": [64], "dtype": "fp32"}, ...]}

Each entry must cover ALL tensor params of the kernel (including out).
``"*"`` applies to every kernel without an explicit entry.

All problems found are reported (not just the first). Exit code 0 iff no
errors. Kernels whose inputs cannot be constructed are SKIPPED (import+AST
only) with an explicit notice — conservative by design; smoke passing never
implies verify passing.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

# Named logger bound to stdout (never the root logger — basicConfig would
# raise the root level and unleash third-party INFO noise into the output).
# The emitted text is the machine-readable protocol agents consume, so it
# stays on stdout, byte-identical to the previous plain-print output.
_log = logging.getLogger("smoke_check_impl")
_log.setLevel(logging.INFO)
_log.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
_log.addHandler(_handler)

_DTYPES = None  # lazily built {pypto dtype name suffix: torch.dtype}


def _dtype_map():
    global _DTYPES
    if _DTYPES is None:
        import torch

        _DTYPES = {
            "FP32": torch.float32, "FLOAT32": torch.float32,
            "FP16": torch.float16, "FLOAT16": torch.float16,
            "BF16": torch.bfloat16, "BFLOAT16": torch.bfloat16,
            "INT32": torch.int32, "INT64": torch.int64,
            "INT16": torch.int16, "INT8": torch.int8,
            "UINT8": torch.uint8, "BOOL": torch.bool,
        }
    return _DTYPES


def _torch_dtype(pto_dtype):
    """Map a pypto DataType (e.g. DataType.DT_FP32) to a torch dtype."""
    name = str(pto_dtype).split(".")[-1].upper()
    if name.startswith("DT_"):
        name = name[3:]
    dt = _dtype_map().get(name)
    if dt is None:
        raise ValueError(f"unsupported pypto dtype for example input: {pto_dtype}")
    return dt


def find_jit_kernels(module) -> list:
    """Module-level JitCallableWrapper objects (duck-typed, version-robust)."""
    kernels = []
    for name in dir(module):
        obj = getattr(module, name)
        if name.startswith("__"):
            continue
        if (hasattr(obj, "compile") and hasattr(obj, "_cached_signature")
                and hasattr(obj, "_create_parser")):
            kernels.append((name, obj))
    return kernels


def build_inputs(name: str, wrapper, shapes_cfg: dict):
    """Build CPU example tensors for one kernel.

    Returns (torch_tensors, tensor_defs) or raises ValueError with the reason
    inputs cannot be constructed.
    """
    import torch

    tensor_defs, non_tensor = wrapper._cached_signature
    if non_tensor:
        raise ValueError(
            f"kernel takes non-tensor params {non_tensor}; cannot auto-build "
            "inputs (conservative skip)")
    spec = shapes_cfg.get(name, shapes_cfg.get("*"))
    if spec is not None:
        if len(spec) != len(tensor_defs):
            raise ValueError(
                f"--shapes entry for {name!r} has {len(spec)} tensors, "
                f"signature expects {len(tensor_defs)}")
        tensors = []
        for t in spec:
            dt = _dtype_map().get(str(t["dtype"]).upper())
            if dt is None:
                raise ValueError(f"--shapes entry for {name!r}: unknown "
                                 f"dtype {t['dtype']!r}")
            tensors.append(torch.zeros(list(t["shape"]), dtype=dt))
        return tensors, tensor_defs
    tensors = []
    for d in tensor_defs:
        shape = getattr(d, "shape", None)
        if not shape or not all(isinstance(dim, int) for dim in shape):
            raise ValueError(
                f"shape {shape!r} is not fully static; pass --shapes to trace "
                "this kernel (skipping trace, import+AST only)")
        tensors.append(torch.zeros(list(shape), dtype=_torch_dtype(d.dtype)))
    return tensors, tensor_defs


def trace_one(name: str, wrapper, shapes_cfg: dict) -> dict:
    """Run the frontend trace for one kernel. Returns a result dict."""
    import pypto.pypto_impl as pypto_impl

    try:
        tensors, tensor_defs = build_inputs(name, wrapper, shapes_cfg)
    except ValueError as e:
        return {"kernel": name, "status": "SKIP", "detail": str(e)}
    try:
        pypto_impl.Reset()  # isolate repeated traces in the same process
        wrapper.compile(tensors, tensor_defs)
    except Exception as e:  # noqa: BLE001 — report every trace-time failure
        return {"kernel": name, "status": "ERROR",
                "detail": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}
    return {"kernel": name, "status": "OK"}


def import_impl(impl_path, module_name="smoke_impl_under_check"):
    """Import the impl file as a module; sibling imports resolvable.

    Bytecode caching is disabled so the check never leaves __pycache__
    next to the impl file.
    """
    sys.path.insert(0, str(impl_path.parent))
    old_dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, impl_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dwb
        try:
            sys.path.remove(str(impl_path.parent))
        except ValueError:
            pass
    return module


def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("impl", help="path to the impl .py file to smoke-check")
    ap.add_argument("--shapes", default=None,
                    help="JSON input specs for kernels with non-static shapes")
    ap.add_argument("--workdir", default=None,
                    help="scratch dir for pypto's output/ scratch (default: "
                         "./_debug/smoke_trace under the current working "
                         "directory); cleaned up after the run. NOTE: "
                         "tempfile.TemporaryDirectory() is forbidden here — "
                         "it lands in the system temporary directory and "
                         "triggers sandbox prompts.")
    return ap.parse_args()


def _load_shapes_cfg(raw) -> dict:
    """Parse --shapes JSON; exits via ValueError-free path on bad input."""
    return json.loads(raw) if raw else {}


def _ast_precheck(impl: Path) -> bool:
    """Syntax precheck. False (error already printed) on SyntaxError."""
    try:
        ast.parse(impl.read_text(encoding="utf-8"), filename=str(impl))
    except SyntaxError as e:
        _log.error(f"[SMOKE ERROR] stage=ast: SyntaxError: {e}")
        return False
    _log.info("[SMOKE] stage=ast: OK")
    return True


def _prepare_workdir(workdir_arg) -> Path:
    """Create a fresh CWD-local scratch directory to avoid sandbox prompts."""
    workdir = Path(workdir_arg).resolve() if workdir_arg else (
        Path.cwd() / "_debug" / "smoke_trace")
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _cleanup_workdir(workdir: Path, old_cwd: str) -> None:
    os.chdir(old_cwd)
    shutil.rmtree(workdir, ignore_errors=True)
    try:
        workdir.parent.rmdir()  # remove _debug too, but only if empty
    except OSError:
        pass


def _trace_kernels(module, shapes_cfg: dict) -> tuple:
    """Trace every module-level jit kernel. Returns (n_errors, n_skips)."""
    n_errors = 0
    n_skips = 0
    kernels = find_jit_kernels(module)
    if not kernels:
        _log.info("[SMOKE] no @pypto.frontend.jit kernel found at module "
              "level; import+AST only")
    for name, wrapper in kernels:
        res = trace_one(name, wrapper, shapes_cfg)
        if res["status"] == "OK":
            _log.info(f"[SMOKE] stage=trace kernel={name}: TRACE OK")
        elif res["status"] == "SKIP":
            n_skips += 1
            _log.info(f"[SMOKE SKIP] kernel={name}: {res['detail']}")
        else:
            n_errors += 1
            _log.error(f"[SMOKE ERROR] stage=trace kernel={name}: {res['detail']}")
    return n_errors, n_skips


def main() -> int:
    args = _parse_cli()
    impl = Path(args.impl).resolve()
    if not impl.is_file():
        _log.error(f"[SMOKE ERROR] file not found: {impl}")
        return 2
    try:
        shapes_cfg = _load_shapes_cfg(args.shapes)
    except json.JSONDecodeError as e:
        _log.error(f"[SMOKE ERROR] --shapes is not valid JSON: {e}")
        return 2

    # ---- 1. AST-level precheck (syntax) ----
    if not _ast_precheck(impl):
        _log.error(f"SMOKE RESULT: FAIL errors=1 file={impl}")
        return 1

    # ---- 2+3. import + trace, inside a throwaway cwd so pypto's output/
    # scratch dir lands there instead of polluting the caller's cwd ----
    workdir = _prepare_workdir(args.workdir)
    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        try:
            module = import_impl(impl)
        except Exception:  # noqa: BLE001
            _log.error(f"[SMOKE ERROR] stage=import: {traceback.format_exc(limit=3)}")
            _log.error(f"SMOKE RESULT: FAIL errors=1 file={impl}")
            return 1
        _log.info("[SMOKE] stage=import: OK")
        n_errors, n_skips = _trace_kernels(module, shapes_cfg)
    finally:
        _cleanup_workdir(workdir, old_cwd)

    if n_errors:
        _log.error(f"SMOKE RESULT: FAIL errors={n_errors} skips={n_skips} file={impl}")
        return 1
    note = " (trace skipped for some kernels: import+AST only)" if n_skips else ""
    _log.info(f"SMOKE RESULT: PASS skips={n_skips}{note} file={impl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
