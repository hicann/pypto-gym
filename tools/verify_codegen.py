#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Drive a @pl.jit kernel to codegen without an NPU, and say where it stopped.

``_TileJitKernel.__call__`` splits into ``_ensure_compiled(args)`` and then the
launch, so compilation can be driven with **CPU** tensors. Parse errors, wrong
argument order, bad tile shapes, address problems and IR failures all surface
here, needing neither a device nor the device lock.

The point of this script is to separate two failures that both raise
``RuntimeError``:

* the kernel is wrong -- parse or IR or codegen never produced a source file;
* the toolchain cannot build for this target -- source was generated fine and
  the backend compiler rejected it.

It decides between them by looking for the generated ``kernel.cpp`` and for the
signature of a toolchain that does not know the target architecture.

Usage
-----
    python tools/verify_codegen.py custom/<op>/test_<op>.py --op <op>
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
import time
import traceback
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)


class VerifyError(RuntimeError):
    """A kernel this script cannot inspect -- reported, then exit 1.

    Raised instead of ``SystemExit`` so the helpers stay callable from another
    module: ``SystemExit`` derives from ``BaseException``, so a caller's
    ``except Exception`` does not see it and the process dies inside what was
    meant to be a recoverable call. Only ``__main__`` turns this into an exit.
    """


# Signature of a backend that does not recognise the A5 target: the aicore
# attribute is a compiler builtin on the device path, and its absence means the
# arch flag was not honoured. Not a kernel defect.
_TOOLCHAIN_MARKERS = (
    "unknown type name '__aicore__'",
    "unknown type name '__gm__'",
)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


# Annotations stringify with the short dtype spelling -- a 3-D dynamic fp16
# tensor renders roughly as three nested DYNAMIC dims followed by "fp16" --
# so match those rather than the DT_* constant names used in source.
_PL_TO_TORCH = {
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp32": torch.float32, "float32": torch.float32,
    "int8": torch.int8, "uint8": torch.uint8,
    "int32": torch.int32, "int64": torch.int64,
}
# Spellings are not uniform: fp16 stringifies as "fp16" but bf16 as "bfloat16".
# Longest alternatives first so "bfloat16" is not shadowed by a shorter match.
_DTYPE_RE = re.compile(
    r"\b(bfloat16|float16|float32|uint8|int64|int32|int8|fp16|fp32|bf16)\b")


def kernel_function(kernel):
    """The decorated function, which is where the annotations live.

    ``signature(kernel)`` gives the launch wrapper's ``(*args, **kwargs)``, not
    the kernel's parameters.
    """
    for attr in ("_func", "fn", "_fn", "func", "__wrapped__"):
        fn = getattr(kernel, attr, None)
        if fn is not None and getattr(fn, "__annotations__", None):
            return fn
    raise VerifyError("cannot find the decorated function behind the kernel object")


def jit_internal(kernel, name: str):
    """A ``@pl.jit`` internal, resolved by name.

    ``_TileJitKernel`` splits ``__call__`` into ``_normalize_launch_args`` and
    ``_ensure_compiled``; driving the second without the launch is the whole
    point of this script, and the public surface (``__call__``) compiles *and*
    launches, so it needs a device. Resolving by name -- the same way
    ``kernel_function`` above finds the decorated function -- keeps the reach
    into framework internals explicit, and turns a rename upstream into a named
    failure here instead of an AttributeError deep inside the call.
    """
    hook = getattr(kernel, name, None)
    if hook is None:
        raise VerifyError(
            f"this pypto_pro build's kernel object has no {name!r}; the launch "
            f"path has been renamed and this script needs updating")
    return hook


def make_args(kernel, rows: int, cols: int) -> list:
    """CPU stand-ins built from the kernel's declared annotations.

    Only rank and dtype matter for compilation; extents are free. The raw
    annotation is printed for any parameter that cannot be read, so a parsing
    failure is diagnosable without another round trip.
    """
    fn = kernel_function(kernel)
    args = []
    for name, ann_obj in fn.__annotations__.items():
        if name == "return":
            continue
        ann = str(ann_obj)
        m = _DTYPE_RE.search(ann)
        dtype = _PL_TO_TORCH[m.group(1)] if m else None
        if dtype is None:
            raise VerifyError(f"parameter {name!r}: no dtype in annotation {ann!r}")
        if "Tensor" not in ann and "Ptr" not in ann:
            args.append(1e-6 if dtype.is_floating_point else 1)
            continue
        # Shape entries are whatever sits in the first bracketed group; count
        # them for the rank and look for a literal leading 1.
        inner = re.search(r"\[\s*([^\[\]]*?)\s*\]", ann)
        entries = [e.strip() for e in inner.group(1).split(",")] if inner else []
        entries = [e for e in entries if e]
        rank = len(entries) or 2
        if rank == 1:
            shape = (cols,)
        elif entries and entries[0] == "1":
            shape = (1, cols)
        else:
            shape = (rows, cols) if rank == 2 else (1,) * (rank - 2) + (rows, cols)
        args.append(torch.zeros(shape, dtype=dtype))
        LOGGER.info("   %-12s %-52s -> %s %s", name, ann[:52], tuple(shape), dtype)
    return args


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--op", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=128)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.path.parent))
    LOGGER.info("== loading %s", args.path)
    try:
        mod = load_module(args.path)
    except Exception:
        LOGGER.info("RESULT: FAIL_IMPORT -- module did not import")
        traceback.print_exc()
        return 2
    LOGGER.info("   import ok")

    kernel = getattr(mod, f"{args.op}_kernel", None)
    if kernel is None:
        LOGGER.info("RESULT: FAIL_IMPORT -- no %s_kernel in module", args.op)
        return 2

    for entry in (f"{args.op}_wrapper", args.op):
        if not callable(getattr(mod, entry, None)):
            LOGGER.info("RESULT: FAIL_CONTRACT -- missing entry point %s", entry)
            return 2
    LOGGER.info("   entry points %s_wrapper / %s present", args.op, args.op)

    LOGGER.info("== compiling with CPU tensors (%sx%s)", args.rows, args.cols)
    # Built outside the guard below: that except is for the framework rejecting
    # the arguments (FAIL_SIGNATURE), not for this script failing to read the
    # annotations. A VerifyError from make_args is now an ordinary Exception, so
    # leaving the call inside would relabel "I cannot parse this kernel" as
    # "the kernel's signature is wrong" and return 3 instead of exiting 1.
    cpu_args = tuple(make_args(kernel, args.rows, args.cols))
    # Both hooks are resolved outside their guards for the same reason: a
    # renamed internal is a defect in this script, not in the kernel, and must
    # not be reported as FAIL_SIGNATURE or as a backend failure.
    normalize = jit_internal(kernel, "_normalize_launch_args")
    ensure_compiled = jit_internal(kernel, "_ensure_compiled")
    try:
        kargs = normalize(cpu_args, {})
    except Exception:
        LOGGER.info("RESULT: FAIL_SIGNATURE -- launch arguments rejected")
        traceback.print_exc()
        return 3
    LOGGER.info("   normalize ok, %d args", len(kargs))

    # The backend's stderr is reported through logging, not raised, so capture
    # it -- otherwise a toolchain failure is indistinguishable from a kernel bug.
    import io

    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    # Stamped before the build so the artifact scan below can prove an emitted
    # source belongs to *this* invocation.
    started_at = time.time()

    err_text = ""
    try:
        ensure_compiled(kargs)
        LOGGER.info("RESULT: PASS_COMPILE -- kernel built to a loadable library")
        return 0
    except Exception as exc:
        err_text = f"{type(exc).__name__}: {exc}"
    finally:
        logging.getLogger().removeHandler(handler)
    err_text += "\n" + captured.getvalue()

    # Did codegen actually emit a source file *in this run*?
    #
    # Taking the newest build/**/kernel.cpp unconditionally let a previous run's
    # artifact stand in for this one: a kernel that failed before codegen would
    # find an older source plus its toolchain-marker log and be reported as
    # PASS_CODEGEN_TOOLCHAIN_BLOCKED, i.e. exit 0. An artifact is only evidence
    # about this kernel if this invocation wrote it, so require that.
    all_generated = sorted(Path("build").rglob("kernel.cpp")) if Path("build").is_dir() else []
    generated = [p for p in all_generated if p.stat().st_mtime >= started_at]
    stale = len(all_generated) - len(generated)
    newest = max(generated, key=lambda p: p.stat().st_mtime) if generated else None

    if newest is None:
        if stale:
            LOGGER.info(
                "   ignored %d pre-existing kernel.cpp under build/ -- none was written "
                "by this run", stale)
        LOGGER.info("RESULT: FAIL_CODEGEN -- no source emitted\n   %s", err_text)
        return 4

    text = newest.read_text(encoding="utf-8", errors="replace")
    n_lines = text.count("\n")
    LOGGER.info("   codegen emitted %s (%d lines)", newest, n_lines)

    # A file existing is not the same as the kernel body being translated:
    # report the emitted tile operations so an empty or signature-only source is
    # visible rather than being read as success.
    ops = re.findall(r"\b(T[A-Z][A-Za-z0-9_]*)\s*[<(]", text)
    if ops:
        counts = {}
        for o in ops:
            counts[o] = counts.get(o, 0) + 1
        top_ops = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
        summary = ", ".join(f"{k}x{v}" for k, v in top_ops)
        LOGGER.info("   emitted tile ops: %s", summary)
    else:
        LOGGER.info("   WARNING: no tile operations found in the emitted source")

    log = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in newest.parent.glob("*.log")
    )
    haystack = err_text + log
    if any(m in haystack for m in _TOOLCHAIN_MARKERS):
        LOGGER.info("RESULT: PASS_CODEGEN_TOOLCHAIN_BLOCKED")
        LOGGER.info("   Source generated cleanly; the backend compiler does not know")
        LOGGER.info("   this target. Everything the DSL can check has passed.")
        return 0

    LOGGER.info("RESULT: FAIL_BACKEND -- source generated but the compiler rejected it")
    LOGGER.info("   for a reason that is not the known toolchain fault:\n   %s", err_text[:900])
    return 5


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        sys.exit(main())
    except VerifyError as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)
