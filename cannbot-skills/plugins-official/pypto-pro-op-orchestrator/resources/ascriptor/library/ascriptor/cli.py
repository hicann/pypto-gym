# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The ``ascriptor`` command line.

M1 ships ``devices``, ``backends``, ``ops``, ``legacy``, ``dump-ir`` and ``check``; M2 lets ``dump-ir`` /
``check`` compile a DSL file (``kernel.py::name``); M4 adds ``dump-ir --after PASS [--explain]``,
``explain --op N`` and ``sim`` (the pipe-level simulator); M5 adds ``compile`` (a backend artifact
directory) and ``run`` (a golden case through a launcher; the device ones run on the box itself).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from . import __version__
from . import backends as _backends
from . import devices as _devices


def _cmd_devices(_: argparse.Namespace) -> int:
    for name in _devices.available():
        p = _devices.load(name)
        print(f"{p.device_type:6s} family={p.family} arch={p.arch} cores={p.cube_cores}c/{p.vec_cores}v "
              f"ub={p.capacities_kb['ub']}KB l0c={p.capacities_kb['l0c']}KB")
    print("aliases:", ", ".join(f"{k}->{v}" for k, v in _devices.FACADE_ALIASES.items()))
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    """What this machine will actually run with, before a kernel is written against it.

    Every line answers a question that otherwise costs a failed run to ask: which interpreter, which
    ascriptor (the imported file, not the metadata beside it), and whether the device launchers will
    work here at all. The card's core count is printed because its absence is a hang rather than an
    error - see `run_pypto`.
    """
    from .runtime.board import Board, BoardError, config_file

    print(f"python     {sys.version.split()[0]}  {sys.executable}")
    print(f"ascriptor  {__version__}  {__file__.rsplit('/', 1)[0]}")
    for name in ("torch", "numpy", "pypto_pro"):
        module = sys.modules.get(name) or _optional(name)
        print(f"{name:10s} {getattr(module, '__version__', 'installed') if module else '-'}")
    print(f"boards     {config_file()}")
    try:
        board = Board.local()
    except BoardError as exc:
        # Print the reason, not a summary of it. "no entry says local", "ASCRIPTOR_BOARD names a
        # key this file does not have" and "there is no config at all" have the same consequence
        # and different repairs, and a workstation alias carried onto a box hits the middle one.
        print("this box   not a board")
        for line in textwrap.wrap(str(exc), 86):
            print(f"           {line}")
        print("           board and pypto refuse here; sim and pipesim do not.")
        return 0
    cores = board.cfg.get("cube_cores")
    print(f"this box   {board.name}  card={board.cfg.get('visible_devices')}  workspace={board.cfg['workspace']}")
    print(f"cores      {cores} AIC / {cores * 2 if cores else '?'} AIV"
          + ("" if cores else "   MISSING: pypto refuses; a fallback launch would deadlock"))
    return 0


def _optional(name: str):
    import importlib

    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001 - absence is the answer, not an error
        return None


def _cmd_backends(_: argparse.Namespace) -> int:
    found = _backends.discover()
    if not found:
        print("no backends registered (built-ins land with M4-M7)")
        return 0
    for name, backend in sorted(found.items()):
        caps = backend.capabilities()
        print(f"{name}: kinds={sorted(caps.function_kinds)} ops={len(caps.opcodes)} devices={sorted(caps.devices)}")
    return 0


def _cmd_ops(args: argparse.Namespace) -> int:
    from .ir import REGISTRY

    if args.json:
        print(REGISTRY.to_json())
        return 0
    for ns, specs in REGISTRY.namespaces().items():
        if args.namespace and ns != args.namespace:
            continue
        print(f"== {ns} ({len(specs)})")
        for s in specs:
            operands = ", ".join(f"{o.name}:{o.pattern}" + ("..." if o.variadic else "") for o in s.operands)
            results = ", ".join(r.pattern for r in s.results)
            head = f"  {s.name}({operands})" + (f" -> {results}" if results else "")
            tags = [s.level] + ([s.side] if s.side != "any" else []) + ([s.pipe] if s.pipe else [])
            if s.legacy:
                tags.append("was " + "/".join(s.legacy))
            print(f"{head}  [{' '.join(tags)}]")
            if args.verbose:
                print(f"      {s.doc}")
                if s.attrs:
                    print("      attrs: " + ", ".join(a.name + ("*" if a.required else "") + ":" + a.type for a in s.attrs))
    if args.namespace is None and not args.verbose:
        print(f"{len(REGISTRY)} ops; {len(REGISTRY.legacy_names())} old names mapped, {len(REGISTRY.retired)} retired")
    return 0


def _cmd_legacy(args: argparse.Namespace) -> int:
    from .ir import REGISTRY

    rc = 0
    for old in args.names:
        new = REGISTRY.successor(old)
        if new is not None:
            print(f"{old} -> {new}")
        elif old in REGISTRY.retired:
            print(f"{old} -> (retired) {REGISTRY.retired[old]}")
        else:
            print(f"{old}: unknown old instruction name", file=sys.stderr)
            rc = 1
    return rc


def _load_module(spec: str):
    """A module from a ``.ascrip`` file, a ``.json`` file, or ``kernel.py::kernel_name`` (compiled)."""
    from .ir import from_json, parse_module

    if "::" in spec:
        path, name = spec.split("::", 1)
        import importlib.util

        mod_spec = importlib.util.spec_from_file_location(Path(path).stem, path)
        if mod_spec is None or mod_spec.loader is None:
            raise SystemExit(f"cannot import {path}")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        kernel = getattr(mod, name, None)
        if kernel is None or not hasattr(kernel, "ir"):
            raise SystemExit(f"{path} has no kernel named {name}")
        return kernel.ir()
    text = Path(spec).read_text(encoding="utf-8")
    if spec.endswith(".json"):
        return from_json(json.loads(text))
    return parse_module(text)


def _lower(m, stop_after: str | None):
    """Run the pass pipeline up to ``stop_after`` (``all`` = the whole pipeline); returns (module, manager)."""
    from .passes import PIPELINE, PassManager

    pm = PassManager(PIPELINE)
    stop = None if stop_after in (None, "all") else stop_after
    if stop is not None and stop not in {p.name for p in PIPELINE}:
        raise SystemExit(f"unknown pass {stop!r}; passes: {', '.join(p.name for p in PIPELINE)}")
    return pm.run(m, stop_after=stop), pm


def _print_lints(m, lowered_ir: bool = False) -> None:
    """The hardware lints (D-051) on stderr: the surface ones before the module is lowered, the DMA ones after."""
    from .ir.lint import REPORTED, format_lints, lint, lint_lowered

    text = format_lints(lint_lowered(m) if lowered_ir else lint(m), seen=REPORTED)
    if text:
        print(text, file=sys.stderr)


def _print_explanations(pm, op_id: int | None) -> None:
    for e in pm.explanations(op_id):
        ops = e.get("ops")
        where = f"#{e['op']}" if "op" in e else (f"#{'/#'.join(str(o) for o in ops)}" if ops else "")
        print(f"[{e['pass']}] {where} {e['message']}")


def _cmd_dump_ir(args: argparse.Namespace) -> int:
    from .ir import print_module, to_json

    m = _load_module(args.file)
    pm = None
    if args.after:
        m, pm = _lower(m, args.after)
    if args.explain and pm is not None:
        _print_explanations(pm, None)
        print()
    if args.json:
        print(json.dumps(to_json(m), indent=1))
        return 0
    source = None
    if args.source:
        source = Path(args.source).read_text(encoding="utf-8")
    elif args.interleave and m.attrs.get("source"):
        candidate = Path(str(m.attrs["source"]))
        if candidate.is_file():
            source = candidate.read_text(encoding="utf-8")
    sys.stdout.write(print_module(m, source=source))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    from .ir import verify

    m = _load_module(args.file)
    diags = verify(m)
    lints: list = []
    if not any(d.severity == "error" for d in diags):
        from .ir.lint import format_lints, lint

        lints = lint(m)
    for d in diags:
        print(str(d), file=sys.stderr)
    if lints:
        print(format_lints(lints), file=sys.stderr)
    errors = [d for d in diags if d.severity == "error"]
    print(f"{args.file}: {len(errors)} error(s), {len(diags) + len(lints) - len(errors)} warning(s), "
          f"{sum(1 for _ in m.walk())} ops")
    return 1 if errors else 0


def _cmd_explain(args: argparse.Namespace) -> int:
    """Everything the compiler decided about one op: its origin chain and every pass note that mentions it."""
    m = _load_module(args.file)
    lowered, pm = _lower(m, "all")
    found = [op for op in lowered.walk() if op.id == args.op]
    if not found:
        found = [op for op in m.walk() if op.id == args.op]
        if not found:
            print(f"no op #{args.op}", file=sys.stderr)
            return 1
        print(f"#{args.op} does not survive lowering; as compiled:")
    from .ir.printer import format_op_line

    for op in found:
        print(format_op_line(op))
        if op.loc:
            print(f"  at {op.loc}")
        for o in op.origin:
            print(f"  origin: {o.pass_name} {o.kind}" + (f" from #{','.join(str(i) for i in o.from_ids)}" if o.from_ids else "")
                  + (f": {o.note}" if o.note else ""))
    _print_explanations(pm, args.op)
    return 0


def _cmd_sim(args: argparse.Namespace) -> int:
    """Lower a kernel and run it on the pipe-level simulator with the inputs of a recorded golden case."""
    from .backends.sim.pipesim import simulate
    from .testing import goldens

    m = _load_module(args.file)
    lowered, _ = _lower(m, "all")
    case = Path(args.case)
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    inputs = goldens.load_args(case, manifest)
    block_dim = manifest.get("launch", {}).get("block_dim")
    r = simulate(lowered, tuple(inputs), block_dim=block_dim, timeout=args.timeout, check_gm=args.check_gm)
    rep = r.report
    print(f"cycles {rep['cycles']}  tasks {rep['tasks']}  hazards {len(rep['hazards'])}" + ("  DEADLOCK" if rep["deadlock"] else ""))
    for h in rep["hazards"]:
        print("  " + h)
    for w in rep["warnings"]:
        print("  " + w)
    if rep["deadlock"]:
        print("  " + rep["deadlock"])
    if args.pipes:
        for name, st in sorted(rep["pipes"].items()):
            print(f"  {name:22s} busy {st['busy']:>9} end {st['end']:>9} tasks {st['tasks']:>6} util {st['utilisation']:.3f}")
    if args.trace:
        r.write_trace(args.trace)
        print(f"trace written to {args.trace}")
    return 1 if rep["hazards"] or rep["deadlock"] else 0


def _cmd_compile(args: argparse.Namespace) -> int:
    """Lower a kernel and print it through a backend into a directory of artifacts."""
    m = _load_module(args.file)
    _print_lints(m)
    lowered, _ = _lower(m, "all")
    _print_lints(lowered, lowered_ir=True)
    backend = _backends.get(args.backend)
    art = backend.compile(lowered, {"block_dim": args.block_dim} if args.block_dim is not None else None)
    out = Path(args.out or f"tmp/compile/{art.metadata.get('kernel', m.name)}")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in art.files.items():
        (out / name).write_bytes(data)
    print(f"{art.entry}: {len(art.files)} files in {out} ({art.metadata.get('ops')} ops)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a kernel on a recorded golden case through a launcher and compare the outputs (bit for bit, or
    within --rtol/--atol / the corpus replay tolerance when they differ)."""
    from .runtime import run_case

    if "::" not in args.file:
        raise SystemExit("run needs kernel.py::name")
    path, name = args.file.split("::", 1)
    import importlib.util

    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load kernel source: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, name)
    kw = {}
    if args.out_dir:
        kw["out_dir"] = args.out_dir
    if args.cann:
        kw["cann_path"] = args.cann
    if args.skip_build:
        kw["skip_build"] = True
    if args.rtol is not None or args.atol is not None:
        kw["tolerance"] = {"rtol": args.rtol or 0.0, "atol": args.atol or 0.0}
    notes: list[str] = []
    outputs, diffs = run_case(kernel, Path(args.case), launcher=args.launcher, notes=notes, **kw)
    for d in diffs + notes:
        print("  " + d)
    verdict = "DIFF" if diffs else ("OK (within tolerance)" if notes else "OK")
    print(f"{args.file} {Path(args.case).name} [{args.launcher}]: {verdict} ({len(outputs)} output(s))")
    return 0 if not diffs else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ascriptor", description="one statement, one instruction")
    parser.add_argument("--version", action="version", version=f"ascriptor {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="what this machine runs with, and whether it is a board").set_defaults(func=_cmd_doctor)
    sub.add_parser("devices", help="list device profiles").set_defaults(func=_cmd_devices)
    sub.add_parser("backends", help="list registered backends").set_defaults(func=_cmd_backends)
    p = sub.add_parser("ops", help="list the op registry")
    p.add_argument("namespace", nargs="?")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true", help="export the registry as JSON")
    p.set_defaults(func=_cmd_ops)
    p = sub.add_parser("legacy", help="map old easyasc instruction names to opcodes")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=_cmd_legacy)
    p = sub.add_parser("dump-ir", help="print a module: a .ascrip / .json file, or kernel.py::name to compile")
    p.add_argument("file")
    p.add_argument("--json", action="store_true", help="print the JSON form")
    p.add_argument("--interleave", action="store_true", help="show the DSL source line above each op (module 'source' attr)")
    p.add_argument("--source", help="DSL source file to interleave")
    p.add_argument("--after", metavar="PASS", help="run the pass pipeline up to PASS ('all' = whole pipeline) before printing")
    p.add_argument("--explain", action="store_true", help="with --after: print every pass decision first")
    p.set_defaults(func=_cmd_dump_ir)
    p = sub.add_parser("check", help="verify a module (.ascrip / .json / kernel.py::name)")
    p.add_argument("file")
    p.set_defaults(func=_cmd_check)
    p = sub.add_parser("compile", help="compile a kernel to a backend artifact directory")
    p.add_argument("file")
    p.add_argument("-o", "--out", help="output directory (default tmp/compile/<kernel>)")
    p.add_argument("--backend", default="cce")
    p.add_argument("--block-dim", type=int)
    p.set_defaults(func=_cmd_compile)
    p = sub.add_parser("run", help="run a kernel on a golden case through a launcher; device launchers run on the box itself")
    p.add_argument("file", help="kernel.py::name")
    p.add_argument("--case", required=True, help="golden case directory (manifest.json + inputs)")
    p.add_argument("--launcher", default="aclnn", choices=["sim", "pipesim", "aclnn", "cannsim", "board", "pypto"])
    p.add_argument("--out-dir")
    p.add_argument("--cann", help="CANN install path (default $ASCEND_HOME_PATH)")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--rtol", type=float, default=None, help="accept numerically equal outputs (with --atol)")
    p.add_argument("--atol", type=float, default=None)
    p.set_defaults(func=_cmd_run)
    p = sub.add_parser("explain", help="trace one op through the passes: origin chain and every decision that mentions it")
    p.add_argument("file")
    p.add_argument("--op", type=int, required=True, help="op id (#N in dump-ir)")
    p.set_defaults(func=_cmd_explain)
    p = sub.add_parser("sim", help="lower a kernel and run the pipe-level simulator on a recorded golden case")
    p.add_argument("file")
    p.add_argument("--case", required=True, help="golden case directory (manifest.json + inputs)")
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--check-gm", action="store_true", help="also report GM hazards")
    p.add_argument("--pipes", action="store_true", help="print per-pipe busy cycles and utilisation")
    p.add_argument("--trace", help="write a Chrome trace JSON")
    p.set_defaults(func=_cmd_sim)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
