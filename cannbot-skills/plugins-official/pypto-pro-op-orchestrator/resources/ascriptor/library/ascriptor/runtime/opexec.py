# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``OpExec``: run a kernel through a launcher and get torch tensors back (the old ``OpExec`` surface, M5 subset).

``OpExec(kernel, launcher=...)`` takes the ``@kernel`` object; calling it with the GM tensors in signature
order followed by the explicit scalars returns the outputs (one tensor, or a list in ``return`` order; a list
output is the list of its members, as ``run_kernel`` returns it).

Every launcher runs on the machine that calls it. There is no launcher that reaches out to another
box: a device launcher runs on the device's own machine, so a run that needs hardware is a script
copied to that box and started there, and the inputs, the reference and the comparison stay on one
side of the link (RFC-0012 §board).

Launchers:

* ``sim`` — the reference interpreter, in process.
* ``pipesim`` — the lowered pipeline under the event/hazard model, in process.
* ``aclnn`` (default) — the cce backend, an aclnn custom-op project built with the local CANN, the host
  harness run on the local card.
* ``cannsim`` — the same package and harness, run under ``cannsim record``.
* ``board`` — the same project, built and run here, on this machine's own card (:mod:`.board`).
* ``pypto`` — the generated PyPTO-Pro sources, run here against the installed PyPTO-Pro wheel.

``board`` and ``pypto`` need this machine to describe itself: an entry marked ``"local": true`` in
the config that ``ASCRIPTOR_BOARDS`` names. A workstation has none, and both refuse there.

The project is regenerated only when the emitted sources change (a hash stamp); the harness reads
shapes and scalars from files, so a new shape never rebuilds anything.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import backends as _backends
from .. import devices as _devices
from ..backends.base import Artifacts
from . import build as _build
from . import harness as _harness
from .project import HostSpec, generate_project

LAUNCHERS = ("sim", "pipesim", "aclnn", "cannsim", "board", "pypto")


def _torch():
    import torch

    return torch


def lower_kernel(kernel: Any):
    from ..ir import LOWERED, check
    from ..ir.lint import REPORTED, format_lints, lint, lint_lowered
    from ..passes import PIPELINE, PassManager

    module = kernel.ir()
    if module.ir == LOWERED:
        check(module)
        deep = format_lints(lint_lowered(module), seen=REPORTED)
        if deep:
            print(deep, file=sys.stderr)
        return module
    # the hardware lints (D-051), named before the first launch. `format_lints` owns the
    # channel prefix and the de-duplication, so this path and `ascriptor check` cannot drift.
    surface = format_lints(lint(module), seen=REPORTED)
    if surface:
        print(surface, file=sys.stderr)
    lowered = PassManager(PIPELINE).run(module)
    deep = format_lints(lint_lowered(lowered), seen=REPORTED)
    if deep:
        print(deep, file=sys.stderr)
    return lowered


def refuse_reserved_entry(kernel: Any, entry: str | None = None) -> None:
    """Refuse a kernel whose C entry the printers would respell (RFC-0007 §1), located at its definition.

    The custom-op build finds the kernel file and symbol under the op type's own spelling, so an entry printed
    as ``name_`` builds nothing that the launch can find. A DSL kernel fails with a frontend diagnostic, an
    imported one with a ProImportError at its Pro definition, before anything is printed or built."""
    from ..backends.cce import cpp
    from ..backends.cce.emit import c_ident

    name = getattr(kernel, "name", None)
    spelled = entry or name
    if not spelled or c_ident(spelled) == spelled:
        return
    printed = c_ident(spelled)
    reason = ("is reserved in the A5 build (a C++ keyword or reserved spelling, a runtime or <math.h> name, or a name "
              "the kernel translation unit declares)" if cpp.kernel_reserved(spelled) or spelled.startswith("__")
              else "is not a C identifier")
    source = f" (the op-type spelling of kernel {name!r})" if spelled != name else ""
    message = (f"the C entry {spelled!r}{source} {reason}: it would be printed as {printed!r}, which the custom-op "
               "build does not look for; rename the kernel")
    fn = getattr(kernel, "fn", None)
    if fn is not None:
        import ast
        import inspect
        import textwrap

        from ..frontend.compiler import _source_path
        from ..frontend.errors import E_RESERVED_NAME, CompileError

        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            ast.increment_lineno(tree, fn.__code__.co_firstlineno - 1)
            node = tree.body[0]
        except (OSError, TypeError, SyntaxError):  # no retrievable source: the file still locates it
            node = None
        raise CompileError(E_RESERVED_NAME, message, _source_path(fn), node)
    location = getattr(kernel, "location", None)
    if location is not None:
        from ..importers.pypto_pro import ProImportError

        raise ProImportError(message, location)
    raise ValueError(f"kernel {name!r}: {message}")


def compile_kernel(kernel: Any, backend: str = "cce", block_dim: int | None = None, entry: str | None = None,
                   bindings: Mapping[str, int] | None = None) -> Artifacts:
    """``entry``: the entry symbol / file base name when the launcher needs its own spelling (see ``aclnn_entry``).

    ``bindings``: a scalar valuation for the call. Only a backend whose tile shapes are
    compile-time needs it (pto_isa, RFC-0011 §3); the others ignore the option.
    """
    from ..backends.cce import CceBackend

    if backend in ("cce", "pto_isa"):  # the printers that spell the entry with c_ident
        refuse_reserved_entry(kernel, entry)
    lowered = lower_kernel(kernel)
    from .launch_config import launch_block_dim

    block_dim = launch_block_dim(lowered, block_dim, "compile_kernel", getattr(kernel, "location", None))
    if backend == "cce":
        be = CceBackend()
    elif backend == "pto_isa":
        # a built-in backend, imported directly: the entry-point group only lists what the
        # installed dist-info recorded, which lags a freshly added backend (RFC-0011 §7.9)
        from ..backends.pto_isa import PtoIsaBackend

        be = PtoIsaBackend()
    else:
        be = _backends.get(backend)
    options = {}
    if block_dim is not None:
        options["block_dim"] = block_dim
    if entry:
        options["entry"] = entry
    if bindings:
        options["bindings"] = dict(bindings)
    return be.compile(lowered, options or None)


def scalar_bindings(kernel: Any, args: tuple[Any, ...]) -> dict[str, int]:
    """The scalar valuation this call implies, for a backend that prints shapes at compile time.

    Returns ``{}`` rather than raising: a kernel with no scalar parameters, or one whose arguments
    do not bind, simply gets no specialisation and the backend refuses on its own terms.
    """
    try:
        from ..backends.sim.launch import bind_arguments
        from ..ir.types import ScalarType

        lowered = lower_kernel(kernel)
        bound = bind_arguments(lowered, args, seed_outputs=True)
        fn = next(f for f in lowered.functions if f.kind == "func")
        return {q.name: int(bound[q.name]) for q in fn.params
                if isinstance(q.type, ScalarType) and q.name in bound}
    except Exception:  # noqa: BLE001 - no valuation is a fact about the call, not an error
        return {}


def aclnn_entry(name: str) -> str:
    """The entry / file name the CANN custom-op build expects for kernel ``name``: its own snake form of the op
    type (``project.optype_snake(project.camel(name))``), which differs from ``name`` when a part starts with a
    digit (``matmul_mknk_2dgrid`` -> ``matmul_mknk2dgrid``)."""
    from . import project as _project

    return _project.optype_snake(_project.camel(name))


def artifacts_hash(a: Artifacts) -> str:
    h = hashlib.sha256()
    for name in sorted(a.files):
        h.update(name.encode())
        h.update(a.files[name])
    return h.hexdigest()


class OpExec:
    def __init__(self, kernel: Any, *, launcher: str = "aclnn", out_dir: str | Path | None = None,
                 cann_path: str | None = None, custom_op_path: str | Path | None = None, block_dim: int | None = None,
                 device: str | None = None, lock_path: str | Path | None = None,
                 timeout: float = 1800.0, skip_build: bool = False, backend: str = "cce",
                 bindings: Mapping[str, int] | None = None, seed_outputs: bool = False,
                 sync_mode: str | None = None, sim_processes: bool | None = None) -> None:
        self.seed_outputs = seed_outputs
        if launcher not in LAUNCHERS:
            raise ValueError(f"launcher must be one of {LAUNCHERS}, got {launcher!r}")
        if launcher != "pypto" and sync_mode is not None:
            raise ValueError("sync_mode options require launcher='pypto'")
        if sync_mode is not None and sync_mode not in ("manual", "auto_mutex"):
            raise ValueError("sync_mode must be 'manual' or 'auto_mutex'")
        self.sync_mode = sync_mode or "manual"   # D-260: every launcher owns its credits
        self.kernel = kernel
        self.launcher = launcher
        self.backend = backend  # which backend prints the artifact: cce (default) or pto_isa
        self.bindings = dict(bindings or {})  # scalar valuation, for a backend that prints shapes
        self._explicit_bindings = dict(self.bindings)
        self.block_dim = block_dim
        if getattr(kernel, "module", None) is not None:  # an imported kernel launches at its exported block_dim
            from .launch_config import launch_block_dim

            self.block_dim = launch_block_dim(kernel.module, block_dim, "OpExec", getattr(kernel, "location", None))
        self.timeout = timeout
        # `sim` and `pipesim` run every core group in its own process by default. Threads in one
        # process hand the GIL back and forth on every tiny torch op, which is why a block_dim>1
        # run can take minutes of wall clock at a few percent CPU. None leaves the simulator's
        # own default; True forces forked processes; False forces threads.
        self.sim_processes = sim_processes
        self.skip_build = skip_build
        self.device = device or getattr(kernel, "device", None) or "950"
        self.profile = _devices.load(self.device)
        self.name = getattr(kernel, "name", getattr(kernel, "__name__", "kernel"))
        self.out_dir = (Path(out_dir) if out_dir else Path("tmp") / "opexec" / self.name).resolve()  # absolute: builds cd elsewhere
        self.cann_path = cann_path or _build.os.environ.get("ASCEND_HOME_PATH")
        self.custom_op_path = (Path(custom_op_path) if custom_op_path else self.out_dir / "custom_op").resolve()
        self.lock_path = Path(lock_path) if lock_path else (Path(_build.os.environ["ASCRIPTOR_NPU_LOCK"])
                                                             if "ASCRIPTOR_NPU_LOCK" in _build.os.environ else self.out_dir / "npu.lock")
        self._artifacts: Artifacts | None = None
        self._spec: HostSpec | None = None
        self._vendor: Path | None = None
        self._harness: Path | None = None

    # ---------------------------------------------------------------- pieces

    @property
    def artifacts(self) -> Artifacts:
        if self._artifacts is None:
            if self.launcher == "pypto":
                # signature manifest only: the real compile happens per call, once the scalar
                # values are known (the pypto backend specialises per scalar valuation)
                from ..backends.pypto_pro import module_manifest

                meta = module_manifest(lower_kernel(self.kernel), block_dim=self.block_dim)
                self._artifacts = Artifacts(files={}, entry=meta["entry"], metadata=meta)
            else:
                entry = aclnn_entry(self.name) if self.launcher in ("aclnn", "cannsim", "board") else None
                # ``backend`` defaults to cce; pto_isa emits the same aclnn-shaped artifact
                # (entry, manifest) and rides this chain unchanged (RFC-0011 §7.9)
                self._artifacts = compile_kernel(self.kernel, self.backend, self.block_dim, entry=entry,
                                                 bindings=self.bindings)
            self._spec = HostSpec(self._artifacts.metadata)
        return self._artifacts

    def pypto_artifacts(self, scalars: dict[str, Any], max_block_dim: int | None = None,
                        lists: dict[str, list[list[int]]] | None = None,
                        shapes: dict[str, list[int]] | None = None) -> Artifacts:
        """The generated pypto sources, specialised to this call's scalar values. ``max_block_dim``
        is the launching card's own core count when it is smaller than the device profile's;
        ``lists`` carries a GMList parameter's member shapes, which pl has no runtime spelling
        for and which therefore join the specialisation; ``shapes`` carries the tensor shapes the
        call will pass, which the printer declares instead of ``pl.DYNAMIC`` so pto can bake the
        GM strides rather than carry them into every TASSIGN (D-160)."""
        from ..backends.pypto_pro import emit_module

        bindings = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in scalars.items()}
        return emit_module(lower_kernel(self.kernel), block_dim=self.block_dim, bindings=bindings,
                           max_block_dim=max_block_dim, lists=lists, shapes=shapes,
                           sync_mode=self.sync_mode)

    @property
    def spec(self) -> HostSpec:
        _ = self.artifacts
        if self._spec is None:
            raise _build.BuildError("compiler artifacts have no host specification")
        return self._spec

    @property
    def project_dir(self) -> Path:
        return self.out_dir / "project"

    @property
    def test_dir(self) -> Path:
        return self.out_dir / "aclnn_test"

    def write_sources(self) -> bool:
        """Generate the project + harness; True when the sources changed since the last build."""
        if not self.cann_path:
            raise _build.BuildError("cann_path is not set and ASCEND_HOME_PATH is empty")
        stamp = self.out_dir / ".source_hash"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "artifacts").mkdir(exist_ok=True)
        for name, data in self.artifacts.files.items():
            (self.out_dir / "artifacts" / name).write_bytes(data)
        generate_project(self.artifacts, self.project_dir, cann_path=self.cann_path, compute_unit=self.profile.compile_unit,
                         block_dim=self.block_dim)
        _harness.write_harness(self.spec, self.test_dir, seed_outputs=self.seed_outputs)
        # this stamp gates the BOX's rebuild (board.run_script compares it with .built), so it
        # has to cover the harness as well as the kernel artifacts: a change to test.cpp alone
        # used to leave the box running its previously compiled executable, silently.
        h = hashlib.sha256(artifacts_hash(self.artifacts).encode())
        for f in ("test.cpp", "ascrip_harness.h"):
            h.update((self.test_dir / f).read_bytes())
        digest = h.hexdigest()
        changed = not stamp.is_file() or stamp.read_text().strip() != digest
        if changed:
            stamp.write_text(digest)
        return changed

    def build(self, force: bool = False) -> None:
        changed = self.write_sources()
        cann_path = self.cann_path
        if cann_path is None:
            raise _build.BuildError("cann_path is not set and ASCEND_HOME_PATH is empty")
        vendor = self.custom_op_path / "vendors" / "customize"
        exe = self.test_dir / "test_aclnnop"
        if self.skip_build and vendor.is_dir() and exe.is_file():
            self._vendor, self._harness = vendor, exe
            return
        if force or changed or not vendor.is_dir():
            import shutil

            shutil.rmtree(self.project_dir / "build_out", ignore_errors=True)
            vendor = _build.build_custom_op(self.project_dir, cann_path, self.custom_op_path, timeout=self.timeout * 2)
            exe = _build.build_harness(self.test_dir, cann_path, vendor, timeout=self.timeout)
        elif not exe.is_file():
            exe = _build.build_harness(self.test_dir, cann_path, vendor, timeout=self.timeout)
        self._vendor, self._harness = vendor, exe

    # ---------------------------------------------------------------- running

    def bind(self, args: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Tensors by parameter name and scalars by name (shape symbols derived like the sim launcher)."""
        from ..backends.sim.launch import bind_arguments

        lowered = lower_kernel(self.kernel)
        bound = bind_arguments(lowered, args, seed_outputs=True)
        if self.backend == "pto_isa":
            from ..ir.types import ScalarType

            inferred = {p.name: bound[p.name] for fn in lowered.functions if fn.kind == "func"
                        for p in fn.params if isinstance(p.type, ScalarType) and p.type.dtype.is_integer
                        and p.name in bound}
            for name, value in self._explicit_bindings.items():
                if name in inferred and inferred[name] != value:
                    raise ValueError(f"PTO scalar binding {name!r} disagrees with the current call")
            current = {**inferred, **self._explicit_bindings}
            if current != self.bindings:
                self.bindings = current
                self._artifacts = self._spec = None
                self._vendor = self._harness = None
        spec = self.spec
        tensors = {p["ir_name"]: bound[p["ir_name"]] for p in spec.tensors + spec.lists}
        scalars = {p["ir_name"]: bound[p["ir_name"]] for p in spec.scalars}
        return tensors, scalars

    def __call__(self, *args: Any) -> Any:
        torch = _torch()
        if self.launcher == "sim":
            from ..backends.sim.launch import run_kernel

            # `timeout` was not being forwarded here, so a slow kernel hit the interpreter's own
            # 120 s default and the message telling the caller to raise the limit named something
            # the caller had no way to reach.
            return run_kernel(self.kernel, *args, block_dim=self.block_dim, timeout=self.timeout,
                              seed_outputs=self.seed_outputs, processes=self.sim_processes)
        if self.launcher == "pipesim":
            # The same three checks the unit runner makes of this stage, so that reaching it
            # through OpExec cannot report a pass the runner would have called a failure.
            from ..backends.sim.pipesim import simulate
            from ..passes.autosync import check_balance

            lowered = lower_kernel(self.kernel)
            balance = check_balance(lowered)
            if balance:
                raise ValueError(f"{self.name}: event balance failed: {balance}")
            run = simulate(lowered, args, block_dim=self.block_dim, timeout=self.timeout,
                           seed_outputs=self.seed_outputs, check_gm=True,
                           processes=self.sim_processes)
            if run.hazards or run.report.get("deadlock"):
                raise ValueError(f"{self.name}: pipe simulation hazards/deadlock: {run.report}")
            return run.outputs[0] if len(run.outputs) == 1 else run.outputs
        tensors, scalars = self.bind(args)
        spec = self.spec
        by_c = {p["name"]: tensors[p["ir_name"]] for p in spec.tensors + spec.lists}
        sc_c = {p["name"]: scalars[p["ir_name"]] for p in spec.scalars}
        if self.launcher == "board":
            from .board import Board

            outputs = Board.local().run_opexec(self, by_c, sc_c)
        elif self.launcher == "pypto":
            from .board import Board

            outputs = Board.local().run_pypto(self, by_c, sc_c)
        else:
            self.build()
            if self._vendor is None or self.cann_path is None:
                raise _build.BuildError("local build did not produce a runnable harness")
            _harness.write_args(spec, self.test_dir, _as_numpy(by_c), sc_c, seed_outputs=self.seed_outputs)
            _build.run_harness(self.test_dir, self.cann_path, self._vendor, mode="npu" if self.launcher == "aclnn" else "cannsim",
                               chipset=self.profile.debug_chipset, lock_path=self.lock_path, timeout=self.timeout)
            outputs = _harness.read_outputs(self.test_dir, spec, by_c)

        def tensor(data: bytes, ref: Any) -> Any:
            return torch.frombuffer(bytearray(data), dtype=ref.dtype).reshape(ref.shape).clone()

        result = []
        for p in spec.outputs:
            ref = by_c[p["name"]]
            if p["kind"] == "list":  # the members, in the caller's shapes
                result.append([tensor(b, m) for b, m in zip(outputs[p["name"]], ref, strict=True)])
            else:
                result.append(tensor(outputs[p["name"]], ref))
        return result[0] if len(result) == 1 else result


def _as_numpy(tensors: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in tensors.items():
        if isinstance(v, (list, tuple)):  # a gmlist argument: its members
            out[k] = [_as_numpy({"m": m})["m"] for m in v]
        elif hasattr(v, "detach"):
            t = v.detach().cpu().contiguous()
            if t.dtype in _no_numpy_dtypes():  # bf16 / fp8: bytes and logical shape, resolved only on execution
                out[k] = (tuple(t.shape), t.view(_torch().uint8).numpy())
            else:
                out[k] = t.numpy()
        else:
            out[k] = v
    return out


def _no_numpy_dtypes() -> set:
    torch = _torch()
    out = {torch.bfloat16}
    for name in ("float8_e4m3fn", "float8_e5m2", "complex32"):
        if hasattr(torch, name):
            out.add(getattr(torch, name))
    return out


def run_case(kernel: Any, case: Path, *, launcher: str = "aclnn", tolerance: dict | None = None,
             notes: list[str] | None = None, **kw: Any) -> tuple[list[Any], list[str]]:
    """Run a recorded golden case through ``launcher`` and compare with the golden; returns (outputs, diffs).

    The comparison is bit for bit first. An explicitly supplied ``tolerance`` (``{"rtol", "atol"}``)
    permits numerical comparison, and ``notes`` receives the bitwise differences that were accepted,
    so a report can still tell "bit-exact" from "within tolerance" (fp32 accumulation order on the cube,
    atomics landing in another order, ...).
    """
    from ..testing import goldens

    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    args = goldens.load_args(case, manifest)
    block_dim = manifest.get("launch", {}).get("block_dim")
    kw.setdefault("bindings", scalar_bindings(kernel, args))
    ex = OpExec(kernel, launcher=launcher, block_dim=block_dim, **kw)
    out = ex(*args)
    outputs = goldens.flatten_outputs(out)  # a list output's members one by one, as the golden records them
    diffs = goldens.compare_outputs(outputs, case, manifest)
    if not diffs:
        return outputs, diffs
    if tolerance:
        failing = goldens.compare_outputs(outputs, case, manifest, tolerance=tolerance)
        tol = {k: tolerance[k] for k in ("rtol", "atol") if k in tolerance}
        per = tolerance.get("outputs") or {}
        failed = {d.split(":")[0] for d in failing}

        def _accepted_by(d: str) -> str:
            # "output <i>: ..." - name what actually accepted this one, which is not always a bound:
            # an index output is checked by what it DENOTES (D-215), and "within tolerance {}" would
            # have reported that as a bound of nothing
            i = d.split(":")[0].split()[-1]
            entry = per.get(i) or per.get(int(i)) or tol
            spec = entry.get("index_gather") if isinstance(entry, dict) else None
            if spec is not None:
                return ("every differing entry denotes the recorded value beside it "
                        f"(board_tolerance index_gather {spec})")
            return f"within tolerance {({k: v for k, v in entry.items() if k in ('rtol', 'atol')} if isinstance(entry, dict) else tol)}"

        if notes is not None:  # the bitwise differences the comparison accepted, output by output
            notes.extend(f"{_accepted_by(d)}: {d}" for d in diffs if d.split(":")[0] not in failed)
        return outputs, failing  # only the outputs that fail the tolerance are differences
    return outputs, diffs


__all__ = ["OpExec", "LAUNCHERS", "compile_kernel", "lower_kernel", "artifacts_hash", "run_case", "aclnn_entry"]
