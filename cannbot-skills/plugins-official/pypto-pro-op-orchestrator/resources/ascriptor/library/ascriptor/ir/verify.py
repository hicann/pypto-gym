# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The verifier (RFC-0001 §11): structural, typing and semantic checks against the registry.

``verify(module)`` returns diagnostics; ``check(module)`` raises :class:`VerifyError` when any
is an error. Every backend calls ``check`` on entry; the pass manager calls it between passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import LOWERED, MODES, SURFACE, Block, FuncRef, Function, Ident, Literal, Module, Op, Value
from .registry import REGISTRY, AttrSpec, OpSpec, Registry, load_builtin_ops, match
from .saturation import SAT_BITS
from .fixpipe_rules import SPLIT_MODES, dual_mode_of, fixpipe_riders
from .sync_rules import raw_flag_error
from .fix_bounds import StaticMemory
from .scalar_math import division_error, vf_extremum_error
from .types import (
    BufType,
    CellType,
    EventType,
    FlagType,
    MaskType,
    MemType,
    Product,
    Ragged,
    RegType,
    ScalarType,
    Type,
    UnalignRegType,
    is_scalar_int,
    type_values,
)


@dataclass(frozen=True)
class Diagnostic:
    severity: str  # error | warning
    message: str
    function: str | None = None
    op_id: int | None = None
    loc: str | None = None
    #: ``trap`` — the interpreter executes it exactly and the hardware does not, so the run is
    #: wrong or faults. ``perf`` — both are correct and one is slower. They travel the same
    #: path and are read very differently, so the channel is part of the diagnostic rather
    #: than something the printer guesses. Everything the verifier raises is a trap.
    kind: str = "trap"
    #: the check that produced it. Identity for de-duplication: one source line expanded
    #: through N ``@vf`` instances is one finding, not N.
    rule: str | None = None

    def __str__(self) -> str:
        where = []
        if self.function:
            where.append(f"@{self.function}")
        if self.op_id is not None:
            where.append(f"#{self.op_id}")
        if self.loc:
            where.append(self.loc)
        prefix = (" ".join(where) + ": ") if where else ""
        return f"{prefix}{self.severity}: {self.message}"


class VerifyError(ValueError):
    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        errors = [d for d in diagnostics if d.severity == "error"]
        super().__init__(f"{len(errors)} error(s):\n" + "\n".join(str(d) for d in errors))


class _Scope:
    def __init__(self, parent: _Scope | None = None) -> None:
        self.parent = parent
        self.values: dict[str, Value] = {}

    def lookup(self, name: str) -> Value | None:
        s: _Scope | None = self
        while s is not None:
            if name in s.values:
                return s.values[name]
            s = s.parent
        return None


class Verifier:
    def __init__(self, module: Module, registry: Registry | None = None) -> None:
        self.m = module
        self.registry = registry or load_builtin_ops()
        self.diags: list[Diagnostic] = []
        self.level = module.level
        self.func: Function | None = None
        self.seen_ids: dict[int, str] = {}
        self.initialised_cells: set[str] = set()
        self.loop_depth = 0
        self.function_kinds = {f.name: f.kind for f in module.functions}
        self.profile = None

    # -- reporting ----------------------------------------------------------------------------

    def error(self, msg: str, op: Op | None = None) -> None:
        self._report("error", msg, op)

    def warn(self, msg: str, op: Op | None = None) -> None:
        self._report("warning", msg, op)

    def _report(self, severity: str, msg: str, op: Op | None) -> None:
        self.diags.append(Diagnostic(severity, msg, self.func.name if self.func else None,
                                     op.id if op else None, op.loc.chain[0] if op and op.loc else None))

    # -- module -------------------------------------------------------------------------------

    def run(self) -> list[Diagnostic]:
        m = self.m
        if m.ir not in (SURFACE, LOWERED):
            self.error(f"unknown ir version {m.ir!r}")
        if m.device is None:
            self.error("module has no device")
        else:
            try:
                from .. import devices

                self.profile = devices.load(m.device)
            except KeyError as exc:
                self.error(str(exc))
        mode = m.attrs.get("mode", "mix")
        mode_name = mode.name if isinstance(mode, Ident) else mode
        if mode_name not in MODES:
            self.error(f"bad mode {mode_name!r}")
        names = [f.name for f in m.functions]
        if len(names) != len(set(names)):
            self.error("duplicate function names")
        kernels = [f for f in m.functions if f.kind == "kernel"]
        if self.level == "surface" and len(kernels) != 1:
            self.error(f"a surface module has exactly one kernel, found {len(kernels)}")
        if self.level == "lowered" and kernels:
            self.error("a lowered module has no kernel; the kernel is split into per-side funcs")
        for f in m.functions:
            self.verify_function(f)
        return self.diags

    # -- functions ----------------------------------------------------------------------------

    def verify_function(self, f: Function) -> None:
        self.func = f
        self.memory = StaticMemory(f)
        self.initialised_cells = set()
        self.loop_depth = 0
        if f.kind not in ("kernel", "vf", "simt", "func"):
            self.error(f"bad function kind {f.kind!r}")
        if f.kind == "func" and self.level != "lowered":
            self.error("'func' is a lowered-only function kind")
        scope = _Scope()
        seen: set[str] = set()
        for p in f.params:
            if p.name in seen:
                self.error(f"duplicate parameter %{p.name}")
            seen.add(p.name)
            scope.values[p.name] = p
        for p in f.params:
            self.check_type_refs(p.type, scope, None, f"parameter %{p.name}")
            if f.kind == "kernel" and not isinstance(p.type, (MemType, ScalarType)):
                self.error(f"kernel parameter %{p.name} must be a GM tensor, a tensor list or a scalar, got {p.type}")
            if f.kind == "kernel" and isinstance(p.type, MemType) and p.type.space not in ("gm", "gmlist"):
                self.error(f"kernel parameter %{p.name} must live in GM, got {p.type}")
        outputs = f.attrs.get("outputs")
        if outputs is not None:
            if f.kind != "kernel":
                self.error("only kernels declare outputs")
            for o in outputs if isinstance(outputs, list) else [outputs]:
                if not isinstance(o, Value) or o.name not in seen:
                    self.error(f"output {o} is not a parameter")
        self.verify_block(f.body, scope, top=True)

    # -- blocks -------------------------------------------------------------------------------

    def verify_block(self, block: Block, parent: _Scope, top: bool = False) -> None:
        scope = _Scope(parent)
        n = len(block.ops)
        for i, op in enumerate(block.ops):
            spec = self.registry.find(op.opcode)
            if spec is not None and spec.terminator and i != n - 1:
                self.error(f"{op.opcode} must be the last op of its block", op)
            if op.opcode == "cf.return" and not top:
                self.error("cf.return is only allowed at the top level of a function body", op)
            self.verify_op(op, scope)

    # -- ops ----------------------------------------------------------------------------------

    def verify_op(self, op: Op, scope: _Scope) -> None:
        if op.id is not None:
            if op.id in self.seen_ids:
                self.error(f"duplicate op id #{op.id} (also in @{self.seen_ids[op.id]})", op)
            self.seen_ids[op.id] = self.func.name if self.func else ""
        spec = self.registry.find(op.opcode)
        if spec is None:
            self.error(f"unknown opcode {op.opcode!r}", op)
            self._define_results(op, scope)
            return
        if self.func is None:
            raise RuntimeError("verifier has no active function")
        if spec.level != "both" and spec.level != self.level:
            self.error(f"{op.opcode} is a {spec.level}-level op in a {self.level} module", op)
        kind = "kernel" if self.func.kind == "func" else self.func.kind
        if kind not in spec.kinds:
            self.error(f"{op.opcode} is not allowed in a {self.func.kind} function (allowed: {sorted(spec.kinds)})", op)
        if spec.devices is not None and self.m.device is not None and self.m.device not in spec.devices:
            self.error(f"{op.opcode} is not available on device {self.m.device} (devices: {sorted(spec.devices)})", op)
        env: dict[str, Any] = {}
        self.check_operands(op, spec, scope, env)
        self.check_results(op, spec, scope, env)
        self.check_attrs(op, spec, scope, env)
        if op.opcode.startswith("sync.local_mutex_"):
            side = str(op.attrs.get("side"))
            pipes = {"cube": {"S", "M", "MTE1", "MTE2", "FIX"}, "vec": {"S", "V", "MTE2", "MTE3"}}
            if str(op.attrs.get("pipe")) not in pipes.get(side, set()):
                self.error("local mutex requires a supported pipe on its cube/vec side", op)
            if self.func.attrs.get("side") is not None and str(self.func.attrs["side"]) != side:
                self.error("local mutex side differs from its function", op)
            if type(op.attrs.get("mode")) is not int or op.attrs["mode"] != 0:
                self.error("local mutex supports only mode=0", op)
            ident = op.attrs.get("id")
            known = self.memory.integer(ident)
            if known is not None and not 0 <= known < 32:
                self.error("local mutex ID must be in 0..31", op)
            if isinstance(ident, Value) and isinstance(ident.type, CellType):
                self.error("local mutex ID must be captured as an immutable scalar", op)
            elif isinstance(ident, Value) and not is_scalar_int(ident.type):
                self.error("local mutex ID must be an integer scalar", op)
        if op.opcode == "mem.alloc" and "mutex_ids" in op.attrs:
            ids = op.attrs["mutex_ids"]
            typ = op.results[0].type if op.results else None
            slots = typ.slots if isinstance(typ, BufType) else 1
            if not isinstance(ids, list) or len(ids) != slots or any(type(i) is not int or not 0 <= i < 32 for i in ids):
                self.error("allocation mutex_ids must contain one ID in 0..31 per physical slot", op)
        if op.opcode == "mem.alloc" and "sync_depth" in op.attrs:
            sync_depth = self.memory.integer(op.attrs["sync_depth"])
            result_type = op.results[0].type if op.results else None
            if not isinstance(result_type, BufType):
                self.error("mem.alloc sync_depth requires a slot buffer result", op)
            elif sync_depth is not None and not 1 <= sync_depth <= result_type.slots:
                self.error(f"mem.alloc sync_depth must be in 1..{result_type.slots}", op)
        if self.profile and (op.opcode.startswith('sync.crosscore.') or op.opcode == 'sync.mutex'):
            key = 'id' if op.opcode == 'sync.mutex' else 'flag_id'
            flag = self.memory.integer(op.attrs.get(key))
            maximum = self.profile.crosscore_id_max
            if flag is not None and not 0 <= flag <= maximum:
                self.error(f'cross-core flag ID must be in 0..{maximum}, got {flag}', op)
            if op.opcode == 'sync.mutex':
                depth = self.memory.integer(op.attrs.get('depth'))
                if depth is not None and not 1 <= depth <= self.profile.crosscore_counter_max:
                    self.error(f'cross-core depth must be in 1..{self.profile.crosscore_counter_max}', op)
        message = division_error(op, self.memory.integer)
        if message:
            self.error(message, op)
        if op.opcode in ("sync.set_flag", "sync.wait_flag"):
            message = raw_flag_error(op.attrs.get("src"), op.attrs.get("dst"), op.attrs.get("event_id"))
            known = self.memory.integer(op.attrs.get("event_id"))
            if message is None and known is not None:
                message = raw_flag_error(op.attrs.get("src"), op.attrs.get("dst"), known)
            if message is not None:
                self.error(message, op)
        if self.m.device in ('950', '950pr'):
            for message in self.memory.errors(op):
                self.error(message, op)
        if op.opcode == "mem.reinterpret" and _widens_a_view(op, self.memory.defs):
            self.error("mem.reinterpret changes the element width of a mem.view window, whose strides count elements "
                       "of the view's dtype (the printers do not rescale them); reinterpret the root before viewing "
                       "it (RFC-0010 §10)", op)
        if op.opcode == "dma.l1_to_bt" and len(op.operands) == 2:
            dst, src = (getattr(value, "type", None) for value in op.operands)
            if isinstance(dst, MemType) and isinstance(src, MemType):
                widening = dst.dtype.name == "f32" and src.dtype.name in ("f16", "bf16") and self.m.device in (None, "950", "950pr")
                if dst.dtype != src.dtype and not widening:
                    self.error("l1_to_bt supports same dtype or A5 f16/bf16 to f32 widening", op)
        if op.opcode in ("scalar.load", "scalar.store") and op.operands:
            memory = op.operands[0]
            if isinstance(memory, Value) and isinstance(memory.type, MemType) and memory.type.space not in ("gm", "ws", "ub"):
                self.error(f"{op.opcode} requires GM/workspace or UB memory, got {memory.type.space}", op)
        if op.opcode in ("core.set_sat_flag", "core.get_sat_flag"):
            mode = op.attrs.get("mode")
            if isinstance(mode, Ident) and mode.name not in SAT_BITS:
                self.error(f"unknown saturation flag mode {mode.name!r}; expected {' | '.join(SAT_BITS)}", op)
            if mode == Ident("global") and self.m.device not in (None, "950", "950pr"):
                self.error("global saturation selection (CTRL[60]) requires an a5 device", op)
            enable = op.attrs.get("enable")
            if isinstance(enable, Value) and (not isinstance(enable.type, ScalarType) or not enable.type.dtype.is_integer):
                self.error("saturation flag value must be a boolean or integer scalar", op)
        self.check_regions(op, spec, scope)
        self.check_cells(op, spec)
        self.check_hardware(op)
        self.check_register_groups(op)
        if not op.regions:
            self._define_results(op, scope)

    def _define_results(self, op: Op, scope: _Scope) -> None:
        for r in op.results:
            if scope.lookup(r.name) is not None:
                self.error(f"value %{r.name} defined twice", op)
            scope.values[r.name] = r

    def check_operands(self, op: Op, spec: OpSpec, scope: _Scope, env: dict[str, Any]) -> None:
        specs = list(spec.operands)
        variadic = bool(specs) and specs[-1].variadic
        fixed = specs[:-1] if variadic else specs
        if len(op.operands) < len(fixed) or (not variadic and len(op.operands) != len(fixed)):
            want = f"at least {len(fixed)}" if variadic else str(len(fixed))
            self.error(f"{op.opcode} takes {want} operand(s), got {len(op.operands)}", op)
            return
        for i, x in enumerate(op.operands):
            ospec = specs[i] if i < len(fixed) else specs[-1]
            if isinstance(x, Value):
                v = scope.lookup(x.name)
                if v is None:
                    self.error(f"operand {ospec.name}: use of undefined value %{x.name}", op)
                    continue
                if v.type != x.type:
                    self.error(f"operand {ospec.name}: %{x.name} has type {v.type} here, op says {x.type}", op)
                if not match(ospec.pat, x.type, env):
                    self.error(f"operand {ospec.name}: {x} : {x.type} does not match pattern {ospec.pattern}", op)
            elif isinstance(x, Literal):
                if ospec.pat.args[0].kind not in ("any", "scalar", "scalar_class", "typevar"):
                    self.error(f"operand {ospec.name}: literal {x} where pattern {ospec.pattern} needs a value", op)
            elif isinstance(x, FuncRef):
                if ospec.pat.args[0].kind != "any":
                    self.error(f"operand {ospec.name}: function reference where pattern {ospec.pattern} needs a value", op)
                if x.name not in self.function_kinds:
                    self.error(f"operand {ospec.name}: unknown function @{x.name}", op)
                elif op.opcode == "cf.call" and self.function_kinds[x.name] != "vf":
                    self.error(f"cf.call target @{x.name} is a {self.function_kinds[x.name]} function, not vf", op)
                elif op.opcode == "simt.launch" and self.function_kinds[x.name] != "simt":
                    self.error(f"simt.launch target @{x.name} is a {self.function_kinds[x.name]} function, not simt", op)
            else:
                self.error(f"operand {ospec.name}: bad operand {x!r}", op)

    def check_results(self, op: Op, spec: OpSpec, scope: _Scope, env: dict[str, Any]) -> None:
        if len(op.results) != len(spec.results):
            self.error(f"{op.opcode} produces {len(spec.results)} result(s), got {len(op.results)}", op)
            return
        for r, rspec in zip(op.results, spec.results, strict=True):
            self.check_type_refs(r.type, scope, op, f"result %{r.name}")
            if not match(rspec.pat, r.type, env):
                self.error(f"result %{r.name} : {r.type} does not match pattern {rspec.pattern}", op)

    def check_attrs(self, op: Op, spec: OpSpec, scope: _Scope, env: dict[str, Any]) -> None:
        for name in op.attrs:
            if spec.attr(name) is None:
                self.error(f"{op.opcode} has no attribute {name!r} (known: {[a.name for a in spec.attrs]})", op)
        for aspec in spec.attrs:
            if aspec.name not in op.attrs:
                if aspec.required:
                    self.error(f"{op.opcode} requires attribute {aspec.name!r}", op)
                continue
            self.check_attr_value(op, aspec, op.attrs[aspec.name], scope, env)

    def check_attr_value(self, op: Op, aspec: AttrSpec, v: Any, scope: _Scope, env: dict[str, Any]) -> None:
        ok = False
        for t in aspec.types:
            if t == "any":
                ok = True
            elif t == "int":
                ok = isinstance(v, int) and not isinstance(v, bool)
            elif t == "float":
                ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            elif t == "bool":
                ok = isinstance(v, bool)
            elif t == "str":
                ok = isinstance(v, str)
            elif t == "ident":
                ok = isinstance(v, Ident)
            elif t == "list":
                ok = isinstance(v, list)
            elif t == "dims":
                ok = isinstance(v, list)
            elif t == "type":
                ok = isinstance(v, Type)
            elif t == "value":
                ok = isinstance(v, Value)
            if ok:
                break
        if not ok:
            self.error(f"attribute {aspec.name} = {v!r} is not of type {aspec.type}", op)
            return
        for val in _values_in(v):
            found = scope.lookup(val.name)
            if found is None:
                self.error(f"attribute {aspec.name}: use of undefined value %{val.name}", op)
            elif found.type != val.type:
                self.error(f"attribute {aspec.name}: %{val.name} has type {found.type} here, op says {val.type}", op)
            elif aspec.pat is not None and isinstance(v, Value) and not match(aspec.pat, val.type, env):
                self.error(f"attribute {aspec.name}: {val} : {val.type} does not match pattern {aspec.pattern}", op)

    def check_regions(self, op: Op, spec: OpSpec, scope: _Scope) -> None:
        if len(op.regions) > len(spec.regions):
            self.error(f"{op.opcode} takes at most {len(spec.regions)} region(s), got {len(op.regions)}", op)
        if spec.regions and not op.regions:
            self.error(f"{op.opcode} needs a region", op)
        if op.opcode in ("cf.break", "cf.continue") and self.loop_depth == 0:
            self.error(f"{op.opcode} outside of a cf.for", op)
        if op.opcode == "cf.for":
            for operand in op.operands:
                value = operand.value if isinstance(operand, Literal) else operand
                integer = isinstance(value, int) or (isinstance(value, Value) and isinstance(value.type, (ScalarType, CellType))
                                                     and (value.type.dtype.is_integer or value.type.dtype.name == "b1"))
                if not integer:
                    self.error("cf.for bounds and step must be integer scalars", op)
            if len(op.operands) == 3:
                step = op.operands[2]
                step = step.value if isinstance(step, Literal) else step
                if isinstance(step, int) and step == 0:
                    self.error("cf.for step must not be zero", op)
        if not op.regions:
            return
        inner = _Scope(scope)
        for r in op.results:
            inner.values[r.name] = r
        if op.opcode == "cf.for":
            self.loop_depth += 1
        for block in op.regions:
            self.verify_block(block, inner)
        if op.opcode == "cf.for":
            self.loop_depth -= 1

    #: Attributes whose *presence* means the default was overridden, with that default. The
    #: frontend omits a rider it left alone, so `"scale" in op.attrs` is nearly the predicate
    #: already — but `dma.copy` carries riders through from a sugar call that may have written the
    #: default explicitly, and a rule that fires on `scale=1.0` would reject a legal kernel.

    def check_hardware(self, op: Op) -> None:
        """Rules the hardware imposes on an op that is already well formed.

        These are not structural or type checks — the op has the right operands, the right
        attributes and the right patterns — but combinations the machine does not perform. They
        live here rather than in a backend because they hold for every target that prints this op
        and for the simulator that models it, and here rather than in the frontend because the
        frontend is not the only producer: `device_lower` synthesises several of these ops from
        `dma.copy`, and the verifier runs after every pass.
        """
        message = vf_extremum_error(op, self.func and self.func.kind, self.profile and self.profile.family)
        if message:
            self.error(message, op)
        if op.opcode in ("vf.store_unalign", "vf.store_unalign_post"):
            # A5 spells an unaligned store and its flush as vstus / vstas, and both post-update: there is no
            # no-advance form (RFC-0001 §6.11). The DSL still accepts PostMode.NORMAL for 0.1.x and canonicalises
            # it to update, so this rejects only hand-written or generated IR that asks for the missing form.
            if str(op.attrs.get("post_mode", "update")) == "normal":
                self.error(f"{op.opcode}: post_mode=normal has no A5 spelling — an unaligned store and its Post "
                           "always advance the cursor; use post_mode=update (I042)", op)
            return
        if op.opcode != "dma.l0c_to_ub":
            return
        mode = dual_mode_of(op)
        if mode not in SPLIT_MODES:
            return
        # The fixpipe's scalar quant rides the deqScalar, which exists only with the dual
        # destination control off. SPLITM / SPLITN split the L0C tile across the two vector
        # sub-blocks and carry the same-type plain copy alone — not relu, not any requant rider,
        # and not even the deqScalar-free float downcast. Stated in full at
        # `easyasc/stub_functions/cube.py:1884`; PTO states half of it, as a static_assert on its
        # quantised TMOV overload only (`npu/a5/TMov.hpp:770`), and would compile the rest.
        # `fixpipe_riders` is the shared predicate; `ir/lint.py` asks it the same question the
        # other way round, to exempt a SINGLE that had no alternative from its performance
        # warning. One list, so the two cannot disagree about what counts as a rider.
        for what, detail in fixpipe_riders(op):
            if what == "dtype":
                self.error(f"dma.l0c_to_ub: dual_mode={mode} carries the same-type plain copy "
                           f"only (f32 -> f32 or i32 -> i32), got {detail}; even an unscaled "
                           "float downcast needs dual_mode=single", op)
            else:
                self.error(f"dma.l0c_to_ub: {detail} needs dual_mode=single — the fixpipe scalar "
                           f"path rides the deqScalar, which the dual destination control "
                           f"({mode}) turns off", op)

    def check_cells(self, op: Op, spec: OpSpec) -> None:
        if op.opcode == "scalar.cell":
            if "init" in op.attrs and op.results:
                self.initialised_cells.add(op.results[0].name)
            return
        if op.opcode == "scalar.set":
            if op.operands and isinstance(op.operands[0], Value):
                self.initialised_cells.add(op.operands[0].name)
            operands = op.operands[1:]
        else:
            operands = op.operands
        for x in list(operands) + list(op.attr_values()):
            if isinstance(x, Value) and isinstance(x.type, CellType) and x.name not in self.initialised_cells:
                self.error(f"cell %{x.name} is read before any scalar.set and has no init", op)

    # -- types --------------------------------------------------------------------------------

    def check_register_groups(self, op: Op) -> None:
        if not op.opcode.startswith("vf."):
            return
        regs = [v for v in op.operands if isinstance(v, Value) and isinstance(v.type, RegType)]
        masks = [v for v in (*op.operands, *op.attr_values()) if isinstance(v, Value) and isinstance(v.type, MaskType)]
        if op.opcode in {"vf.mask_and", "vf.mask_or", "vf.mask_xor", "vf.mask_not", "vf.mask_mov", "vf.mask_sel",
                         "vf.mask_interleave", "vf.mask_deinterleave"} and any(v.type.n == 2 for v in masks):
            if len({v.type.lanes for v in masks}) > 1:
                self.error(f"{op.opcode}: grouped mask operands must have matching predicate counts", op)
        if op.opcode == "vf.mod" and any(v.type.dtype.name not in ("i64", "u64") for v in regs):
            self.error("vf.mod requires i64 or u64 registers", op)
        if not any(v.type.n == 2 for v in regs):
            return
        matching = {"add", "sub", "mul", "div", "mod", "max", "min", "and", "or", "xor", "copy", "select",
                    "cmp", "abssub", "muladddst", "muldstadd", "interleave", "deinterleave"}
        if op.opcode.removeprefix("vf.") in matching and len({v.type for v in regs}) > 1:
            self.error(f"{op.opcode}: grouped operands must have matching dtype and register count", op)
        count = min(v.type.lanes for v in regs) if op.opcode == "vf.cast" else regs[0].type.lanes
        if any(v.type.lanes < count for v in masks):
            self.error(f"{op.opcode}: register group needs {count} predicate lanes; use a matching reg_num mask", op)

    def check_type_refs(self, t: Type, scope: _Scope, op: Op | None, what: str) -> None:
        for name in sorted(type_values(t)):
            v = scope.lookup(name)
            if v is None:
                self.error(f"{what}: dimension %{name} is not a value in scope", op)
            elif not (is_scalar_int(v.type) or (isinstance(v.type, CellType) and v.type.dtype.is_integer)):
                self.error(f"{what}: dimension %{name} must be an integer scalar or cell, got {v.type}", op)
        if isinstance(t, MemType) and t.space == "gmlist" and self.func is not None and op is not None:
            self.error(f"{what}: gmlist values are only kernel parameters", op)
        if isinstance(t, BufType):
            self.check_type_refs(t.elem, scope, op, what)
        if isinstance(t, (RegType, MaskType, UnalignRegType)) and self.func is not None and self.func.kind != "vf":
            self.error(f"{what}: register types exist only inside vf functions", op)
        if isinstance(t, RegType) and t.n == 2 and t.dtype.name not in ("i64", "u64", "c32", "c64"):
            self.error(f"{what}: c310 register groups support i64/u64/c32/c64, got {t}", op)
        if isinstance(t, (EventType, FlagType)) and self.func is not None and self.func.kind not in ("kernel", "func"):
            self.error(f"{what}: events and flags exist only in kernels", op)
        if isinstance(t, MemType) and t.is_local:
            if t.rank != 2:
                self.error(f"{what}: on-chip tensors are two-dimensional, got rank {t.rank} ({t})", op)
            for d in t.dims:
                if isinstance(d, Ragged) or (isinstance(d, Product) and any(isinstance(f, Ragged) for f in d.factors)):
                    self.error(f"{what}: on-chip tensors cannot have '?' dimensions", op)


def _widens_a_view(op: Op, defs: dict[str, Op]) -> bool:
    """A plain reinterpret to another element width whose window lies in a ``mem.view`` coordinate system."""
    types = [getattr(v, "type", None) for v in (*op.operands[:1], *op.results[:1])]
    if "tile" in op.attrs or len(types) != 2 or not all(isinstance(t, MemType) for t in types) \
            or max(types[0].dtype.bits, 8) == max(types[1].dtype.bits, 8):
        return False
    value = op.operands[0]
    while isinstance(value, Value) and (d := defs.get(value.name)) is not None and d.operands:
        if d.opcode == "mem.view":
            return True
        if d.opcode != "mem.slice" and (d.opcode != "mem.reinterpret" or "tile" in d.attrs):
            return False  # a root, slot, reshape or tile starts a row-major system
        value = d.operands[0]
    return False


def _values_in(v: Any):
    if isinstance(v, Value):
        yield v
    elif isinstance(v, list):
        for x in v:
            yield from _values_in(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _values_in(x)


def verify(module: Module, registry: Registry | None = None) -> list[Diagnostic]:
    return Verifier(module, registry or REGISTRY).run()


def check(module: Module, registry: Registry | None = None) -> Module:
    diags = verify(module, registry)
    if any(d.severity == "error" for d in diags):
        raise VerifyError(diags)
    return module


__all__ = ["Diagnostic", "VerifyError", "Verifier", "verify", "check"]
