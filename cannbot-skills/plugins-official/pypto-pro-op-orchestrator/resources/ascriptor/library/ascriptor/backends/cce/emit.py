# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The cce printer: a Lowered module -> CCE source files.

One op prints as one statement (annotated ``// #id``, D-015): kernel-level ops as one call of the
```` wrapper named after them in ``tensorutils_cce.h``, ``@vf`` ops as one compiler vector
intrinsic on ``vector_*`` registers, ``@simt`` ops as plain C on the compiler's SIMT layer. Scalars are
``const`` locals, cells are mutable locals, views are locals built from their root allocation plus the
byte offset :mod:`.views` folds, events are ``Event`` objects with the static ids of the events
pass, addresses are the ``addr`` attributes of ``addr_alloc``.

There is no logic here beyond table lookup and printing: anything the printer cannot express raises
:class:`CceGap` with the op and its source location (RFC-0007 §5), never a silent fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...ir import Block, FuncRef, Function, Ident, Literal, Module, Op, Value
from ...ir.scalar_range import ScalarRanges
from ...ir.saturation import TRUNCATING_CASTS
from ...ir.types import (
    BufType,
    CellType,
    DType,
    EventType,
    MaskType,
    MemType,
    Product,
    RegType,
    ScalarType,
    UnalignRegType,
)
from ...ir.types import dtype as _dtype
from ...passes.addr_alloc import ALIGN
from ...passes.util import Defs, dim_scalar, literal_or_value
from ..base import Artifacts
from ..shared.fold_plan import MEM_READS, paren, plan_folds, unparen
from . import cpp, views
from ...devices import load as _load_device
from .arch import c310
from .counters import mask_update
from .emit_vec import C220Vec
from .host import HostPrinter, dim_str
from .host import ident as _ident

HEADER = Path(__file__).with_name("include") / "tensorutils_cce.h"
F32 = _dtype("f32")


BUFF_ALIAS = {2: "DBuff", 3: "TBuff", 4: "QBuff", 5: "PBuff"}
EVENT_ALIAS = {1: "SEvent", 2: "DEvent", 3: "TEvent", 4: "QEvent"}


def buff_type(T: str, pos: str, slots: int) -> str:
    """The slot-buffer type: the old DBuff / TBuff / QBuff / PBuff names for depths 2..5, Buff<..., N> beyond."""
    if slots in BUFF_ALIAS:
        return f"{BUFF_ALIAS[slots]}<{T}, {pos}>"
    return f"Buff<{T}, {pos}, {slots}>"


def cast_call(d: str, s: str, shape: str, *, m: str = "pset_b8(PAT_ALL)", rnd: str = "ROUND_R", sat: str = "RS_DISABLE",
              part: str = "PART_EVEN", part_t: str = "PART_P0", mode: str = "MODE_ZEROING") -> str:
    """The `vcvt` statement for a cast pair of `c310.CAST_SHAPES`: the header's argument order per shape."""
    tails = {
        "part": f"{m}, {part}, {mode}", "part_t": f"{m}, {part_t}, {mode}",
        "sat_part": f"{m}, {sat}, {part}, {mode}", "sat_part_t": f"{m}, {sat}, {part_t}, {mode}",
        "rnd_sat_part": f"{m}, {rnd}, {sat}, {part}, {mode}", "rnd_sat_part_t": f"{m}, {rnd}, {sat}, {part_t}, {mode}",
        "rnd_sat": f"{m}, {rnd}, {sat}, {mode}", "sat_rnd": f"{m}, {sat}, {rnd}, {mode}",
        "rnd_part": f"{m}, {rnd}, {part}, {mode}", "rnd_part_t": f"{m}, {rnd}, {part_t}, {mode}",
        "rnd": f"{m}, {rnd}, {mode}",
        "b64_from_f32": f"{rnd}, {sat}", "f32_from_b64": rnd, "b64_widen": None,
    }
    tail = tails[shape]
    return f"vcvt({d}, {s}{', ' + tail if tail else ''});"


class CceGap(Exception):
    """An op the printer has no line for: reported with the op and its source location."""

    def __init__(self, op: Op | None, why: str) -> None:
        self.op = op
        self.why = why
        if op is None:
            super().__init__(f"cce backend: {why}")
        else:
            loc = f" at {op.loc}" if op.loc else ""
            super().__init__(f"cce backend: {op.opcode} #{op.id}{loc}: {why}")


def c_ident(name: str) -> str:
    s = cpp.api(name)  # the rest of C_KEYWORDS takes the underscore only inside the kernel translation unit
    return s + "_" if cpp.kernel_reserved(s) else s



def _elem(t: Any) -> MemType:
    return t.elem if isinstance(t, BufType) else t


# =============================================================================== function printers


class FnPrinter(HostPrinter, C220Vec):
    """Prints one function; the kernel-level flavour. Subclasses change the memory model.
    ``C220Vec`` contributes the ``vec.*`` handlers of the a2 family (guarded on ``mp.arch``)."""

    kind = "kernel"

    def __init__(self, mp: ModulePrinter, fn: Function) -> None:
        self.mp = mp
        self.fn = fn
        self.ranges = ScalarRanges(fn)
        self.defs = Defs(Module(fn.name, {}, (fn,)))  # per function: value names repeat across functions
        self.lines: list[str] = []
        self.epilogue: list[str] = []  # mutex token drains, printed before every return (sync.mutex)
        self.indent = 1
        self.names: dict[str, str] = {}
        self.used: set[str] = set()
        self.geo_cache: dict[str, views.Geo] = {}
        # Folding (formatting only): a scalar temporary the frontend named after its opcode (add_5, mul_4, ceil_div_1,
        # load, cube_idx ...) that is used exactly once, by a later op of the same block with nothing but pure
        # definitions in between, is printed inside that use instead of as a `const` local. User-named values keep
        # their own statement. The folded ops' ids ride on the statement that absorbs them.
        self.foldable: dict[str, int | None] = plan_folds(fn)  # temporary -> id of the op that absorbs it
        self.inline: dict[str, str] = {}
        self.inline_ids: dict[str, list[int]] = {}  # the ids a folded temporary carries (its own and its operands')
        self.absorbed: list[int] = []  # ids of the temporaries folded into the statement being built
        self.atomic: str | None = None  # between atomic.begin and atomic.end (the armed SPR)

    def gap(self, op: Op | None, why: str) -> Exception:
        return CceGap(op, why)

    # ---------------------------------------------------------------- names and operands

    def name(self, v: Value) -> str:
        if v.name in self.inline:  # a folded temporary: its expression (callers that build operators wrap it)
            self.absorbed.extend(self.inline_ids.pop(v.name, []))
            return self.inline[v.name]
        n = self.names.get(v.name)
        if n is None:
            n = c_ident(v.name)
            while n in self.used:
                n += "_"
            self.used.add(n)
            self.names[v.name] = n
        return n

    def val(self, x: Any, dt: DType | None = None) -> str:
        if isinstance(x, Value):
            return self.name(x)
        if isinstance(x, Literal):
            return cpp.literal(x, dt)
        if isinstance(x, FuncRef):
            return self.mp.fname(x.name)
        if isinstance(x, bool | int | float):
            return cpp.literal(x, dt)
        if isinstance(x, Ident):
            return x.name
        raise TypeError(f"cannot print operand {x!r}")

    def cexpr(self, e: views.Expr) -> str:
        return views.cexpr(e, lambda v: paren(self.name(v)))

    def operand(self, x: Any, dt: DType | None = None) -> str:
        """``val`` for an operand of a printed operator: a folded expression is parenthesised."""
        return paren(self.val(x, dt))

    def attr(self, op: Op, key: str, default: Any = None, dt: DType | None = None) -> str | None:
        v = op.attrs.get(key, default)
        if v is None:
            return None
        return self.val(v, dt)

    def aexpr(self, op: Op, key: str, default: Any = 0) -> views.Expr:
        """An attribute as an offset expression (int or Value)."""
        v = op.attrs.get(key, default)
        if isinstance(v, Value):
            return v
        return literal_or_value(v)

    # ---------------------------------------------------------------- output

    def emit(self, text: str, op: Op | None = None, note: str | None = None) -> None:
        ids = self.absorbed + ([op.id] if op is not None and op.id is not None else [])
        self.absorbed = []
        parts = ([note] if note else []) + [" ".join(f"#{i}" for i in ids)] if ids or note else []
        tag = "  // " + " ".join(p for p in parts if p) if parts else ""
        self.lines.append("    " * self.indent + text + tag)

    def bare(self, x: Any, dt: DType | None = None) -> str:
        """``val`` for a whole right-hand side or call argument: a folded expression without outer parentheses."""
        return unparen(self.val(x, dt))

    def cexpr_bare(self, e: views.Expr) -> str:
        return unparen(self.cexpr(e))

    def run_op(self, op: Op) -> None:
        handler = getattr(self, "op_" + op.opcode.replace(".", "_"), None)
        if handler is None:
            raise CceGap(op, "no printer for this opcode")
        n_before = len(self.lines)
        handler(op)
        # a folded temporary this op consumes only through an attribute the printer never prints (a view's extents):
        # its ids still ride on this op's statement, so every op keeps a printed trace
        left = [n for n, c in self.foldable.items() if c == op.id and n in self.inline_ids]
        if left:
            ids = [i for n in left for i in self.inline_ids.pop(n)]
            if len(self.lines) > n_before:
                last = self.lines[-1]
                self.lines[-1] = last + (" " if "  // #" in last else "  // ") + " ".join(f"#{i}" for i in ids)
            else:
                self.absorbed.extend(ids)

    # ---------------------------------------------------------------- types

    def scalar_ctype(self, t: Any) -> str:
        if isinstance(t, ScalarType | CellType):
            return cpp.ctype(t.dtype)
        raise TypeError(f"not a scalar type: {t}")

    def window_type(self, t: Any) -> str:
        mt = _elem(t)
        if not isinstance(mt, MemType):
            raise TypeError(f"not a memory type: {t}")
        T = cpp.ctype(mt.dtype)
        if isinstance(t, BufType):
            return buff_type(T, cpp.POS[mt.space], t.slots)
        if mt.space in ("gm", "ws", "gmlist"):
            return f"GMTensor<{T}>"
        return f"Tensor<{T}, {cpp.POS[mt.space]}>"

    # ---------------------------------------------------------------- scalars (all kinds)

    def _def(self, op: Op, expr: str) -> None:
        r = op.results[0]
        if r.name in self.foldable:
            self.inline[r.name] = expr
            self.inline_ids[r.name] = self.absorbed + ([op.id] if op.id is not None else [])
            self.absorbed = []
            return
        self.emit(f"const {self.scalar_ctype(r.type)} {self.name(r)} = {expr};", op)

    def _binop(self, op: Op, sym: str) -> None:
        dt = self._rdt(op)
        a, b = (self.operand(x, dt) for x in op.operands[:2])
        self._def(op, f"{a} {sym} {b}")

    def op_scalar_min(self, op: Op, kind: str = "min") -> None:
        dt = self._rdt(op)
        a, b = (self.val(x, dt) for x in op.operands[:2])
        if dt == F32 and self.kind == "simt":  # a vf cannot hold one on A5 (verifier)
            raise CceGap(op, f"f32 scalar {kind}: no measured SIMT spelling (RFC-0001 §6.16)")
        self._def(op, cpp.f32_extremum(kind, a, b) if dt == F32 else f"{kind.title()}({a}, {b})")

    def op_scalar_ceil_div(self, op: Op) -> None:
        self._def(op, f"CeilDiv({self.val(op.operands[0])}, {self.val(op.operands[1])})")

    def op_scalar_align(self, op: Op) -> None:
        self._def(op, f"AlignUp({self.val(op.operands[0])}, {int(op.attrs['n'])})")

    def op_scalar_sqrt(self, op: Op) -> None:
        dt = self._rdt(op)
        if self.mp.arch == 'c310' and dt.name == 'bf16':
            raise CceGap(op, 'scalar sqrt requires a target-supported BF16 scalar conversion; use an FP32 scalar or an explicit register cast (M10-055)')
        callee = ('__sqrtf' if self.fn.kind == 'simt' else '::sqrt') if self.mp.arch == 'c310' else '__builtin_sqrtf'
        if dt.name == "f32":
            self._def(op, f"{callee}({self.val(op.operands[0], dt)})")
        elif dt.name in ("f16", "bf16"):  # through fp32: the scalar unit has no half sqrt
            self._def(op, f"({cpp.ctype(dt)}){callee}((float)({self.val(op.operands[0], dt)}))")
        else:
            raise CceGap(op, f"scalar sqrt of {dt} has no CCE spelling (f32 / f16 / bf16)")

    def op_scalar_cmp(self, op: Op) -> None:
        pred = self.ident(op, "pred")
        sym = cpp.CMP_OPS[pred]
        a, b = op.operands[:2]
        dt = a.type.dtype if isinstance(a, Value) and isinstance(a.type, ScalarType | CellType) else None
        if dt is None and isinstance(b, Value) and isinstance(b.type, ScalarType | CellType):
            dt = b.type.dtype
        self._def(op, f"{self.operand(a, dt)} {sym} {self.operand(b, dt)}")

    # -- gmlist parameters: the ListTensorDesc reads of the GMList<T> wrapper (RFC-0001 §13) ----------------

    def op_list_count(self, op: Op) -> None:
        self._def(op, f"{self.name(op.operands[0])}.count()")

    def op_list_item_dim(self, op: Op) -> None:
        self._def(op, f"(int32_t){self.name(op.operands[0])}.dim({self.val(op.operands[1])}, {int(op.attrs['dim'])})")

    def op_list_load_ptr(self, op: Op) -> None:  # the explicit descriptor-read forms of RFC-0001 §13 (no pass emits them yet)
        self._def(op, f"(uint64_t){self.name(op.operands[0])}.ptr({self.val(op.operands[1])})")

    def op_list_load_dim(self, op: Op) -> None:
        self._def(op, f"{self.name(op.operands[0])}.dim({self.val(op.operands[1])}, {int(op.attrs['dim'])})")

    def op_list_item(self, op: Op) -> None:
        r = op.results[0]
        T = cpp.ctype(self._dtype_of(r))
        self.emit(f"const GMTensor<{T}> {self.name(r)}({self.name(op.operands[0])}.ptr({self.val(op.operands[1])}));", op)

    def op_scalar_load(self, op: Op) -> None:
        src, idx = op.operands[:2]
        self._def(op, f"{self.name(src)}.load({self.val(idx)})")  # type: ignore[arg-type]

    def op_scalar_store(self, op: Op) -> None:
        dst, idx, src = op.operands[:3]
        dt = _elem(dst.type).dtype  # type: ignore[union-attr]
        self.emit(f"{self.name(dst)}.store({self.val(idx)}, ({cpp.ctype(dt)})({self.val(src, dt)}));", op)  # type: ignore[arg-type]

    # ---------------------------------------------------------------- control flow (all kinds)

    def op_cf_for(self, op: Op) -> None:
        i = op.results[0]
        lo, hi, step = op.operands[:3]
        T = self.scalar_ctype(i.type)
        n = self.name(i)
        s = self.val(step)
        if isinstance(step, Literal) and int(step.value) > 0:
            cond = f"{n} < {self.val(hi)}"
        else:
            cond = f"(({s}) > 0 ? {n} < {self.val(hi)} : {n} > {self.val(hi)})"
        self.emit(f"for ({T} {n} = {self.val(lo)}; {cond}; {n} += {s}) {{", op)
        self.indent += 1
        fork = None
        if self.mp.arch == "c220":
            self.v_hazard_seed_loop(op.regions[0])  # loop-carried V-V hazards (the body's writes reach its first reads)
            pr, pw = self.v_hazard_state()
            fork = (set(pr), set(pw))
        self.run_block(op.regions[0])
        self.indent -= 1
        if fork is not None:
            # the loop may run zero times: a barrier printed inside the body discharges nothing on the
            # fall-through path, so the pre-loop pendings (incl. the seeded body writes) survive the join
            pr, pw = self.v_hazard_state()
            pr |= fork[0]
            pw |= fork[1]
        self.emit("}")

    def op_cf_if(self, op: Op) -> None:
        self.emit(f"if ({self.val(op.operands[0])}) {{", op)
        self.indent += 1
        # c220 V-V hazards: the pending-access tracker is linear but control flow forks here. A barrier
        # printed inside one branch discharges nothing on the other path (it does not execute there), so
        # each branch must start from the pre-branch state, and the join keeps the union of every path
        # that can reach the tail (including the fall-through when there is no else).
        fork = None
        if getattr(self.mp, "arch", None) == "c220":
            pr, pw = self.v_hazard_state()
            fork = (set(pr), set(pw))
        self.run_block(op.regions[0])
        self.indent -= 1
        outs = []
        if fork is not None:
            pr, pw = self.v_hazard_state()
            outs.append((set(pr), set(pw)))
        has_else = len(op.regions) > 1 and len(op.regions[1]) > 0
        if has_else:
            self.emit("} else {")
            self.indent += 1
            if fork is not None:
                pr, pw = self.v_hazard_state()
                pr.clear()
                pw.clear()
                pr |= fork[0]
                pw |= fork[1]
            self.run_block(op.regions[1])
            self.indent -= 1
            if fork is not None:
                pr, pw = self.v_hazard_state()
                outs.append((set(pr), set(pw)))
        if fork is not None:
            if not has_else:
                outs.append(fork)  # the branch may not execute: the pre-branch pendings survive
            pr, pw = self.v_hazard_state()
            pr.clear()
            pw.clear()
            for r, w in outs:
                pr |= r
                pw |= w
        self.emit("}")

    def op_cf_call(self, op: Op) -> None:
        callee = op.operands[0]
        assert isinstance(callee, FuncRef)
        fn = self.mp.module.function(callee.name)
        args = []
        for p, a in zip(fn.params, op.operands[1:], strict=True):
            args.append(self.call_arg(p, a))
        self.emit(f"{self.mp.fname(callee.name)}({', '.join(args)});", op)

    def call_arg(self, param: Value, arg: Any) -> str:
        t = param.type
        if isinstance(t, MemType):
            if t.space in ("ub", "gm", "ws"):
                return f"{self.name(arg)}.ptr()"
            raise CceGap(None, f"cannot pass a {t.space} window to {param}")
        if isinstance(t, ScalarType | CellType):
            return f"({cpp.ctype(t.dtype)})({self.val(arg, t.dtype)})"
        raise CceGap(None, f"cannot pass {arg} as {param}: {t}")

    # ---------------------------------------------------------------- core

    def op_core_cube_idx(self, op: Op) -> None:
        self._def(op, "GetCubeIdx()")

    def op_core_cube_num(self, op: Op) -> None:
        self._def(op, "GetCubeNum()")

    def op_core_vec_idx(self, op: Op) -> None:
        self._def(op, "GetVecIdx()")

    def op_core_vec_num(self, op: Op) -> None:
        self._def(op, "GetVecNum()")

    def op_core_sub_block_idx(self, op: Op) -> None:
        self._def(op, "GetSubBlockIdx()")

    def op_core_set_hf32(self, op: Op) -> None:
        self.emit(f"SetHF32Mode({self.flag(op, 'enable', True)});", op)

    def op_core_set_sat_flag(self, op: Op) -> None:
        self.emit(f"SetSatFlag({self._sat_bit(op)}, {self.flag(op, 'enable', True)});", op)

    def op_core_get_sat_flag(self, op: Op) -> None:
        self._def(op, f"GetSatFlag({self._sat_bit(op)})")

    def op_core_clean_dcache(self, op: Op) -> None:
        dst = op.attrs.get("dst")
        entire = self.ident(op, "entire_type", "ENTIRE_DATA_CACHE")
        target = self.ident(op, "dcci_dst", "CACHELINE_OUT")
        if not isinstance(dst, Value):
            raise CceGap(op, "clean_dcache needs a GM window in 'dst'")
        if entire not in ("ENTIRE_DATA_CACHE", "SINGLE_CACHE_LINE") or target not in (
                "CACHELINE_ALL", "CACHELINE_UB", "CACHELINE_OUT", "CACHELINE_ATOMIC"):
            raise CceGap(op, f"unknown dcache mode {entire}/{target}")
        self.emit(f"DataCacheCleanAndInvalid({self.name(dst)}, (uint64_t)AscendC::CacheLine::{entire}, "
                  f"(uint64_t)AscendC::DcciDst::{target});", op)

    # ---------------------------------------------------------------- debug (comments only)

    # ---------------------------------------------------------------- memory (kernel level)

    def _bytes_expr(self, mt: MemType) -> views.Expr:
        numel = views.prod([dim_scalar(d) if not isinstance(d, Product) else views.prod([dim_scalar(f) for f in d.factors])
                            for d in mt.dims])
        if mt.dtype.bits >= 8:
            return views.mul(numel, mt.dtype.bits // 8)
        return views.floordiv(views.add(views.mul(numel, mt.dtype.bits), 7), 8)

    def op_mem_alloc(self, op: Op) -> None:
        r = op.results[0]
        t = r.type
        if "addr" not in op.attrs:
            raise CceGap(op, "allocation without an address (run addr_alloc)")
        addr = self.bare(op.attrs["addr"])
        mt = _elem(t)
        assert isinstance(mt, MemType)
        T = cpp.ctype(mt.dtype)
        pos = cpp.POS[mt.space]
        if isinstance(t, BufType):
            b = self._bytes_expr(mt)
            align = ALIGN[mt.space]
            slot = (b + align - 1) // align * align if isinstance(b, int) else f"AlignUp({self.cexpr_bare(b)}, {align})"
            self.emit(f"const {buff_type(T, pos, t.slots)} {self.name(r)}((uint64_t)({addr}), (uint64_t)({slot}));", op)
        else:
            self.emit(f"const Tensor<{T}, {pos}> {self.name(r)}((uint64_t)({addr}));", op)

    def op_mem_workspace(self, op: Op) -> None:
        r = op.results[0]
        mt = r.type
        assert isinstance(mt, MemType)
        T = cpp.ctype(mt.dtype)
        offset = self.attr(op, "offset", 0)
        self.mp.note_workspace(op, self)
        self.emit(f"const GMTensor<{T}> {self.name(r)}((__gm__ {T}*)(workspace + ({offset})));", op)

    def _def_view(self, op: Op) -> None:
        r = op.results[0]
        g = self.geo(r)
        self.emit(f"const {self.window_type(r.type)} {self.name(r)} = {self.window_expr(g)};", op)

    def window_expr(self, g: views.Geo) -> str:
        root = self.name(g.root)
        root_t = _elem(g.root.type)
        assert isinstance(root_t, MemType)
        if g.slot is not None:
            slots = g.root.type.slots if isinstance(g.root.type, BufType) else 0
            root = (f"{root}.slot[{self.cexpr(g.slot)}]" if self.ranges.normalized(g.slot, slots)
                    else f"{root}.get({self.cexpr(g.slot)})")
        return self.displaced(root, g.space, g.dtype, views.byte_offset(g), same=root_t.dtype == g.dtype)

    def displaced(self, root: str, space: str, dtype: DType, off: views.Expr, *, same: bool = True) -> str:
        """The window ``root`` (a C expression) re-viewed as ``dtype`` and displaced by ``off`` bytes: ``root[elems]``
        when the offset is provably whole elements, else a window built from the byte address."""
        T = cpp.ctype(dtype)
        base = root if same else f"{root}.as<{T}>()"
        if isinstance(off, int) and off == 0:
            return base
        elems = views.div_exact(off, cpp.esize(dtype))
        if elems is not None:
            return f"{base}[{self.cexpr(elems)}]"
        if space in ("gm", "ws", "gmlist"):
            return f"GMTensor<{T}>((__gm__ {T}*)((__gm__ uint8_t*){root}.ptr() + ({self.cexpr(off)})))"
        return f"Tensor<{T}, {cpp.POS[space]}>({root}.addr + ({self.cexpr(off)}))"

    def op_mem_get_buf(self, op: Op) -> None:
        self._def_view(op)

    def op_mem_slice(self, op: Op) -> None:
        self._def_view(op)

    def op_mem_reinterpret(self, op: Op) -> None:
        self._def_view(op)

    def op_mem_reshape(self, op: Op) -> None:
        self._def_view(op)

    def op_mem_view(self, op: Op) -> None:
        self._def_view(op)

    def at(self, v: Value, off: views.Expr) -> str:
        """The window ``v`` displaced by ``off`` bytes."""
        mt = _elem(v.type)
        assert isinstance(mt, MemType)
        return self.displaced(self.name(v), mt.space, mt.dtype, off)

    def nz_at(self, v: Value, row: views.Expr, col: views.Expr, rows_total: views.Expr) -> str:
        mt = _elem(v.type)
        assert isinstance(mt, MemType)
        return self.at(v, views.nz_offset_bytes(mt.space, mt.dtype, row, col, rows_total))

    def _geom(self, v: Any, op: Op) -> tuple[str, str]:
        """A tile's declared [rows, cols] as C++ text - the fill ops address the NZ geometry."""
        mt = _elem(v.type)
        assert isinstance(mt, MemType)
        if len(mt.dims) != 2:
            raise CceGap(op, f"{v.name} has {len(mt.dims)} dimensions; the NZ fill addresses a 2-D tile")
        return (self.cexpr(dim_scalar(mt.dims[0])), self.cexpr(dim_scalar(mt.dims[1])))

    def _dtype_of(self, v: Any) -> DType:
        mt = _elem(v.type)
        assert isinstance(mt, MemType)
        return mt.dtype

    # Synchronization

    def op_sync_event(self, op: Op) -> None:
        r = op.results[0]
        t = r.type
        assert isinstance(t, EventType)
        ids = op.attrs.get("ids")
        if ids is None:
            if t.id is None:
                raise CceGap(op, "event without flag ids (run the events pass)")
            ids = [t.id]
        ids = [int(i) for i in ids]
        if t.set_pipe is None or t.wait_pipe is None:
            raise CceGap(op, "event without pipes")
        preset = op.attrs.get("preset", 0)
        if isinstance(preset, bool):
            preset = len(ids) if preset else 0
        cls = EVENT_ALIAS.get(len(ids), "Event")
        guards = op.attrs.get("guards") or ()
        self.emit(f"{cls}<{cpp.PIPE[t.set_pipe]}, {cpp.PIPE[t.wait_pipe]}, {int(preset)}, {', '.join(str(i) for i in ids)}> {self.name(r)};",
                  op, note=f"guards {', '.join(str(g) for g in guards)}" if guards else None)

    def op_sync_set_flag(self, op: Op) -> None:
        self.emit(f"SetFlag<{cpp.PIPE[self.ident(op, 'src')]}, {cpp.PIPE[self.ident(op, 'dst')]}>("
                  f"{self.attr(op, 'event_id')});", op)

    def op_sync_wait_flag(self, op: Op) -> None:
        self.emit(f"WaitFlag<{cpp.PIPE[self.ident(op, 'src')]}, {cpp.PIPE[self.ident(op, 'dst')]}>("
                  f"{self.attr(op, 'event_id')});", op)

    def op_sync_barrier(self, op: Op) -> None:
        if self.mp.arch == "c220" and self.ident(op, "pipe", "ALL") in ("V", "ALL"):
            self.v_hazard_clear()  # an explicit V (or ALL) barrier discharges the V-V hazard tracker
        self.emit(f"PipeBarrier<{cpp.PIPE[self.ident(op, 'pipe', 'ALL')]}>();", op)

    def op_sync_mutex(self, op: Op) -> None:
        """The old kernelbase prologue / epilogue of a cross-core mutex: the consumer side publishes ``depth`` free
        tokens before the body (what the interpreter models), and the producer side drains them at the end so no
        flag is left set for the next launch. ``vc``: the vector cores produce, the cube core consumes; ``cv`` the
        other way round. The lock protocol between them is the crosscore ops of the body."""
        kind = self.ident(op, "kind")
        fid = self.attr(op, "id")
        depth = int(op.attrs["depth"])
        side = _ident(self.fn.attrs.get("side"), "vec")
        set_pipe = cpp.PIPE[self.ident(op, "dst_end_pipe", "FIX" if kind == "vc" else "MTE3")]
        wait_pipe = cpp.PIPE[self.ident(op, "src_start_pipe", "S")]
        consumer, publish, drain = ("cube", "CUBE_READY", "WAIT_CUBE") if kind == "vc" else ("vec", "VEC_READY", "WAIT_VEC")
        if side == consumer:
            for _ in range(depth):
                self.emit(f"{publish}<{set_pipe}>({fid});", op)
        else:
            self.comment(f"sync.mutex {kind} id={fid}: the producer side; its {depth} tokens are drained before the return", op)
            self.epilogue.extend(f"{drain}<{wait_pipe}>({fid});" for _ in range(depth))

    def _crosscore(self, op: Op) -> None:
        fn = c310.CROSSCORE[op.opcode]
        self.emit(f"{fn}<{cpp.PIPE[self.ident(op, 'pipe')]}>({self.attr(op, 'flag_id')});", op)

    op_sync_crosscore_cube_ready = _crosscore
    op_sync_crosscore_wait_vec = _crosscore
    op_sync_crosscore_vec_ready = _crosscore
    op_sync_crosscore_wait_cube = _crosscore
    op_sync_crosscore_allcube_ready = _crosscore
    op_sync_crosscore_allcube_wait = _crosscore
    op_sync_crosscore_allvec_ready = _crosscore
    op_sync_crosscore_allvec_wait = _crosscore
    op_sync_crosscore_intracore_allvec_ready = _crosscore
    op_sync_crosscore_intracore_allvec_wait = _crosscore

    # ---------------------------------------------------------------- vector-core SPRs, atomics

    def op_vec_set_mask(self, op: Op) -> None:
        self.emit(f"SetVectorMask((uint64_t)({self.attr(op, 'high')}), (uint64_t)({self.attr(op, 'low')}));", op)

    def op_vec_reset_mask(self, op: Op) -> None:
        self.emit("ResetMask();", op)

    def op_vec_set_mask_by_count(self, op: Op) -> None:
        self.emit(f"SetVectorMaskByCount({self.attr(op, 'count')});", op)

    # Counter mode is c220's. It is not that the c310 compiler lacks the builtin -- CANN 9.2.0
    # declares __builtin_cce_set_mask_count with no arch guard and a TU calling it assembles for
    # dav-c310-vec -- it is that c310 has NO OPERATION the mode would govern: its vector unit is
    # register-level, an op's extent comes from the register width, the repeat count and an explicit
    # MaskReg, and the mask SPR is a carrier a @vf reads with move_mask_spr rather than something
    # that gates a store. Measured, not assumed: tools/diag/probes/mask_counter_mode.py sets
    # set_mask_by_count(20) around a register store on the board and all 64 lanes still land.
    # So the refusal is here, at print time, where every other gap is reported (D-217).
    _NO_COUNTER_MODE = ("counter mode is the c220 bracket around a counted tensor-level op (D-062); "
                        "c310's vector unit is register-level and has no operation whose extent the mode "
                        "would govern -- the mask SPR does not gate a register store there "
                        "(tools/diag/probes/mask_counter_mode.py)")

    def op_vec_set_mask_count(self, op: Op) -> None:
        if self.mp.arch != "c220":
            raise CceGap(op, self._NO_COUNTER_MODE)
        self.emit("SetMaskCount();", op)

    def op_vec_set_mask_normal(self, op: Op) -> None:
        self.emit("SetMaskNorm();", op)

    def op_vec_set_mask_counter(self, op: Op) -> None:
        if self.mp.arch != "c220":
            raise CceGap(op, self._NO_COUNTER_MODE)
        self.emit(f"SetVectorMask({self.attr(op, 'count')});", op)

    def op_atomic_begin(self, op: Op) -> None:
        kind = self.ident(op, "op", "add")
        if kind not in ("add", "max", "min"):
            raise CceGap(op, f"atomic op {kind!r} has no CCE spelling")
        self.atomic = kind
        self.emit(f"SetAtomicOp{kind.capitalize()}();", op)

    def op_atomic_set_type(self, op: Op) -> None:
        dt = self.ident(op, "dtype")
        self.emit(f"SetAtomicType<{cpp.CTYPE[dt]}>();", op)

    def op_atomic_end(self, op: Op) -> None:
        self.atomic = None
        self.emit("SetAtomicNone();", op)

    # ---------------------------------------------------------------- DMA

    def _atomic_wrap(self, op: Op, dt: DType) -> tuple[str | None, str | None]:
        """The SPR pair around one GM store, given the accumulate type of its DESTINATION.

        Accumulation is an armed hardware state, not a property of one instruction, so the
        disarm belongs to whoever armed it. Inside a `with atomic_*()` region that is
        ``atomic.begin`` / ``atomic.end``: the op SPR is already set, this store contributes
        only its own dtype SPR, and it must NOT print `SetAtomicNone()` -- a bare
        `set_atomic_none()` is a clear and not a restore, so a trailing one here would disarm
        the region and every later GM write in it would run plain (the scalar `SetValueTo`
        after a copy is the reachable case). Standing alone -- the recorded corpus carries the
        attr without the markers -- the store owns both halves and brackets itself.
        """
        kind = self.ident(op, "atomic")
        if kind is None or kind == "none":
            return None, None
        if kind not in ("add", "max", "min"):
            raise CceGap(op, f"atomic {kind!r} has no CCE spelling")
        if self.atomic is not None:
            if kind != self.atomic:
                raise CceGap(op, f"atomic {kind!r} store inside `with atomic_{self.atomic}()`: "
                                 "one armed SPR cannot hold both operations")
            return f"SetAtomicType<{cpp.ctype(dt)}>();", None
        return f"SetAtomic{kind.capitalize()}<{cpp.ctype(dt)}>();", "SetAtomicNone();"

    def op_dma_gm_to_ub_pad(self, op: Op) -> None:
        """``DataCopyPad`` GM -> UB. An explicit `pad` fills the destination's 32-byte alignment
        tail through the `set_mov_pad_val` SPR, which is where AscendC's own `DataCopyPadGm2UBImpl`
        puts it; without one the tail keeps whatever the transfer found there."""
        dst, src = op.operands[:2]
        tail = ""
        pad = op.attrs.get("pad")
        if pad is not None:
            dt = self._dtype_of(dst)
            v = pad.value if isinstance(pad, Literal) else pad
            if isinstance(v, Value):
                raise CceGap(op, "a run-time pad value: the SPR takes the value's bit pattern and "
                                 "the printer bitcasts it at compile time, so only a literal is reachable")
            if dt.bits > 32 and v not in (0, 0.0, False):
                raise CceGap(op, f"a 64-bit pad value ({v!r}): AscendC's own DataCopyPad asserts "
                                 "paddingValue == 0 for b64 on this device, so only a zero tail is reachable")
            try:
                bits = cpp.bit_pattern(dt, v)
            except ValueError as exc:
                raise CceGap(op, f"pad value {v!r}: {exc}") from None
            tail = f", true, 0x{bits:X}ULL"
        self.emit(f"gm_to_ub_pad({self.name(dst)}, {self.name(src)}, {self.attr(op, 'n_burst')}, "
                  f"{self.attr(op, 'burst_len_byte')}, {self.attr(op, 'src_stride_byte', 0)}, "
                  f"{self.attr(op, 'dst_stride', 0)}{tail});", op)

    def _sort_f32(self, op: Op) -> None:
        for v in op.operands:
            if self._dtype_of(v).name not in ("f32", "u32"):
                raise CceGap(op, f"the sort family is printed for fp32 (score, index) records, got {self._dtype_of(v)}")

    def op_vec_sort32(self, op: Op) -> None:
        self._sort_f32(op)
        dst, src, idx = op.operands[:3]
        self.emit(f"sort32({self.name(dst)}, {self.name(src)}, {self.name(idx)}, {self.attr(op, 'repeat', 0)});", op)

    def op_vec_mergesort4(self, op: Op) -> None:
        self._sort_f32(op)
        dst, src = op.operands[:2]
        self.emit(f"mergesort4({self.name(dst)}, {self.name(src)}, {self.attr(op, 'length_per_seq')}, {self.attr(op, 'repeat', 1)});", op)

    def op_vec_mergesort_2seq(self, op: Op) -> None:
        self._sort_f32(op)
        dst, src1, src2 = op.operands[:3]
        self.emit(f"mergesort_2seq({self.name(dst)}, {self.name(src1)}, {self.name(src2)}, {self.attr(op, 'size1')}, {self.attr(op, 'size2')});", op)

    def op_dma_gm_to_ub_nd(self, op: Op) -> None:
        """``NdLoops<DIM>`` (index 0 innermost, strides in elements, pads per loop) then ``gm_to_ub_nd``; the
        ``fence`` attr ``all`` (the old default) closes the transfer with ``pipe_barrier(PIPE_ALL)``."""
        dst, src = op.operands[:2]
        dim, dt = int(op.attrs["dim"]), self._dtype_of(dst)
        if not 1 <= dim <= 5:
            raise CceGap(op, f"nd dma of dim {dim} (the hardware has 5 loops)")
        if dt.bits > 32:
            raise CceGap(op, "64-bit nd dma (two b32 passes) is not printed")
        if op.attrs.get("asc_optimize"):
            raise CceGap(op, "asc_optimize is not printed")
        fence = op.attrs.get("fence", "all")
        if fence not in ("all", "mte2"):
            raise CceGap(op, f"fence {fence!r} (all | mte2)")

        def entries(key: str, ct: str, default: int, cfg: str | None = None) -> str:
            c = op.attrs.get(cfg) if cfg else None
            v = [c] * dim if c is not None else op.attrs.get(key)
            if v is None:
                v = [default] * dim
            return "{" + ", ".join(f"({ct})({self.val(x)})" for x in v) + "}"

        loops = f"nd{op.id}"
        self.emit(f"NdLoops<{dim}> {loops}{{{entries('loop_src_stride', 'uint64_t', 1)}, {entries('loop_dst_stride', 'uint32_t', 1)}, "
                  f"{entries('loop_size', 'uint32_t', 1)}, {entries('loop_left_pad', 'uint8_t', 0, 'config_left_pad')}, "
                  f"{entries('loop_right_pad', 'uint8_t', 0, 'config_right_pad')}}};", op)
        nearest = op.attrs.get("nearest_value_mode", False)
        self.emit(f"gm_to_ub_nd({self.name(dst)}, {self.name(src)}, {loops}, ({cpp.ctype(dt)})({self.val(op.attrs.get('constant_value', 0), dt)}), "
                  f"{self.val(bool(nearest) if isinstance(nearest, bool | int) else nearest)});", op)
        if fence == "all":
            self.emit("pipe_barrier(PIPE_ALL);", op)

    def op_dma_ub_to_gm_pad(self, op: Op) -> None:
        dst, src = op.operands[:2]
        pre, post = self._atomic_wrap(op, self._dtype_of(dst))
        if pre:
            self.emit(pre, op)
        self.emit(f"ub_to_gm_pad({self.name(dst)}, {self.name(src)}, {self.attr(op, 'n_burst')}, "
                  f"{self.attr(op, 'burst_len_byte')}, {self.attr(op, 'src_stride', 0)}, {self.attr(op, 'dst_stride_byte', 0)});", op)
        if post:
            self.emit(post, op)

    def _blocks(self, op: Op, fn: str) -> None:
        dst, src = op.operands[:2]
        self.emit(f"{fn}({self.name(dst)}, {self.name(src)}, {self.attr(op, 'n_burst')}, {self.attr(op, 'burst_len')}, "
                  f"{self.attr(op, 'src_stride', 0)}, {self.attr(op, 'dst_stride', 0)});", op)

    def op_dma_ub_to_ub(self, op: Op) -> None:
        # c220's copy_ubuf_to_ubuf issues on the V queue (the op's IR pipe is V): it races the vector
        # ops around it exactly as D-065 describes, so it joins the V-V hazard tracking like any vec.* op.
        if getattr(self.mp, "arch", None) == "c220":
            self._vhazard(op)
        self._blocks(op, "ub_to_ub")

    def op_dma_ub_to_l1(self, op: Op) -> None:
        self._blocks(op, "ub_to_l1")

    def op_dma_gm_to_l1(self, op: Op) -> None:
        self._blocks(op, "gm_to_l1")

    def op_dma_gm_to_l1_pad(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"gm_to_l1_pad({self.name(dst)}, {self.name(src)}, {self.attr(op, 'n_burst')}, "
                  f"{self.attr(op, 'burst_len_byte')}, {self.attr(op, 'src_stride_byte', 0)}, {self.attr(op, 'dst_stride', 0)});", op)

    # NZ tile origins (``*_row0 / *_col0``) repeat the operand view's offsets (RFC-0007 §2): the folded view is the
    # address and the attrs are never added again.
    def op_dma_ub_to_l1_nd2nz(self, op: Op) -> None:
        dst, src = op.operands[:2]
        n_src = self.attr(op, "n_src")
        self.emit(f"ub_to_l1_nd2nz({self.name(dst)}, {self.name(src)}, {self.attr(op, 'm_src')}, {n_src}, "
                  f"{self.attr(op, 'm_dst')}, {self.attr(op, 'n_dst')}, {self.attr(op, 'N_src', op.attrs['n_src'])});", op)

    def op_dma_ub_to_l1_nz(self, op: Op) -> None:
        dst, src = op.operands[:2]
        M_src = self.aexpr(op, "M_src", op.attrs["m_src"])
        self.emit(f"ub_to_l1_nz({self.name(dst)}, {self.name(src)}, {self.attr(op, 'm_src')}, {self.attr(op, 'n_src')}, "
                  f"{self.attr(op, 'm_dst')}, {self.attr(op, 'n_dst')}, {self.cexpr(M_src)});", op)

    def _nd2nz(self, op: Op, fn: str) -> None:
        dst, src = op.operands[:2]
        self.emit(f"{fn}({self.name(dst)}, {self.name(src)}, {self.attr(op, 'M')}, {self.attr(op, 'N')}, "
                  f"{self.attr(op, 'M_dst', op.attrs['M'])}, {self.attr(op, 'N_src', op.attrs['N'])});", op)

    def op_dma_gm_to_l1_nd2nz(self, op: Op) -> None:
        self._nd2nz(op, "gm_to_l1_nd2nz")

    def op_dma_gm_to_l1_dn2nz(self, op: Op) -> None:
        self._nd2nz(op, "gm_to_l1_dn2nz")

    def op_dma_gm_to_l1_mx_scale_nd2nz(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"gm_to_l1_mx_scale_nd2nz({self.name(dst)}, {self.name(src)}, {self.attr(op, 'rows')}, "
                  f"{self.attr(op, 'k_groups')}, {self.attr(op, 'src_k_groups', op.attrs['k_groups'])});", op)

    def op_dma_set_constant_to_l1(self, op: Op) -> None:
        t = op.operands[0]
        dt = self._dtype_of(t)
        self.emit(f"set_constant_to_l1({self.name(t)}, ({cpp.ctype(dt)})({self.attr(op, 'val', 0, dt)}), "
                  f"{self.attr(op, 'n_blocks')});", op)

    def op_dma_l1_to_l0(self, op: Op) -> None:
        dst, src = op.operands[:2]
        m_src = self.aexpr(op, "m_src")
        trans = self.flag(op, "src_is_transpose", False)
        self.emit(f"l1_to_l0<{trans}>({self.name(dst)}, {self.name(src)}, {self.cexpr(m_src)}, {self.attr(op, 'n_src')}, "
                  f"{self.attr(op, 'm_copy' if 'm_copy' in op.attrs else 'm_dst')}, {self.attr(op, 'n_dst')});", op)

    def op_dma_l1_to_l0_mx(self, op: Op) -> None:
        dst, src = op.operands[:2]
        src_mx = op.attrs.get("src_mx")
        if not isinstance(src_mx, Value):
            raise CceGap(op, "l1_to_l0.mx needs the scale window in 'src_mx'")
        m_src, n_src = self.aexpr(op, "m_src"), self.aexpr(op, "n_src")
        trans = bool(op.attrs.get("src_is_transpose", False))
        scale_src_rows = n_src if trans else m_src
        mx_off = views.add(self.aexpr(op, "src_mx_offset_element", 0),
                           views.add(views.mul(self.aexpr(op, "src_mx_row0", 0), 32),
                                     views.mul(self.aexpr(op, "src_mx_col0", 0), scale_src_rows)))
        mx_at = self.at(src_mx, views.mul(mx_off, cpp.esize(self._dtype_of(src_mx))))
        self.emit(f"l1_to_l0_mx<{'true' if trans else 'false'}>({self.name(dst)}, {self.name(src)}, {mx_at}, "
                  f"{self.cexpr(m_src)}, {self.cexpr(n_src)}, {self.attr(op, 'm_dst')}, {self.attr(op, 'n_dst')});", op)

    def op_dma_l1_to_l0_img2col(self, op: Op) -> None:
        dst, src = op.operands[:2]
        a = {k: self.attr(op, k, d) for k, d in (("h", None), ("w", None), ("c", None), ("kh", None), ("kw", None),
                                                  ("pad_l", 0), ("pad_r", 0), ("pad_t", 0), ("pad_b", 0),
                                                  ("stride_h", 1), ("stride_w", 1), ("dil_h", 1), ("dil_w", 1),
                                                  ("k0", 0), ("m0", 0), ("k_ext", None), ("m_ext", None))}
        missing = [k for k, v in a.items() if v is None]
        if missing:
            raise CceGap(op, f"img2col without {missing}")
        self.emit(f"l1_to_l0_img2col({self.name(dst)}, {self.name(src)}, {a['h']}, {a['w']}, {a['c']}, {a['kh']}, "
                  f"{a['kw']}, {a['pad_l']}, {a['pad_r']}, {a['pad_t']}, {a['pad_b']}, {a['stride_h']}, {a['stride_w']}, "
                  f"{a['dil_h']}, {a['dil_w']}, {a['k0']}, {a['m0']}, {a['k_ext']}, {a['m_ext']});", op)

    def op_dma_l1_to_bt(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"l1_to_bt({self.name(dst)}, {self.name(src)}, {self.attr(op, 'n')});", op)

    def _mmad(self, op: Op, fn: str) -> None:
        dst, a, b = op.operands[:3]
        dst_at = self.name(dst)
        M, N, K = (self.attr(op, k) for k in ("M", "N", "K"))
        init = bool(op.attrs.get("is_init", False))
        bias = op.attrs.get("bias")
        if isinstance(bias, Value) and init:
            self.emit(f"{fn}_bias({dst_at}, {self.name(a)}, {self.name(b)}, {self.name(bias)}, {M}, {N}, {K});", op)
        else:
            self.emit(f"{fn}({dst_at}, {self.name(a)}, {self.name(b)}, {M}, {N}, {K}, {'true' if init else 'false'});", op)

    def op_cube_mmad(self, op: Op) -> None:
        self._mmad(op, "mmad")

    def op_cube_mmad_mx(self, op: Op) -> None:
        self._mmad(op, "mmad_mx")

    def _fixpipe_tail(self, op: Op) -> str:
        relu = self.flag(op, "relu", False)
        scale = op.attrs.get("scale")
        scaled = scale is not None
        scale_s = self.val(scale, F32) if scaled else "0.0f"
        offset = self.attr(op, "offset", 0)
        hybrid = self.flag(op, "hif8_hybrid", False)
        return f"{relu}, {scale_s}, {offset}, {'true' if scaled else 'false'}, {hybrid}"

    def _fixpipe_atomic(self, op: Op) -> tuple[str | None, str | None]:
        """The SPR pair around an L0C -> GM store (mapping B.13).

        c220 compiles its two units separately and its atomic helpers carry no core guard, so
        the SPR applies on whichever core issues the store -- the AIC, for a fixpipe. On c310
        both halves of that are false: every helper in that arch's block is wrapped in
        ``if ASCEND_IS_AIV`` (tensorutils_cce.h), so an SPR armed from the cube side does
        nothing, and ``l0c_to_gm_*`` takes no atomic argument to carry the mode on the store
        instead (its body is ``if ASCEND_IS_AIC``). There is nothing to reach, which is why
        this refuses rather than printing something that would quietly not accumulate.

        Inside a region the disarm belongs to ``atomic.end`` -- see ``_atomic_wrap``.
        """
        kind = self.ident(op, "atomic")
        if kind is None or kind == "none":
            return None, None
        if self.mp.arch != "c220":
            raise CceGap(op, f"atomic fixpipe stores are printed for c220 only: on {self.mp.arch} "
                             "the atomic SPR helpers are ASCEND_IS_AIV-guarded and the fixpipe "
                             "store issues on the AIC, while l0c_to_gm_* carries no atomic "
                             "argument to hold the mode instead - so neither half is reachable "
                             "(pypto reaches it as a TSTORE template argument)")
        if kind not in ("add", "max", "min"):
            raise CceGap(op, f"atomic {kind!r} has no CCE spelling")
        dt = self._dtype_of(op.operands[0])
        if self.atomic is not None:
            if kind != self.atomic:
                raise CceGap(op, f"atomic {kind!r} store inside `with atomic_{self.atomic}()`: "
                                 "one armed SPR cannot hold both operations")
            return f"SetAtomicType<{cpp.ctype(dt)}>();", None
        return f"SetAtomic{kind.capitalize()}<{cpp.ctype(dt)}>();", "SetAtomicNone();"

    def _l0c_to_gm(self, op: Op, fn: str, third: str, like: str) -> None:
        dst, src = op.operands[:2]
        pre, post = self._fixpipe_atomic(op)
        if pre:
            self.emit(pre, op)
        self.emit(f"{fn}({self.name(dst)}, {self.name(src)}, {self.attr(op, 'M')}, {self.attr(op, 'N')}, "
                  f"{self.attr(op, third, op.attrs[like])}, {self.attr(op, 'M_src', op.attrs['M'])}, {self._fixpipe_tail(op)});", op)
        if post:
            self.emit(post, op)

    def op_dma_l0c_to_gm_nz2nd(self, op: Op) -> None:
        self._l0c_to_gm(op, "l0c_to_gm_nz2nd", "N_dst", "N")

    def op_dma_l0c_to_gm_nz2nz(self, op: Op) -> None:
        self._l0c_to_gm(op, "l0c_to_gm_nz2nz", "M_pad", "M")

    def op_dma_l0c_to_gm_nz2dn(self, op: Op) -> None:
        self._l0c_to_gm(op, "l0c_to_gm_nz2dn", "M_dst", "M")

    def op_dma_l0c_to_l1(self, op: Op) -> None:
        dst, src = op.operands[:2]
        if self.mp.arch == "c220" and self._dtype_of(dst).name == "f32" and self._dtype_of(src).name == "f32":
            raise CceGap(op, "fp32 -> fp32 l0c_to_l1 has no c220 spelling (the builtin's dst overloads carry no "
                            "float; the old framework's field code disables this path too) — quant to half/bf16, "
                            "or route through UB")
        self.emit(f"l0c_to_l1({self.name(dst)}, {self.name(src)}, {self.attr(op, 'M')}, {self.attr(op, 'N')}, "
                  f"{self.attr(op, 'M_dst', op.attrs['M'])}, {self.attr(op, 'M_src', op.attrs['M'])}, {self.flag(op, 'relu', False)});", op)

    def op_dma_l0c_to_ub(self, op: Op) -> None:
        if self.ident(op, "atomic") not in (None, "none"):
            raise CceGap(op, "atomic on a UB-destination fixpipe")
        dst, src = op.operands[:2]
        dual = c310.DUAL_MODE[self.ident(op, "dual_mode", "splitm")]
        sub = self.attr(op, "sub_block_id", 0)
        self.emit(f"l0c_to_ub({self.name(dst)}, {self.name(src)}, {self.attr(op, 'M')}, {self.attr(op, 'N')}, "
                  f"{self.attr(op, 'N_dst', op.attrs['N'])}, {self.attr(op, 'M_src', op.attrs['M'])}, {dual}, (bool)({sub}), "
                  f"{self._fixpipe_tail(op)});", op)

    # ---------------------------------------------------------------- SIMT launch

    def op_simt_launch(self, op: Op) -> None:
        callee = op.operands[0]
        assert isinstance(callee, FuncRef)
        fn = self.mp.module.function(callee.name)
        threads = int(op.attrs["threads"])
        self.mp.threads[callee.name] = max(threads, self.mp.threads.get(callee.name, 0))
        args = [self.call_arg(p, a) for p, a in zip(fn.params, op.operands[1:], strict=True)]
        self.emit(f"simt::launch<{self.mp.fname(callee.name)}>({threads}{''.join(', ' + a for a in args)});", op)

    # ---------------------------------------------------------------- rendering

    def signature(self) -> str:
        params = []
        for p in self.fn.params:
            t = p.type
            if isinstance(t, MemType) and t.space in ("gm", "gmlist"):
                params.append(f"GM_ADDR {self.name(p)}_")
            elif isinstance(t, ScalarType):
                params.append(f"{cpp.ctype(t.dtype)} {self.name(p)}")
            else:
                raise CceGap(None, f"kernel parameter {p}: {t} cannot be passed from the host")
        params.append("GM_ADDR workspace")
        return f"__aicore__ inline void {self.mp.fname(self.fn.name)}({', '.join(params)})"

    def render(self) -> str:
        head = self.signature()
        for p in self.fn.params:
            t = p.type
            if isinstance(t, MemType) and t.space == "gmlist":
                self.emit(f"GMList<{cpp.ctype(t.dtype)}> {self.name(p)}({self.name(p)}_);")
            elif isinstance(t, MemType):
                self.emit(f"GMTensor<{cpp.ctype(t.dtype)}> {self.name(p)}((__gm__ {cpp.ctype(t.dtype)}*){self.name(p)}_);")
        if self.mp.arch == "c220":
            # The launch state this family's sticky SPRs are specified to start in, established
            # rather than inherited. The vendor's `matmul::clearWorkspace` establishes the same ones
            # and does it at entry, not exit, which is itself the statement that a previous kernel
            # may have left them armed; but the generated wrapper calls it for a mix op only and
            # only under `g_coreType == AscendC::AIC`, so the vector core never runs it and a pure
            # vec or pure cube op runs it on neither core. An op whose first mask-dependent
            # instruction precedes its own `vec.set_mask`, or whose GM store precedes any
            # `atomic.begin`, would otherwise take whatever the last kernel left -- which is why
            # such a unit passes its own test and fails inside a network (M10-095). Accumulation is
            # cleared on both sides, as the vendor clears it; the mask is the vector unit's.
            self.emit("SetAtomicNone();", None)
            if _ident(self.fn.attrs.get("side"), "vec") == "vec":
                self.emit("ResetMask();", None)
                self.emit("SetMaskNorm();", None)
        self.run_block(self.fn.body)
        if self.epilogue and not (self.fn.body.ops and self.fn.body.ops[-1].opcode == "cf.return"):
            for line in self.epilogue:
                self.emit(line, None)
        body = "\n".join(self.lines)
        return f"{head}\n{{\n{body}\n}}\n"


# =============================================================================== vf


class VfPrinter(FnPrinter):
    """A ``@vf`` function: bare vector intrinsics inside ``__VEC_SCOPE__``."""

    kind = "vf"

    def run_op(self, op: Op) -> None:
        grouped = [v for v in op.operands if isinstance(v, Value) and isinstance(v.type, RegType)
                   and v.type.dtype.name == "c32" and v.type.n == 2]
        if not grouped:
            super().run_op(op)
            return
        if op.opcode == "vf.reinterpret":
            src, dst = op.operands[0], op.results[0]
            if dst.type == src.type:
                self.emit(f"vector_u32x2_t& {self.name(dst)} = {self.name(src)};", op)
                return
            raise CceGap(op, "cross-width reinterpret of a complex32 register group")
        supported = {"vf.load_cont", "vf.store_cont", "vf.copy", "vf.dup", "vf.add", "vf.sub", "vf.mul",
                     "vf.div", "vf.adds", "vf.muls", "vf.select"}
        if op.opcode not in supported:
            raise CceGap(op, f"{op.opcode} on a complex32 register group")
        # c32's group ABI is two packed 256-byte carriers. Reuse the existing component printer
        # in two lexical scopes, with field names rather than new IR values (D-229 ABI issue).
        aliases = {v.name: self.name(v) for v in grouped}
        for part in range(2):
            self.emit("{", op)
            self.indent += 1
            operands = tuple(replace(v, type=RegType(v.type.dtype)) if isinstance(v, Value) and v.name in aliases else v for v in op.operands)
            attrs = dict(op.attrs)
            mask = attrs.get("mask")
            split_mask = None
            if isinstance(mask, Value):
                split_mask = Value(f"group_mask_{op.id}", MaskType(32))
                m = self.name(split_mask)
                self.emit(f"vector_bool {m};", op)
                self.emit(f"punpack({m}, {self.name(mask)}, {'LOWER' if part == 0 else 'HIGHER'});", op)
                attrs["mask"] = split_mask
            if op.opcode in ("vf.load_cont", "vf.store_cont"):
                off = attrs.get("offset", 0)
                if isinstance(off, Literal):
                    off = off.value
                if isinstance(off, int):
                    attrs["offset"] = off + part * 64
                elif part:
                    off_value = Value(f"group_offset_{op.id}", ScalarType(_dtype("i32")))
                    self.emit(f"int32_t {self.name(off_value)} = {self.val(off)} + 64;", op)
                    attrs["offset"] = off_value
            compute_mask = split_mask is not None and op.opcode in ("vf.dup", "vf.add", "vf.sub", "vf.mul", "vf.div", "vf.adds", "vf.muls")
            if compute_mask:
                attrs.pop("mask")
            for name, base in aliases.items():
                self.names[name] = f"{base}.val[{part}]"
            try:
                super().run_op(replace(op, operands=operands, attrs=attrs))
                if compute_mask:
                    dst = operands[0]
                    self.emit(f"vand({self.reg(dst)}, {self.reg(dst)}, {self.reg(dst)}, {self.name(split_mask)}, MODE_ZEROING);", op)
            finally:
                self.names.update(aliases)
            self.indent -= 1
            self.emit("}")

    # -- operands ------------------------------------------------------------------------------------

    def reg_dtype(self, v: Any) -> DType:
        t = v.type
        if isinstance(t, RegType):
            return t.dtype
        raise CceGap(None, f"{v} is not a register")

    def reg(self, v: Any) -> str:
        return self.name(v)

    def creg(self, v: Any) -> str:
        """The register as its unsigned carrier type (loads / stores / gathers)."""
        return f"({cpp.carrier_vtype(self.reg_dtype(v))}&){self.name(v)}"

    def sreg(self, v: Any) -> str:
        return f"({cpp.signed_carrier_vtype(self.reg_dtype(v))}&){self.name(v)}"

    def mask(self, op: Op, key: str = "mask") -> str:
        m = op.attrs.get(key)
        if isinstance(m, Value):
            return self.name(m)
        if any(isinstance(x, Value) and isinstance(x.type, RegType) and x.type.dtype.bits == 64 and x.type.n == 1 for x in op.operands):
            return "pset_b32(PAT_VL32)"  # a 64-bit register is 32 lanes (D-050): the two-register form's upper lanes stay out
        return "pset_b8(PAT_ALL)"

    def mode(self, op: Op) -> str:
        return c310.MERGE[self.ident(op, "merge", "zeroing")]

    def scalar(self, x: Any, dt: DType) -> str:
        """A scalar operand cast to the register element type."""
        return f"({cpp.ctype(dt)})({self.val(x, dt)})"

    def ptr(self, v: Any, off: Any = 0, ct: str | None = None) -> str:
        """A UB window as a typed pointer displaced by ``off`` elements (of the window's dtype)."""
        n = self.name(v)
        base = n if (isinstance(off, int) and off == 0) else f"({n} + ({self.val(off)}))"
        if ct is None:
            return base
        return f"(__ubuf__ {ct}*){base}"

    def pdt(self, v: Any) -> DType:
        return self._dtype_of(v)

    # -- declarations ----------------------------------------------------------------------------------

    def op_vf_reg(self, op: Op) -> None:
        r = op.results[0]
        t = r.type
        assert isinstance(t, RegType)
        vt = "vector_u32x2_t" if t.dtype.name == "c32" and t.n == 2 else cpp.vtype(t.dtype)
        self.emit(f"{vt} {self.name(r)};", op)

    def op_vf_mask(self, op: Op) -> None:
        r = op.results[0]
        t = r.type
        assert isinstance(t, MaskType)
        pat = c310.MASK_PATTERN.get(self.ident(op, "init", "all"))
        if pat is None:
            raise CceGap(op, f"unknown mask pattern {op.attrs.get('init')}")
        if t.width == 64 and t.n == 1:
            pat = {"PAT_ALL": "PAT_VL32", "PAT_H": "PAT_VL16", "PAT_Q": "PAT_VL8"}.get(pat, pat)
        self.emit(f"vector_bool {self.name(r)} = pset_b{min(t.width // t.n, 32)}({pat});", op)

    def op_vf_unalign(self, op: Op) -> None:
        r = op.results[0]
        assert isinstance(r.type, UnalignRegType)
        self.emit(f"vector_align {self.name(r)};", op)

    def op_vf_reinterpret(self, op: Op) -> None:
        r = op.results[0]
        t = r.type
        assert isinstance(t, RegType)
        if t.n == 2 and t.dtype.name == "c32":
            raise CceGap(op, "cross-width reinterpret into a complex32 register group")
        self.emit(f"{cpp.vtype(t.dtype)}& {self.name(r)} = ({cpp.vtype(t.dtype)}&){self.name(op.operands[0])};", op)  # type: ignore[arg-type]

    def op_mem_alloc(self, op: Op) -> None:
        r = op.results[0]
        t = r.type
        if isinstance(t, BufType):
            raise CceGap(op, "slot buffers inside a vf function are not printed")
        assert isinstance(t, MemType)
        if "addr" not in op.attrs:
            raise CceGap(op, "allocation without an address (run addr_alloc)")
        T = cpp.ctype(t.dtype)
        self.emit(f"__ubuf__ {T}* {self.name(r)} = (__ubuf__ {T}*)({self.val(op.attrs['addr'])});", op)

    def op_mem_workspace(self, op: Op) -> None:
        raise CceGap(op, "workspaces are kernel-level")

    def space_qual(self, space: str) -> str:
        return "__gm__" if space in ("gm", "ws", "gmlist") else "__ubuf__"

    def _def_view(self, op: Op) -> None:
        r = op.results[0]
        g = self.geo(r)
        if g.slot is not None:
            raise CceGap(op, "slot buffers inside a vf function are not printed")
        q = self.space_qual(g.space)
        T = cpp.ctype(g.dtype)
        off = views.byte_offset(g)
        root = self.name(g.root)
        if isinstance(off, int) and off == 0:
            expr = f"({q} {T}*){root}"
        else:
            expr = f"({q} {T}*)(({q} uint8_t*){root} + ({self.cexpr(off)}))"
        self.emit(f"{q} {T}* {self.name(r)} = {expr};", op)

    def op_scalar_load(self, op: Op) -> None:
        src, idx = op.operands[:2]
        self._def(op, f"{self.name(src)}[{self.val(idx)}]")  # type: ignore[arg-type]

    def op_scalar_store(self, op: Op) -> None:
        dst, idx, src = op.operands[:3]
        dt = self._dtype_of(dst)
        self.emit(f"{self.name(dst)}[{self.val(idx)}] = ({cpp.ctype(dt)})({self.val(src, dt)});", op)  # type: ignore[arg-type]

    def op_cf_for(self, op: Op) -> None:
        # the compiler accepts only ``for (uint16_t i = lo; i < hi; i += step)`` inside __VEC_SCOPE__
        i = op.results[0]
        lo, hi, step = op.operands[:3]
        n = self.name(i)
        if isinstance(step, Literal):
            if int(step.value) <= 0:
                raise CceGap(op, "vf loops count upwards: the step must be positive (the induction variable is uint16_t)")
            inc = self.val(step)
        else:  # a scalar step: taken as positive (the interpreter's cf.for stops at hi; a non-positive step never terminates here)
            inc = f"(uint16_t)({self.val(step)})"
        self.emit(f"for (uint16_t {n} = {self.val(lo)}; {n} < {self.val(hi)}; {n} += {inc}) {{", op)
        self.indent += 1
        self.run_block(op.regions[0])
        self.indent -= 1
        self.emit("}")

    def op_vf_ub_cursor(self, op: Op) -> None:
        r = op.results[0]
        src = op.operands[0]
        T = cpp.ctype(self._dtype_of(src))
        self.emit(f"__ubuf__ {T}* {self.name(r)} = {self.name(src)};", op)  # type: ignore[arg-type]

    # -- loads / stores --------------------------------------------------------------------------------

    def op_vf_load_cont(self, op: Op) -> None:
        dst, src = op.operands[:2]
        mode = self.ident(op, "mode", "norm")
        dist = c310.LOAD_DIST.get(mode)
        if dist is None:
            raise CceGap(op, f"load_cont mode {mode!r}")
        dt = self.reg_dtype(dst)
        off = self.attr(op, "offset", 0)
        if cpp.esize(dt) == 8:  # 32 lanes: a single-register load, deinterleaved into the two-register halves (D-050)
            if mode != "norm":
                raise CceGap(op, f"64-bit load_cont with mode {mode!r}")
            if dst.type.n == 2:
                self.emit(f"vlds({self.creg(dst)}, {self.ptr(src, 0, 'uint64_t')}, {off});", op)
                return
            t = self._tmp(op, "vector_u64", "w")
            self.emit(f"vlds({t}, {self.ptr(src, 0, 'uint64_t')}, {off}, NORM);", op)
            self.emit(f"vdintlv((vector_u32&){self.reg(dst)}.val[0], (vector_u32&){self.reg(dst)}.val[1], (vector_u32&){t}, (vector_u32&){t});", op)
            return
        self.emit(f"vlds({self.creg(dst)}, {self.ptr(src, 0, cpp.carrier_ctype(dt))}, {off}, {dist});", op)

    def op_vf_load_interleave(self, op: Op) -> None:
        d0, d1, src = op.operands[:3]
        mode = self.ident(op, "mode", "dintlv_b16")
        dist = c310.LOAD_INTLV_DIST.get(mode)
        if dist is None:
            raise CceGap(op, f"load_interleave mode {mode!r}")
        dt = self.reg_dtype(d0)
        self.emit(f"vlds({self.creg(d0)}, {self.creg(d1)}, {self.ptr(src, 0, cpp.carrier_ctype(dt))}, "
                  f"{self.attr(op, 'offset', 0)}, {dist});", op)

    def op_vf_store_cont(self, op: Op) -> None:
        dst, src = op.operands[:2]
        mode = self.ident(op, "mode", "norm")
        dt = self.reg_dtype(src)
        if mode == "norm":
            mode = f"norm_b{min(cpp.esize(dt), 4) * 8}"
        dist = c310.STORE_DIST.get(mode)
        if dist is None:
            raise CceGap(op, f"store_cont mode {mode!r}")
        off = self.attr(op, "offset", 0)
        if cpp.esize(dt) == 8:
            # One 32-bit store of the halves interleaved back into memory order: exactly the 32 elements (256 bytes).
            # The compiler header's two-register vsts issues a second 256-byte store behind them (masked off in
            # principle) — it zeroed the next UB tile on the board and cannsim (the scatter kernels' index tile, D-050).
            if not mode.startswith("norm"):
                raise CceGap(op, f"64-bit store_cont with mode {mode!r}")
            if src.type.n == 2:
                self.emit(f"vsts({self.creg(src)}, {self.ptr(dst, 0, 'uint64_t')}, {off}, {self.mask(op)});", op)
                return
            n = op.id
            r = self.reg(src)
            self.emit(f"vector_u32 s0{n}, s1{n};", op)
            self.emit(f"vintlv(s0{n}, s1{n}, (vector_u32&){r}.val[0], (vector_u32&){r}.val[1]);", op)
            if "mask" in op.attrs:  # the 32 lane bits become (low, high) pairs, as the header's macro does it
                self.emit(f"vector_bool lm{n}, hm{n};", op)
                self.emit(f"pintlv_b32(lm{n}, hm{n}, {self.mask(op)}, {self.mask(op)});", op)
                m = f"lm{n}"
            else:
                m = "pset_b32(PAT_ALL)"
            off2 = int(off) * 2 if str(off).lstrip("-").isdigit() else f"({off}) * 2"
            self.emit(f"vsts(s0{n}, {self.ptr(dst, 0, 'uint32_t')}, {off2}, NORM_B32, {m});", op)
            return
        pdt = self.pdt(dst)
        ct = cpp.carrier_ctype(pdt) if mode.startswith("pack") or mode.startswith("first") else cpp.carrier_ctype(dt)
        self.emit(f"vsts({self.creg(src)}, {self.ptr(dst, 0, ct)}, {off}, {dist}, {self.mask(op)});", op)

    def op_vf_store_interleave(self, op: Op) -> None:
        dst, s0, s1 = op.operands[:3]
        mode = self.ident(op, "mode", "intlv_b16")
        dist = c310.STORE_INTLV_DIST.get(mode)
        if dist is None:
            raise CceGap(op, f"store_interleave mode {mode!r}")
        dt = self.reg_dtype(s0)
        self.emit(f"vsts({self.creg(s0)}, {self.creg(s1)}, {self.ptr(dst, 0, cpp.carrier_ctype(dt))}, "
                  f"{self.attr(op, 'offset', 0)}, {dist}, {self.mask(op)});", op)

    def op_vf_load(self, op: Op) -> None:
        """The 32-byte block copy UB -> register. Stride 1 is the contiguous load (``vlds``), zeroed to the mask when
        one is given (the interpreter's masked-off lanes are zero, D-052). A real stride is ``vsldb`` in the register's
        own element type (``c310.VSLDB_ELEM``): issued through the unsigned carrier — a reinterpret-cast of the register
        around the intrinsic — the consumers read the register's previous value at the board's -O3 (D-052 / D-055; the
        root cause and AscendC's immunity, D-057). A dtype without a native form keeps the carrier and the idempotent
        ``vor`` that made every consumer see the loaded blocks."""
        dst, src = op.operands[:2]
        dt = self.reg_dtype(dst)
        stride = self.attr(op, "blk_stride", 1)
        off = op.attrs.get("offset", 0)
        if str(stride) == "1":
            reg, ptr = self.creg(dst), self.ptr(src, off, cpp.carrier_ctype(dt))
            suffix = "" if cpp.esize(dt) == 8 and dst.type.n == 2 else ", NORM"
            self.emit(f"vlds({reg}, {ptr}, 0{suffix});", op)
            if "mask" in op.attrs:
                self.emit(f"vand({reg}, {reg}, {reg}, {self.mask(op)}, MODE_ZEROING);", op)
            return
        elem = c310.VSLDB_ELEM.get(dt.name)
        if elem is not None:
            self.emit(f"vsldb({self.reg(dst)}, {self.ptr(src, off, elem)}, (int32_t)(({stride}) << 16), {self.mask(op)});", op)
            return
        reg, ptr = self.creg(dst), self.ptr(src, off, cpp.carrier_ctype(dt))
        self.emit(f"vsldb({reg}, {ptr}, (int32_t)(({stride}) << 16), {self.mask(op)});", op)
        self.emit(f"vor({reg}, {reg}, {reg}, pset_b8(PAT_ALL), MODE_ZEROING);", op)

    def op_vf_store(self, op: Op) -> None:
        """The 32-byte block copy register -> UB: stride 1 is the contiguous masked store, a real stride ``vsstb`` in
        the register's own element type (``c310.VSSTB_ELEM``, the mirror of the load's rule, D-057)."""
        dst, src = op.operands[:2]
        dt = self.reg_dtype(src)
        stride = self.attr(op, "blk_stride", 1)
        off = op.attrs.get("offset", 0)
        if str(stride) == "1":
            dist = "" if cpp.esize(dt) == 8 and src.type.n == 2 else f", NORM_B{min(cpp.esize(dt), 4) * 8}"
            self.emit(f"vsts({self.creg(src)}, {self.ptr(dst, off, cpp.carrier_ctype(dt))}, 0{dist}, "
                      f"{self.mask(op)});", op)
            return
        elem = c310.VSSTB_ELEM.get(dt.name)
        if elem is not None:
            self.emit(f"vsstb({self.reg(src)}, {self.ptr(dst, off, elem)}, (int32_t)(({stride}) << 16), {self.mask(op)});", op)
            return
        self.emit(f"vsstb({self.creg(src)}, {self.ptr(dst, off, cpp.carrier_ctype(dt))}, (int32_t)(({stride}) << 16), {self.mask(op)});", op)

    def op_vf_load_unalign_pre(self, op: Op) -> None:
        ureg, src = op.operands[:2]
        dt = self.pdt(src)
        self.emit(f"vldas({self.name(ureg)}, {self.ptr(src, op.attrs.get('offset', 0), cpp.carrier_ctype(dt))});", op)  # type: ignore[arg-type]

    def op_vf_load_unalign(self, op: Op) -> None:
        dst, src, ureg = op.operands[:3]
        dt = self.reg_dtype(dst)
        ct = cpp.carrier_ctype(dt)
        post = self.ident(op, "post_mode", "normal")
        stride = op.attrs.get("stride")
        if stride is not None or post == "update":
            off = op.attrs.get("offset", 0)
            if not (isinstance(off, int) and off == 0):
                raise CceGap(op, "post-updating unaligned load with a non-zero offset")
            self.emit(f"vldus({self.creg(dst)}, {self.name(ureg)}, (__ubuf__ {ct}*&){self.name(src)}, "  # type: ignore[arg-type]
                      f"(uint32_t)({self.val(stride if stride is not None else 0)}), POST_UPDATE);", op)
            return
        self.emit(f"vldus({self.creg(dst)}, {self.name(ureg)}, {self.ptr(src, op.attrs.get('offset', 0), ct)});", op)  # type: ignore[arg-type]

    def op_vf_store_unalign(self, op: Op) -> None:
        dst, src, ureg = op.operands[:3]
        dt = self.reg_dtype(src)
        ct = cpp.carrier_ctype(dt)
        count = self.attr(op, "count")
        if count is None:
            raise CceGap(op, "store_unalign without count")
        off = op.attrs.get("offset", 0)
        if not (isinstance(off, int) and off == 0):
            raise CceGap(op, "store_unalign with a non-zero offset (the cursor advances in place)")
        self.emit(f"vstus({self.name(ureg)}, (uint32_t)({count}), {self.creg(src)}, (__ubuf__ {ct}*&){self.name(dst)}, "  # type: ignore[arg-type]
                  "POST_UPDATE);", op)

    def op_vf_store_unalign_post(self, op: Op) -> None:
        dst, ureg = op.operands[:2]
        ct = cpp.carrier_ctype(self.pdt(dst))
        stride = self.attr(op, "stride", 0)
        # A5 has no flush that leaves the cursor: vstas always post-updates (RFC-0001 §6.11, I042).
        self.emit(f"vstas({self.name(ureg)}, (__ubuf__ {ct}*&){self.name(dst)}, (int32_t)({stride}), POST_UPDATE);", op)  # type: ignore[arg-type]

    def op_vf_gather_copy(self, op: Op) -> None:
        dst, src, index = op.operands[:3]
        dt = self.reg_dtype(dst)
        pdt = self.pdt(src)
        idx = self.reg(index)
        if self.reg_dtype(index).bits == 64:  # the element indices are the low halves of the two-register index
            idx = f"(vector_u32){idx}.val[0]"
        self.emit(f"vgather2({self.sreg(dst)}, {self.ptr(src, op.attrs.get('offset', 0), cpp.signed_carrier_ctype(pdt))}, "
                  f"{idx}, {self.mask(op)});", op)

    def op_vf_gatherb(self, op: Op) -> None:
        dst, src, index = op.operands[:3]
        self._no64(op)
        dt = self.reg_dtype(dst)
        if cpp.esize(dt) == 8:  # the single-register (32-lane) vgatherb, deinterleaved into the two halves (D-050)
            if "mask" in op.attrs:
                raise CceGap(op, "masked 64-bit gatherb (the native v32 form takes a b64 predicate, not the two-register mask)")
            t = self._tmp(op, "vector_u64", "g")
            self.emit(f"vgatherb((vector_s64&){t}, {self.ptr(src, op.attrs.get('offset', 0), 'int64_t')}, {self.reg(index)}, pset_b8(PAT_ALL));", op)
            self.emit(f"vdintlv((vector_u32&){self.reg(dst)}.val[0], (vector_u32&){self.reg(dst)}.val[1], (vector_u32&){t}, (vector_u32&){t});", op)
            return
        self.emit(f"vgatherb({self.sreg(dst)}, {self.ptr(src, op.attrs.get('offset', 0), cpp.signed_carrier_ctype(dt))}, "
                  f"{self.reg(index)}, {self.mask(op)});", op)

    def op_vf_scatter_copy(self, op: Op) -> None:
        dst, src, index = op.operands[:3]
        idx = self.reg(index)
        if self.reg_dtype(index).bits == 64:  # the element indices are the low halves of the two-register index
            idx = f"(vector_u32){idx}.val[0]"
        dt = self.reg_dtype(src)
        base = self.ptr(dst, op.attrs.get("offset", 0), cpp.carrier_ctype(dt))
        if dt.bits == 64:
            if src.type.n == 2:
                self.emit(f"vscatter({self.creg(src)}, {base}, {idx}, {self.mask(op)});", op)
                return
            # AscendC's ScatterImplB64 (dav_c310): the 32 values are the 64 32-bit lanes of the register in memory order
            # (low, high, low, high ...), the index register becomes (2 i, 2 i + 1) pairs and one 32-bit vscatter moves
            # every half. The compiler header's two-register vscatter macro scattered every lane to element 0 when the
            # index came from a deinterleaved 64-bit register (board and cannsim, D-050).
            if "mask" in op.attrs:
                raise CceGap(op, "masked 64-bit scatter (the 32-bit scatter of the halves takes a 64-lane predicate)")
            n = op.id
            s = self.reg(src)
            self.emit(f"vector_u32 i2{n}, i3{n}, ix{n}, ixh{n}, d0{n}, d1{n};", op)
            self.emit(f"vmuls(i2{n}, {idx}, (uint32_t)2, pset_b32(PAT_ALL), MODE_ZEROING);", op)
            self.emit(f"vadds(i3{n}, i2{n}, (uint32_t)1, pset_b32(PAT_ALL), MODE_ZEROING);", op)
            self.emit(f"vintlv(ix{n}, ixh{n}, i2{n}, i3{n});", op)
            self.emit(f"vintlv(d0{n}, d1{n}, (vector_u32&){s}.val[0], (vector_u32&){s}.val[1]);", op)
            self.emit(f"vscatter(d0{n}, (__ubuf__ uint32_t*){base}, ix{n}, pset_b32(PAT_ALL));", op)
            return
        self.emit(f"vscatter({self.creg(src)}, {base}, {idx}, {self.mask(op)});", op)

    def op_vf_ub_to_mask(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"plds({self.name(dst)}, {self.ptr(src, op.attrs.get('offset', 0), 'uint32_t')}, 0, NORM);", op)  # type: ignore[arg-type]

    def op_vf_mask_to_ub(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"psts({self.name(src)}, {self.ptr(dst, op.attrs.get('offset', 0), 'uint32_t')}, 0, NORM);", op)  # type: ignore[arg-type]

    # -- compute ---------------------------------------------------------------------------------------

    # The two-register (vector_2xvl_*) forms the c310 compiler header provides (its __VF_*_B64 macros: add / sub with
    # carry builtins, mul, div, mod, max / min, and / or / xor / not, neg, abs, shifts, the *s scalar forms, cadd / cmax /
    # cmin, sel, dup / br / mov, cmp, cvt, lds / sts, gather2 / scatter with a 32-bit index) plus the sequences printed
    # below (abssub, muladddst, axpy, gatherb).
    B64_OK = {"vf.add", "vf.sub", "vf.mul", "vf.div", "vf.max", "vf.min", "vf.and", "vf.or", "vf.xor", "vf.shiftl",
              "vf.shiftr", "vf.neg", "vf.abs", "vf.not", "vf.adds", "vf.muls", "vf.maxs", "vf.mins", "vf.shiftls",
              "vf.shiftrs", "vf.cadd", "vf.cmax", "vf.cmin", "vf.select", "vf.gather", "vf.cmp", "vf.cmps", "vf.dup",
              "vf.copy", "vf.cast", "vf.load_cont", "vf.store_cont", "vf.gather_copy", "vf.scatter_copy", "vf.gatherb",
              "vf.reinterpret", "vf.abssub", "vf.muladddst", "vf.muldstadd", "vf.mod", "vf.axpy", "vf.interleave", "vf.deinterleave"}
    COMPLEX_OK = {"vf.add", "vf.sub", "vf.mul", "vf.div", "vf.copy", "vf.load_cont", "vf.store_cont", "vf.reinterpret",
                  "vf.dup", "vf.adds", "vf.muls", "vf.abs"}

    def _no64(self, op: Op) -> None:
        for x in op.operands:
            if isinstance(x, Value) and isinstance(x.type, RegType):
                if x.type.dtype.kind == "complex" and op.opcode not in self.COMPLEX_OK:
                    raise CceGap(op, f"{op.opcode} on complex registers: only add / sub / mul / div / copy / load / store are "
                                     "printed (the parts are the halves of the carrier register)")
                if x.type.dtype.bits == 64 and op.opcode not in self.B64_OK:
                    raise CceGap(op, f"{op.opcode} on 64-bit registers: the c310 compiler header has no two-register form of it")

    def _tmp(self, op: Op, vt: str, tag: str) -> str:
        """A per-op temporary register of the vector type ``vt``."""
        n = f"{tag}{op.id}"
        self.emit(f"{vt} {n};", op)
        return n

    def _unary(self, op: Op) -> None:
        self._no64(op)
        dst, src = op.operands[:2]
        self.emit(f"{c310.UNARY[op.opcode]}({self.reg(dst)}, {self.reg(src)}, {self.mask(op)}, MODE_ZEROING);", op)

    def op_vf_log2(self, op: Op) -> None:
        self._log_base(op, 1.4426950408889634)  # 1 / ln 2

    def op_vf_log10(self, op: Op) -> None:
        self._log_base(op, 0.4342944819032518)  # 1 / ln 10

    def _log_base(self, op: Op, scale: float) -> None:
        """c310 has one logarithm (``vln``): log2 / log10 are ``vln`` then a ``vmuls`` by the constant, both under the op's
        mask (the interpreter's exact log2 / log10 differ by the rounding of that product)."""
        self._no64(op)
        dst, src = op.operands[:2]
        dt = self.reg_dtype(dst)
        self.emit(f"vln({self.reg(dst)}, {self.reg(src)}, {self.mask(op)}, MODE_ZEROING);", op)
        self.emit(f"vmuls({self.reg(dst)}, {self.reg(dst)}, {self.scalar(scale, dt)}, {self.mask(op)}, MODE_ZEROING);", op)

    def _binary(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        dt = self.reg_dtype(dst)
        if dt.kind == "complex":
            self._complex_binary(op)
            return
        self._no64(op)
        rb = self.reg(b)
        if op.opcode in ("vf.shiftl", "vf.shiftr") and dt.bits == 64:  # the two-register shifts take a 32-bit amount vector
            rb = f"(vector_s32){rb}.val[0]" if self.reg_dtype(b).bits == 64 else f"(vector_s32){rb}"
        self.emit(f"{c310.BINARY[op.opcode]}({self.reg(dst)}, {self.reg(a)}, {rb}, {self.mask(op)}, MODE_ZEROING);", op)

    def _scalar_binary(self, op: Op) -> None:
        dst, src, v = op.operands[:3]
        dt = self.reg_dtype(src)
        if dt.kind == "complex" and op.opcode in ("vf.adds", "vf.muls"):
            self._complex_scalar_binary(op)
            return
        self._no64(op)
        s = self.val(v) if op.opcode in ("vf.shiftls", "vf.shiftrs") else self.scalar(v, dt)
        self.emit(f"{c310.SCALAR_BINARY[op.opcode]}({self.reg(dst)}, {self.reg(src)}, {s}, {self.mask(op)}, MODE_ZEROING);", op)

    def _reduce(self, op: Op) -> None:
        self._no64(op)
        dst, src = op.operands[:2]
        dt = self.reg_dtype(src)
        index = op.attrs.get("index") is True
        if dt.bits == 64:  # the __VF_VCADD_B64 / __VF_VCMAX_MIN_B64 forms take (dst, src, mask)
            if index:
                raise CceGap(op, "64-bit vcmax/vcmin has no qualified index lane")
            self.emit(f"{c310.REDUCE[op.opcode]}({self.reg(dst)}, {self.reg(src)}, {self.mask(op)});", op)
            if op.opcode in ("vf.cmax", "vf.cmin"):  # vcmax / vcmin leave the index in lane 1; the register semantics is value, zeros
                self.emit(f"vand({self.reg(dst)}.val[0], {self.reg(dst)}.val[0], {self.reg(dst)}.val[0], pset_b32(PAT_VL1), MODE_ZEROING);", op)
            return
        self.emit(f"{c310.REDUCE[op.opcode]}({self.reg(dst)}, {self.reg(src)}, {self.mask(op)}, MODE_ZEROING);", op)
        if op.opcode in ("vf.cmax", "vf.cmin") and not index:  # every width (cannsim and board, D-050): keep lane 0
            c = self.creg(dst)
            self.emit(f"vand({c}, {c}, {c}, pset_b{min(dt.bits, 32)}(PAT_VL1), MODE_ZEROING);", op)

    for _name in c310.UNARY:
        locals()["op_" + _name.replace(".", "_")] = _unary
    for _name in c310.BINARY:
        locals()["op_" + _name.replace(".", "_")] = _binary
    for _name in c310.SCALAR_BINARY:
        locals()["op_" + _name.replace(".", "_")] = _scalar_binary
    for _name in c310.REDUCE:
        locals()["op_" + _name.replace(".", "_")] = _reduce
    del _name

    # -- 64-bit sequences (no single two-register intrinsic) and complex arithmetic -----------------------

    def op_vf_abssub(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        dt = self.reg_dtype(dst)
        if dt.bits != 64:
            self._binary(op)
            return
        m, t = self.mask(op), self._tmp(op, cpp.vtype(dt), "d")
        if dt.kind == "int":  # |a - b| in two's complement
            self.emit(f"vsub({t}, {self.reg(a)}, {self.reg(b)}, {m}, MODE_ZEROING);", op)
            self.emit(f"vabs({self.reg(dst)}, {t}, {m}, MODE_ZEROING);", op)
        else:  # unsigned: max - min
            self.emit(f"vmax({t}, {self.reg(a)}, {self.reg(b)}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmin({self.reg(dst)}, {self.reg(a)}, {self.reg(b)}, {m}, MODE_ZEROING);", op)
            self.emit(f"vsub({self.reg(dst)}, {t}, {self.reg(dst)}, {m}, MODE_ZEROING);", op)

    def op_vf_muladddst(self, op: Op) -> None:  # dst = a * b + dst
        dst, a, b = op.operands[:3]
        dt = self.reg_dtype(dst)
        if dt.bits != 64:
            self._binary(op)
            return
        m, t = self.mask(op), self._tmp(op, cpp.vtype(dt), "p")
        self.emit(f"vmul({t}, {self.reg(a)}, {self.reg(b)}, {m}, MODE_ZEROING);", op)
        self.emit(f"vadd({self.reg(dst)}, {t}, {self.reg(dst)}, {m}, MODE_ZEROING);", op)

    def op_vf_muldstadd(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        dt = self.reg_dtype(dst)
        if dt.bits != 64:
            self._binary(op)
            return
        m, t = self.mask(op), self._tmp(op, cpp.vtype(dt), "p")
        self.emit(f"vmul({t}, {self.reg(dst)}, {self.reg(a)}, {m}, MODE_ZEROING);", op)
        self.emit(f"vadd({self.reg(dst)}, {t}, {self.reg(b)}, {m}, MODE_ZEROING);", op)

    def op_vf_mod(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        if any(self.reg_dtype(v).name not in ("i64", "u64") for v in (dst, a, b)):
            raise CceGap(op, "integer remainder requires i64 or u64 registers")
        self.emit(f"vmod({self.reg(dst)}, {self.reg(a)}, {self.reg(b)}, {self.mask(op)}, MODE_ZEROING);", op)

    def op_vf_axpy(self, op: Op) -> None:  # dst = src * scalar + dst
        dst, src, v = op.operands[:3]
        dt = self.reg_dtype(dst)
        if dt.bits != 64:
            self._scalar_binary(op)
            return
        m, t = self.mask(op), self._tmp(op, cpp.vtype(dt), "p")
        self.emit(f"vmuls({t}, {self.reg(src)}, {self.scalar(v, dt)}, {m}, MODE_ZEROING);", op)
        self.emit(f"vadd({self.reg(dst)}, {t}, {self.reg(dst)}, {m}, MODE_ZEROING);", op)

    def _complex_imm(self, op: Op, dt: DType) -> tuple[str, str]:
        """The (re, im) component literals of a complex immediate, rounded the way the board's constructor does
        (complex32((half)re, (half)im) / complex64(re f, im f)): c32 parts round to half here so the fp32 compute
        below sees the value the silicon sees."""
        import struct

        x = op.operands[-1]
        k = x.value if isinstance(x, Literal) else None
        if not isinstance(k, complex):
            raise CceGap(op, f"{op.opcode}: a complex register needs a complex immediate, got {x!r}")
        if dt.name == "c32":
            cre, cim = (struct.unpack("e", struct.pack("e", v))[0] for v in (k.real, k.imag))
        else:
            cre, cim = float(k.real), float(k.imag)
        return cre, cim

    def _complex_scalar_binary(self, op: Op) -> None:
        """dst = src (+|*) K for a complex immediate K, on the (re, im) parts like _complex_binary: c64 works on the
        fp32 plane halves with vadds / vmuls; c32 deinterleaves (add) or widens to fp32 (mul, one rounding back)."""
        dst, src = op.operands[:2]
        dt = self.reg_dtype(dst)
        cre, cim = self._complex_imm(op, dt)
        half = dt.name == "c32"
        real_part, imag_part = (f"(half){cre}f", f"(half){cim}f") if half else (f"{cre}f", f"{cim}f")
        n = op.id
        add = op.opcode == "vf.adds"
        if dt.bits == 64:
            d = [f"(vector_f32&){self.reg(v)}.val[{i}]" for v in (dst, src) for i in (0, 1)]
            if add:
                self.emit(f"vadds({d[0]}, {d[2]}, {real_part}, {self.mask(op)}, MODE_ZEROING);", op)
                self.emit(f"vadds({d[1]}, {d[3]}, {imag_part}, {self.mask(op)}, MODE_ZEROING);", op)
                return
            self._complex_muls_parts(op, "vector_f32", d[0], d[1], d[2], d[3], real_part, imag_part, self.mask(op))
            return
        if "mask" in op.attrs:
            raise CceGap(op, "masked complex32 ops (the mask would need re-laning after the deinterleave)")
        if add:
            self.emit(f"vector_f16 z{n}, ar{n}, ai{n}, dr{n}, di{n}, hi{n};", op)
            self.emit(f"vbr(z{n}, (half)0.0f);", op)
            self.emit(f"vdintlv(ar{n}, ai{n}, (vector_f16&){self.reg(src)}, z{n});", op)
            self.emit(f"vadds(dr{n}, ar{n}, {real_part}, pset_b8(PAT_ALL), MODE_ZEROING);", op)
            self.emit(f"vadds(di{n}, ai{n}, {imag_part}, pset_b8(PAT_ALL), MODE_ZEROING);", op)
            self.emit(f"vintlv((vector_f16&){self.reg(dst)}, hi{n}, dr{n}, di{n});", op)
            return
        # mul widens to fp32 and rounds once back, like _complex_binary's c32 mul (the immediate's parts are already
        # half-rounded by _complex_imm, spelled as fp32 literals for the plane compute)
        self.emit(f"vector_f32 ar{n}, ai{n}, dr{n}, di{n};", op)
        self.emit(f"vcvt(ar{n}, (vector_f16&){self.reg(src)}, pset_b8(PAT_ALL), PART_EVEN, MODE_ZEROING);", op)
        self.emit(f"vcvt(ai{n}, (vector_f16&){self.reg(src)}, pset_b8(PAT_ALL), PART_ODD, MODE_ZEROING);", op)
        self._complex_muls_parts(op, "vector_f32", f"dr{n}", f"di{n}", f"ar{n}", f"ai{n}",
                                 f"{cre}f", f"{cim}f", "pset_b8(PAT_ALL)")
        self.emit(f"vector_f16 e{n}, o{n};", op)
        self.emit(f"vcvt(e{n}, dr{n}, pset_b8(PAT_ALL), ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);", op)
        self.emit(f"vcvt(o{n}, di{n}, pset_b8(PAT_ALL), ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);", op)
        self.emit(f"vor((vector_u16&){self.reg(dst)}, (vector_u16&)e{n}, (vector_u16&)o{n}, pset_b8(PAT_ALL), MODE_ZEROING);", op)

    def _complex_muls_parts(self, op: Op, vt: str, dr: str, di: str, ar: str, ai: str,
                            real_part: str, imag_part: str, m: str) -> None:
        """(ar + ai i) * (re + im i) with scalar parts: every product before any write (dst may alias src)."""
        n = op.id
        self.emit(f"{vt} t1{n}, t2{n}, t3{n}, t4{n};", op)
        self.emit(f"vmuls(t1{n}, {ar}, {real_part}, {m}, MODE_ZEROING);", op)
        self.emit(f"vmuls(t2{n}, {ai}, {imag_part}, {m}, MODE_ZEROING);", op)
        self.emit(f"vmuls(t3{n}, {ar}, {imag_part}, {m}, MODE_ZEROING);", op)
        self.emit(f"vmuls(t4{n}, {ai}, {real_part}, {m}, MODE_ZEROING);", op)
        self.emit(f"vsub({dr}, t1{n}, t2{n}, {m}, MODE_ZEROING);", op)
        self.emit(f"vadd({di}, t3{n}, t4{n}, {m}, MODE_ZEROING);", op)

    def op_vf_abs(self, op: Op) -> None:
        dst, src = op.operands[:2]
        sdt = self.reg_dtype(src)
        if sdt.kind != "complex":
            self._unary(op)
            return
        # |z| is real (MicroAPI::Abs's complex form on the board): sqrt(re^2 + im^2) on the parts, into the
        # low src-lanes of the real destination (the half-register result the old script describes).
        if "mask" in op.attrs:
            raise CceGap(op, "masked complex abs (the result re-lanes into the low half)")
        n = op.id
        if sdt.bits == 64:
            m = self.mask(op)
            planes = [f"(vector_f32&){self.reg(src)}.val[{i}]" for i in (0, 1)]
            self.emit(f"vector_f32 r2{n}, i2{n};", op)
            self.emit(f"vmul(r2{n}, {planes[0]}, {planes[0]}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmul(i2{n}, {planes[1]}, {planes[1]}, {m}, MODE_ZEROING);", op)
            self.emit(f"vadd(r2{n}, r2{n}, i2{n}, {m}, MODE_ZEROING);", op)
            self.emit(f"vsqrt({self.reg(dst)}, r2{n}, {m}, MODE_ZEROING);", op)
            return
        # c32: widen the packed (re, im) lanes to fp32, |z| there, narrow once (even lanes) and compact
        self.emit(f"vector_f32 ar{n}, ai{n};", op)
        self.emit(f"vcvt(ar{n}, (vector_f16&){self.reg(src)}, pset_b8(PAT_ALL), PART_EVEN, MODE_ZEROING);", op)
        self.emit(f"vcvt(ai{n}, (vector_f16&){self.reg(src)}, pset_b8(PAT_ALL), PART_ODD, MODE_ZEROING);", op)
        self.emit(f"vmul(ar{n}, ar{n}, ar{n}, pset_b8(PAT_ALL), MODE_ZEROING);", op)
        self.emit(f"vmul(ai{n}, ai{n}, ai{n}, pset_b8(PAT_ALL), MODE_ZEROING);", op)
        self.emit(f"vadd(ar{n}, ar{n}, ai{n}, pset_b8(PAT_ALL), MODE_ZEROING);", op)
        self.emit(f"vsqrt(ar{n}, ar{n}, pset_b8(PAT_ALL), MODE_ZEROING);", op)
        self.emit(f"vector_f16 e{n}, z{n}, hi{n};", op)
        self.emit(f"vcvt(e{n}, ar{n}, pset_b8(PAT_ALL), ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);", op)
        self.emit(f"vbr(z{n}, (half)0.0f);", op)
        self.emit(f"vdintlv((vector_f16&){self.reg(dst)}, hi{n}, e{n}, z{n});", op)

    def _complex_binary(self, op: Op) -> None:
        """Complex add / sub / mul / div on the (re, im) parts: for c64 the two halves of the 2xvl carrier are the fp32
        parts (a 64-bit load deinterleaves them); for c32 add / sub the packed (re16, im16) lanes are deinterleaved into
        two fp16 registers and interleaved back, mul / div widen the parts to fp32 and round once (unmasked: the mask
        would need re-laning)."""
        dst, a, b = op.operands[:3]
        dt = self.reg_dtype(dst)
        n = op.id
        if dt.bits == 64:
            parts = [(f"(vector_f32&){self.reg(v)}.val[0]", f"(vector_f32&){self.reg(v)}.val[1]") for v in (dst, a, b)]
            self._complex_parts(op, "vector_f32", *parts[0], *parts[1], *parts[2], self.mask(op))
            return
        if "mask" in op.attrs:
            raise CceGap(op, "masked complex32 ops (the mask would need re-laning after the deinterleave)")
        if op.opcode in ("vf.add", "vf.sub"):
            self.emit(f"vector_f16 z{n}, ar{n}, ai{n}, br{n}, bi{n}, dr{n}, di{n}, hi{n};", op)
            self.emit(f"vbr(z{n}, (half)0.0f);", op)
            self.emit(f"vdintlv(ar{n}, ai{n}, (vector_f16&){self.reg(a)}, z{n});", op)
            self.emit(f"vdintlv(br{n}, bi{n}, (vector_f16&){self.reg(b)}, z{n});", op)
            self._complex_parts(op, "vector_f16", f"dr{n}", f"di{n}", f"ar{n}", f"ai{n}", f"br{n}", f"bi{n}", "pset_b8(PAT_ALL)")
            self.emit(f"vintlv((vector_f16&){self.reg(dst)}, hi{n}, dr{n}, di{n});", op)
            return
        # mul / div compute in fp32 and round once, as torch's ComplexHalf (the goldens) does: the parts widen straight
        # out of the packed lanes (PART_EVEN = re, PART_ODD = im) and the results narrow back into them; every part is
        # read before the destination is written, so it may alias an operand. Four fp16 products were up to 7.8e-3 off
        # on the board and cannsim.
        self.emit(f"vector_f32 ar{n}, ai{n}, br{n}, bi{n}, dr{n}, di{n};", op)
        for part, src in (("ar", a), ("ai", a), ("br", b), ("bi", b)):
            which = "PART_EVEN" if part.endswith("r") else "PART_ODD"
            self.emit(f"vcvt({part}{n}, (vector_f16&){self.reg(src)}, pset_b8(PAT_ALL), {which}, MODE_ZEROING);", op)
        self._complex_parts(op, "vector_f32", f"dr{n}", f"di{n}", f"ar{n}", f"ai{n}", f"br{n}", f"bi{n}", "pset_b8(PAT_ALL)")
        # c310's f32 -> f16 vcvt is zeroing-only (the header asserts MODE_MERGING away): the even and odd halves are
        # converted into two registers and or-ed together (the lanes a part does not write are zero).
        self.emit(f"vector_f16 e{n}, o{n};", op)
        self.emit(f"vcvt(e{n}, dr{n}, pset_b8(PAT_ALL), ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);", op)
        self.emit(f"vcvt(o{n}, di{n}, pset_b8(PAT_ALL), ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);", op)
        self.emit(f"vor((vector_u16&){self.reg(dst)}, (vector_u16&)e{n}, (vector_u16&)o{n}, pset_b8(PAT_ALL), MODE_ZEROING);", op)

    def _complex_parts(self, op: Op, vt: str, dr: str, di: str, ar: str, ai: str, br: str, bi: str, m: str) -> None:
        n = op.id
        if op.opcode in ("vf.add", "vf.sub"):
            fn = "vadd" if op.opcode == "vf.add" else "vsub"
            self.emit(f"{fn}({dr}, {ar}, {br}, {m}, MODE_ZEROING);", op)
            self.emit(f"{fn}({di}, {ai}, {bi}, {m}, MODE_ZEROING);", op)
            return
        self.emit(f"{vt} t1{n}, t2{n}, t3{n}, t4{n};", op)  # every product before any write: dst may alias a or b
        if op.opcode == "vf.mul":  # (ar br - ai bi) + (ar bi + ai br) i
            self.emit(f"vmul(t1{n}, {ar}, {br}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmul(t2{n}, {ai}, {bi}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmul(t3{n}, {ar}, {bi}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmul(t4{n}, {ai}, {br}, {m}, MODE_ZEROING);", op)
            self.emit(f"vsub({dr}, t1{n}, t2{n}, {m}, MODE_ZEROING);", op)
            self.emit(f"vadd({di}, t3{n}, t4{n}, {m}, MODE_ZEROING);", op)
            return
        if op.opcode == "vf.div":  # ((ar br + ai bi) + (ai br - ar bi) i) / (br br + bi bi)
            self.emit(f"{vt} d{n};", op)
            self.emit(f"vmul(t1{n}, {br}, {br}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmul(t2{n}, {bi}, {bi}, {m}, MODE_ZEROING);", op)
            self.emit(f"vadd(d{n}, t1{n}, t2{n}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmul(t1{n}, {ar}, {br}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmul(t2{n}, {ai}, {bi}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmul(t3{n}, {ai}, {br}, {m}, MODE_ZEROING);", op)
            self.emit(f"vmul(t4{n}, {ar}, {bi}, {m}, MODE_ZEROING);", op)
            self.emit(f"vadd(t1{n}, t1{n}, t2{n}, {m}, MODE_ZEROING);", op)
            self.emit(f"vsub(t3{n}, t3{n}, t4{n}, {m}, MODE_ZEROING);", op)
            self.emit(f"vdiv({dr}, t1{n}, d{n}, {m}, MODE_ZEROING);", op)
            self.emit(f"vdiv({di}, t3{n}, d{n}, {m}, MODE_ZEROING);", op)
            return
        raise CceGap(op, f"{op.opcode} on complex registers")

    def op_vf_copy(self, op: Op) -> None:
        dst, src = op.operands[:2]
        if "mask" in op.attrs:  # a zeroing copy: vmov merges, so select between the source and a zero register
            dt = self.reg_dtype(dst)
            carrier = dt.name == "hif8"
            reg = self.creg if carrier else self.reg
            z = self._tmp(op, cpp.carrier_vtype(dt) if carrier else cpp.vtype(dt), "z")
            self.emit(f"vbr({z}, {self.scalar(0, _dtype('u8') if carrier else dt)});", op)
            self.emit(f"vsel({reg(dst)}, {reg(src)}, {z}, {self.mask(op)});", op)
            return
        self.emit(f"vmov({self.reg(dst)}, {self.reg(src)});", op)

    def op_vf_dup(self, op: Op) -> None:
        dst, src = op.operands[:2]
        dt = self.reg_dtype(dst)
        if isinstance(src, Value) and isinstance(src.type, RegType):
            reg = self.creg if dt.name == "hif8" else self.reg
            self.emit(f"vdup({reg(dst)}, {reg(src)}, {self.mask(op)}, POS_LOWEST, MODE_ZEROING);", op)
            return
        if dt.name == "hif8":
            raise CceGap(op, "scalar broadcast to hif8 has no C310 intrinsic; use explicit byte-carrier bits or a float-register cast")
        if dt.kind == "complex":
            if "mask" in op.attrs:
                raise CceGap(op, "a masked complex dup (the mask would need re-laning)")
            cre, cim = self._complex_imm(op, dt)
            if dt.bits == 64:  # the carrier halves are the fp32 planes
                self.emit(f"vbr((vector_f32&){self.reg(dst)}.val[0], {cre}f);", op)
                self.emit(f"vbr((vector_f32&){self.reg(dst)}.val[1], {cim}f);", op)
                return
            n = op.id
            self.emit(f"vector_f16 kr{n}, ki{n}, kh{n};", op)
            self.emit(f"vbr(kr{n}, (half){cre}f);", op)
            self.emit(f"vbr(ki{n}, (half){cim}f);", op)
            self.emit(f"vintlv((vector_f16&){self.reg(dst)}, kh{n}, kr{n}, ki{n});", op)
            return
        if "mask" in op.attrs:
            self.emit(f"vdup({self.reg(dst)}, {self.scalar(src, dt)}, {self.mask(op)}, MODE_ZEROING);", op)
        else:
            self.emit(f"vbr({self.reg(dst)}, {self.scalar(src, dt)});", op)

    def op_vf_cmp(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        self.emit(f"vcmp_{c310.CMP[self.ident(op, 'mode')]}({self.name(dst)}, {self.reg(a)}, {self.reg(b)}, {self.mask(op)});", op)  # type: ignore[arg-type]

    def op_vf_cmps(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        dt = self.reg_dtype(a)
        self.emit(f"vcmps_{c310.CMP[self.ident(op, 'mode')]}({self.name(dst)}, {self.reg(a)}, {self.scalar(b, dt)}, {self.mask(op)});", op)  # type: ignore[arg-type]

    def op_vf_select(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        # hif8 has no native vsel overload: select through the uint8 carrier (a bitwise pick is dtype-blind)
        r = self.creg if self.reg_dtype(dst).name == "hif8" else self.reg
        self.emit(f"vsel({r(dst)}, {r(a)}, {r(b)}, {self.mask(op)});", op)

    def op_vf_arange(self, op: Op) -> None:
        dst = op.operands[0]
        dt = self.reg_dtype(dst)
        if cpp.esize(dt) == 8 and dt.name != "i64":
            raise CceGap(op, "64-bit arange requires i64")
        order = c310.ORDER[self.ident(op, "mode", "increase")]
        start = self.scalar(op.attrs.get('v', 0), _dtype("i32") if dt.name == "i64" else dt)
        self.emit(f"vci({self.reg(dst)}, {start}, {order});", op)

    def op_vf_interleave(self, op: Op) -> None:
        d0, d1, s0, s1 = op.operands[:4]
        self.emit(f"vintlv({self.creg(d0)}, {self.creg(d1)}, {self.creg(s0)}, {self.creg(s1)});", op)

    def op_vf_deinterleave(self, op: Op) -> None:
        d0, d1, s0, s1 = op.operands[:4]
        self.emit(f"vdintlv({self.creg(d0)}, {self.creg(d1)}, {self.creg(s0)}, {self.creg(s1)});", op)

    def op_vf_gather(self, op: Op) -> None:
        dst, src, index = op.operands[:3]
        self.emit(f"vselr({self.creg(dst)}, {self.creg(src)}, {self.creg(index)});", op)

    def op_vf_gathermask(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"vsqz({self.reg(dst)}, {self.reg(src)}, {self.mask(op)}, MODE_NO_STORED);", op)

    op_vf_squeeze = op_vf_gathermask

    def op_vf_unsqueeze(self, op: Op) -> None:
        dst = op.operands[0]
        self.emit(f"vusqz({self.reg(dst)}, {self.mask(op)});", op)

    def op_vf_expsub(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        self.emit(f"vexpdif({self.reg(dst)}, {self.reg(a)}, {self.reg(b)}, {self.mask(op)}, {c310.PART[self.ident(op, 'layout', 'zero')]});", op)

    def op_vf_mulscast(self, op: Op) -> None:
        dst, src, v = op.operands[:3]
        dt = self.reg_dtype(src)
        self.emit(f"vmulscvt({self.reg(dst)}, {self.reg(src)}, {self.scalar(v, dt)}, {self.mask(op)}, "
                  f"{c310.PART[self.ident(op, 'layout', 'zero')]});", op)

    def op_vf_histograms(self, op: Op) -> None:
        dst, src = op.operands[:2]
        fn = "chistv2" if self.ident(op, "mode", "frequency") == "accumulate" else "dhistv2"
        self.emit(f"{fn}({self.reg(dst)}, {self.reg(src)}, {self.mask(op)}, Bin_N{int(op.attrs.get('bin_group', 0))});", op)

    def op_vf_pack(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"vpack({self.reg(dst)}, {self.reg(src)}, {c310.HILO[self.ident(op, 'part', 'lowest')]});", op)

    def op_vf_cast(self, op: Op) -> None:
        dst, src = op.operands[:2]
        ddt, sdt = self.reg_dtype(dst), self.reg_dtype(src)
        if (sdt.name, ddt.name) in TRUNCATING_CASTS and op.attrs.get("saturate", False):
            raise CceGap(op, "cast i64 -> i32 has no saturation selector; clamp before narrowing")
        shape = c310.CAST_SHAPES.get((ddt.name, sdt.name))
        if shape is None:
            raise CceGap(op, f"cast {sdt} -> {ddt} is not in the c310 cast matrix (arch/c310.CAST_SHAPES mirrors the Cast API's "
                             "tables 3 and 6-9 for Ascend 950PR/950DT; the missing families are unsigned <-> float, narrowing to "
                             "int8 and the mixed-sign widenings, and they go through the signed or the wider type)")
        if self.ident(op, "merge", "zeroing") == "merging" and (ddt.name, sdt.name) in c310.MERGE_REJECTED:
            raise CceGap(op, f"merging cast {sdt} -> {ddt}: c310 has no merging cast at all — every row of the Cast API's "
                             "tables 6-9 is MaskMergeMode::ZEROING, and the compiler refuses MODE_MERGING on every pair "
                             "probed — convert into a temporary and select")
        layout = self.ident(op, "layout", "zero")
        rnd = c310.ROUND[self.ident(op, "round", "rint")]
        allowed = c310.CAST_ROUNDS.get((ddt.name, sdt.name))
        if allowed is not None and rnd not in allowed:
            raise CceGap(op, f"cast {sdt} -> {ddt} rounds {' / '.join(sorted(allowed))} only on c310 (the compiler header's "
                             f"static_assert); this cast asks for {rnd} (round={self.ident(op, 'round', 'rint')!r})")
        self.emit(cast_call(self.reg(dst), self.reg(src), shape, m=self.mask(op), rnd=rnd,
                            sat="RS_ENABLE" if bool(op.attrs.get("saturate", False)) else "RS_DISABLE",
                            part=c310.PART.get(layout, "PART_EVEN"), part_t=c310.PART_T.get(layout, "PART_P0"), mode=self.mode(op)), op)

    # -- masks -----------------------------------------------------------------------------------------

    def _mask_binary(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        self.emit(f"{c310.MASK_BINARY[op.opcode]}({self.name(dst)}, {self.name(a)}, {self.name(b)}, {self.mask(op)});", op)  # type: ignore[arg-type]

    for _name in c310.MASK_BINARY:
        locals()["op_" + _name.replace(".", "_")] = _mask_binary
    del _name

    def op_vf_mask_not(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"pnot({self.name(dst)}, {self.name(src)}, {self.mask(op)});", op)  # type: ignore[arg-type]

    def op_vf_mask_mov(self, op: Op) -> None:
        dst, src = op.operands[:2]
        if "mask" in op.attrs:
            self.emit(f"pmov({self.name(dst)}, {self.name(src)}, {self.mask(op)});", op)  # type: ignore[arg-type]
        else:
            self.emit(f"pmov({self.name(dst)}, {self.name(src)});", op)  # type: ignore[arg-type]

    def op_vf_mask_sel(self, op: Op) -> None:
        dst, a, b = op.operands[:3]
        self.emit(f"psel({self.name(dst)}, {self.name(a)}, {self.name(b)}, {self.mask(op)});", op)  # type: ignore[arg-type]

    def op_vf_mask_pack(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"ppack({self.name(dst)}, {self.name(src)}, {c310.HILO[self.ident(op, 'mode', 'lowest')]});", op)  # type: ignore[arg-type]

    def op_vf_mask_unpack(self, op: Op) -> None:
        dst, src = op.operands[:2]
        self.emit(f"punpack({self.name(dst)}, {self.name(src)}, {c310.HILO[self.ident(op, 'mode', 'lowest')]});", op)  # type: ignore[arg-type]

    def _mask_width(self, v: Any) -> int:
        t = v.type
        assert isinstance(t, MaskType)
        return min(t.width // t.n, 32)

    def op_vf_mask_interleave(self, op: Op) -> None:
        d0, d1, s0, s1 = op.operands[:4]
        self.emit(f"pintlv_b{self._mask_width(d0)}({self.name(d0)}, {self.name(d1)}, {self.name(s0)}, {self.name(s1)});", op)  # type: ignore[arg-type]

    def op_vf_mask_deinterleave(self, op: Op) -> None:
        d0, d1, s0, s1 = op.operands[:4]
        self.emit(f"pdintlv_b{self._mask_width(d0)}({self.name(d0)}, {self.name(d1)}, {self.name(s0)}, {self.name(s1)});", op)  # type: ignore[arg-type]

    def op_vf_mask_update(self, op: Op) -> None:
        mask_update(self, op)  # counters.py (I040)

    def op_vf_mask_from_spr(self, op: Op) -> None:
        dst = op.operands[0]
        w = self._mask_width(dst)
        if w == 8:
            raise CceGap(op, "movp has no b8 form")
        self.emit(f"{self.name(dst)} = movp_b{w}();", op)  # type: ignore[arg-type]

    def op_vf_barrier(self, op: Op) -> None:
        key = (self.ident(op, "src"), self.ident(op, "dst"))
        tag = c310.MEM_BAR.get(key)  # type: ignore[arg-type]
        if tag is None:
            raise CceGap(op, f"no mem_bar for {key}")
        self.emit(f"mem_bar({tag});", op)

    def op_vf_clear_spr(self, op: Op) -> None:
        self.emit("sprclr(SPR_AR);", op)

    # -- rendering -------------------------------------------------------------------------------------

    def signature(self) -> str:
        params = []
        for p in self.fn.params:
            t = p.type
            if isinstance(t, MemType) and t.space == "ub":
                params.append(f"__ubuf__ {cpp.ctype(t.dtype)}* {self.name(p)}")
            elif isinstance(t, ScalarType | CellType):
                params.append(f"{cpp.ctype(t.dtype)} {self.name(p)}")
            else:
                raise CceGap(None, f"vf parameter {p}: {t} cannot be passed to a vf function")
        return f"__aicore__ inline void {self.mp.fname(self.fn.name)}({', '.join(params)})"

    def render(self) -> str:
        from .vf_render import render_vf

        return render_vf(self)


# =============================================================================== simt


class SimtPrinter(VfPrinter):
    """A ``@simt`` function: plain C on the compiler's SIMT layer."""

    kind = "simt"

    op_cf_for = FnPrinter.op_cf_for

    def op_simt_thread_id(self, op: Op) -> None:
        self._def(op, "(int32_t)simt::thread_id()")

    def op_simt_thread_num(self, op: Op) -> None:
        self._def(op, "(int32_t)simt::thread_num()")

    def op_simt_block_idx(self, op: Op) -> None:
        # `simt_block_idx()` is the CUBE CORE id - the frontend's own doc, the interpreter's answer
        # (`lane.cube_idx`), and what every ported kernel assumes. The compiler's `blockIdx.x` inside
        # a `__simt_vf__` is NOT that in a MIX binary: a board probe says it equals `GetVecIdx()`
        # exactly (0 .. 2N-1) while `gridDim.x` still counts AICs, so the pair is inconsistent and
        # `simt_block_idx() * 2 + GetSubBlockIdx()` - the sharding the kernels are written with -
        # skipped half its slots (D-161: half of `simt_atomic_add`'s input was never accumulated).
        # Derive the cube id from the vec index with the architecture's participants-per-cube ratio,
        # exactly as the pypto printer does (D-113 (4)). An AIV_ONLY binary has no sub-block layer -
        # probed on the same board - and there `blockIdx.x` IS the vec index the contract asks for.
        if self.mp.mode == "vec" or self.mp.vec_per_cube == 1:
            self._def(op, "(int32_t)simt::blk_idx()")
        else:
            self._def(op, f"((int32_t)simt::blk_idx() / {self.mp.vec_per_cube})")

    def op_simt_block_num(self, op: Op) -> None:
        # `gridDim.x` needs no correction in either mode: it answered the CUBE count in the mix
        # binary and the VEC count in the AIV-only one, which is the contract both times.
        self._def(op, "(int32_t)simt::blk_num()")

    def op_simt_barrier(self, op: Op) -> None:
        self.emit("simt::barrier();", op)

    def op_simt_load(self, op: Op) -> None:
        src, idx = op.operands[:2]
        self._def(op, f"{self.name(src)}[{self.val(idx)}]")  # type: ignore[arg-type]

    def op_simt_store(self, op: Op) -> None:
        dst, idx, src = op.operands[:3]
        dt = self._dtype_of(dst)
        self.emit(f"{self.name(dst)}[{self.val(idx)}] = ({cpp.ctype(dt)})({self.val(src, dt)});", op)  # type: ignore[arg-type]

    def op_simt_atomic(self, op: Op) -> None:
        dst, idx, src = op.operands[:3]
        kind = self.ident(op, "op", "add")
        if kind not in ("add", "sub", "max", "min", "exch", "and", "or", "xor", "cas", "inc", "dec"):
            raise CceGap(op, f"simt atomic {kind!r}")
        dt = self._dtype_of(dst)
        if kind in ("and", "or", "xor") and dt.name not in ("i32", "u32", "i64", "u64"):
            raise CceGap(op, f"simt atomic {kind} on {dt}: the c310 builtins take 32- / 64-bit integers")
        if kind in ("inc", "dec") and dt.name != "u32":
            raise CceGap(op, f"simt atomic {kind} on {dt}: atomicInc / atomicDec take uint32 only")
        ptr = f"{self.name(dst)} + ({self.val(idx)})"  # type: ignore[arg-type]
        if kind == "cas":
            if len(op.operands) < 4:
                raise CceGap(op, "simt atomic cas without its compare operand")
            expr = f"simt::atomic_cas({ptr}, ({cpp.ctype(dt)})({self.val(op.operands[3], dt)}), ({cpp.ctype(dt)})({self.val(src, dt)}))"
        else:
            expr = f"simt::atomic_{kind}({ptr}, ({cpp.ctype(dt)})({self.val(src, dt)}))"
        if op.results:
            self._def(op, expr)
        else:
            self.emit(expr + ";", op)

    def op_simt_threadfence(self, op: Op) -> None:
        self.emit("simt::threadfence();", op)

    def op_simt_threadfence_block(self, op: Op) -> None:
        self.emit("simt::threadfence_block();", op)

    _SIMT_F32 = ("exp", "exp2", "log", "log2", "log1p", "sin", "cos", "tanh", "rsqrt",
                 "rint", "round", "floor", "ceil", "trunc")

    def _simt_math(self, op: Op, name: str, n: int, want_float: bool = True) -> None:
        for x in op.operands[:n]:
            dt = _value_dtype(x)
            if want_float and dt is not None and dt.name != "f32":
                raise CceGap(op, f"simt.{name} on {dt.name}: the SIMT math layer is f32 "
                                 "(cast the operand first)")
        args = ", ".join(self.val(x) for x in op.operands[:n])
        self._def(op, f"simt::{name}({args})")

    def op_simt_fmod(self, op: Op) -> None:
        self._simt_math(op, "fmod", 2)

    def op_simt_fma(self, op: Op) -> None:
        for x in op.operands[:3]:
            dt = _value_dtype(x)
            if dt is not None and dt.name != "f32":
                raise CceGap(op, f"simt.fma on {dt.name}: __fma is f32")
        self._def(op, f"__fma({self.val(op.operands[0])}, {self.val(op.operands[1])}, {self.val(op.operands[2])})")

    def op_simt_isnan(self, op: Op) -> None:
        self._def(op, f"(int32_t)__isnan({self.val(op.operands[0])})")

    def op_simt_isinf(self, op: Op) -> None:
        self._def(op, f"(int32_t)__isinf({self.val(op.operands[0])})")

    def op_simt_isfinite(self, op: Op) -> None:
        self._def(op, f"(int32_t)__isfinite({self.val(op.operands[0])})")

    def op_simt_popc(self, op: Op) -> None:
        self._def(op, f"(int32_t)__popc((uint32_t)({self.val(op.operands[0])}))")

    def op_simt_ffs(self, op: Op) -> None:
        self._def(op, f"simt::ffs((uint32_t)({self.val(op.operands[0])}))")

    def op_simt_mul_hi(self, op: Op) -> None:
        t0 = _value_dtype(op.operands[0])
        signed = t0 is None or t0.kind == "int"
        fn = "__mulhi" if signed else "__umulhi"
        self._def(op, f"{fn}({self.val(op.operands[0])}, {self.val(op.operands[1])})")

    def op_mem_alloc(self, op: Op) -> None:
        raise CceGap(op, "allocations inside a simt function are not printed")

    def signature(self) -> str:
        params = []
        for p in self.fn.params:
            t = p.type
            if isinstance(t, MemType) and t.space in ("ub", "gm", "ws"):
                params.append(f"{self.space_qual(t.space)} {cpp.ctype(t.dtype)}* {self.name(p)}")
            elif isinstance(t, ScalarType | CellType):
                params.append(f"{cpp.ctype(t.dtype)} {self.name(p)}")
            else:
                raise CceGap(None, f"simt parameter {p}: {t} cannot be passed to a simt function")
        threads = self.mp.threads.get(self.fn.name, 1024)
        return f"__simt_vf__ __launch_bounds__({threads}) inline void {self.mp.fname(self.fn.name)}({', '.join(params)})"

    def render(self) -> str:
        head = self.signature()
        self.indent = 1
        self.run_block(self.fn.body)
        body = "\n".join(self.lines)
        return f"{head}\n{{\n{body}\n}}\n"


def _value_dtype(x):
    """The scalar dtype of an operand, or None for a literal / non-scalar."""
    t = getattr(x, "type", None)
    return getattr(t, "dtype", None) if isinstance(t, ScalarType | CellType) else None


def _simt_unary_method(name: str):
    def method(self, op):
        self._simt_math(op, name, 1)
    method.__name__ = f"op_simt_{name}"
    return method


for _n in SimtPrinter._SIMT_F32:
    setattr(SimtPrinter, f"op_simt_{_n}", _simt_unary_method(_n))


# =============================================================================== module


class ModulePrinter:
    def __init__(self, module: Module, block_dim: int | None = None, entry: str | None = None) -> None:
        if module.level != "lowered":
            raise CceGap(None, f"the cce backend prints Lowered modules only (got {module.ir!r}); run the pipeline first")
        self.module = module
        self.defs = Defs(module)
        meta = dict(module.attrs.get("meta", {}))
        self.kernel = str(meta.get("kernel", module.name))
        # the entry symbol and the .cpp base name; a launcher may need a spelling of its own (the CANN custom-op
        # build derives both from the op type with its own camel -> snake rule, which drops the underscore
        # before a digit: matmul_mknk_2dgrid -> MatmulMknk2dgrid -> matmul_mknk2dgrid)
        self.entry_name = c_ident(entry) if entry else c_ident(self.kernel)
        self.mode = _ident(module.attrs.get("mode"), "mix")
        self.device = module.device
        prof = _load_device(self.device) if self.device else None
        self.arch = prof.arch if prof else "c310"  # c310 (a5) | c220 (a2)
        # vector participants per cube core (2 on both a5 and a2); the ratio the SIMT block id is
        # derived with, see SimtPrinter.op_simt_block_idx
        self.vec_per_cube = max(1, prof.vec_cores // prof.cube_cores) if prof and prof.cube_cores else 2
        views.set_l1_fp32_zz(self.arch == "c220")  # fp32 L1 windows fold as ZZ on the c220 cube (RFC-0008 §5)
        self.outputs = [v.name if isinstance(v, Value) else str(v) for v in meta.get("outputs", [])]
        self.block_dim = block_dim if block_dim is not None else meta.get("block_dim")
        self.threads: dict[str, int] = {}
        self.workspaces: list[dict[str, Any]] = []
        self._fnames: dict[str, str] = {}

    def fname(self, ir_name: str) -> str:
        n = self._fnames.get(ir_name)
        if n is None:
            n = c_ident(ir_name)
            if ir_name.startswith(self.kernel + "."):
                n = c_ident(self.kernel) + "_" + ir_name[len(self.kernel) + 1:]
            self._fnames[ir_name] = n
        return n

    def note_workspace(self, op: Op, fp: FnPrinter) -> None:
        name = str(op.attrs.get("name", op.results[0].name))
        if any(w["name"] == name for w in self.workspaces):
            return
        t = op.results[0].type
        assert isinstance(t, MemType)
        numel = op.attrs.get("numel")
        offset = op.attrs.get("offset", 0)
        params = {p.name for p in fp.fn.params}
        self.workspaces.append({
            "name": name, "dtype": t.dtype.name,
            "numel": numel if isinstance(numel, int) else self.host_expr(numel, fp, params, op),
            "offset": offset if isinstance(offset, int) else self.host_expr(offset, fp, params, op),
            "dims": [dim_str(d) for d in t.dims],
        })

    def host_expr(self, x: Any, fp: FnPrinter, params: set[str], at: Op) -> str:
        """A scalar value as a C expression over the kernel parameters only (what the host tiling function can
        evaluate): the defining scalar ops are folded back to the parameters."""
        if isinstance(x, Literal):
            return cpp.literal(x)
        if isinstance(x, bool | int | float):
            return cpp.literal(x)
        if not isinstance(x, Value):
            raise CceGap(at, f"workspace size {x!r} is not a scalar")
        if x.name in params:
            return cpp.api(x.name)
        d = fp.defs.op(x)
        if d is not None and d.opcode in ("core.cube_num", "core.vec_num", "simt.block_num"):
            # a core-count query in a workspace shape: the host knows the launch geometry once block_dim is fixed
            if not isinstance(self.block_dim, int):
                raise CceGap(at, f"workspace size depends on {d.opcode} but the module carries no block_dim; "
                                 "record / launch with a fixed block_dim to size it on the host")
            n = self.block_dim
            if d.opcode == "core.vec_num":
                n = n if self.mode == "vec" else 2 * n
            elif d.opcode == "core.cube_num" and self.mode == "vec":
                n = 0
            return str(n)
        if d is not None and d.opcode == "scalar.cell":
            # a Var the kernel never re-assigns is its initial value; a re-assigned cell has no host-time value
            target = d.results[0].name if d.results else None
            sets = [o for o in fp.fn.body.walk()
                    if o.opcode == "scalar.set" and o.operands and isinstance(o.operands[0], Value) and o.operands[0].name == target]
            if not sets:
                return self.host_expr(d.attrs.get("init"), fp, params, at)
            raise CceGap(at, f"workspace size depends on {x}, a Var re-assigned in the kernel; the host tiling function "
                             "reads only parameters and write-once Vars")
        if d is None or not d.opcode.startswith("scalar."):
            raise CceGap(at, f"workspace size depends on {x} ({d.opcode if d else 'a parameter of another kind'}), which the "
                             "host tiling function cannot evaluate; sizes must be arithmetic of the kernel parameters")
        kind = d.opcode[7:]
        if kind == "const":
            return cpp.literal(d.attrs["value"])
        a = [self.host_expr(o, fp, params, at) for o in d.operands]
        if kind == "cmp":
            return f"({a[0]} {cpp.CMP_OPS[str(d.attrs['pred'])]} {a[1]})"
        if kind in ("add", "sub", "mul", "div", "mod", "shl", "shr", "and", "or", "xor"):
            sym = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%", "shl": "<<", "shr": ">>", "and": "&", "or": "|",
                   "xor": "^"}[kind]
            return f"({a[0]} {sym} {a[1]})"
        if kind == "ceil_div":
            return f"(({a[0]} + {a[1]} - 1) / {a[1]})"
        if kind == "min":
            return f"std::min<int64_t>({a[0]}, {a[1]})"
        if kind == "max":
            return f"std::max<int64_t>({a[0]}, {a[1]})"
        if kind == "align":
            n = int(d.attrs["n"])
            return f"(({a[0]} + {n - 1}) / {n} * {n})"
        if kind == "neg":
            return f"(-{a[0]})"
        if kind == "cast":
            return a[0]
        if kind == "select":
            return f"({a[0]} ? {a[1]} : {a[2]})"
        raise CceGap(at, f"workspace size uses {d.opcode}, which the host tiling function cannot evaluate")

    def params(self) -> list[Value]:
        sides = [f for f in self.module.functions if f.kind == "func"]
        if not sides:
            raise CceGap(None, "no side functions (run split_sides)")
        return list(sides[0].params)

    def entry(self, sides: dict[str, Function]) -> str:
        params = self.params()
        tensors = cpp.entry_tensors(params, self.outputs)
        scalars = [p for p in params if isinstance(p.type, ScalarType)]
        for p in params:
            if not isinstance(p.type, MemType | ScalarType):
                raise CceGap(None, f"kernel parameter {p}: {p.type} cannot be passed from the host")
        pnames = {p.name: c_ident(p.name) for p in params}
        sig = ", ".join(f"GM_ADDR {pnames[p.name]}" for p in tensors) + (", " if tensors else "") + "GM_ADDR workspace, GM_ADDR tiling"
        lines = ['#include "tensorutils_cce.h"', "using namespace ascrip;"]
        for side in ("cube", "vec"):
            if side in sides:
                lines.append(f'#include "{self.entry_name}_{side}.h"')
        lines += ["", "", f'extern "C" __global__ __aicore__ void {self.entry_name}({sig})', "{",
                  f"    KERNEL_TASK_TYPE_DEFAULT({c310.TASK_TYPE[self.mode]});"]
        if scalars:
            lines.append("    GET_TILING_DATA(tiling_data, tiling);")
        lines.append("    pipe_barrier(PIPE_ALL);")
        for p in scalars:
            lines.append(f"    const {cpp.ctype(p.type.dtype)} {pnames[p.name]} = tiling_data.{cpp.api(p.name)};")  # type: ignore[union-attr]
        args = ", ".join(pnames[p.name] for p in params) + (", " if params else "") + "workspace"
        if "cube" in sides and self.mode in ("mix", "cube"):
            lines += ["    if ASCEND_IS_AIC {", f"        {self.fname(sides['cube'].name)}({args});", "    }"]
        if "vec" in sides and self.mode in ("mix", "vec"):
            lines += ["    if ASCEND_IS_AIV {", f"        {self.fname(sides['vec'].name)}({args});", "    }"]
        lines += ["}", ""]
        return "\n".join(lines)

    def compile(self) -> Artifacts:
        sides: dict[str, Function] = {}
        vfs: list[Function] = []
        simts: list[Function] = []
        for f in self.module.functions:
            if f.kind == "func":
                sides[_ident(f.attrs.get("side"), "vec")] = f  # type: ignore[index]
            elif f.kind == "vf":
                if self.arch != "c310":
                    raise CceGap(None, f"@vf functions are a c310 (a5) form; the {self.arch} family has none")
                vfs.append(f)
            elif f.kind == "simt":
                if self.arch != "c310":
                    raise CceGap(None, f"@simt functions are a c310 (a5) form; the {self.arch} family has none")
                simts.append(f)
            else:
                raise CceGap(None, f"function {f.name} of kind {f.kind} in a Lowered module")
        files: dict[str, bytes] = {}
        kname = self.entry_name
        includes = "".join(f'#include "{self.fname(f.name)}.h"\n' for f in vfs + simts)
        side_text: dict[str, str] = {}
        for side, fn in sides.items():
            side_text[side] = FnPrinter(self, fn).render()
        for fn in simts:  # after the sides: their launch sites fix the thread counts
            files[f"{self.fname(fn.name)}.h"] = ('#pragma once\n#include "tensorutils_cce.h"\nusing namespace ascrip;\n\n' + SimtPrinter(self, fn).render()).encode()
        for fn in vfs:
            files[f"{self.fname(fn.name)}.h"] = ('#pragma once\n#include "tensorutils_cce.h"\nusing namespace ascrip;\n\n' + VfPrinter(self, fn).render()).encode()
        for side, text in side_text.items():
            files[f"{kname}_{side}.h"] = (f'#pragma once\n#include "tensorutils_cce.h"\nusing namespace ascrip;\n{includes}\n' + text).encode()
        files[f"{kname}.cpp"] = self.entry(sides).encode()
        files["tensorutils_cce.h"] = HEADER.read_bytes()
        files["scalar_math.h"] = (HEADER.parents[2] / "shared/include/scalar_math.h").read_bytes()
        meta = self.metadata(sides, vfs, simts, sorted(files))
        files["manifest.json"] = (json.dumps(meta, indent=1) + "\n").encode()
        return Artifacts(files=files, entry=f"{kname}.cpp", metadata=meta)

    def metadata(self, sides: dict[str, Function], vfs: list[Function], simts: list[Function], names: list[str]) -> dict[str, Any]:
        params = []
        for p in self.params():
            t = p.type
            if isinstance(t, MemType):
                params.append({"name": cpp.api(p.name), "ir_name": p.name, "kind": "list" if t.space == "gmlist" else "tensor", "dtype": t.dtype.name,
                               "dims": [dim_str(d) for d in t.dims], "output": p.name in self.outputs})
            else:
                assert isinstance(t, ScalarType)
                params.append({"name": cpp.api(p.name), "ir_name": p.name, "kind": "scalar", "dtype": t.dtype.name})
        ops = sum(1 for _ in self.module.walk())
        return {
            "backend": "cce", "arch": self.arch, "kernel": self.entry_name, "ir_kernel": self.kernel,
            "device": self.device, "mode": self.mode, "task_type": c310.TASK_TYPE[self.mode],
            "block_dim": self.block_dim, "params": params, "outputs": [cpp.api(o) for o in self.outputs],
            "workspaces": self.workspaces, "sides": sorted(sides), "vf": [self.fname(f.name) for f in vfs],
            "simt": {self.fname(f.name): self.threads.get(f.name, 1024) for f in simts},
            "ops": ops, "files": names, "entry": f"{self.entry_name}.cpp",
        }


def emit_module(module: Module, block_dim: int | None = None, entry: str | None = None) -> Artifacts:
    from ...passes import PassManager
    from ...passes.integer_division import PASS_DEF
    from ...passes.bf16_scalar import PASS_DEF as BF16_SCALAR
    from ...passes.scalar_simplify import PASS_DEF as SCALAR_SIMPLIFY

    module = PassManager((PASS_DEF, BF16_SCALAR, SCALAR_SIMPLIFY), options={"compact_integer_mod": True}).run(module)
    try:
        return ModulePrinter(module, block_dim, entry).compile()
    except KeyError as exc:  # a dtype / ident without a C spelling: a gap, not a crash
        raise CceGap(None, str(exc.args[0]) if exc.args else repr(exc)) from exc


def _shape_dims(dims: tuple[Any, ...]) -> list[Any]:
    return [d if isinstance(d, int) else str(d) for d in dims]


__all__ = ["CceGap", "ModulePrinter", "FnPrinter", "VfPrinter", "SimtPrinter", "emit_module", "c_ident", "HEADER"]
