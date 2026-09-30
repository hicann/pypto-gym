# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The host-language layer the ``cce`` and ``pto_isa`` function printers share.

Scalars are C++ expressions, control flow is C++ control flow and a debug op is a comment on both
targets (RFC-0011 §7.5, §7.20), so these handlers print the same text whichever header the kernel
is built against. What differs is how text reaches the buffer, and that stays with each printer:

* ``emit`` / ``name`` / ``val`` / ``bare`` -- cce folds a temporary into its use and carries the
  folded ids on the statement; pto_isa prints one statement per op;
* ``_def`` / ``_binop`` / ``op_scalar_min`` -- the statement and operator shapes that follow from it;
* ``gap`` -- the backend's located refusal (``CceGap`` / ``PtoIsaGap``);
* ``_event_name`` -- pto_isa refuses a set / wait on an event whose declaration refused.

Only opcodes BOTH printers handle belong here: ``cce/__init__._opcodes`` and ``pto_isa.HANDLED`` are
derived from the ``op_*`` attributes, so a handler added to this class is a capability both declare.

It lives with cce because it prints through cce's ``cpp`` and ``views``, which pto_isa already imports
(RFC-0011 §7.7 reuses the vector and SIMT printers the same way); ``backends/shared`` holds only what
depends on no backend.
"""

from __future__ import annotations

from typing import Any

from ...ir import Block, Ident, Op, Value
from ...ir.saturation import SAT_BITS
from ...ir.types import DType
from ...ir.types import dtype as _dtype
from . import cpp, views


def ident(x: Any, default: str | None = None) -> str | None:
    if x is None:
        return default
    return x.name if isinstance(x, Ident) else str(x)


def dim_str(d: Any) -> str | int:
    """A manifest dimension: an int, or the scalar's name without the IR sigil."""
    return d if isinstance(d, int) else str(d).lstrip("%")


class HostPrinter:
    """Mixin for a function printer; see the module docstring for what the printer provides."""

    def gap(self, op: Op | None, why: str) -> Exception:
        raise NotImplementedError

    # ---------------------------------------------------------------- attributes, output

    def ident(self, op: Op, key: str, default: str | None = None) -> str | None:
        return ident(op.attrs.get(key), default)

    def flag(self, op: Op, key: str, default: bool = False) -> str:
        v = op.attrs.get(key, default)
        if isinstance(v, Value):
            return self.name(v)
        return "true" if v else "false"

    def geo(self, v: Value) -> views.Geo:
        g = self.geo_cache.get(v.name)
        if g is None:
            g = views.fold(v, self.defs)
            self.geo_cache[v.name] = g
        return g

    def comment(self, text: str, op: Op | None = None) -> None:
        self.emit("// " + text, op)

    def run_block(self, block: Block) -> None:
        for op in block.ops:
            self.run_op(op)

    # ---------------------------------------------------------------- scalars

    def _rdt(self, op: Op) -> DType:
        return op.results[0].type.dtype  # type: ignore[union-attr]

    def op_scalar_add(self, op: Op) -> None:
        self._binop(op, "+")

    def op_scalar_sub(self, op: Op) -> None:
        self._binop(op, "-")

    def op_scalar_mul(self, op: Op) -> None:
        self._binop(op, "*")

    def op_scalar_div(self, op: Op) -> None:
        # Integer operands: C truncation, where the reference interpreter floors. The corpus divides non-negative
        # sizes and offsets (RFC-0007 §4), and shared/ScalarFolder refuses to fold a negative operand for the same reason.
        self._binop(op, "/")

    def op_scalar_mod(self, op: Op) -> None:
        from ...ir.scalar_math import rounding
        dt = self._rdt(op)
        if dt.is_integer and rounding(op) == "floor":
            a, b = (self.val(x, dt) for x in op.operands)
            # A `__simt_vf__` body may only call a `__simt_callee__` function, and one of those
            # may only be called by one. So the helper exists twice and the caller decides.
            scope = "ascrip::simt::" if self.fn.kind == "simt" else "ascrip::"
            self._def(op, f"{scope}FloorMod<{cpp.ctype(dt)}>({a}, {b})")
        else:
            self._binop(op, "%")

    def op_scalar_shl(self, op: Op) -> None:
        self._binop(op, "<<")

    def op_scalar_shr(self, op: Op) -> None:
        self._binop(op, ">>")

    def op_scalar_xor(self, op: Op) -> None:
        self._binop(op, "^")

    def op_scalar_and(self, op: Op) -> None:
        self._binop(op, "&&" if self._rdt(op).name == "b1" else "&")

    def op_scalar_or(self, op: Op) -> None:
        self._binop(op, "||" if self._rdt(op).name == "b1" else "|")

    def op_scalar_max(self, op: Op) -> None:
        self.op_scalar_min(op, "max")

    def op_scalar_neg(self, op: Op) -> None:
        self._def(op, f"-({self.val(op.operands[0], self._rdt(op))})")

    def op_scalar_not(self, op: Op) -> None:
        a = self.val(op.operands[0])
        self._def(op, f"!({a})" if self._rdt(op).name == "b1" else f"~({a})")

    def op_scalar_abs(self, op: Op) -> None:
        dt = self._rdt(op)
        try:
            expr = cpp.scalar_abs_expr(dt, self.val(op.operands[0], dt), arch=self.mp.arch, kind=self.fn.kind)
        except ValueError as exc:
            raise self.gap(op, str(exc)) from exc
        self._def(op, expr)

    def op_scalar_select(self, op: Op) -> None:
        dt = self._rdt(op)
        c, a, b = op.operands[:3]
        self._def(op, f"({self.val(c)}) ? ({self.val(a, dt)}) : ({self.val(b, dt)})")

    def op_scalar_cast(self, op: Op) -> None:
        target = self._rdt(op)
        source = getattr(getattr(op.operands[0], 'type', None), 'dtype', None)
        if self.mp.arch == 'c310' and self.fn.kind == 'func':
            if target.name == 'bf16' and (source is None or source.name == 'f32'):
                self._def(op, f"AscendC::Cast({self.val(op.operands[0], _dtype('f32'))})")
                return
            if target.name == 'f32' and source is not None and source.name == 'bf16':
                self._def(op, f"AscendC::Cast({self.val(op.operands[0])})")
                return
        self._def(op, f"({self.scalar_ctype(op.results[0].type)})({self.val(op.operands[0])})")

    def op_scalar_const(self, op: Op) -> None:
        self._def(op, cpp.literal(op.attrs["value"], self._rdt(op)))

    def op_scalar_cell(self, op: Op) -> None:
        """A DSL cell is a mutable C++ local (RFC-0001 §5.2) -- not ``const``, unlike an SSA def."""
        r = op.results[0]
        dt = r.type.dtype  # type: ignore[union-attr]
        init = op.attrs.get("init", 0)
        self.emit(f"{cpp.ctype(dt)} {self.name(r)} = {self.bare(init, dt)};", op)

    def op_scalar_set(self, op: Op) -> None:
        cell, src = op.operands[:2]
        self.emit(f"{self.name(cell)} = {self.bare(src, cell.type.dtype)};", op)  # type: ignore[union-attr]

    # ---------------------------------------------------------------- control flow

    def op_cf_break(self, op: Op) -> None:
        self.emit("break;", op)

    def op_cf_continue(self, op: Op) -> None:
        self.emit("continue;", op)

    def op_cf_return(self, op: Op) -> None:
        for line in self.epilogue:  # a return leaves the kernel: the mutex tokens are drained first
            self.emit(line, op)
        self.emit("return;", op)

    # ---------------------------------------------------------------- core

    _SAT_BITS = SAT_BITS

    def _sat_bit(self, op: Op) -> int:
        mode = self.ident(op, "mode", "")
        bit = self._SAT_BITS.get(mode)
        if bit is None:
            raise self.gap(op, f"saturation flag mode {mode!r} (float | float8 | int | cast | global)")
        return bit

    # ---------------------------------------------------------------- debug (comments only)

    def op_debug_print(self, op: Op) -> None:
        self.comment(f"debug.print {op.attrs.get('fmt', '')!r}", op)

    def op_debug_dump(self, op: Op) -> None:
        name = self.name(op.operands[0]) if op.operands else "?"  # type: ignore[arg-type]
        self.comment(f"debug.dump {name} {op.attrs.get('desc', '')!r}", op)

    def op_debug_assert(self, op: Op) -> None:
        self.comment(f"debug.assert {op.attrs.get('msg', '')!r}", op)

    def op_debug_print_reg(self, op: Op) -> None:
        self.comment(f"debug.print_reg {op.attrs.get('label', '')!r}", op)

    # ---------------------------------------------------------------- event objects, the local mutex

    def _event_name(self, op: Op) -> str:
        return self.name(op.operands[0])  # type: ignore[arg-type]

    def op_sync_set(self, op: Op) -> None:
        self.emit(f"{self._event_name(op)}.set();", op)

    def op_sync_wait(self, op: Op) -> None:
        self.emit(f"{self._event_name(op)}.wait();", op)

    def op_sync_set_all(self, op: Op) -> None:
        self.emit(f"{self._event_name(op)}.set_all();", op)

    def op_sync_release(self, op: Op) -> None:
        self.emit(f"{self._event_name(op)}.release();", op)

    def op_sync_local_mutex_get(self, op: Op) -> None:
        self.emit(f"get_buf({cpp.PIPE[self.ident(op, 'pipe')]}, {self.val(op.attrs['id'])}, 0);", op)

    def op_sync_local_mutex_release(self, op: Op) -> None:
        self.emit(f"rls_buf({cpp.PIPE[self.ident(op, 'pipe')]}, {self.val(op.attrs['id'])}, 0);", op)


__all__ = ["HostPrinter", "dim_str", "ident"]
