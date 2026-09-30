# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The a2 family's tensor-vector instructions (``vec.*``) in the frontend (RFC-0008 §3, D-061).

Every rule reproduces the old stub (``easyasc/stub_functions/vec/*.py``): the operand checks (UB windows, the dtype
allow-lists), the inference of ``repeat`` from the destination's span and of the strides from each operand's
``(span, shape)`` (``vecutils.infer_repeat`` / ``infer_strides``: a view whose row is exactly 8 blocks walks the
allocation's row pitch, a one-block row is a broadcast layout with ``blk_stride = 0``, anything else takes the
contiguous ``1 / 8``). The ``count=`` / ``count_per_rep=`` keywords become **attributes of the op itself** (D-062):
the IR carries no vector-mode state — the old stubs' source-order mode tracker (and its automatic ``bar_v`` /
lazily-emitted switches, which broke whenever source order was not execution order) is gone. The interpreter
executes each op by its own attributes; the c220 backend materialises the SPR switches around each op from the
lowered CFG, the way CANN's own dav_c220 Level-2 calls bracket a counted instruction (no barrier: the mask SPR
writes dispatch in order with the vector instructions). The IR therefore always carries explicit repeat / stride
attributes; the interpreter and the printer never infer.
"""

from __future__ import annotations

import ast
from typing import Any

from ..ir.types import MemType
from . import dsl
from .errors import E_BAD_OPERAND, E_UNSUPPORTED
from .rules_mem import _bytes, _c0
from .values import Dyn, ElemOffset

FH = ("f32", "f16")
FHI = ("f32", "f16", "i32")
_BINARY = {"vec.add": FHI, "vec.sub": FHI, "vec.mul": FHI, "vec.div": FH, "vec.max": ("f32", "f16", "i16", "i32"), "vec.min": FHI,
           "vec.and": ("i16", "u16"), "vec.or": ("i16", "u16")}
_UNARY = {"vec.exp": FH, "vec.ln": FH, "vec.abs": FH, "vec.rec": FH, "vec.sqrt": FH, "vec.rsqrt": FH, "vec.not": ("i16", "u16"), "vec.relu": FH}
_UNARY_SCALAR = {"vec.adds": FHI, "vec.muls": FHI, "vec.maxs": FHI, "vec.mins": FHI, "vec.lrelu": FH, "vec.axpy": FH,
                 "vec.shiftls": ("i16", "u16", "i32", "u32"), "vec.shiftrs": ("i16", "u16", "i32", "u32")}
_REDUCE = ("vec.cmax", "vec.cmin", "vec.cadd", "vec.cgmax", "vec.cgmin", "vec.cgadd", "vec.cpadd")
_MULADDDST_PAIRS = (("f32", "f32"), ("f16", "f16"), ("f32", "f16"))  # (dst, src) — the CANN doc's Atlas A2 modes

# cast pairs and their rounding modes on a2 (the old stub_functions/cast_rules.py A2_CAST_ROUND_MODES, IR dtype names)
_INT_MODES = frozenset({"rint", "floor", "ceil", "round", "trunc"})
_NONE_ONLY = frozenset({"none"})
_INT_OR_NONE = _INT_MODES | _NONE_ONLY
_FLOAT_REDUCE = _INT_MODES | {"odd", "none"}
A2_CAST_MODES: dict[tuple[str, str], frozenset[str]] = {
    ("f16", "f32"): _NONE_ONLY, ("f16", "i32"): _INT_MODES, ("f16", "i16"): _INT_MODES, ("f16", "i8"): _INT_OR_NONE,
    ("f16", "u8"): _INT_OR_NONE, ("f16", "i4"): _INT_OR_NONE,
    ("f32", "f32"): _INT_MODES, ("f32", "f16"): _FLOAT_REDUCE, ("f32", "i32"): _INT_MODES, ("f32", "i64"): _INT_MODES,
    ("f32", "i16"): _INT_MODES, ("f32", "bf16"): _INT_MODES,
    ("bf16", "f32"): _NONE_ONLY, ("bf16", "i32"): _INT_MODES,
    ("i4", "f16"): _NONE_ONLY, ("u8", "f16"): _NONE_ONLY, ("i8", "f16"): _NONE_ONLY, ("i16", "f16"): _INT_OR_NONE,
    ("i16", "f32"): _NONE_ONLY, ("i32", "f32"): _INT_OR_NONE, ("i32", "i16"): _NONE_ONLY, ("i32", "i64"): _NONE_ONLY,
    ("i64", "i32"): _NONE_ONLY, ("i64", "f32"): _INT_MODES,
}


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


class VecRules:
    """Mixin of :class:`FunctionCompiler` (a2 family)."""

    # -- operands and inference ------------------------------------------------------------------------------------

    def _vec_ub(self, v: Any, what: str, callee: Any, node: ast.AST, dtypes: tuple[str, ...] | None = None) -> Dyn:
        if isinstance(v, ElemOffset):
            v = v.base
        if not (isinstance(v, Dyn) and isinstance(v.type, MemType) and v.type.space == "ub"):
            raise self.err(E_BAD_OPERAND, f"{callee.__name__}: {what} must be a UB tensor, got {v!r}", node)  # type: ignore[attr-defined]
        if dtypes is not None and v.type.dtype.name not in dtypes:
            raise self.err(E_BAD_OPERAND, f"{callee.__name__}: {what} does not support data type {v.type.dtype} "  # type: ignore[attr-defined]
                                          f"(one of {', '.join(dtypes)})", node)
        return v

    def _vec_span(self, v: Dyn, node: ast.AST) -> tuple[Any, Any, Any]:
        """``(span0, span1, shape1)`` of a view: rows (the leading dims folded), the row span, the allocation's row."""
        g = self.geom(v, node)  # type: ignore[attr-defined]
        span0 = self.prod(list(g.span[:-1]), node) if len(g.span) > 1 else 1  # type: ignore[attr-defined]
        return span0, g.span[-1], g.shape[-1]

    def vec_numel(self, v: Dyn, node: ast.AST) -> Any:
        return self.prod(list(self.geom(v, node).span), node)  # type: ignore[attr-defined]

    def vec_infer_repeat(self, v: Dyn, node: ast.AST, elem_bytes: int | None = None) -> Any:
        """``CeilDiv(numel(span), 256 / sizeof)``: the repeats that cover the view (the old ``infer_repeat``)."""
        denom = 256 // (elem_bytes or _bytes(v.type.dtype))  # type: ignore[union-attr]
        return self.ceil_div(self.vec_numel(v, node), denom, node)  # type: ignore[attr-defined]

    def _rep_of_row(self, shape1: Any, c0: int, node: ast.AST) -> Any:
        return shape1 // c0 if _is_int(shape1) else self.ceil_div(shape1, c0, node)  # type: ignore[attr-defined]

    def vec_infer_strides(self, v: Dyn, node: ast.AST) -> tuple[Any, Any]:
        """The old ``infer_strides``: a row of exactly 8 blocks -> ``(1, shape1 / C0)``, a row of one block -> the
        broadcast layout ``(0, shape1 / C0)``, anything else -> ``(1, 8)``; a single matched row has ``rep_stride 0``."""
        c0 = _c0(v.type.dtype)  # type: ignore[union-attr]
        span0, span1, shape1 = self._vec_span(v, node)
        blk: Any = 1
        rep: Any = 8
        matched = False
        if _is_int(span1):
            if span1 == 8 * c0:
                blk, rep, matched = 1, self._rep_of_row(shape1, c0, node), True
            elif span1 == c0:
                blk, rep, matched = 0, self._rep_of_row(shape1, c0, node), True
        if matched and _is_int(span0) and span0 == 1:
            rep = 0
        return blk, rep

    def vec_resolve(self, v: Dyn, blk: Any, rep: Any, node: ast.AST) -> tuple[Any, Any]:
        if blk is None or rep is None:
            ablk, arep = self.vec_infer_strides(v, node)
            blk = ablk if blk is None else blk
            rep = arep if rep is None else rep
        return blk, rep

    # -- the count modes (attributes, not state: D-062) --------------------------------------------------------------

    def vec_mode_attrs(self, count: Any, count_per_rep: Any, limit: int | None, node: ast.AST) -> dict[str, Any]:
        """Validate the two keyword modes and return them as the op's own attributes."""
        if count is not None and count_per_rep is not None:
            raise self.err(E_BAD_OPERAND, "count and count_per_rep are mutually exclusive", node)  # type: ignore[attr-defined]
        if count_per_rep is not None and _is_int(count_per_rep) and limit is not None and not 0 <= count_per_rep <= limit:
            raise self.err(E_BAD_OPERAND, f"count_per_rep is the active lane count inside one repeat; valid range for this op is "  # type: ignore[attr-defined]
                                          f"[0, {limit}], got {count_per_rep} (use repeat + rep_stride for multiple chunks)", node)
        return {"count": count, "count_per_rep": count_per_rep}

    # -- the rule ----------------------------------------------------------------------------------------------------

    def rule_vec(self, rule: str, callee: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if self.kind != "kernel":  # type: ignore[attr-defined]
            raise self.err(E_UNSUPPORTED, f"{callee.__name__} is a kernel-level instruction", node)  # type: ignore[attr-defined]
        a, kw = list(args), dict(kwargs)
        if rule == "vec.abs" and len(a) == 1 and not kw:  # the old facade's abs(x) on a scalar
            x = self.rvalue(a[0], node)  # type: ignore[attr-defined]
            return self.emit("scalar.abs", (x,), {}, self._scalar_type(x, node), None, node)  # type: ignore[attr-defined]

        def arg(i: int, key: str, default: Any = None) -> Any:
            return a[i] if len(a) > i else kw.pop(key, default)

        def done(opcode: str, operands: tuple[Any, ...], attrs: dict[str, Any]) -> Any:
            if kw:
                raise self.err(E_BAD_OPERAND, f"{callee.__name__}: unexpected argument(s) {', '.join(sorted(kw))}", node)  # type: ignore[attr-defined]
            clean = {k: (self.iv(v, node) if k not in ("mode", "tmp_addr_buf", "round_en") else v) for k, v in attrs.items() if v is not None}
            self.emit(opcode, operands, clean, None, None, node)  # type: ignore[attr-defined]
            return None

        if rule in _BINARY or rule == "vec.muladddst":
            dst, src1, src2 = (self._vec_ub(arg(i, k), k, callee, node) for i, k in enumerate(("dst", "src1", "src2")))
            if rule == "vec.muladddst":
                if src1.type.dtype != src2.type.dtype or (dst.type.dtype.name, src1.type.dtype.name) not in _MULADDDST_PAIRS:  # type: ignore[union-attr]
                    raise self.err(E_BAD_OPERAND, "muladddst supports (dst, src) dtype pairs (float, float) / (half, half) / (float, half); "  # type: ignore[attr-defined]
                                                  f"got ({dst.type.dtype}, {src1.type.dtype})", node)
            else:
                if not dst.type.dtype == src1.type.dtype == src2.type.dtype:  # type: ignore[union-attr]
                    raise self.err(E_BAD_OPERAND, f"{callee.__name__}: dst / src1 / src2 data types must match", node)  # type: ignore[attr-defined]
                self._vec_ub(dst, "dst", callee, node, _BINARY[rule])
            repeat = arg(3, "repeat")
            dst_blk, src1_blk, src2_blk = arg(4, "dst_blk_stride"), arg(5, "src1_blk_stride"), arg(6, "src2_blk_stride")
            dst_rep, src1_rep, src2_rep = arg(7, "dst_rep_stride"), arg(8, "src1_rep_stride"), arg(9, "src2_rep_stride")
            count, per_rep = kw.pop("count", None), kw.pop("count_per_rep", None)
            if repeat is None:
                repeat = self.vec_infer_repeat(dst, node)
            dst_blk, dst_rep = self.vec_resolve(dst, dst_blk, dst_rep, node)
            src1_blk, src1_rep = self.vec_resolve(src1, src1_blk, src1_rep, node)
            src2_blk, src2_rep = self.vec_resolve(src2, src2_blk, src2_rep, node)
            mode = self.vec_mode_attrs(count, per_rep, 256 // _bytes(dst.type.dtype), node)  # type: ignore[union-attr]
            return done(rule, (dst.plain(), src1.plain(), src2.plain()),
                        {"repeat": repeat, "dst_blk_stride": dst_blk, "src1_blk_stride": src1_blk, "src2_blk_stride": src2_blk,
                         "dst_rep_stride": dst_rep, "src1_rep_stride": src1_rep, "src2_rep_stride": src2_rep, **mode})

        if rule in _UNARY or rule in _UNARY_SCALAR:
            scalar = rule in _UNARY_SCALAR
            dst = self._vec_ub(arg(0, "dst"), "dst", callee, node, (_UNARY_SCALAR if scalar else _UNARY)[rule])
            src = self._vec_ub(arg(1, "src"), "src", callee, node)
            if dst.type.dtype != src.type.dtype:  # type: ignore[union-attr]
                raise self.err(E_BAD_OPERAND, f"{callee.__name__}: dst / src data types must match", node)  # type: ignore[attr-defined]
            k = 2
            val = None
            if scalar:
                val = self.rvalue(arg(2, "val"), node)  # type: ignore[attr-defined]
                if val is None:
                    raise self.err(E_BAD_OPERAND, f"{callee.__name__}(dst, src, val, ...): the scalar is missing", node)  # type: ignore[attr-defined]
                if rule in ("vec.shiftls", "vec.shiftrs"):
                    bits = dst.type.dtype.bits  # type: ignore[union-attr]
                    if isinstance(val, float) or (_is_int(val) and not 0 <= val <= bits):
                        raise self.err(E_BAD_OPERAND, f"{callee.__name__}: the shift count must be an integer in [0, {bits}], got {val!r}", node)  # type: ignore[attr-defined]
                k = 3
            repeat = arg(k, "repeat")
            dst_blk, src_blk = arg(k + 1, "dst_blk_stride"), arg(k + 2, "src_blk_stride")
            dst_rep, src_rep = arg(k + 3, "dst_rep_stride"), arg(k + 4, "src_rep_stride")
            count, per_rep = kw.pop("count", None), kw.pop("count_per_rep", None)
            round_en = kw.pop("round_en", None) if rule == "vec.shiftrs" else None
            if repeat is None:
                repeat = self.vec_infer_repeat(dst, node)
            dst_blk, dst_rep = self.vec_resolve(dst, dst_blk, dst_rep, node)
            src_blk, src_rep = self.vec_resolve(src, src_blk, src_rep, node)
            mode = self.vec_mode_attrs(count, per_rep, 256 // _bytes(dst.type.dtype), node)  # type: ignore[union-attr]
            operands = (dst.plain(), src.plain()) + ((val,) if scalar else ())
            return done(rule, operands, {"repeat": repeat, "dst_blk_stride": dst_blk, "src_blk_stride": src_blk, "dst_rep_stride": dst_rep,
                                         "src_rep_stride": src_rep, "round_en": bool(round_en) if round_en else None, **mode})

        if rule in _REDUCE:
            dst = self._vec_ub(arg(0, "dst"), "dst", callee, node, FH)
            src = self._vec_ub(arg(1, "src"), "src", callee, node, FH)
            if dst.type.dtype != src.type.dtype:  # type: ignore[union-attr]
                raise self.err(E_BAD_OPERAND, f"{callee.__name__}: dst / src data types must match", node)  # type: ignore[attr-defined]
            repeat, dst_rep = arg(2, "repeat"), arg(3, "dst_rep_stride", 1)
            src_blk, src_rep = arg(4, "src_blk_stride"), arg(5, "src_rep_stride")
            per_rep = kw.pop("count_per_rep", None)
            if dst_rep is None:
                raise self.err(E_BAD_OPERAND, f"{callee.__name__}: dst_rep_stride does not support None, please provide explicitly", node)  # type: ignore[attr-defined]
            if repeat is None:
                repeat = self.vec_infer_repeat(src, node)
            src_blk, src_rep = self.vec_resolve(src, src_blk, src_rep, node)
            mode = self.vec_mode_attrs(None, per_rep, 256 // _bytes(src.type.dtype), node)  # type: ignore[union-attr]
            return done(rule, (dst.plain(), src.plain()), {"repeat": repeat, "dst_rep_stride": dst_rep, "src_blk_stride": src_blk,
                                                           "src_rep_stride": src_rep, **mode})

        if rule == "vec.dup":
            dst = self._vec_ub(arg(0, "dst"), "dst", callee, node, ("f32", "f16", "i32", "u32"))
            value = self.rvalue(arg(1, "value"), node)  # type: ignore[attr-defined]
            if value is None:
                raise self.err(E_BAD_OPERAND, "dup(dst, value, ...): the value is missing", node)  # type: ignore[attr-defined]
            repeat, dst_blk, dst_rep = arg(2, "repeat"), arg(3, "dst_blk_stride"), arg(4, "dst_rep_stride")
            count, per_rep = kw.pop("count", None), kw.pop("count_per_rep", None)
            if repeat is None:
                repeat = self.vec_infer_repeat(dst, node)
            if count is not None:  # AscendC's counter-mode Duplicate hardwires blk 1 / rep 8 (vector_dup walks the strides)
                dst_blk = 1 if dst_blk is None else dst_blk
                dst_rep = 8 if dst_rep is None else dst_rep
                if (_is_int(dst_blk) and dst_blk != 1) or (_is_int(dst_rep) and dst_rep != 8):
                    raise self.err(E_BAD_OPERAND, f"dup count mode requires contiguous strides (blk=1, rep=8); got blk={dst_blk}, rep={dst_rep}", node)  # type: ignore[attr-defined]
            dst_blk, dst_rep = self.vec_resolve(dst, dst_blk, dst_rep, node)
            mode = self.vec_mode_attrs(count, per_rep, 256 // _bytes(dst.type.dtype), node)  # type: ignore[union-attr]
            return done(rule, (dst.plain(), value), {"repeat": repeat, "dst_blk_stride": dst_blk, "dst_rep_stride": dst_rep, **mode})

        if rule == "vec.brcb":
            dst = self._vec_ub(arg(0, "dst"), "dst", callee, node)
            src = self._vec_ub(arg(1, "src"), "src", callee, node)
            if dst.type.dtype != src.type.dtype:  # type: ignore[union-attr]
                raise self.err(E_BAD_OPERAND, "brcb: dst / src data types must match", node)  # type: ignore[attr-defined]
            dst_blk, dst_rep, repeat = arg(2, "dst_blk_stride"), arg(3, "dst_rep_stride"), arg(4, "repeat")
            if repeat is None:  # one source element per block: the repeats that cover the source (the old infer_repeat_brcb)
                n = self.vec_numel(src, node)
                repeat = n // 8 if _is_int(n) else self.ceil_div(n, 8, node)  # type: ignore[attr-defined]
            dst_blk, dst_rep = self.vec_resolve(dst, dst_blk, dst_rep, node)
            return done(rule, (dst.plain(), src.plain()), {"repeat": repeat, "dst_blk_stride": dst_blk, "dst_rep_stride": dst_rep})

        if rule == "vec.cast":
            return self._rule_vec_cast(callee, arg, kw, done, node)
        if rule in ("vec.compare", "vec.compare_scalar"):
            return self._rule_vec_compare(rule, callee, arg, kw, done, node)
        if rule == "vec.select":
            return self._rule_vec_select(callee, arg, kw, done, node)
        if rule in ("vec.gather", "vec.scatter", "vec.gather_block"):
            return self._rule_vec_gather(rule, callee, arg, kw, done, node)
        if rule == "vec.transdata5hd":
            return self._rule_vec_transdata(callee, arg, kw, done, node)
        raise self.err(E_UNSUPPORTED, f"{callee.__name__}: {rule} is not a frontend instruction", node)  # type: ignore[attr-defined]

    # -- the special forms -------------------------------------------------------------------------------------------

    def _mode_name(self, v: Any, what: str, callee: Any, node: ast.AST) -> str:
        if isinstance(v, dsl.EnumValue):
            return v.name
        raise self.err(E_BAD_OPERAND, f"{callee.__name__}: {what} must be a mode value, got {v!r}", node)  # type: ignore[attr-defined]

    def _rule_vec_cast(self, callee: Any, arg: Any, kw: dict[str, Any], done: Any, node: ast.AST) -> Any:
        dst = self._vec_ub(arg(0, "dst"), "dst", callee, node)
        src = self._vec_ub(arg(1, "src"), "src", callee, node)
        repeat = arg(2, "repeat")
        dst_blk, src_blk = arg(3, "dst_blk_stride"), arg(4, "src_blk_stride")
        dst_rep, src_rep = arg(5, "dst_rep_stride"), arg(6, "src_rep_stride")
        mode_v = arg(7, "round_mode", dsl.RoundMode.AWAY_FROM_ZERO)
        count, per_rep = kw.pop("count", None), kw.pop("count_per_rep", None)
        sd, dd = src.type.dtype, dst.type.dtype  # type: ignore[union-attr]
        modes = A2_CAST_MODES.get((sd.name, dd.name))
        if modes is None:
            raise self.err(E_BAD_OPERAND, f"A2 cast only supports listed Cast pairs, got: {sd} -> {dd}", node)  # type: ignore[attr-defined]
        mode = self._mode_name(mode_v, "round_mode", callee, node)
        if modes == _NONE_ONLY and dd.bits > sd.bits:  # a widening pair has no rounding: the mode is forced (the old should_force_none_round_mode)
            mode = "none"
        if mode not in modes:
            raise self.err(E_BAD_OPERAND, f"{sd} -> {dd} cast supports round_mode in {', '.join(sorted(modes))}, got: {mode}", node)  # type: ignore[attr-defined]
        wide = max(_bytes(sd), _bytes(dd))
        if repeat is None:
            repeat = self.vec_infer_repeat(dst, node, wide)
        if dst_blk is None:
            dst_blk, _ = self.vec_resolve(dst, None, 8, node)
        if src_blk is None:
            src_blk, _ = self.vec_resolve(src, None, 8, node)
        if (dst_rep is None) != (src_rep is None):
            raise self.err(E_BAD_OPERAND, "cast: src_rep_stride and dst_rep_stride must all be None or simultaneously specified", node)  # type: ignore[attr-defined]
        if dst_rep is None:  # the narrower side's repeat covers fewer blocks (the old _resolve_cast_rep_strides)
            c0s, c0d = _c0(sd), _c0(dd)
            src_rep, dst_rep = (8 * c0d // c0s, 8) if c0s > c0d else (8, 8 * c0s // c0d) if c0s < c0d else (8, 8)
        counts = self.vec_mode_attrs(count, per_rep, 256 // wide, node)
        return done("vec.cast", (dst.plain(), src.plain()),
                    {"mode": dsl.EnumValue("RoundMode", mode), "repeat": repeat, "dst_blk_stride": dst_blk, "src_blk_stride": src_blk,
                     "dst_rep_stride": dst_rep, "src_rep_stride": src_rep, **counts})

    def _rule_vec_compare(self, rule: str, callee: Any, arg: Any, kw: dict[str, Any], done: Any, node: ast.AST) -> Any:
        dst = self._vec_ub(arg(0, "dst"), "dst", callee, node, ("i8", "u8"))
        src1 = self._vec_ub(arg(1, "src1"), "src1", callee, node, ("f32", "f16", "i16", "i32"))
        scalar = rule == "vec.compare_scalar"
        if scalar:
            src2 = self.rvalue(arg(2, "src2"), node)  # type: ignore[attr-defined]
            float_src = src1.type.dtype.name in ("f32", "f16", "bf16")  # type: ignore[union-attr]
            if isinstance(src2, Dyn):
                float_scalar = src2.type.dtype.name in ("f32", "f16", "bf16")  # type: ignore[union-attr]
            elif isinstance(src2, (bool, int, float)):
                float_scalar = isinstance(src2, float)
            else:
                raise self.err(E_BAD_OPERAND, "compare_scalar(dst, src1, scalar, mode, ...): the scalar is missing", node)  # type: ignore[attr-defined]
            if float_src != float_scalar:
                raise self.err(E_BAD_OPERAND, f"compare_scalar requires matching float/int families, src1 dtype: {src1.type.dtype}, scalar: {src2!r}", node)  # type: ignore[attr-defined]
        else:
            src2 = self._vec_ub(arg(2, "src2"), "src2", callee, node)
            if src1.type.dtype != src2.type.dtype:  # type: ignore[union-attr]
                raise self.err(E_BAD_OPERAND, "compare: src1 / src2 data types must match", node)  # type: ignore[attr-defined]
        mode = self._mode_name(arg(3, "mode"), "mode", callee, node)
        repeat = arg(4, "repeat")
        if scalar:
            dst_blk, src1_blk, dst_rep, src1_rep = arg(5, "dst_blk_stride"), arg(6, "src1_blk_stride"), arg(7, "dst_rep_stride"), arg(8, "src1_rep_stride")
            src2_blk = src2_rep = None
        else:
            dst_blk, src1_blk, src2_blk = arg(5, "dst_blk_stride"), arg(6, "src1_blk_stride"), arg(7, "src2_blk_stride")
            dst_rep, src1_rep, src2_rep = arg(8, "dst_rep_stride"), arg(9, "src1_rep_stride"), arg(10, "src2_rep_stride")
        if repeat is None:
            repeat = self.vec_infer_repeat(src1, node)
        dst_blk, dst_rep = self.vec_resolve(dst, dst_blk, dst_rep, node)
        src1_blk, src1_rep = self.vec_resolve(src1, src1_blk, src1_rep, node)
        attrs = {"mode": dsl.EnumValue("CompareMode", mode), "repeat": repeat, "dst_blk_stride": dst_blk, "src1_blk_stride": src1_blk,
                 "dst_rep_stride": dst_rep, "src1_rep_stride": src1_rep}
        if not scalar:
            src2_blk, src2_rep = self.vec_resolve(src2, src2_blk, src2_rep, node)
            attrs.update({"src2_blk_stride": src2_blk, "src2_rep_stride": src2_rep})
        operands = (dst.plain(), src1.plain(), src2 if scalar else src2.plain())
        return done(rule, operands, attrs)

    def _rule_vec_select(self, callee: Any, arg: Any, kw: dict[str, Any], done: Any, node: ast.AST) -> Any:
        dst = self._vec_ub(arg(0, "dst"), "dst", callee, node, ("f32", "f16", "i16", "i32"))
        selmask = self._vec_ub(arg(1, "selmask"), "selmask", callee, node, ("u8",))
        src1 = self._vec_ub(arg(2, "src1"), "src1", callee, node)
        src2 = self._vec_ub(arg(3, "src2"), "src2", callee, node)
        if not dst.type.dtype == src1.type.dtype == src2.type.dtype:  # type: ignore[union-attr]
            raise self.err(E_BAD_OPERAND, "select: dst / src1 / src2 data types must match", node)  # type: ignore[attr-defined]
        mode = self._mode_name(arg(4, "mode"), "mode", callee, node)
        repeat = arg(5, "repeat")
        dst_blk, src1_blk, src2_blk = arg(6, "dst_blk_stride"), arg(7, "src1_blk_stride"), arg(8, "src2_blk_stride")
        dst_rep, src1_rep, src2_rep = arg(9, "dst_rep_stride"), arg(10, "src1_rep_stride"), arg(11, "src2_rep_stride")
        tmp = arg(12, "tmp_addr_buf")
        if mode == "tensor_tensor":
            gd = self.geom(dst, node)  # type: ignore[attr-defined]
            for s in (src1, src2):
                gs = self.geom(s, node)  # type: ignore[attr-defined]
                if gd.root == gs.root and gd.offset == gs.offset:
                    raise self.err(E_BAD_OPERAND, "SelectMode.TENSOR_TENSOR requires dst address to differ from src1 and src2", node)  # type: ignore[attr-defined]
            if tmp is None:
                raise self.err(E_BAD_OPERAND, "SelectMode.TENSOR_TENSOR requires an explicit `tmp_addr_buf`: a uint32 UB tensor of >= 8 lanes "  # type: ignore[attr-defined]
                                              "(one 32 B block) to stage the selMask address", node)
            tmp = self._vec_ub(tmp, "tmp_addr_buf", callee, node, ("u32",))
            n = self.vec_numel(tmp, node)
            if _is_int(n) and n < 8:
                raise self.err(E_BAD_OPERAND, f"tmp_addr_buf must hold >= 8 uint32 (one 32B block); got {n} element(s)", node)  # type: ignore[attr-defined]
        elif tmp is not None:
            raise self.err(E_BAD_OPERAND, "tmp_addr_buf is only used by SelectMode.TENSOR_TENSOR", node)  # type: ignore[attr-defined]
        if repeat is None:
            repeat = self.vec_infer_repeat(dst, node)
        dst_blk, dst_rep = self.vec_resolve(dst, dst_blk, dst_rep, node)
        src1_blk, src1_rep = self.vec_resolve(src1, src1_blk, src1_rep, node)
        src2_blk, src2_rep = self.vec_resolve(src2, src2_blk, src2_rep, node)
        return done("vec.select", (dst.plain(), selmask.plain(), src1.plain(), src2.plain()),
                    {"mode": dsl.EnumValue("SelectMode", mode), "repeat": repeat, "dst_blk_stride": dst_blk, "src1_blk_stride": src1_blk,
                     "src2_blk_stride": src2_blk, "dst_rep_stride": dst_rep, "src1_rep_stride": src1_rep, "src2_rep_stride": src2_rep,
                     "tmp_addr_buf": tmp})

    def _rule_vec_gather(self, rule: str, callee: Any, arg: Any, kw: dict[str, Any], done: Any, node: ast.AST) -> Any:
        dst = self._vec_ub(arg(0, "dst"), "dst", callee, node)
        src = self._vec_ub(arg(1, "src"), "src", callee, node)
        offset = self._vec_ub(arg(2, "offset"), "offset", callee, node, ("u32",))
        if dst.type.dtype != src.type.dtype:  # type: ignore[union-attr]
            raise self.err(E_BAD_OPERAND, f"{callee.__name__}: dst / src data types must match", node)  # type: ignore[attr-defined]
        if rule == "vec.gather_block":
            repeat, dst_blk, dst_rep = arg(3, "repeat"), arg(4, "dst_blk_stride"), arg(5, "dst_rep_stride")
            if repeat is None:
                repeat = self.vec_infer_repeat(dst, node)
            dst_blk, dst_rep = self.vec_resolve(dst, dst_blk, dst_rep, node)
            return done(rule, (dst.plain(), src.plain(), offset.plain()), {"repeat": repeat, "dst_blk_stride": dst_blk, "dst_rep_stride": dst_rep})
        start_idx, repeat = arg(3, "start_idx", 0), arg(4, "repeat")
        stride_key = "dst_rep_stride" if rule == "vec.gather" else "src_rep_stride"
        rep = arg(5, stride_key)
        if kw.pop("count", None) is not None:  # the old stub: AscendC's counted overload gathers the wrong lanes on c220
            raise self.err(E_UNSUPPORTED, f"{callee.__name__}: count-mode (count=...) is not supported -- it silently gathers the wrong "  # type: ignore[attr-defined]
                                          "lanes on real A2/c220 while passing in the simulator; use repeat mode instead", node)
        walker = dst if rule == "vec.gather" else src
        if repeat is None:
            repeat = self.vec_infer_repeat(walker, node)
        if rep is None:
            _, rep = self.vec_resolve(walker, None, None, node)
        return done(rule, (dst.plain(), src.plain(), offset.plain()), {"repeat": repeat, stride_key: rep, "start_idx": start_idx})

    def _rule_vec_transdata(self, callee: Any, arg: Any, kw: dict[str, Any], done: Any, node: ast.AST) -> Any:
        b16 = ("f16", "bf16", "i16", "u16")
        dst = self._vec_ub(arg(0, "dst"), "dst", callee, node, b16)
        src = self._vec_ub(arg(1, "src"), "src", callee, node, b16)
        if dst.type.dtype != src.type.dtype:  # type: ignore[union-attr]
            raise self.err(E_BAD_OPERAND, "transdata5hd: dst / src data types must match", node)  # type: ignore[attr-defined]
        repeat, src_row, dst_row = arg(2, "repeat"), arg(3, "src_row_stride"), arg(4, "dst_row_stride")
        src_rep, dst_rep = arg(5, "src_rep_stride", 1), arg(6, "dst_rep_stride", 16)
        _, _, src_shape1 = self._vec_span(src, node)
        if src_row is None:
            src_row = src_shape1
        if dst_row is None:
            dst_row = 16
        if repeat is None:
            if not _is_int(src_shape1):
                raise self.err(E_BAD_OPERAND, "transdata5hd: repeat is required when the source row length is dynamic", node)  # type: ignore[attr-defined]
            repeat = src_shape1 // 16
        if _is_int(repeat) and not 1 <= repeat <= 255:
            raise self.err(E_BAD_OPERAND, f"transdata5hd: repeat must be in [1, 255], current value: {repeat}", node)  # type: ignore[attr-defined]
        for v, lbl in ((src_row, "src_row_stride"), (dst_row, "dst_row_stride")):
            if _is_int(v) and v % 16:
                raise self.err(E_BAD_OPERAND, f"transdata5hd: {lbl} must be a multiple of 16 (32B rows), got {v}", node)  # type: ignore[attr-defined]
        return done("vec.transdata5hd", (dst.plain(), src.plain()),
                    {"repeat": repeat, "src_row_stride": src_row, "dst_row_stride": dst_row, "src_rep_stride": src_rep, "dst_rep_stride": dst_rep})
