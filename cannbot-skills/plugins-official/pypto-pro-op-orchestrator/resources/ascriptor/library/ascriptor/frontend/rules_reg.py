# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Register semantics of the frontend: ``Reg`` / ``RegList`` / ``MaskReg`` and their expressions.

The DSL writes vector code as register expressions: ``x <<= x.abs() + 1.0``. Like the old
``RegOP``, an expression is deferred (:class:`RegExpr`) until it meets ``<<=``; the outermost
op then writes the target directly and only inner links get temporaries, emitted in source
order. ``RegList`` is a static list of registers: element access needs a static index and
whole-list operations expand element by element; reductions use the old pairwise tree.
Loads and stores between UB and registers pick the distribution from the tensor's rider
(``.single()`` ...) and from the element sizes, exactly as ``Reg.__ilshift__`` /
``Tensor.__ilshift__`` did.
"""

from __future__ import annotations

import ast
from typing import Any

from ..ir import Ident
from ..ir.types import DTYPES, CellType, DType, MaskType, MemType, RegType, ScalarType
from . import dsl
from .errors import E_BAD_COPY, E_BAD_OPERAND, E_BAD_SIGNATURE, E_UNSUPPORTED
from .values import Dyn, ElemOffset, RegExpr, RegList

UNARY = {"exp": "exp", "abs": "abs", "sqrt": "sqrt", "relu": "relu", "ln": "ln", "log": "log", "log2": "log2", "log10": "log10",
         "neg": "neg", "vnot": "not", "vcopy": "copy"}
SCALAR = {"shiftls": "shiftls", "shiftrs": "shiftrs", "axpy": "axpy", "lrelu": "lrelu", "vmins": "mins", "vmaxs": "maxs"}
BINARY = {"vand": "and", "vor": "or", "vxor": "xor", "prelu": "prelu", "vmax": "max", "vmin": "min"}
REDUCE = ("cadd", "cmax", "cmin", "cgadd", "cgmax", "cgmin", "cpadd")
LIST_REDUCE = {"cadd": "add", "cmax": "max", "cmin": "min"}
BINOPS = {ast.Add: ("add", "adds"), ast.Sub: ("sub", None), ast.Mult: ("mul", "muls"), ast.Div: ("div", None), ast.Mod: ("mod", None)}
CMP = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}
MASK_BINOPS = {ast.BitAnd: "mask_and", ast.BitOr: "mask_or", ast.BitXor: "mask_xor"}
MASK_OPS = {"mask_not", "mask_and", "mask_or", "mask_xor", "mask_sel", "mask_mov", "mask_pack", "mask_unpack", "mask_update",
            "mask_from_spr", "cmp", "cmps"}
STORE_PREFIX = "store:"

LOAD_MODES = {  # (mode, element bytes) -> ident of vf.load_cont
    ("normal", 1): "norm", ("normal", 2): "norm", ("normal", 4): "norm", ("normal", 8): "norm",
    ("downsample", 1): "ds_b8", ("downsample", 2): "ds_b16",
    ("upsample", 1): "us_b8", ("upsample", 2): "us_b16",
    ("single", 1): "brc_b8", ("single", 2): "brc_b16", ("single", 4): "brc_b32",
    ("brcb", 2): "e2b_b16", ("brcb", 4): "e2b_b32",
    ("unpack", 1): "unpack_b8", ("unpack", 2): "unpack_b16", ("unpack", 4): "unpack_b32",
    ("unpack4", 1): "unpack4_b8",
}
STORE_MODES = {
    ("normal", 1): "norm_b8", ("normal", 2): "norm_b16", ("normal", 4): "norm_b32", ("normal", 8): "norm",
    ("downsample", 1): "pack_b16", ("downsample", 2): "pack_b32", ("downsample", 4): "pack_b64",
    ("pack4", 1): "pack4_b32",
    ("single", 1): "first_element_b8", ("single", 2): "first_element_b16", ("single", 4): "first_element_b32",
}
INTERLEAVE_MODES = {1: "b8", 2: "b16", 4: "b32"}


def elem_bytes(dt: DType) -> int:
    return max(dt.bits, 8) // 8


def _mask_bits(dt: DType) -> int:
    return dt.bits if dt.bits in (8, 16, 32, 64) else 8


class RegRules:
    """Mixin of :class:`FunctionCompiler`; every method uses the compiler's ``emit`` / ``err``."""

    # -- declarations ----------------------------------------------------------------------------

    def rule_reg(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        dt = self.need_dtype(args[0] if args else kwargs.get("dtype"), node)
        name = args[1] if len(args) > 1 else kwargs.get("name", "")
        n = args[2] if len(args) > 2 else kwargs.get("reg_num", 1)
        if type(n) is not int or n not in (1, 2):
            raise self.err(E_BAD_OPERAND, "reg_num must be 1 or 2", node)
        if n == 2 and dt.name not in ("i64", "u64", "c32", "c64"):
            raise self.err(E_BAD_OPERAND, f"reg_num=2 supports i64/u64/c32/c64, got {dt}", node)
        attrs = {"name": name} if name else {}
        return self.emit("vf.reg", (), attrs, RegType(dt, n), name or "reg", node)

    def rule_maskreg(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        dt = self.need_dtype(args[0] if args else kwargs.get("dtype"), node)
        init = args[1] if len(args) > 1 else kwargs.get("init_mode")
        name = args[2] if len(args) > 2 else kwargs.get("name", "")
        n = kwargs.get("reg_num", 1)
        if type(n) is not int or n not in (1, 2):
            raise self.err(E_BAD_OPERAND, "MaskReg reg_num must be 1 or 2", node)
        if n == 2 and dt.name not in ("i64", "u64", "c32", "c64"):
            raise self.err(E_BAD_OPERAND, f"reg_num=2 masks support i64/u64/c32/c64, got {dt}", node)
        attrs: dict[str, Any] = {}
        if name:
            attrs["name"] = name
        if isinstance(init, dsl.EnumValue) and init.name != "all":
            attrs["init"] = Ident(init.name)
        elif init is not None and not isinstance(init, dsl.EnumValue):
            raise self.err(E_BAD_OPERAND, "MaskReg init_mode must be a MaskType", node)
        return self.emit("vf.mask", (), attrs, MaskType(_mask_bits(dt), n), name or "mask", node)

    def rule_reglist(self, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        dt = self.need_dtype(args[0] if args else kwargs.get("dtype"), node)
        length = args[1] if len(args) > 1 else kwargs.get("length")
        name = args[2] if len(args) > 2 else kwargs.get("name", "")
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise self.err(E_BAD_OPERAND, "RegList length must be a static positive int", node)
        elems = [self.emit("vf.reg", (), {"name": f"{name or 'rl'}_{i}"}, RegType(dt, 1), f"{name or 'rl'}_{i}", node) for i in range(length)]
        return RegList(elems, dt, name)

    # -- classification helpers ------------------------------------------------------------------

    @staticmethod
    def is_reg(v: Any) -> bool:
        return isinstance(v, Dyn) and isinstance(v.type, RegType)

    @staticmethod
    def is_mask(v: Any) -> bool:
        return isinstance(v, Dyn) and isinstance(v.type, MaskType)

    @staticmethod
    def is_ub(v: Any) -> bool:
        return (isinstance(v, Dyn) and isinstance(v.type, MemType) and v.type.space == "ub") or \
            (isinstance(v, ElemOffset) and isinstance(v.base.type, MemType))

    def reg_dtype(self, v: Any) -> DType:
        if isinstance(v, RegList):
            return v.dtype
        assert isinstance(v, Dyn) and isinstance(v.type, RegType)
        return v.type.dtype

    def expr_type(self, e: RegExpr, node: ast.AST) -> RegType | MaskType:
        if e.op in MASK_OPS:
            src = e.inputs[0] if e.inputs else None
            if self.is_mask(src):
                return src.type  # type: ignore[return-value]
            if self.is_reg(src):
                return MaskType(_mask_bits(src.type.dtype), src.type.n)  # type: ignore[union-attr]
            return MaskType(32)
        if e.op == "cast" and e.dtype is not None:
            first = e.inputs[0]
            n = first.type.n if self.is_reg(first) else 1
            if e.dtype.name not in ("i64", "u64", "c32", "c64"):
                n = 1
            return RegType(e.dtype, n)
        first = e.inputs[0]
        if isinstance(first, RegList):
            return RegType(first.dtype, 1)
        if self.is_reg(first):
            return first.type  # type: ignore[return-value]
        raise self.err(E_BAD_OPERAND, f"cannot type register expression {e.op}", node)

    def materialize(self, e: RegExpr, node: ast.AST) -> Dyn:
        """Emit ``e`` into a fresh temporary register (the old ``run_regop``)."""
        if e.op.startswith(STORE_PREFIX):
            raise self.err(E_BAD_OPERAND, f"{e.op[len(STORE_PREFIX):]}() is a store form; write it as `ub[...] <<= reg.{e.op[len(STORE_PREFIX):]}()`", node)
        if e.op in MASK_OPS:
            raise self.err(E_BAD_OPERAND, "a compare / mask expression must be assigned to a MaskReg with <<=", node)
        if isinstance(e.inputs[0], RegList) and e.op not in LIST_REDUCE:
            raise self.err(E_BAD_OPERAND, "a RegList expression must be assigned to a RegList with <<=", node)
        t = self.expr_type(e, node)
        tmp = self.emit("vf.reg", (), {}, t, "tmp", node)
        self.emit_regexpr(e, tmp, node)
        return tmp

    def reg_operand(self, v: Any, node: ast.AST) -> Any:
        """A register-expression input: expressions are materialized first (old order)."""
        if isinstance(v, RegExpr):
            return self.materialize(v, node)
        return v

    # -- building expressions ----------------------------------------------------------------------

    def reg_method(self, reg: Any, name: str, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        """Methods of Reg, RegList, MaskReg and of a deferred expression."""
        if isinstance(reg, RegExpr):
            reg = self.materialize(reg, node)
        if isinstance(reg, RegList):
            return self.reglist_method(reg, name, args, kwargs, node)
        if self.is_mask(reg):
            return self.mask_method(reg, name, args, kwargs, node)
        rt = reg.type
        if name in UNARY:
            return RegExpr(UNARY[name], (reg,))
        if name in SCALAR:
            return RegExpr(SCALAR[name], (reg, self.reg_scalar(self.rvalue(args[0], node), rt.dtype, node)))
        if name in BINARY:
            return RegExpr(BINARY[name], (reg, self.reg_operand(args[0], node)))
        if name in REDUCE:
            return RegExpr(name, (reg,))
        if name == "dup":
            return RegExpr("dup", (reg,))
        if name in ("downsample", "pack4", "single_value"):
            return RegExpr(STORE_PREFIX + ("single" if name == "single_value" else name), (reg,))
        if name == "reinterpret":
            dt = self.need_dtype(args[0] if args else kwargs.get("target_dtype", kwargs.get("dtype")), node)
            n = kwargs.get("name") or (args[1] if len(args) > 1 else "") or "view"
            return self.emit("vf.reinterpret", (reg,), {}, RegType(dt, rt.n), n, node)
        if name in ("cast", "astype"):
            if name == "astype":
                dt = self.need_dtype(args[0] if args else kwargs.get("dtype"), node)
                cfg = args[1] if len(args) > 1 else kwargs.get("cfg", kwargs.get("config"))
            else:
                dt = None
                cfg = args[0] if args else kwargs.get("cfg", kwargs.get("config"))
            return RegExpr("cast", (reg,), attrs=self.cast_attrs(cfg, node), dtype=dt)
        if name == "fill":
            self.emit("vf.dup", (reg, self.reg_scalar(self.rvalue(args[0], node), rt.dtype, node)), {}, None, None, node)
            return None
        if name == "arange":
            start = self.rvalue(args[0] if args else kwargs.get("start", 0), node)
            increase = args[1] if len(args) > 1 else kwargs.get("increase", True)
            self.emit("vf.arange", (reg,), {"v": self.attr(self.reg_scalar(start, rt.dtype, node)), "mode": Ident("increase" if increase else "decrease")},
                      None, None, node)
            return None
        if name == "ub_gather":
            src, index = args[0], args[1]
            mask = args[2] if len(args) > 2 else kwargs.get("mask")
            base, off = self.ub_parts(src, node)
            self.emit("vf.gather_copy", (reg, base, index), {"offset": off, "mask": mask}, None, None, node)
            return None
        if name == "gather":
            self.emit("vf.gather", (reg, args[0], args[1]), {}, None, None, node)
            return None
        if name == "gather_mask":
            self.emit("vf.gathermask", (reg, args[0]), {"mask": args[1] if len(args) > 1 else kwargs.get("mask")}, None, None, node)
            return None
        raise self.err(E_UNSUPPORTED, f"register method {name!r} is not supported", node)

    def reglist_method(self, rl: RegList, name: str, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if name in UNARY:
            return RegExpr(UNARY[name], (rl,))
        if name in SCALAR:
            return RegExpr(SCALAR[name], (rl, self.reg_scalar(self.rvalue(args[0], node), rl.dtype, node)))
        if name in BINARY:
            return RegExpr(BINARY[name], (rl, self.reg_operand(args[0], node)))
        if name in LIST_REDUCE:
            return RegExpr(name, (rl,))
        if name == "fill":
            v = self.reg_scalar(self.rvalue(args[0], node), rl.dtype, node)
            for e in rl.elems:
                self.emit("vf.dup", (e, v), {}, None, None, node)
            return None
        raise self.err(E_UNSUPPORTED, f"RegList has no method {name!r}", node)

    def mask_method(self, m: Dyn, name: str, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if name == "update":
            self.emit("vf.mask_update", (m,), {"cnt": self.attr(self.rvalue(args[0], node))}, None, None, node)
            return None
        if name == "mov":
            return RegExpr("mask_mov", (args[0],))
        if name == "sel":
            return RegExpr("mask_sel", (args[0], args[1]))
        if name in ("pack", "unpack"):
            low = args[0] if args else kwargs.get("low_part", True)
            return RegExpr(f"mask_{name}", (m,), attrs={"mode": Ident("lowest" if low else "highest")})
        if name == "move_to_spr":
            return RegExpr("mask_from_spr", ())
        if name == "select":
            return RegExpr("select", (self.reg_operand(args[0], node), self.reg_operand(args[1], node)), mask=m)
        raise self.err(E_UNSUPPORTED, f"MaskReg has no method {name!r}", node)

    def reg_binop(self, op: ast.operator, lhs: Any, rhs: Any, node: ast.AST) -> Any:
        """``+ - * /`` and ``& | ^`` with a register, register list, mask or expression on either side."""
        # mask attachment: mask * expr, expr * mask
        if isinstance(op, ast.Mult) and (self.is_mask(lhs) or self.is_mask(rhs)):
            mask, other = (lhs, rhs) if self.is_mask(lhs) else (rhs, lhs)
            if isinstance(other, RegExpr):
                other.mask = mask
                return other
            raise self.err(E_BAD_OPERAND, "a mask multiplies a register expression (reg.abs() * mask), not a register", node)
        if type(op) in MASK_BINOPS and self.is_mask(lhs) and self.is_mask(rhs):
            return RegExpr(MASK_BINOPS[type(op)], (lhs, rhs))
        if type(op) not in BINOPS:
            raise self.err(E_UNSUPPORTED, f"operator {type(op).__name__} is not defined on registers", node)
        vec, scal = BINOPS[type(op)]
        lhs, rhs = self.reg_operand(lhs, node), self.reg_operand(rhs, node)
        lreg = self.is_reg(lhs) or isinstance(lhs, RegList)
        rreg = self.is_reg(rhs) or isinstance(rhs, RegList)
        if lreg and rreg:
            return RegExpr(vec, (lhs, rhs))
        if lreg:
            if scal is None and isinstance(op, ast.Mod):
                raise self.err(E_BAD_OPERAND, "register remainder takes two integer registers", node)
            dt = self.reg_dtype(lhs)
            s = self.rvalue(rhs, node)
            if isinstance(op, ast.Sub):
                return RegExpr("adds", (lhs, self.reg_scalar(self.negate(s, node), dt, node)))
            if isinstance(op, ast.Div):
                return RegExpr("muls", (lhs, self.reg_scalar(self.reciprocal(s, node), dt, node)))
            return RegExpr(scal, (lhs, self.reg_scalar(s, dt, node)))
        if rreg and isinstance(op, (ast.Add, ast.Mult)):
            return RegExpr(scal, (rhs, self.reg_scalar(self.rvalue(lhs, node), self.reg_dtype(rhs), node)))
        raise self.err(E_UNSUPPORTED, f"`scalar {type(op).__name__} register` is not supported; write the register first", node)

    def reg_compare(self, op: ast.cmpop, lhs: Any, rhs: Any, node: ast.AST) -> RegExpr:
        if type(op) not in CMP:
            raise self.err(E_UNSUPPORTED, "unsupported register comparison", node)
        lhs = self.reg_operand(lhs, node)
        if not self.is_reg(lhs):
            raise self.err(E_UNSUPPORTED, "a register comparison needs the register on the left", node)
        rhs = self.reg_operand(rhs, node)
        mode = Ident(CMP[type(op)])
        if self.is_reg(rhs):
            return RegExpr("cmp", (lhs, rhs), attrs={"mode": mode})
        return RegExpr("cmps", (lhs, self.reg_scalar(self.rvalue(rhs, node), lhs.type.dtype, node)), attrs={"mode": mode})

    def negate(self, s: Any, node: ast.AST) -> Any:
        if isinstance(s, Dyn):
            return self.emit("scalar.neg", (s,), {}, self._scalar_type(s, node), None, node)
        return -s

    def reciprocal(self, s: Any, node: ast.AST) -> Any:
        if isinstance(s, Dyn):
            return self.emit("scalar.div", (1.0, s), {}, ScalarType(DTYPES["f32"]), None, node)
        return 1 / s

    def reg_scalar(self, s: Any, dt: DType, node: ast.AST) -> Any:
        """A scalar operand of a register op, adapted to the register dtype the old way."""
        if isinstance(s, Dyn):
            if isinstance(s.type, (ScalarType, CellType)):
                return s
            raise self.err(E_BAD_OPERAND, f"expected a scalar, got {s}", node)
        if isinstance(s, bool):
            return int(s)
        if isinstance(s, int):
            return float(s) if dt.is_float else s
        if isinstance(s, float):
            return s if dt.is_float else int(s)
        if isinstance(s, complex):
            if dt.kind != "complex":
                raise self.err(E_BAD_OPERAND, f"a complex immediate needs a complex register, got {dt}", node)
            return s
        raise self.err(E_BAD_OPERAND, f"expected a scalar, got {s!r}", node)

    def cast_attrs(self, cfg: Any, node: ast.AST) -> dict[str, Any]:
        if cfg is None:
            return {}
        if not isinstance(cfg, dsl.CastConfig):
            raise self.err(E_BAD_OPERAND, "cast needs a CastConfig", node)
        attrs: dict[str, Any] = {"round": Ident(cfg.round_mode.name), "layout": Ident(cfg.reg_layout.name)}
        if cfg.saturate:
            attrs["saturate"] = True
        if cfg.merge_mode.name != "zeroing":
            attrs["merge"] = Ident(cfg.merge_mode.name)
        return attrs

    # -- emission ---------------------------------------------------------------------------------

    def emit_regexpr(self, e: RegExpr, dst: Any, node: ast.AST) -> None:
        """Emit ``e`` writing ``dst`` (a register, a mask or a RegList)."""
        attrs = dict(e.attrs)
        if e.mask is not None:
            attrs["mask"] = e.mask
        first = e.inputs[0] if e.inputs else None
        if isinstance(first, RegList):
            if e.op in LIST_REDUCE:
                self.emit_list_reduce(e, dst, attrs, node)
                return
            if not isinstance(dst, RegList):
                raise self.err(E_BAD_COPY, "a RegList expression must be assigned to a RegList", node)
            self.emit_list_elementwise(e, dst, attrs, node)
            return
        if isinstance(dst, RegList):
            raise self.err(E_BAD_COPY, "only RegList expressions can be assigned to a RegList", node)
        if e.op == "cast":
            self.emit("vf.cast", (dst, first), attrs, None, None, node)
            return
        if e.op == "dup":
            self.emit("vf.dup", (dst, first), attrs, None, None, node)
            return
        if e.op in ("cmp", "cmps", "select", "mask_and", "mask_or", "mask_xor", "mask_sel", "mask_not", "mask_mov", "mask_pack",
                    "mask_unpack", "mask_from_spr"):
            self.emit(f"vf.{e.op}", (dst, *e.inputs), attrs, None, None, node)
            return
        self.emit(f"vf.{e.op}", (dst, *e.inputs), attrs, None, None, node)

    def emit_list_elementwise(self, e: RegExpr, dst: RegList, attrs: dict[str, Any], node: ast.AST) -> None:
        rl = e.inputs[0]
        assert isinstance(rl, RegList)
        if len(dst) != len(rl):
            raise self.err(E_BAD_COPY, f"RegList lengths differ: {len(dst)} vs {len(rl)}", node)
        other = e.inputs[1] if len(e.inputs) > 1 else None
        if isinstance(other, RegList) and len(other) != len(rl):
            raise self.err(E_BAD_COPY, f"RegList lengths differ: {len(rl)} vs {len(other)}", node)
        for i, (d, s) in enumerate(zip(dst.elems, rl.elems, strict=True)):
            if other is None:
                self.emit(f"vf.{e.op}", (d, s), attrs, None, None, node)
            elif isinstance(other, RegList):
                self.emit(f"vf.{e.op}", (d, s, other.elems[i]), attrs, None, None, node)
            else:
                self.emit(f"vf.{e.op}", (d, s, other), attrs, None, None, node)

    def emit_list_reduce(self, e: RegExpr, dst: Any, attrs: dict[str, Any], node: ast.AST) -> None:
        """The old pairwise tree: pair up, then fold in place, then one cross-lane reduce."""
        rl = e.inputs[0]
        assert isinstance(rl, RegList)
        pair = LIST_REDUCE[e.op]
        if len(rl) == 1:
            self.emit(f"vf.{e.op}", (dst, rl.elems[0]), attrs, None, None, node)
            return
        live: list[Dyn] = []
        for i in range(len(rl) // 2):
            tmp = self.emit("vf.reg", (), {}, RegType(rl.dtype, 1), "tmp", node)
            self.emit(f"vf.{pair}", (tmp, rl.elems[2 * i], rl.elems[2 * i + 1]), attrs, None, None, node)
            live.append(tmp)
        if len(rl) % 2 == 1:
            live.append(rl.elems[-1])
        while len(live) > 1:
            nxt: list[Dyn] = []
            read = 0
            while read + 1 < len(live):
                self.emit(f"vf.{pair}", (live[read], live[read], live[read + 1]), attrs, None, None, node)
                nxt.append(live[read])
                read += 2
            if read < len(live):
                nxt.append(live[read])
            live = nxt
        self.emit(f"vf.{e.op}", (dst, live[0]), attrs, None, None, node)

    # -- <<= --------------------------------------------------------------------------------------

    def ub_parts(self, v: Any, node: ast.AST) -> tuple[Dyn, Any]:
        """(base UB value, element offset attr) of a UB tensor or ``tensor[k]``."""
        if isinstance(v, ElemOffset):
            return v.base.plain(), v.offset
        if isinstance(v, Dyn) and isinstance(v.type, MemType):
            return v.plain(), 0
        raise self.err(E_BAD_OPERAND, f"expected a UB tensor, got {v!r}", node)

    def ub_riders(self, v: Any) -> Any:
        return v.base.riders if isinstance(v, ElemOffset) else v.riders

    def copy_into_reg(self, dst: Dyn, src: Any, node: ast.AST) -> None:
        rt = dst.type
        assert isinstance(rt, RegType)
        if isinstance(src, RegExpr):
            self.emit_regexpr(src, dst, node)
            return
        if self.is_reg(src):
            self.emit("vf.copy", (dst, src), {}, None, None, node)
            return
        if isinstance(src, RegList):
            raise self.err(E_BAD_COPY, "cannot copy a RegList into one register", node)
        if self.is_ub(src):
            self.load_reg(dst, src, node)
            return
        src = self.rvalue(src, node)
        if isinstance(src, (int, float, bool)) or (isinstance(src, Dyn) and isinstance(src.type, (ScalarType, CellType))):
            self.emit("vf.dup", (dst, self.reg_scalar(src, rt.dtype, node)), {}, None, None, node)
            return
        raise self.err(E_BAD_COPY, f"cannot copy {src!r} into a register", node)

    def load_reg(self, dst: Dyn, src: Any, node: ast.AST) -> None:
        """``reg <<= ub[...]`` with the tensor's load rider; converting loads go through a temp."""
        base, off = self.ub_parts(src, node)
        mode = self.ub_riders(src).mode or "normal"
        st = base.type
        assert isinstance(st, MemType)
        rt = dst.type
        assert isinstance(rt, RegType)
        if st.dtype == rt.dtype:
            self.emit("vf.load_cont", (dst, base), {"offset": off, "mode": Ident(self.load_mode(mode, st.dtype, node))}, None, None, node)
            return
        tmp = self.emit("vf.reg", (), {}, RegType(st.dtype, rt.n), "tmp", node)
        dsz, ssz = elem_bytes(rt.dtype), elem_bytes(st.dtype)
        if mode == "single":
            m = "single"
        elif (dsz, ssz) in ((4, 2), (2, 1)):
            m = "unpack"
        elif (dsz, ssz) == (4, 1):
            m = "unpack4"
        elif dsz < ssz:
            m = "normal"
        else:
            raise self.err(E_BAD_COPY, f"a converting load from {st.dtype} to {rt.dtype} is not supported", node)
        self.emit("vf.load_cont", (tmp, base), {"offset": off, "mode": Ident(self.load_mode(m, st.dtype, node))}, None, None, node)
        self.emit("vf.cast", (dst, tmp), {}, None, None, node)

    def load_mode(self, mode: str, dt: DType, node: ast.AST) -> str:
        key = (mode, elem_bytes(dt))
        if key not in LOAD_MODES:
            raise self.err(E_BAD_COPY, f"load distribution {mode!r} is not defined for {dt}", node)
        return LOAD_MODES[key]

    def store_mode(self, mode: str, dt: DType, node: ast.AST) -> str:
        key = (mode, elem_bytes(dt))
        if key not in STORE_MODES:
            raise self.err(E_BAD_COPY, f"store distribution {mode!r} is not defined for {dt}", node)
        return STORE_MODES[key]

    def copy_into_mask(self, dst: Dyn, src: Any, node: ast.AST) -> None:
        if isinstance(src, RegExpr):
            if src.op not in MASK_OPS:
                raise self.err(E_BAD_COPY, f"only compare / mask expressions can be assigned to a MaskReg, not {src.op}", node)
            self.emit_regexpr(src, dst, node)
            return
        src = self.rvalue(src, node)
        if isinstance(src, Dyn) and isinstance(src.type, (ScalarType, CellType)):
            self.emit("vf.mask_update", (dst,), {"cnt": src}, None, None, node)
            return
        if self.is_mask(src):
            self.emit("vf.mask_mov", (dst, src), {}, None, None, node)
            return
        raise self.err(E_BAD_COPY, f"cannot copy {src!r} into a mask register", node)

    def copy_into_reglist(self, dst: RegList, src: Any, node: ast.AST) -> None:
        if isinstance(src, RegExpr):
            self.emit_regexpr(src, dst, node)
            return
        if isinstance(src, RegList):
            if len(src) != len(dst):
                raise self.err(E_BAD_COPY, "RegList lengths differ", node)
            for d, s in zip(dst.elems, src.elems, strict=True):
                self.emit("vf.copy", (d, s), {}, None, None, node)
            return
        if self.is_ub(src):
            base, off = self.ub_parts(src, node)
            block = 256 // elem_bytes(dst.dtype)  # every register takes one lane's worth of source elements per lane
            for i, d in enumerate(dst.elems):
                self.load_reg(d, ElemOffset(base, self.offset_add(off, block * i, node)), node)
            return
        src = self.rvalue(src, node)
        if isinstance(src, (int, float)) or (isinstance(src, Dyn) and isinstance(src.type, (ScalarType, CellType))):
            for d in dst.elems:
                self.emit("vf.dup", (d, self.reg_scalar(src, dst.dtype, node)), {}, None, None, node)
            return
        raise self.err(E_BAD_COPY, f"cannot copy {src!r} into a RegList", node)

    def offset_add(self, a: Any, b: Any, node: ast.AST) -> Any:
        if isinstance(a, int) and isinstance(b, int):
            return a + b
        if b == 0:
            return a
        if a == 0:
            return b
        return self.binop(ast.Add(), a, b, node)

    def store_to_ub(self, dst: Any, src: Any, node: ast.AST) -> None:
        """``ub[...] <<= reg | expr | RegList`` (the register half of ``Tensor.__ilshift__``)."""
        base, off = self.ub_parts(dst, node)
        st = base.type
        assert isinstance(st, MemType)
        if isinstance(src, RegList):
            block = 256 // elem_bytes(src.dtype)  # one register of source lanes per element, in destination elements
            for i, s in enumerate(src.elems):
                self.store_to_ub(ElemOffset(base, self.offset_add(off, block * i, node)), s, node)
            return
        if isinstance(src, RegExpr):
            if src.op.startswith(STORE_PREFIX):
                reg = src.inputs[0]
                mode = src.op[len(STORE_PREFIX):]
                attrs = {"offset": off, "mode": Ident(self.store_mode(mode, self.reg_dtype(reg), node)), "mask": src.mask}
                self.emit("vf.store_cont", (base, reg), attrs, None, None, node)
                return
            if src.op == "copy":
                reg = src.inputs[0]
                self.emit("vf.store_cont", (base, reg), {"offset": off, "mode": Ident(self.store_mode("normal", self.reg_dtype(reg), node)), "mask": src.mask},
                          None, None, node)
                return
            src = self.materialize(src, node)
        if not self.is_reg(src):
            raise self.err(E_BAD_COPY, f"cannot store {src!r} into UB", node)
        rt = src.type
        assert isinstance(rt, RegType)
        if rt.dtype == st.dtype:
            self.emit("vf.store_cont", (base, src), {"offset": off, "mode": Ident(self.store_mode("normal", rt.dtype, node))}, None, None, node)
            return
        tmp = self.emit("vf.reg", (), {}, RegType(st.dtype, rt.n), "tmp", node)
        self.emit("vf.cast", (tmp, src), {}, None, None, node)
        dsz, ssz = elem_bytes(st.dtype), elem_bytes(rt.dtype)
        if (dsz, ssz) == (1, 4):
            mode = "pack4"
        elif (dsz, ssz) in ((1, 2), (2, 4)):
            mode = "downsample"
        else:
            raise self.err(E_BAD_COPY, f"a converting store from {rt.dtype} to {st.dtype} is not supported (the old DSL silently dropped it)", node)
        self.emit("vf.store_cont", (base, tmp), {"offset": off, "mode": Ident(self.store_mode(mode, st.dtype, node))}, None, None, node)

    # -- explicit vf instructions ----------------------------------------------------------------

    def rule_vf(self, rule: str, callee: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        """The stub-function forms (``ub_to_reg_normal(reg, ub)``, ``expsub(dst, a, b, mask)`` ...)."""
        if self.kind != "vf":
            raise self.err(E_UNSUPPORTED, f"{callee.__name__} is only available inside vf functions", node)
        op, _, variant = rule.partition(":")
        name = op[3:]
        a = [self.reg_operand(x, node) for x in args]
        kw = {k: self.reg_operand(v, node) for k, v in kwargs.items()}

        def arg(i: int, key: str, default: Any = None) -> Any:
            return a[i] if len(a) > i else kw.get(key, default)

        if op == "debug.print_reg":
            reg, label, lanes = arg(0, "reg"), arg(1, "label", ""), arg(2, "lanes", 8)
            if not self.is_reg(reg):
                raise self.err(E_BAD_OPERAND, "print_reg requires a vector register", node)
            if not isinstance(label, str) or type(lanes) is not int or lanes < 0:
                raise self.err(E_BAD_SIGNATURE, "print_reg requires a static string label and nonnegative integer lanes", node)
            self.emit(op, (reg,), {"label": label, "lanes": lanes}, None, None, node)
            return None

        mask = None
        if name in ("load", "store"):
            dst, src = a[0], a[1]
            blk = arg(2, "blk_stride", 1)
            mask = arg(3, "mask")
            if name == "load":
                base, off = self.ub_parts(src, node)
                self.emit("vf.load", (dst, base), {"offset": off, "blk_stride": self.attr(self.rvalue(blk, node)), "mask": mask}, None, None, node)
            else:
                base, off = self.ub_parts(dst, node)
                self.emit("vf.store", (base, src), {"offset": off, "blk_stride": self.attr(self.rvalue(blk, node)), "mask": mask}, None, None, node)
            return None
        if name == "load_cont":
            dst, src = a[0], a[1]
            mode = variant or self.dist_name(arg(2, "loaddist"), node)
            base, off = self.ub_parts(src, node)
            self.emit("vf.load_cont", (dst, base), {"offset": off, "mode": Ident(self.load_mode(mode, self.reg_dtype(dst), node))}, None, None, node)
            return None
        if name == "store_cont":
            dst, src = a[0], a[1]
            if variant:
                mode, mask = variant, arg(2, "mask")
            else:
                mask, mode = arg(2, "mask"), self.dist_name(arg(3, "storedist"), node)
            base, off = self.ub_parts(dst, node)
            self.emit("vf.store_cont", (base, src), {"offset": off, "mode": Ident(self.store_mode(mode, self.reg_dtype(src), node)), "mask": mask}, None, None, node)
            return None
        if name == "load_interleave":
            base, off = self.ub_parts(a[2], node)
            self.emit("vf.load_interleave", (a[0], a[1], base), {"offset": off, "mode": Ident("dintlv_" + INTERLEAVE_MODES[elem_bytes(self.reg_dtype(a[0]))])},
                      None, None, node)
            return None
        if name == "store_interleave":
            base, off = self.ub_parts(a[0], node)
            self.emit("vf.store_interleave", (base, a[1], a[2]), {"offset": off, "mode": Ident("intlv_" + INTERLEAVE_MODES[elem_bytes(self.reg_dtype(a[1]))]),
                                                                  "mask": arg(3, "mask")}, None, None, node)
            return None
        if name in ("gather_copy", "gatherb"):
            base, off = self.ub_parts(a[1], node)
            self.emit(f"vf.{name}", (a[0], base, a[2]), {"offset": off, "mask": arg(3, "mask")}, None, None, node)
            return None
        if name == "scatter_copy":
            base, off = self.ub_parts(a[0], node)
            self.emit("vf.scatter_copy", (base, a[1], a[2]), {"offset": off, "mask": arg(3, "mask")}, None, None, node)
            return None
        if name == "gather":
            self.emit("vf.gather", (a[0], a[1], a[2]), {}, None, None, node)
            return None
        if name == "gathermask":
            self.emit("vf.gathermask", (a[0], a[1]), {"mask": arg(2, "mask")}, None, None, node)
            return None
        if name == "squeeze":
            self.emit("vf.squeeze", (a[0], a[1]), {"mask": arg(2, "mask"), "store": bool(arg(3, "store", False)) or None}, None, None, node)
            return None
        if name == "unsqueeze":
            self.emit("vf.unsqueeze", (a[0],), {"mask": arg(1, "mask")}, None, None, node)
            return None
        if name == "clear_spr":
            self.emit("vf.clear_spr", (), {}, None, None, node)
            return None
        if name in ("ub_to_mask", "mask_to_ub"):
            if name == "ub_to_mask":
                base, off = self.ub_parts(a[1], node)
                self.emit("vf.ub_to_mask", (a[0], base), {"offset": off}, None, None, node)
            else:
                base, off = self.ub_parts(a[0], node)
                self.emit("vf.mask_to_ub", (base, a[1]), {"offset": off}, None, None, node)
            return None
        if name == "ub_cursor":
            base, off = self.ub_parts(a[0], node)
            root = self.roots.get(base.name, base.name)
            src = base if off == 0 else self.emit("mem.slice", (base,), {"offsets": [0, off], "extents": [self.attr(self.dim_value(d, node)) for d in base.type.dims]},
                                                  base.type, "cursor_src", node)
            self.roots[src.name] = root
            res = self.emit("vf.ub_cursor", (src,), {}, base.type, kw.get("name") or "cursor", node)
            self.roots[res.name] = root  # loads and stores through the cursor are accesses of the parameter (cf.call read / write lists)
            return res
        if name == "load_unalign_pre":
            base, off = self.ub_parts(a[1], node)
            self.emit("vf.load_unalign_pre", (a[0], base), {"offset": off}, None, None, node)
            return None
        if name == "load_unalign":
            if variant == "once":
                dst, src = a[0], a[1]
                ureg = arg(2, "ureg") or self.emit("vf.unalign", (), {}, self.unalign_type("load"), "ureg", node)
                base, off = self.ub_parts(src, node)
                self.emit("vf.load_unalign_pre", (ureg, base), {"offset": off}, None, None, node)
                self.emit("vf.load_unalign", (dst, base, ureg), {"offset": off}, None, None, node)
                return None
            dst, ureg, src = a[0], a[1], a[2]
            stride = arg(3, "stride")
            post = arg(4, "post_mode")
            base, off = self.ub_parts(src, node)
            attrs: dict[str, Any] = {"offset": off}
            if stride is not None:
                attrs["stride"] = self.attr(self.rvalue(stride, node))
                attrs["post_mode"] = Ident(post.name if isinstance(post, dsl.EnumValue) else "update")
            self.emit("vf.load_unalign", (dst, base, ureg), attrs, None, None, node)
            return None
        if name == "store_unalign":
            if variant == "once":
                dst, src, count = a[0], a[1], a[2]
                ureg = arg(3, "ureg") or self.emit("vf.unalign", (), {}, self.unalign_type("store"), "ureg", node)
                base, off = self.ub_parts(dst, node)
                self.emit("vf.store_unalign", (base, src, ureg), {"offset": off, "count": self.attr(self.rvalue(count, node)), "post_mode": Ident("update")}, None, None, node)
                self.emit("vf.store_unalign_post", (base, ureg), {"offset": off, "stride": 0, "post_mode": Ident("update")}, None, None, node)
                return None
            dst, src, ureg, count = a[0], a[1], a[2], a[3]
            base, off = self.ub_parts(dst, node)
            # A5 has no store that leaves the cursor, so PostMode.NORMAL is accepted and means UPDATE (I042).
            self.emit("vf.store_unalign", (base, src, ureg), {"offset": off, "count": self.attr(self.rvalue(count, node)),
                                                               "post_mode": Ident("update")}, None, None, node)
            return None
        if name == "store_unalign_post":
            dst, ureg = a[0], a[1]
            stride = arg(2, "stride", 0)
            base, off = self.ub_parts(dst, node)
            self.emit("vf.store_unalign_post", (base, ureg), {"offset": off, "stride": self.attr(self.rvalue(stride, node)),
                                                              "post_mode": Ident("update")}, None, None, node)
            return None
        if name == "barrier":
            src, dst = a[0], a[1]
            self.emit("vf.barrier", (), {"src": Ident(src.name), "dst": Ident(dst.name)}, None, None, node)
            return None
        if name in ("cmp",):
            dst, src1, src2, mode = a[0], a[1], a[2], a[3]
            mask = arg(4, "mask")
            if self.is_reg(src2):
                self.emit("vf.cmp", (dst, src1, src2), {"mode": Ident(mode.name), "mask": mask}, None, None, node)
            else:
                self.emit("vf.cmps", (dst, src1, self.reg_scalar(self.rvalue(src2, node), self.reg_dtype(src1), node)), {"mode": Ident(mode.name), "mask": mask},
                          None, None, node)
            return None
        if name == "select":
            self.emit("vf.select", (a[0], a[1], a[2]), {"mask": arg(3, "mask")}, None, None, node)
            return None
        if name == "arange":
            dst = a[0]
            start = self.rvalue(arg(1, "start", 0), node)
            increase = arg(2, "increase", True)
            self.emit("vf.arange", (dst,), {"v": self.attr(self.reg_scalar(start, self.reg_dtype(dst), node)), "mode": Ident("increase" if increase else "decrease")},
                      None, None, node)
            return None
        if name == "dup":
            dst, src = a[0], self.rvalue(a[1], node)
            v = src if self.is_reg(src) else self.reg_scalar(src, self.reg_dtype(dst), node)
            self.emit("vf.dup", (dst, v), {"mask": arg(2, "mask")}, None, None, node)
            return None
        if name == "pack":
            part = arg(2, "low_or_high")
            self.emit("vf.pack", (a[0], a[1]), {"part": Ident(part.name if isinstance(part, dsl.EnumValue) else "lowest")}, None, None, node)
            return None
        if name in ("interleave", "deinterleave", "mask_interleave", "mask_deinterleave"):
            self.emit(f"vf.{name}", (a[0], a[1], a[2], a[3]), {}, None, None, node)
            return None
        if name in ("mask_not", "mask_mov"):
            self.emit(f"vf.{name}", (a[0], a[1]), {"mask": arg(2, "mask")}, None, None, node)
            return None
        if name in ("mask_and", "mask_or", "mask_xor", "mask_sel"):
            self.emit(f"vf.{name}", (a[0], a[1], a[2]), {"mask": arg(3, "mask")}, None, None, node)
            return None
        if name in ("mask_pack", "mask_unpack"):
            low = arg(2, "low_part", True)
            self.emit(f"vf.{name}", (a[0], a[1]), {"mode": Ident("lowest" if low else "highest")}, None, None, node)
            return None
        if name == "mask_from_spr":
            self.emit("vf.mask_from_spr", (a[0],), {}, None, None, node)
            return None
        if name == "mask_update":
            self.emit("vf.mask_update", (a[0],), {"cnt": self.attr(self.rvalue(a[1], node))}, None, None, node)
            return None
        if name == "cast":
            dst, src = a[0], a[1]
            cfg = arg(2, "config", kw.get("cfg"))
            self.emit("vf.cast", (dst, src), {**self.cast_attrs(cfg, node), "mask": arg(3, "mask")}, None, None, node)
            return None
        if name in ("expsub", "mulscast"):
            dst = a[0]
            layout = kw.get("layout")
            attrs = {"mask": arg(3, "mask")}
            if isinstance(layout, dsl.EnumValue):
                attrs["layout"] = Ident(layout.name)
            if name == "expsub":
                self.emit("vf.expsub", (dst, a[1], a[2]), attrs, None, None, node)
            else:
                self.emit("vf.mulscast", (dst, a[1], self.reg_scalar(self.rvalue(a[2], node), DTYPES["f32"], node)), attrs, None, None, node)
            return None
        if name == "histograms":
            bin_group = arg(2, "bin_group")
            mode = arg(3, "mode")
            attrs = {"mask": arg(4, "mask")}
            if isinstance(bin_group, dsl.EnumValue):
                attrs["bin_group"] = int(bin_group.name[3:])
            if isinstance(mode, dsl.EnumValue):
                attrs["mode"] = Ident(mode.name)
            self.emit("vf.histograms", (a[0], a[1]), attrs, None, None, node)
            return None
        if name in UNARY.values() or name in ("copy", "not"):
            self.emit(f"vf.{name}", (a[0], a[1]), {"mask": arg(2, "mask")}, None, None, node)
            return None
        if name in SCALAR.values() or name in ("adds", "muls"):
            dst = a[0]
            self.emit(f"vf.{name}", (dst, a[1], self.reg_scalar(self.rvalue(a[2], node), self.reg_dtype(a[1]), node)), {"mask": arg(3, "mask")}, None, None, node)
            return None
        if name in BINARY.values() or name in ("add", "sub", "mul", "div", "mod", "shiftl", "shiftr", "abssub", "muldstadd", "muladddst"):
            self.emit(f"vf.{name}", (a[0], a[1], a[2]), {"mask": arg(3, "mask")}, None, None, node)
            return None
        if name in REDUCE:
            self.emit(f"vf.{name}", (a[0], a[1]), {"mask": arg(2, "mask")}, None, None, node)
            return None
        raise self.err(E_UNSUPPORTED, f"no compile rule for {rule!r}", node)

    def dist_name(self, v: Any, node: ast.AST) -> str:
        if isinstance(v, dsl.EnumValue) and v.family in ("LoadDist", "StoreDist"):
            return v.name
        raise self.err(E_BAD_SIGNATURE, "expected a LoadDist / StoreDist value", node)

    def unalign_type(self, role: str) -> Any:
        from ..ir.types import UnalignRegType

        return UnalignRegType(role)
