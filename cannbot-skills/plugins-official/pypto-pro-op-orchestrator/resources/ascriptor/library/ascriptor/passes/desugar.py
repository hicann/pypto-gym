# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``desugar``: expand the cube shortcuts into the instructions the hardware runs.

``cube.matmul``, ``cube.matmul_mx`` and ``cube.conv2d`` are Surface-only conveniences (the old
``easyasc.shortcuts``). Each becomes the old stub's instruction stream — slot views of the
kernel-owned L0A / L0B / BT buffers, ``dma.l1_to_l0`` (``.mx`` / ``.img2col``) loads,
``dma.l1_to_bt`` bias staging, ``cube.mmad`` (``.mx``) and the counter bumps — so that autosync
sees the real pipes, the pipe simulator times the real instructions and every backend emits the
same sequence. The tile loops (``splitn`` / ``splitk``, conv's dynamic K) stay device loops
(D-027); a static conv K loop is unrolled like the old shortcut did.

The scratch buffers (``_l0a`` / ``_l0b``: two 32 KB half slots each, ``_btbuf``: two BT slots)
and their counters are created once per function on first use, at the top of the body, as the
old ``KernelBase`` created them for every kernel.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..ir import Block, Function, Ident, Literal, Module, Op, Value
from ..ir.builder import Rewriter
from ..ir.types import BufType, CellType, DimValue, DType, MemType, RegType, ScalarType, dtype
from .manager import Pass, PassContext, PassError
from .util import I32, Defs, Emit, Scalar, View, as_int, function_names, view_of

PASS = "desugar"
B1 = ScalarType(dtype("b1"))
U8 = dtype("u8")
F32 = dtype("f32")
INT32 = dtype("i32")
L0_ROWS = 128
L0_BYTES_PER_ROW = 256  # a slot is 32 KB of bytes (the old half [128, 128]); a tile of any dtype is viewed onto it


def c0_of(dt: DType) -> int:
    """Elements per 32-byte fractal row."""
    return 64 if dt.bits < 8 else 32 * 8 // dt.bits


def dim_of(x: Scalar) -> Any:
    return DimValue(x.name) if isinstance(x, Value) else int(x)


class _Scratch:
    """The function's L0A / L0B / BT slot buffers and counters (old ``KernelBase._l0a/_l0b/_btbuf`` + ``*cnt``)."""

    def __init__(self, rw: Rewriter, names: set[str], anchor: Op, bt_slot_elems: int) -> None:
        self.rw = rw
        self.names = names
        self.anchor = anchor
        self.decls: list[Op] = []
        self.l0a = self._buf("_l0a", MemType("l0a", U8, (L0_ROWS, L0_BYTES_PER_ROW), "nz"))
        self.l0b = self._buf("_l0b", MemType("l0b", U8, (L0_ROWS, L0_BYTES_PER_ROW), "nz"))
        self.bt = self._buf("_btbuf", MemType("bt", F32, (1, bt_slot_elems), None))
        self.l0acnt = self._cell("_l0acnt")
        self.l0bcnt = self._cell("_l0bcnt")
        self.btcnt = self._cell("_btbufcnt")

    def fresh(self, base: str) -> str:
        name, n = base, 0
        while name in self.names:
            n += 1
            name = f"{base}.{n}"
        self.names.add(name)
        return name

    def _buf(self, base: str, elem: MemType) -> Value:
        v = Value(self.fresh(base), BufType(elem, 2))
        self.decls.append(self.rw.make("mem.alloc", (), results=(v,), attrs={"name": v.name}, from_ops=(self.anchor,), kind="expanded",
                                       note="scratch slot buffer of the cube shortcuts"))
        return v

    def _cell(self, base: str) -> Value:
        v = Value(self.fresh(base), CellType(INT32))
        self.decls.append(self.rw.make("scalar.cell", (), results=(v,), attrs={"init": 0}, from_ops=(self.anchor,), kind="expanded",
                                       note="slot counter of the cube shortcuts"))
        return v


class _Expander:
    """One shortcut op -> its instruction stream."""

    def __init__(self, rw: Rewriter, fn: Function, names: set[str], scratch: _Scratch, defs: Defs, ctx: PassContext, op: Op) -> None:
        self.rw = rw
        self.fn = fn
        self.names = names
        self.s = scratch
        self.defs = defs
        self.ctx = ctx
        self.op = op
        self.ops: list[Op] = []  # the current block being filled
        self.e = Emit(rw, fn, op, names)

    # -- helpers -----------------------------------------------------------------------------

    def err(self, msg: str) -> PassError:
        return PassError(PASS, f"{self.op.opcode} #{self.op.id} ({self.op.loc}): {msg}")

    def make(self, opcode: str, operands: tuple = (), results: tuple = (), attrs: dict | None = None, note: str | None = None,
             regions: tuple = ()) -> Op:
        self.flush()
        op = self.rw.make(opcode, operands, results=results, attrs=attrs, from_ops=(self.op,), kind="expanded", note=note or self.op.opcode,
                          loc=self.op.loc, regions=regions)
        self.ops.append(op)
        return op

    def flush(self) -> None:
        self.ops.extend(self.e.pre)
        self.e.pre.clear()

    def value(self, x: Any) -> Scalar:
        return self.e.value(x)

    def fresh(self, base: str, t: Any) -> Value:
        return Value(self.s.fresh(base), t)

    def view(self, v: Value) -> View:
        if not isinstance(v, Value):
            raise self.err("operands must be tensors")
        w = view_of(v, self.defs)
        if w.rank != 2 or not all(w.kept):
            raise self.err(f"%{v.name} must be a two-dimensional on-chip tile")
        return w

    def slot(self, buf: Value, cnt: Value, dt: DType, rows: Scalar, cols: Scalar) -> Value:
        """``buf[cnt]`` seen as a ``[rows, cols]`` tile of the operand dtype at the slot start (old ``_prepare_l0_tensors``
        re-viewed the half slot; the tile shape is what ``mmad`` reads)."""
        elem = buf.type.elem
        assert isinstance(elem, MemType)
        s = self.fresh(buf.name.lstrip("_") + "_slot", elem)
        self.make("mem.get_buf", (buf, cnt), results=(s,))
        rows, cols = self.value(rows), self.value(cols)
        if isinstance(rows, int) and isinstance(cols, int):
            nbytes = rows * cols * dt.bits // 8
            if nbytes > L0_ROWS * L0_BYTES_PER_ROW:
                raise self.err(f"a [{rows}, {cols}] {dt.name} tile ({nbytes} bytes) exceeds the 32 KB L0 slot; use splitn / splitk")
        v = self.fresh(buf.name.lstrip("_") + "_view", MemType(elem.space, dt, (dim_of(rows), dim_of(cols)), elem.layout))
        self.make("mem.reinterpret", (s,), results=(v,), attrs={"packed_axis": 1, "tile": [rows, cols]})
        return v

    def slice2(self, src: Value, w: View, off: tuple[Scalar, Scalar], ext: tuple[Scalar, Scalar]) -> tuple[Value, tuple[Scalar, Scalar]]:
        """A rectangular window of ``src``; returns the view and its absolute offsets in the root."""
        t = src.type
        assert isinstance(t, MemType)
        off = (self.value(off[0]), self.value(off[1]))
        ext = (self.value(ext[0]), self.value(ext[1]))
        abs_off = (self.e.add(self.value(w.offsets[0]), off[0]), self.e.add(self.value(w.offsets[1]), off[1]))
        if all(isinstance(o, int) and o == 0 for o in off) and all(a == b for a, b in zip(ext, (self.value(x) for x in w.span), strict=True)):
            return src, abs_off
        v = self.fresh(src.name + "_tile", MemType(t.space, t.dtype, (dim_of(ext[0]), dim_of(ext[1])), t.layout))
        self.make("mem.slice", (src,), results=(v,), attrs={"offsets": list(off), "extents": list(ext)})
        return v, abs_off

    def l1_to_l0(self, dst: Value, src: Value, w: View, src_off: tuple[Scalar, Scalar], m_dst: Scalar, n_dst: Scalar, transpose: bool,
                 extra: dict[str, Any] | None = None, opcode: str = "dma.l1_to_l0", logical: int = 1) -> None:
        # ``w.shape`` is the *carrier's*, and ``m_dst``/``n_dst`` are already logical (`l0_tile`
        # scaled them). A packed mx source has to say the same thing on both sides or the two
        # halves of the op are in different units: `mxfp4_dense_scale_matmul` calls the low-level
        # `l1_to_l0_mx` and passes n_src = K = 128 logical fp4, while `mxfp4_carrier_matmul` came
        # through here and said 32 *bytes* for the same quantity. cce never notices (its
        # `mx_src_stride = CeilDiv(n_src, C0 * 2)` is 1 either way at these sizes); the scale
        # plane's own shape does, since it is `n_src / C0` wide.
        packed = 0 if transpose else 1
        src_shape = [self.value(w.shape[0]), self.value(w.shape[1])]
        if logical != 1:
            src_shape[packed] = self.e.mul(src_shape[packed], logical)
        attrs: dict[str, Any] = {"m_src": src_shape[0], "n_src": src_shape[1], "m_dst": self.value(m_dst),
                                 "n_dst": self.value(n_dst), "src_row0": self.value(src_off[0]), "src_col0": self.value(src_off[1]),
                                 "src_is_transpose": bool(transpose), "dst_position": Ident(dst.type.space)}
        if extra:
            attrs.update(extra)
        self.make(opcode, (dst, src), attrs=attrs)

    def stage_bias(self, bias: Value, col0: Scalar, width: Scalar) -> Value:
        """``bias[:, col0:col0+width]`` -> the current BT slot (old ``_stage_bias_tile``)."""
        bw = self.view(bias)
        bt_elem = self.s.bt.type.elem
        assert isinstance(bt_elem, MemType)
        cap = int(bt_elem.dims[1])
        if isinstance(width, int) and width > cap:
            raise self.err(f"bias tile width {width} exceeds one BT slot ({cap} fp32 elements); use a smaller splitn")
        bt = self.fresh("bt_slot", bt_elem)
        self.make("mem.get_buf", (self.s.bt, self.s.btcnt), results=(bt,))
        bdt = bias.type.dtype
        if bdt.name != bt_elem.dtype.name and bdt.bits == bt_elem.dtype.bits:  # int32 bias: the fp32 slot viewed as int32
            v = self.fresh("bt_view", MemType(bt_elem.space, bdt, bt_elem.dims, bt_elem.layout))
            self.make("mem.reinterpret", (bt,), results=(v,))
            bt = v
        # The bias row is contiguous in L1 (a plain byte copy wrote it, l1_to_bt reads it as one run), whatever
        # layout its L1 type defaults to: slice it as ND so a column offset is an element offset, not a fractal one.
        if bias.type.layout == "nz":
            nd = self.fresh("bias_nd", MemType(bias.type.space, bias.type.dtype, bias.type.dims, None))
            self.make("mem.reinterpret", (bias,), attrs={"layout": Ident("nd")}, results=(nd,))
            bias = nd
        src, _ = self.slice2(bias, bw, (0, col0), (bw.span[0], width))
        self.make("dma.l1_to_bt", (bt, src), attrs={"n": self.value(width)})
        return bt

    def mmad(self, dst: Value, dw: View, dst_off: tuple[Scalar, Scalar], a: Value, b: Value, M: Scalar, N: Scalar, K: Scalar, init: bool,
             bias: Value | None, opcode: str = "cube.mmad") -> None:
        attrs: dict[str, Any] = {"M": self.value(M), "N": self.value(N), "K": self.value(K), "is_init": bool(init),
                                 "dst_row0": self.value(dst_off[0]), "dst_col0": self.value(dst_off[1]),
                                 "dst_rows": self.value(dw.shape[0]), "dst_cols": self.value(dw.shape[1])}
        if bias is not None and init:
            attrs["bias"] = bias
        self.make(opcode, (dst, a, b), attrs=attrs)

    def inc(self, cell: Value) -> None:
        t = self.fresh(cell.name.lstrip("_") + "_next", I32)
        self.make("scalar.add", (cell, 1), results=(t,))
        self.make("scalar.set", (cell, t))

    def loop(self, name: str, lo: Scalar, hi: Scalar, step: Scalar, body: Any) -> Value:
        """``for name in range(lo, hi, step)`` as a device loop; ``body(induction)`` fills the body block."""
        self.flush()
        i = self.fresh(name, I32)
        outer, self.ops = self.ops, []
        body(i)
        self.flush()
        inner, self.ops = self.ops, outer
        self.make("cf.for", (lo, hi, step), results=(i,), attrs={"name": i.name}, regions=(Block(tuple(inner)),))
        return i

    def branch(self, cond: Value, then: Any, otherwise: Any | None = None) -> None:
        self.flush()
        outer, self.ops = self.ops, []
        then()
        self.flush()
        then_ops, self.ops = self.ops, []
        if otherwise is not None:
            otherwise()
            self.flush()
        else_ops, self.ops = self.ops, outer
        self.make("cf.if", (cond,), regions=(Block(tuple(then_ops)), Block(tuple(else_ops))))

    def eq0(self, i: Value) -> Value:
        c = self.fresh(i.name + "_is0", B1)
        self.make("scalar.cmp", (i, 0), results=(c,), attrs={"pred": Ident("eq")})
        return c

    # -- matmul ------------------------------------------------------------------------------

    def matmul(self, mx: bool) -> list[Op]:
        op = self.op
        dst, a, b = op.operands[0], op.operands[1], op.operands[2]
        sa, sb = (op.operands[3], op.operands[4]) if mx else (None, None)
        A, B, D = self.view(a), self.view(b), self.view(dst)
        at, bt = bool(op.attrs.get("a_transpose", False)), bool(op.attrs.get("b_transpose", False))
        dta, dtb = a.type.dtype, b.type.dtype
        int4_k: Any = None
        if not mx and (dta.name == "i4" or dtb.name == "i4"):
            # a2 int4 mmad (the old framework's DT.int4 contract): both operands are DT.int4 views of
            # DT.int carriers. The loads move the CARRIERS - an i32 zZ fractal and an i4 one lay out
            # the same bytes (equal 32-byte fractal rows, the packing runs along the row) - and only
            # the mmad reads the tile as int4, with K counted in logical int4 elements.
            if getattr(self.ctx.device, "arch", "") != "c220":
                raise self.err("DT.int4 mmad is a c220 (a2) feature; this device has no int4 cube path")
            if dta.name != "i4" or dtb.name != "i4":
                raise self.err("an int4 matmul needs BOTH operands as DT.int4 views (got "
                               f"{dta.name} / {dtb.name})")
            if dst.type.dtype.name != "i32":
                raise self.err(f"an int4 matmul accumulates in int32; the L0C tile is {dst.type.dtype.name}")

            def _carrier(v: Value, label: str) -> Value:
                d = self.defs.op(v)
                if d is None or d.opcode != "mem.reinterpret":
                    raise self.err(f"the int4 {label} operand must come from carrier.reinterpret(DT.int4)")
                src0 = d.operands[0]
                if src0.type.dtype.name != "i32":
                    raise self.err(f"the int4 {label} carrier must be DT.int (i32), got {src0.type.dtype.name}")
                return src0

            int4_k = self.value(op.attrs["k"]) if "k" in op.attrs else None
            a, b = _carrier(a, "A"), _carrier(b, "B")
            A, B = self.view(a), self.view(b)
            dta, dtb = a.type.dtype, b.type.dtype
            if int4_k is None:
                int4_k = self.e.mul(self.value(A.span[1]), 8)  # logical K = carrier columns * 8
        elif dta.name == "i32" or (dta.bits == 4 and not mx):
            raise self.err("int4 / int32 matmul (a2 carriers) is not supported on this device")
        logical = 2 if (mx and dta.bits < 8) else 1  # fp4 carriers hold two logical elements per view element on the packed axis
        a_rows, a_cols = self.value(A.span[0]), self.e.mul(self.value(A.span[1]), logical)
        b_rows, b_cols = self.value(B.span[0]), self.e.mul(self.value(B.span[1]), logical)
        m = self.value(op.attrs["m"]) if "m" in op.attrs else (a_cols if at else a_rows)
        k = self.value(op.attrs["k"]) if "k" in op.attrs else (a_rows if at else a_cols)
        if int4_k is not None:
            k = int4_k  # logical int4 K; the carrier span (K/8) drives only the loads
        n = self.value(op.attrs["n"]) if "n" in op.attrs else (b_cols if bt else b_rows)
        if not mx and int4_k is None:
            # MMAD derives its physical L0 pitch from the consumed dimensions.
            # Keep the full L1 source pitch, but stage only that logical window.
            A = replace(A, extents=(k, m) if at else (m, k))
            B = replace(B, extents=(k, n) if bt else (n, k))
            if any(key in op.attrs for key in ("m", "n", "k")):
                self.ctx.explain.note("matmul extents also select L0 staging geometry; L1 source pitch is preserved", op=op.id, kind="expanded")
        init = bool(op.attrs.get("init", True))
        bias = op.attrs.get("bias")
        if bias is not None and not init:
            raise self.err("a bias needs is_init=True (C = bias + A @ B)")
        splitn, splitk = as_int(op.attrs.get("splitn")), as_int(op.attrs.get("splitk"))
        if "splitn" in op.attrs and splitn is None or "splitk" in op.attrs and splitk is None:
            raise self.err("splitn / splitk must be static ints")
        if splitn and splitk:
            raise self.err("splitn and splitk cannot both be set")
        mm = "cube.mmad.mx" if mx else "cube.mmad"
        ld = "dma.l1_to_l0.mx" if mx else "dma.l1_to_l0"
        s = self.s

        def mx_attrs(scale: Value | None, offset: Scalar = 0) -> dict[str, Any] | None:
            if scale is None:
                return None
            sw = self.view(scale)
            out: dict[str, Any] = {"src_mx": scale, "src_mx_row0": self.value(sw.offsets[0]), "src_mx_col0": self.value(sw.offsets[1])}
            if not (isinstance(offset, int) and offset == 0):
                out["src_mx_offset_element"] = self.value(offset)
            return out

        def l0_tile(w: View, rows: Scalar | None = None, cols: Scalar | None = None, transpose: bool = False) -> tuple[Scalar, Scalar]:
            """The L0 tile a load of ``[rows, cols]`` (default: the whole view) leaves: transposed loads swap the axes."""
            r = w.span[0] if rows is None else rows
            c = w.span[1] if cols is None else cols
            if mx:
                c = self.e.mul(self.value(c), logical) if not transpose else c
                r = self.e.mul(self.value(r), logical) if transpose else r
            return (c, r) if transpose else (r, c)

        def i4v(l0: Value, rows: Scalar, cols: Scalar) -> Value:
            """int4 mmad operand: the carrier-typed L0 tile seen as [rows, cols * 8] int4 (same bytes)."""
            if int4_k is None:
                return l0
            t = l0.type
            assert isinstance(t, MemType)
            lcols = self.e.mul(self.value(cols), 8)
            v = self.fresh(l0.name + "_i4", MemType(t.space, dtype("i4"), (dim_of(self.value(rows)), dim_of(lcols)), t.layout))
            self.make("mem.reinterpret", (l0,), results=(v,), attrs={"packed_axis": 1, "tile": [self.value(rows), lcols]})
            return v

        if splitn:
            def body(i: Value) -> None:
                valid_n = self.e.min(self.e.sub(n, i), splitn)
                ar, ac = l0_tile(A, transpose=at)
                br, bc = l0_tile(B, cols=valid_n, transpose=True) if bt else l0_tile(B, rows=valid_n)
                a_l0c = self.slot(s.l0a, s.l0acnt, dta, ar, ac)
                b_l0c = self.slot(s.l0b, s.l0bcnt, dtb, br, bc)
                a_l0, b_l0 = i4v(a_l0c, ar, ac), i4v(b_l0c, br, bc)
                self.branch(self.eq0(i), lambda: self.l1_to_l0(a_l0c, a, A, A.offsets, A.span[0], A.span[1], at, mx_attrs(sa), ld, logical))
                if bt:
                    tile, off = self.slice2(b, B, (0, i), (B.span[0], valid_n))
                    self.l1_to_l0(b_l0c, tile, B, off, B.span[0], valid_n, True, mx_attrs(sb, self.scale_row_offset(i, k)), ld, logical)
                else:
                    tile, off = self.slice2(b, B, (i, 0), (valid_n, B.span[1]))
                    self.l1_to_l0(b_l0c, tile, B, off, valid_n, B.span[1], False, mx_attrs(sb, self.scale_row_offset(i, k)), ld, logical)
                bt_v = self.stage_bias(bias, i, valid_n) if bias is not None else None
                sub, doff = self.slice2(dst, D, (0, i), (D.span[0], valid_n))
                self.mmad(sub, D, doff, a_l0, b_l0, m, valid_n, k, init, bt_v, mm)
                if bias is not None:
                    self.inc(s.btcnt)
                self.inc(s.l0bcnt)

            self.loop("_subn", 0, n, splitn, body)
            self.inc(s.l0acnt)
            return self.ops

        if splitk:
            k_given = "k" in op.attrs
            bt_v = self.stage_bias(bias, 0, n) if bias is not None else None

            def body(i: Value) -> None:
                valid_k = self.e.min(self.e.sub(k, i), splitk)
                copies = {}
                for w, dt in ((A, dta), (B, dtb)):
                    if mx:
                        copies[id(w)] = valid_k
                    elif int4_k is not None:
                        # `k` is logical int4 elements, but an int4 operand's L1 tile, its L0 slot
                        # and `i4v` are all counted in int32 carriers of 8 -- the no-split path
                        # passes `A.span[1]` here for exactly that reason. Passing `valid_k` made
                        # every consumer eight times too wide and the L1 window start eight times
                        # too far along; only `mmad` stayed right, because it takes `valid_k`
                        # directly. Exact on an A2 card without a split and wrong with one, at
                        # both 8 and 16 carriers per chunk, while the same split in fp16 is exact.
                        copies[id(w)] = self.e.div(self.e.add(valid_k, 7), 8)  # ceil: a tail chunk
                    else:
                        copies[id(w)] = self.e.align(valid_k, c0_of(dt)) if k_given else valid_k
                ar, ac = l0_tile(A, rows=copies[id(A)], transpose=True) if at else l0_tile(A, cols=copies[id(A)])
                br, bc = l0_tile(B, rows=copies[id(B)], transpose=True) if bt else l0_tile(B, cols=copies[id(B)])
                a_l0c = self.slot(s.l0a, s.l0acnt, dta, ar, ac)
                b_l0c = self.slot(s.l0b, s.l0bcnt, dtb, br, bc)
                a_l0, b_l0 = i4v(a_l0c, ar, ac), i4v(b_l0c, br, bc)
                for src, w, tr, l0, scale in ((a, A, at, a_l0c, sa), (b, B, bt, b_l0c, sb)):
                    if mx:
                        copy = valid_k
                        if tr:
                            tile, off = self.slice2(src, w, (i, 0), (copy, w.span[1]))
                            self.l1_to_l0(l0, tile, w, off, copy, w.span[1], True, mx_attrs(scale, self.e.mul(self.e.div(i, 64), 32)), ld, logical)
                        else:
                            tile, off = self.slice2(src, w, (0, self.e.div(i, logical)), (w.span[0], self.e.div(copy, logical)))
                            self.l1_to_l0(l0, tile, w, off, w.span[0], copy, False, mx_attrs(scale, self.e.mul(self.e.div(i, 64), 32)), ld, logical)
                        continue
                    copy = copies[id(w)]
                    start = self.e.div(i, 8) if int4_k is not None else i  # the same carrier unit
                    if tr:
                        tile, off = self.slice2(src, w, (start, 0), (copy, w.span[1]))
                        self.l1_to_l0(l0, tile, w, off, copy, w.span[1], True)
                    else:
                        tile, off = self.slice2(src, w, (0, start), (w.span[0], copy))
                        self.l1_to_l0(l0, tile, w, off, w.span[0], copy, False)
                if init:
                    self.branch(self.eq0(i), lambda: self.mmad(dst, D, D.offsets, a_l0, b_l0, m, n, valid_k, True, bt_v, mm),
                                lambda: self.mmad(dst, D, D.offsets, a_l0, b_l0, m, n, valid_k, False, None, mm))
                else:
                    self.mmad(dst, D, D.offsets, a_l0, b_l0, m, n, valid_k, False, None, mm)
                # The c220 A2 family (910B and 910_93) does not interlock two short
                # MMADs that update the same L0C accumulator.  M=16 exposes the RAW hazard:
                # the following accumulate can read L0C before the preceding
                # bias/init MMAD has settled.  PIPE_M is the narrow hardware fix;
                # keep it in the IR so simulation, scheduling and CCE emission all
                # see the same ordering.  The missing interlock is a writeback, which
                # does not read the accumulator dtype (M10-081, 2026-09-18), so the
                # settle does not either.  Other device families keep their own
                # measured ordering policy and are not widened into this.
                if self.ctx.device.family == "a2":
                    self.make("sync.barrier", (), attrs={"pipe": Ident("M")},
                              note="a2 family: settle the split-K L0C accumulator between MMADs")
                self.inc(s.l0acnt)
                self.inc(s.l0bcnt)

            self.loop("_subk", 0, k, splitk, body)
            if bias is not None:
                self.inc(s.btcnt)
            return self.ops

        if mx:
            a_l0 = self.slot(s.l0a, s.l0acnt, dta, *((m, k) if not at else (m, k)))
            b_l0 = self.slot(s.l0b, s.l0bcnt, dtb, n, k)
        else:
            ar, ac = l0_tile(A, transpose=at)
            br, bc = l0_tile(B, transpose=bt)
            a_l0c = self.slot(s.l0a, s.l0acnt, dta, ar, ac)
            b_l0c = self.slot(s.l0b, s.l0bcnt, dtb, br, bc)
            a_l0, b_l0 = i4v(a_l0c, ar, ac), i4v(b_l0c, br, bc)
        if mx:
            self.l1_to_l0(a_l0, a, A, A.offsets, k if at else m, m if at else k, at, mx_attrs(sa), ld, logical)
            self.l1_to_l0(b_l0, b, B, B.offsets, k if bt else n, n if bt else k, bt, mx_attrs(sb), ld, logical)
        else:
            self.l1_to_l0(a_l0c, a, A, A.offsets, A.span[0], A.span[1], at)
            self.l1_to_l0(b_l0c, b, B, B.offsets, B.span[0], B.span[1], bt)
        bt_v = self.stage_bias(bias, 0, n) if bias is not None else None
        self.mmad(dst, D, D.offsets, a_l0, b_l0, m, n, k, init, bt_v, mm)
        if bias is not None:
            self.inc(s.btcnt)
        self.inc(s.l0acnt)
        self.inc(s.l0bcnt)
        return self.ops

    def scale_row_offset(self, i: Value, k: Scalar) -> Scalar:
        """Byte offset of the 16-row scale tile that starts at row ``i``: (i // 16) * ceil(k / 64) * 32."""
        return self.e.mul(self.e.mul(self.e.div(i, 16), self.e.ceil_div(k, 64)), 32)

    # -- conv2d ------------------------------------------------------------------------------

    def conv2d(self) -> list[Op]:
        op = self.op
        dst, fmap, weight = op.operands[0], op.operands[1], op.operands[2]
        D, F, W = self.view(dst), self.view(fmap), self.view(weight)
        if any(not (isinstance(o, int) and o == 0) for o in F.offsets):
            raise self.err("the feature map must be a whole L1 tensor, not a slice")
        dt = fmap.type.dtype
        c0 = c0_of(dt)
        h, w, c, cout = (self.value(op.attrs[x]) for x in ("h", "w", "c", "cout"))
        kh, kw = int(op.attrs["kh"]), int(op.attrs["kw"])
        tile_m, cout_p_dst = as_int(D.span[0]), as_int(D.span[1])
        if tile_m is None or cout_p_dst is None:
            raise self.err("the L0C tile must have a static shape")
        cout_p = -(-cout // 16) * 16 if isinstance(cout, int) else cout_p_dst
        if cout_p != cout_p_dst:
            raise self.err(f"l0c must be [tile_m, {cout_p}], got [{tile_m}, {cout_p_dst}]")
        if tile_m % 16:
            raise self.err(f"tile_m must be 16-aligned, got {tile_m}")
        c1 = self.e.ceil_div(c, c0)
        K = self.e.mul(c1, kh * kw * c0)
        tile_k = self.value(op.attrs["tile_k"]) if "tile_k" in op.attrs else None
        if tile_k is None:
            if not isinstance(K, int):
                raise self.err("tile_k is required when c is a runtime scalar (K is dynamic)")
            tile_k = _pick_tile_k(K, cout_p, max(dt.bits, 8) // 8)
        if not isinstance(tile_k, int):
            raise self.err("tile_k must be static")
        slot = 32 * 1024 // (max(dt.bits, 8) // 8)
        if tile_m * tile_k > slot or tile_k * cout_p > slot:
            raise self.err(f"tile [{tile_m}, {tile_k}] x [{cout_p}, {tile_k}] exceeds the 32 KB L0 slot")
        bias = op.attrs.get("bias")
        s = self.s
        bt_v = self.stage_bias(bias, 0, cout_p) if bias is not None else None
        c_hw = self.e.mul(c1, c0)
        geom = {k2: self.value(op.attrs.get(k2, d)) for k2, d in (("stride_h", 1), ("stride_w", 1), ("dil_h", 1), ("dil_w", 1), ("pad_t", 0),
                                                                    ("pad_b", 0), ("pad_l", 0), ("pad_r", 0))}
        m0 = self.value(op.attrs.get("m0", 0))

        def k_tile(k0: Scalar, init: bool) -> None:
            a_l0, b_l0 = self.slot(s.l0a, s.l0acnt, dt, tile_m, tile_k), self.slot(s.l0b, s.l0bcnt, dt, cout_p, tile_k)
            attrs = {"h": h, "w": w, "c": c_hw, "c0": c0, "kh": kh, "kw": kw, "k0": self.value(k0), "m0": m0, "k_ext": tile_k, "m_ext": tile_m,
                     **geom, "dst_position": Ident("l0a")}
            self.make("dma.l1_to_l0.img2col", (a_l0, fmap), attrs=attrs)
            tile, off = self.slice2(weight, W, (0, k0), (W.span[0], tile_k))
            self.l1_to_l0(b_l0, tile, W, off, cout_p, tile_k, False)
            self.mmad(dst, D, D.offsets, a_l0, b_l0, tile_m, cout_p, tile_k, init, bt_v if init else None)
            self.make("sync.barrier", (), attrs={"pipe": Ident("M")}, note="conv2d: L0C reuse across K tiles (old PipeBarrier<PIPE_M>)")
            self.inc(s.l0acnt)
            self.inc(s.l0bcnt)

        if isinstance(K, int):
            for k0 in range(0, K, tile_k):
                k_tile(k0, k0 == 0)
        else:
            k_tile(0, True)
            self.loop("_k0", tile_k, K, tile_k, lambda i: k_tile(i, False))
        if bias is not None:
            self.inc(s.btcnt)
        return self.ops


def _pick_tile_k(K: int, cout_p: int, esize: int) -> int:
    """The old ``_pick_tile_k``: the largest 16-multiple divisor of K whose [tile_k, cout_p] L0B tile fits one slot."""
    cap = 32 * 1024 // (cout_p * esize)
    for t in range(min(K, cap // 16 * 16), 0, -16):
        if K % t == 0:
            return t
    return 16


SHORTCUTS = ("cube.matmul", "cube.matmul_mx", "cube.conv2d")


def run(module: Module, ctx: PassContext) -> Module:
    rw = Rewriter(module, PASS)
    defs = Defs(module)
    bt_cap_kb = float(ctx.device.capacities_kb.get("bt", 4))
    bt_slot = int(bt_cap_kb * 1024) // (2 * 4)  # two fp32 slots share the BT capacity
    functions = []

    def expand_function(f: Function) -> Function:
        """Expand one function with its own shape names and scratch declarations."""
        if f.kind == "vf":
            def lower_arange(block: Block) -> Block:
                ops = []
                for op in block.ops:
                    if op.regions:
                        op = replace(op, regions=tuple(lower_arange(b) for b in op.regions))
                    if op.opcode == "vf.arange" and isinstance(op.operands[0].type, RegType):
                        dt = op.operands[0].type.dtype
                        decreasing = str(op.attrs.get("mode", "increase")) == "decrease"
                        start = op.attrs.get("v", 0)
                        value = start.value if isinstance(start, Literal) else start
                        if (dt.name == "i64" and (isinstance(value, Value) or value != 0)) or (dt.is_integer and decreasing):
                            dst = op.operands[0]
                            ops.append(rw.make("vf.arange", (dst,), attrs={**op.attrs, "v": 0, "mode": Ident("increase")}, from_ops=(op,), kind="expanded",
                                               note="integer arange: generate zero-based increasing lane indices"))
                            if decreasing:
                                ops.append(rw.make("vf.neg", (dst, dst), from_ops=(op,), kind="expanded",
                                                   note="integer arange: negate lane indices; DEC_ORDER reverses an increasing interval"))
                            if isinstance(value, Value) or value != 0:
                                operand = start if isinstance(start, (Value, Literal)) else Literal(start)
                                ops.append(rw.make("vf.adds", (dst, dst, operand), from_ops=(op,), kind="expanded",
                                                   note="integer arange: add the full-width starting value"))
                            ctx.explain.note("integer arange lowered to lane indices, optional negation and a full-width add", op=op.id, kind="expanded")
                            continue
                    ops.append(op)
                return Block(tuple(ops))
            f = replace(f, body=lower_arange(f.body))
        if not any(o.opcode in SHORTCUTS for o in f.walk()):
            return f
        names = function_names(f)
        first = next(o for o in f.walk() if o.opcode in SHORTCUTS)
        scratch = _Scratch(rw, names, first, bt_slot)

        def walk(block: Block) -> Block:
            out: list[Op] = []
            for op in block.ops:
                if op.regions:
                    op = replace(op, regions=tuple(walk(r) for r in op.regions))
                if op.opcode in SHORTCUTS:
                    ex = _Expander(rw, f, names, scratch, defs, ctx, op)
                    if op.opcode == "cube.conv2d":
                        new = ex.conv2d()
                    else:
                        new = ex.matmul(op.opcode == "cube.matmul_mx")
                    ctx.explain.note(f"{op.opcode} expanded into {len(new)} ops (#{new[0].id}..#{new[-1].id})", op=op.id, kind="expanded")
                    out.extend(new)
                else:
                    out.append(op)
            return Block(tuple(out))

        body = walk(f.body)
        return replace(f, body=Block(tuple(scratch.decls) + body.ops))

    for f in module.functions:
        functions.append(expand_function(f))
    attrs = dict(module.attrs)
    attrs["next_id"] = rw._next_id
    return Module(module.name, attrs, tuple(functions))


PASS_DEF = Pass(PASS, run, doc="expand cube.matmul / matmul_mx / conv2d into l1_to_l0 + mmad streams over the L0 slot buffers")

__all__ = ["PASS_DEF", "run"]
