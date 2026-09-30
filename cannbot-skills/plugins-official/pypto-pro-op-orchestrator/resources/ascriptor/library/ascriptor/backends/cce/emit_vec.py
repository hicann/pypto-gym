# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The c220 (a2 family) tensor-vector printer: ``vec.*`` -> dav-c220 CCE intrinsics (RFC-0008 phase B).

One instruction per op, spelled exactly as CANN's own ``dav_c220`` implementations issue them (argument
order, integer casts, trailing mode bits) — recorded in the D-061 trace (``tmp/a2/c220_mapping.md``) from
handler -> impl -> intrinsic. The count / count_per_rep attributes (D-062) materialise here as the same
mask-SPR bracket CANN's Level-2 counted calls emit, with no barrier: SPR writes dispatch in order on the
V queue. Mask-state wrappers (``SetMaskCount`` & co.) and the ``transdata5hd`` VA-register sequence live
in ``tensorutils_cce.h`` under the c220 sections, keeping printed lines one-per-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...ir import Op, Value
from ...ir.types import DType
from . import cpp
from .arch import c220

if TYPE_CHECKING:  # pragma: no cover
    pass


class VecGap(Exception):
    """Raised as CceGap by the assembly in emit.py (import cycle: CceGap lives there)."""


class C220Vec:
    """Mixin for FnPrinter: the ``vec.*`` handlers. Uses FnPrinter's emit / name / attr / operand helpers."""

    # -- helpers -----------------------------------------------------------------------------------------------------

    def _vgap(self, op: Op, why: str) -> Exception:
        from .emit import CceGap  # late: emit.py imports this module

        return CceGap(op, why)

    def _varch(self, op: Op) -> None:
        if getattr(self.mp, "arch", "c310") != "c220":  # type: ignore[attr-defined]
            raise self._vgap(op, f"{op.opcode} is a c220 (a2 family) tensor-vector instruction; this module targets "
                                 f"{getattr(self.mp, 'arch', 'c310')}")  # type: ignore[attr-defined]

    def _vp(self, v: Value) -> str:
        """The operand's UB pointer."""
        return f"{self.name(v)}.ptr()"  # type: ignore[attr-defined]

    def _vdt(self, op: Op, v: Value, allowed: tuple[str, ...] | None, what: str = "operand") -> DType:
        dt = self._dtype_of(v)  # type: ignore[attr-defined]
        if allowed is not None and dt.name not in allowed:
            raise self._vgap(op, f"{op.opcode}: {what} dtype {dt} has no c220 intrinsic (allowed: {', '.join(allowed)})")
        return dt

    def _vattr(self, op: Op, key: str, default: Any) -> str:
        return self.attr(op, key, default)  # type: ignore[attr-defined]

    def _vrep(self, op: Op) -> str:
        # the polymorphic builtins resolve their overload by the ARGUMENT TYPE PATTERN: repeat must be
        # the uint8_t CANN's own impls pass, or the frontend picks another arch's variant (found on the
        # board: an int32 repeat made vadds resolve to a form that read its scalar as a vector address)
        return f"(uint8_t)({self.attr(op, 'repeat', 1)})"  # type: ignore[attr-defined]

    def _vmode_bracket(self, op: Op) -> tuple[list[str], list[str]]:
        """The D-062 bracket of a counted / count_per_rep op (empty for a plain repeat op)."""
        if op.attrs.get("count") is not None:
            n = self.attr(op, "count")  # type: ignore[attr-defined]
            return [f"SetMaskCount();", f"SetVectorMask({n});"], ["SetMaskNorm();", "ResetMask();"]
        if op.attrs.get("count_per_rep") is not None:
            n = self.attr(op, "count_per_rep")  # type: ignore[attr-defined]
            return [f"SetVectorMaskByCount({n});"], ["ResetMask();"]
        return [], []

    # -- the c220 V-pipe hazard barrier (the old framework's auto bar_v, data half) ----------------------------------
    # The c220 vector pipe does not interlock UB accesses between its own instructions: a vector op that
    # reads (or rewrites) what an earlier vector op wrote needs an explicit pipe_barrier(PIPE_V) between
    # them, or it observes stale bytes (board-verified: a muls -> adds -> exp chain read garbage without
    # them; the old framework's `_insert_b_device_vec_barriers` encoded exactly this RAW / WAR / WAW rule).
    # The tracker mirrors that algorithm over roots: pending read / write sets, cleared by every barrier.

    def _vroots(self, op: Op) -> tuple[set, set]:
        from ...ir import REGISTRY

        reads: set = set()
        writes: set = set()
        spec = REGISTRY.get(op.opcode)
        defs = [getattr(o, "access", "read") for o in getattr(spec, "operands", ())] if spec else []
        for i, v in enumerate(op.operands):
            if not isinstance(v, Value):
                continue
            try:
                root = self.geo(v).root.name  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - a non-window operand (scalar): no UB key
                continue
            (writes if (defs[i] if i < len(defs) else "read") == "write" else reads).add(root)
        tmp = op.attrs.get("tmp_addr_buf")
        if isinstance(tmp, Value):
            try:
                writes.add(self.geo(tmp).root.name)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        return reads, writes

    def v_hazard_state(self) -> tuple[set, set]:
        if not hasattr(self, "_v_pending_reads"):
            self._v_pending_reads: set = set()
            self._v_pending_writes: set = set()
        return self._v_pending_reads, self._v_pending_writes

    def v_hazard_clear(self) -> None:
        pr, pw = self.v_hazard_state()
        pr.clear()
        pw.clear()

    def v_hazard_seed_loop(self, body) -> None:
        """Entering a loop body: its own vector accesses may reach back around the back-edge — a write
        to the next iteration's first reads (RAW/WAW), and a READ to the next iteration's first writes
        (WAR: a younger write's ack can overtake an older multi-repeat read's operand fetch). Seed both
        pending sets so the first hazardous op of the body prints the barrier."""
        from ...ir import REGISTRY

        pr, pw = self.v_hazard_state()
        for o in body.walk():
            if not o.opcode.startswith("vec."):
                continue
            spec = REGISTRY.get(o.opcode)
            defs = [getattr(x, "access", "read") for x in getattr(spec, "operands", ())] if spec else []
            for i, v in enumerate(o.operands):
                if isinstance(v, Value):
                    try:
                        root = self.geo(v).root.name  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001
                        continue
                    if (defs[i] if i < len(defs) else "read") == "write":
                        pw.add(root)
                    else:
                        pr.add(root)

    def _vhazard(self, op: Op) -> None:
        reads, writes = self._vroots(op)
        pr, pw = self.v_hazard_state()
        if (pw & reads) or (pr & writes) or (pw & writes):
            self.emit("PipeBarrier<PIPE_V>();", op, note="c220: V-V hazard")  # type: ignore[attr-defined]
            pr.clear()
            pw.clear()
        pr |= reads
        pw |= writes

    def _vemit(self, op: Op, *lines: str) -> None:
        self._vhazard(op)
        pre, post = self._vmode_bracket(op)
        for t in pre + list(lines) + post:
            self.emit(t, op)  # type: ignore[attr-defined]

    # -- element-wise families ---------------------------------------------------------------------------------------

    def _vbinary(self, op: Op, kind: str) -> None:
        self._varch(op)
        ins, allowed = c220.VEC_BINARY[kind]
        dst, s1, s2 = op.operands[:3]
        self._vdt(op, dst, allowed, "dst")
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        b1, r1 = self._vattr(op, "src1_blk_stride", 1), self._vattr(op, "src1_rep_stride", 8)
        b2, r2 = self._vattr(op, "src2_blk_stride", 1), self._vattr(op, "src2_rep_stride", 8)
        self._vemit(op, f"{ins}({self._vp(dst)}, {self._vp(s1)}, {self._vp(s2)}, {rep}, (uint8_t){db}, (uint8_t){b1}, "
                        f"(uint8_t){b2}, (uint8_t){dr}, (uint8_t){r1}, (uint8_t){r2});")

    def op_vec_add(self, op: Op) -> None:
        self._vbinary(op, "add")

    def op_vec_sub(self, op: Op) -> None:
        self._vbinary(op, "sub")

    def op_vec_mul(self, op: Op) -> None:
        self._vbinary(op, "mul")

    def op_vec_div(self, op: Op) -> None:
        self._vbinary(op, "div")

    def op_vec_max(self, op: Op) -> None:
        self._vbinary(op, "max")

    def op_vec_min(self, op: Op) -> None:
        self._vbinary(op, "min")

    def op_vec_and(self, op: Op) -> None:
        self._vbinary(op, "and")

    def op_vec_or(self, op: Op) -> None:
        self._vbinary(op, "or")

    def op_vec_muladddst(self, op: Op) -> None:
        self._varch(op)
        dst, s1, s2 = op.operands[:3]
        dd = self._vdt(op, dst, ("f16", "f32"), "dst")
        sd = self._vdt(op, s1, ("f16", "f32"), "src1")
        if (dd.name, sd.name) not in (("f16", "f16"), ("f32", "f32"), ("f32", "f16")):
            raise self._vgap(op, f"vmla supports (dst, src) in (f16,f16) (f32,f32) (f32,f16), got ({dd}, {sd})")
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        b1, r1 = self._vattr(op, "src1_blk_stride", 1), self._vattr(op, "src1_rep_stride", 8)
        b2, r2 = self._vattr(op, "src2_blk_stride", 1), self._vattr(op, "src2_rep_stride", 8)
        self._vemit(op, f"vmla({self._vp(dst)}, {self._vp(s1)}, {self._vp(s2)}, {rep}, (uint8_t){db}, (uint8_t){b1}, "
                        f"(uint8_t){b2}, (uint8_t){dr}, (uint8_t){r1}, (uint8_t){r2});")

    def _vunary(self, op: Op, kind: str) -> None:
        self._varch(op)
        ins, allowed = c220.VEC_UNARY[kind]
        dst, src = op.operands[:2]
        self._vdt(op, dst, allowed, "dst")
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        sb, sr = self._vattr(op, "src_blk_stride", 1), self._vattr(op, "src_rep_stride", 8)
        self._vemit(op, f"{ins}({self._vp(dst)}, {self._vp(src)}, {rep}, (uint16_t){db}, (uint16_t){sb}, "
                        f"(uint8_t){dr}, (uint8_t){sr});")

    def op_vec_exp(self, op: Op) -> None:
        self._vunary(op, "exp")

    def op_vec_ln(self, op: Op) -> None:
        self._vunary(op, "ln")

    def op_vec_abs(self, op: Op) -> None:
        self._vunary(op, "abs")

    def op_vec_rec(self, op: Op) -> None:
        self._vunary(op, "rec")

    def op_vec_sqrt(self, op: Op) -> None:
        self._vunary(op, "sqrt")

    def op_vec_rsqrt(self, op: Op) -> None:
        self._vunary(op, "rsqrt")

    def op_vec_not(self, op: Op) -> None:
        self._vunary(op, "not")

    def op_vec_relu(self, op: Op) -> None:
        self._vunary(op, "relu")

    def _vscalar(self, op: Op, kind: str) -> None:
        self._varch(op)
        ins, allowed, trailing = c220.VEC_SCALAR[kind]
        dst, src, v = op.operands[:3]
        dd = self._vdt(op, dst, allowed, "dst")
        sdt = self._dtype_of(src) if kind == "axpy" else dd  # type: ignore[attr-defined]  # vaxpy scalar rides the src dtype
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        sb, sr = self._vattr(op, "src_blk_stride", 1), self._vattr(op, "src_rep_stride", 8)
        if kind in ("shiftls", "shiftrs"):
            carrier = "int32_t" if dd.name.startswith("i") else "uint32_t"
            scalar = f"({carrier}){self.operand(v)}"  # type: ignore[attr-defined]
        else:
            # the intrinsics take the scalar in the register dtype exactly (a half op rounds a wider scalar
            # to half on silicon — the cast is the hardware's own behaviour, and the old handler's spelling)
            scalar = f"({cpp.ctype(sdt)})({self.bare(v, sdt)})"  # type: ignore[attr-defined]
        if kind == "shiftrs":
            tail = f", {self.flag(op, 'round_en', False)}"  # type: ignore[attr-defined]
            self._vemit(op, f"{ins}({self._vp(dst)}, {self._vp(src)}, {scalar}, {rep}, (uint16_t){db}, (uint16_t){sb}, "
                            f"(uint16_t){dr}, (uint16_t){sr}{tail});")
        elif kind in ("maxs", "mins"):
            self._vemit(op, f"{ins}({self._vp(dst)}, {self._vp(src)}, {scalar}, {rep}, (uint16_t){db}, (uint16_t){sb}, "
                            f"(uint8_t){dr}, (uint8_t){sr}{trailing});")
        elif kind == "axpy":
            self._vemit(op, f"vaxpy({self._vp(dst)}, {self._vp(src)}, {scalar}, {rep}, (uint16_t){db}, (uint16_t){sb}, "
                            f"(uint8_t){dr}, (uint8_t){sr});")
        else:
            self._vemit(op, f"{ins}({self._vp(dst)}, {self._vp(src)}, {scalar}, {rep}, (uint16_t){db}, (uint16_t){sb}, "
                            f"(uint16_t){dr}, (uint16_t){sr}{trailing});")

    def op_vec_adds(self, op: Op) -> None:
        self._vscalar(op, "adds")

    def op_vec_muls(self, op: Op) -> None:
        self._vscalar(op, "muls")

    def op_vec_maxs(self, op: Op) -> None:
        self._vscalar(op, "maxs")

    def op_vec_mins(self, op: Op) -> None:
        self._vscalar(op, "mins")

    def op_vec_lrelu(self, op: Op) -> None:
        self._vscalar(op, "lrelu")

    def op_vec_shiftls(self, op: Op) -> None:
        self._vscalar(op, "shiftls")

    def op_vec_shiftrs(self, op: Op) -> None:
        self._vscalar(op, "shiftrs")

    def op_vec_axpy(self, op: Op) -> None:
        self._vscalar(op, "axpy")

    # -- reductions --------------------------------------------------------------------------------------------------

    def _vreduce(self, op: Op, kind: str) -> None:
        self._varch(op)
        if op.attrs.get("count") is not None:
            raise self._vgap(op, f"{op.opcode}: a group reduction has no counter-mode form")
        ins = c220.VEC_REDUCE[kind]
        dst, src = op.operands[:2]
        self._vdt(op, src, ("f16", "f32"), "src")
        rep = self._vrep(op)
        dr = self._vattr(op, "dst_rep_stride", 1)
        sb, sr = self._vattr(op, "src_blk_stride", 1), self._vattr(op, "src_rep_stride", 8)
        tail = ", 0" if kind == "cadd" else ", Order_t::ONLY_VALUE" if kind in ("cmax", "cmin") else ""
        self._vemit(op, f"{ins}({self._vp(dst)}, {self._vp(src)}, {rep}, {dr}, {sb}, {sr}{tail});")

    def op_vec_cadd(self, op: Op) -> None:
        self._vreduce(op, "cadd")

    def op_vec_cmax(self, op: Op) -> None:
        self._vreduce(op, "cmax")

    def op_vec_cmin(self, op: Op) -> None:
        self._vreduce(op, "cmin")

    def op_vec_cgadd(self, op: Op) -> None:
        self._vreduce(op, "cgadd")

    def op_vec_cgmax(self, op: Op) -> None:
        self._vreduce(op, "cgmax")

    def op_vec_cgmin(self, op: Op) -> None:
        self._vreduce(op, "cgmin")

    def op_vec_cpadd(self, op: Op) -> None:
        self._vreduce(op, "cpadd")

    # -- fills, broadcasts, casts ------------------------------------------------------------------------------------

    def op_vec_dup(self, op: Op) -> None:
        self._varch(op)
        dst, v = op.operands[:2]
        dd = self._vdt(op, dst, c220.DUP_DTYPES, "dst")
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        self._vemit(op, f"vector_dup({self._vp(dst)}, ({cpp.ctype(dd)})({self.bare(v, dd)}), {rep}, (uint16_t){db}, 1, "
                        f"(uint8_t){dr}, 0);")  # type: ignore[attr-defined]

    def op_vec_brcb(self, op: Op) -> None:
        self._varch(op)
        dst, src = op.operands[:2]
        dd = self._dtype_of(dst)  # type: ignore[attr-defined]
        carrier = {2: "uint16_t", 4: "uint32_t"}.get(dd.bits // 8)
        if carrier is None:
            raise self._vgap(op, f"vbrcb rides 2- or 4-byte carriers, got {dd}")
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        # BrcbImpl forces the full mask before the intrinsic; the SPR is left FULL after
        self._vemit(op, "ResetMask();",
                    f"vbrcb((__ubuf__ {carrier}*)({self._vp(dst)}), (__ubuf__ {carrier}*)({self._vp(src)}), (uint16_t){db}, "
                    f"(uint16_t){dr}, {rep});")

    def op_vec_cast(self, op: Op) -> None:
        self._varch(op)
        dst, src = op.operands[:2]
        dd, sd = self._dtype_of(dst), self._dtype_of(src)  # type: ignore[attr-defined]
        entry = c220.CAST_INTRINSICS.get((dd.name, sd.name))
        if entry is None:
            raise self._vgap(op, f"cast {sd} -> {dd} has no vconv on c220 (arch/c220.CAST_INTRINSICS lists the pairs)")
        base, modes = entry
        mode = self.ident(op, "mode", "none") or "none"  # type: ignore[attr-defined]
        suffix = c220.CAST_SUFFIX.get(mode)
        if suffix is None or suffix not in modes:  # the table stores the suffixes that exist for the pair
            names = sorted(m for m, x in c220.CAST_SUFFIX.items() if x in modes)
            raise self._vgap(op, f"cast {sd} -> {dd} supports round modes {names}, got {mode!r}")
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        sb, sr = self._vattr(op, "src_blk_stride", 1), self._vattr(op, "src_rep_stride", 8)
        self._vemit(op, f"{base}{suffix}({self._vp(dst)}, {self._vp(src)}, {rep}, (uint16_t){db}, (uint16_t){sb}, "
                        f"(uint16_t){dr}, (uint16_t){sr});")

    # -- compares and selects ----------------------------------------------------------------------------------------

    def _vcmp_mode(self, op: Op) -> str:
        mode = self.ident(op, "mode") or ""  # type: ignore[attr-defined]
        if mode not in c220.VEC_CMP:
            raise self._vgap(op, f"{op.opcode}: c220 compare modes are {', '.join(c220.VEC_CMP)}, got {mode!r}")
        return mode

    def _vcmp_src(self, op: Op, v: Value, mode: str) -> None:
        dt = self._dtype_of(v)  # type: ignore[attr-defined]
        if dt.name == "i32" and mode == "eq":
            return
        self._vdt(op, v, ("f16", "f32"), "src")

    def op_vec_compare(self, op: Op) -> None:
        self._varch(op)
        mode = self._vcmp_mode(op)
        dst, s1, s2 = op.operands[:3]
        self._vcmp_src(op, s1, mode)
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        b1, r1 = self._vattr(op, "src1_blk_stride", 1), self._vattr(op, "src1_rep_stride", 8)
        b2, r2 = self._vattr(op, "src2_blk_stride", 1), self._vattr(op, "src2_rep_stride", 8)
        self._vemit(op, f"vcmpv_{mode}((__ubuf__ uint8_t*)({self._vp(dst)}), {self._vp(s1)}, {self._vp(s2)}, {rep}, "
                        f"(uint8_t){db}, (uint8_t){b1}, (uint8_t){b2}, (uint8_t){dr}, (uint8_t){r1}, (uint8_t){r2});")

    def op_vec_compare_scalar(self, op: Op) -> None:
        self._varch(op)
        mode = self._vcmp_mode(op)
        dst, s1, v = op.operands[:3]
        self._vcmp_src(op, s1, mode)
        sdt = self._dtype_of(s1)  # type: ignore[attr-defined]
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        b1, r1 = self._vattr(op, "src1_blk_stride", 1), self._vattr(op, "src1_rep_stride", 8)
        self._vemit(op, f"vcmpvs_{mode}((__ubuf__ uint8_t*)({self._vp(dst)}), {self._vp(s1)}, ({cpp.ctype(sdt)})({self.bare(v, sdt)}), {rep}, "
                        f"(uint16_t){db}, (uint16_t){b1}, (uint16_t){dr}, (uint16_t){r1});")  # type: ignore[attr-defined]

    def op_vec_select(self, op: Op) -> None:
        self._varch(op)
        dst, sel, s1, s2 = op.operands[:4]
        dd = self._vdt(op, dst, ("f16", "f32"), "dst")
        mode = self.ident(op, "mode", "") or ""  # type: ignore[attr-defined]
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        b1, r1 = self._vattr(op, "src1_blk_stride", 1), self._vattr(op, "src1_rep_stride", 8)
        b2, r2 = self._vattr(op, "src2_blk_stride", 1), self._vattr(op, "src2_rep_stride", 8)
        if mode == "tensor_scalar":
            # CMPMASK := the scalar operand block; vsel's third operand is the selmask (src1-slot strides,
            # provisional until the T1 diff against the old backend settles them, D-061)
            self._vemit(op, f"set_cmpmask({self._vp(s2)});",
                        "pipe_barrier(PIPE_V);",
                        f"vsel({self._vp(dst)}, {self._vp(s1)}, (__ubuf__ {cpp.ctype(dd)}*)({self._vp(sel)}), {rep}, "
                        f"(uint8_t){db}, (uint8_t){b1}, (uint8_t){b2}, (uint8_t){dr}, (uint8_t){r1}, (uint8_t){r2}, (uint8_t)1);")
            return
        if mode != "tensor_tensor":
            raise self._vgap(op, f"select mode must be TENSOR_TENSOR or TENSOR_SCALAR, got {mode!r}")
        tmp = op.attrs.get("tmp_addr_buf")
        if not isinstance(tmp, Value):
            raise self._vgap(op, "TENSOR_TENSOR select needs the tmp_addr_buf staging block")
        # CMPMASK := a block holding the selmask's UB address; the HW advances that address per repeat.
        # The counted dup clobbers VMASK to FULL / normal (the old backend's documented behaviour).
        self._vemit(op, "SetMaskCount();",
                    "SetVectorMask(8);",
                    f"vector_dup((__ubuf__ uint32_t*)({self.name(tmp)}.ptr()), (uint32_t)(uint64_t)({self._vp(sel)}), "
                    "(uint8_t)1, (uint16_t)1, 1, (uint8_t)8, 0);",  # type: ignore[attr-defined]
                    "SetMaskNorm();",
                    "ResetMask();",
                    "pipe_barrier(PIPE_V);",
                    f"set_cmpmask({self.name(tmp)}.ptr());",  # type: ignore[attr-defined]
                    "pipe_barrier(PIPE_V);",
                    f"vsel({self._vp(dst)}, {self._vp(s1)}, {self._vp(s2)}, {rep}, (uint8_t){db}, (uint8_t){b1}, (uint8_t){b2}, "
                    f"(uint8_t){dr}, (uint8_t){r1}, (uint8_t){r2}, (uint8_t)2);")

    # -- gathers, scatters, transposes -------------------------------------------------------------------------------

    def op_vec_gather(self, op: Op) -> None:
        self._varch(op)
        dst, src, off = op.operands[:3]
        dd = self._dtype_of(dst)  # type: ignore[attr-defined]
        carrier = {2: "uint16_t", 4: "uint32_t"}.get(dd.bits // 8)
        if carrier is None:
            raise self._vgap(op, f"vgather rides 2- or 4-byte carriers, got {dd}")
        if op.attrs.get("count") is not None:
            raise self._vgap(op, "counted gather is rejected on c220 (it gathers the wrong lanes on silicon)")
        rep = self._vrep(op)
        dr = self._vattr(op, "dst_rep_stride", 8)
        start = self._vattr(op, "start_idx", 0)
        # GatherImpl always programs the mask: easyasc passed VECTORFULLMASK, so the SPR is left FULL
        # start_idx is in BYTES (the old framework's printed contract: "gather/scatter offset and
        # start_idx are both in bytes"); the intrinsic's base is a byte address, so it adds unscaled.
        self._vemit(op, "ResetMask();",
                    f"vgather((__ubuf__ {carrier}*)({self._vp(dst)}), {self._vp(off)}, "
                    f"(uint32_t)((uint64_t)({self._vp(src)}) + (uint64_t)({start})), (uint16_t){dr}, {rep});")

    def op_vec_gather_block(self, op: Op) -> None:
        self._varch(op)
        dst, src, off = op.operands[:3]
        dd = self._dtype_of(dst)  # type: ignore[attr-defined]
        if dd.bits not in (16, 32):
            raise self._vgap(op, f"vgatherb rides 2- or 4-byte carriers, got {dd}")
        rep = self._vrep(op)
        db, dr = self._vattr(op, "dst_blk_stride", 1), self._vattr(op, "dst_rep_stride", 8)
        # C220 vgatherb copies eight 32-byte blocks through its u16 pointer
        # overload regardless of the logical element type (M10-098).
        self._vemit(op, "ResetMask();",
                    f"vgatherb((__ubuf__ uint16_t*)({self._vp(dst)}), {self._vp(off)}, (uint32_t)(uint64_t)({self._vp(src)}), "
                    f"(uint16_t){dr}, (uint8_t){db}, {rep});")

    def op_vec_scatter(self, op: Op) -> None:
        self._varch(op)
        raise self._vgap(op, "vec.scatter has no c220 intrinsic (both ScatterImpl overloads are "
                             "ASCENDC_REPORT_NOT_SUPPORT; vscatter arrives on later archs)")

    def op_vec_transdata5hd(self, op: Op) -> None:
        self._varch(op)
        dst, src = op.operands[:2]
        self._vdt(op, dst, ("f16", "bf16", "i16", "u16"), "dst")
        rep = self._vrep(op)
        sr = self._vattr(op, "src_rep_stride", 1)
        dr = self._vattr(op, "dst_rep_stride", 16)
        srow = self._vattr(op, "src_row_stride", 16)
        drow = self._vattr(op, "dst_row_stride", 16)
        self._vemit(op, f"transdata5hd({self.name(dst)}, {self.name(src)}, {rep}, {srow}, {drow}, {sr}, {dr});")  # type: ignore[attr-defined]
