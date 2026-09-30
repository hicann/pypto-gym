# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The a2 family's tensor-vector instructions in the reference interpreter (RFC-0008 §4, D-061).

The model is the old simulator's ``VPipe`` (``easyasc/simulator/pipe_vec.py``), ported op by op so the goldens
recorded from it replay bit for bit:

* every vector lane owns a 256-entry mask (``Lane.spr_mask``, all ones at launch) and a mode: **repeat** — an
  instruction walks ``repeat`` repeats of 8 blocks of 32 bytes, the block / repeat strides in blocks, the mask's
  active prefix (``256 / sizeof(dtype)`` lanes, the same for every repeat) gating the write-back — or **count**
  (``set_mask_count`` + ``set_mask_counter``): the first ``count`` elements, contiguous, no strides, no mask;
* element-wise ops keep the old value of a masked-off lane; ``cadd`` / ``cgadd`` / ``cpadd`` treat it as 0, ``cmax`` /
  ``cgmax`` as -inf, ``cmin`` / ``cgmin`` as +inf, and a reduction whose lanes are all off leaves its result alone
  (``cpadd`` always writes); ``brcb``, ``compare*``, ``select``, ``gather*``, ``scatter`` and ``transdata5hd`` ignore
  the mask (``compare`` / ``select`` use a packed bit tensor instead);
* half and bf16 compute in fp32 and round once on the store; ``div`` is the fp32 division (the board's ``Div<float>``
  is 1 ulp off in a deterministic pattern — RFC-0008 §1); ``compare`` treats NaN as unordered-false, ``NE`` included;
* an op whose addressed footprint runs past the end of its allocation is an error (the sim-hidden overrun into
  the neighbouring UB tensor); the footprint of a strided op counts only the lanes the mask enables. An operand
  whose start is not a 32-byte block boundary is the same rule's other half (D-017, M10-065) and is rejected
  whatever the mask does, since an all-off predicate suppresses an instruction's data effects and not its base.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from ...ir.core import Op
from .cast_rounding import apply_cast_dtype

# -- numerics ------------------------------------------------------------------------------------------------------


def _compute_dtype(dt: torch.dtype) -> torch.dtype:
    return torch.float32 if dt in (torch.float16, torch.bfloat16) else dt


def _signed_view(dt: torch.dtype) -> torch.dtype | None:
    for u, s in (("uint16", torch.int16), ("uint32", torch.int32), ("uint64", torch.int64)):
        if hasattr(torch, u) and dt == getattr(torch, u):
            return s
    return None


def _masked_blend(mask: torch.Tensor, on_true: torch.Tensor, on_false: torch.Tensor) -> torch.Tensor:
    sv = _signed_view(on_true.dtype)  # torch.where lacks some unsigned CPU kernels: blend through the signed view
    if sv is not None:
        return torch.where(mask, on_true.view(sv), on_false.view(sv)).view(on_true.dtype)
    return torch.where(mask, on_true, on_false)


def _int_view_dtype(elem: int) -> torch.dtype:
    return {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[elem]


def _bitwise_not(src: torch.Tensor) -> torch.Tensor:
    if src.dtype == torch.uint8:
        return torch.bitwise_not(src.to(torch.int8)).to(torch.uint8)
    sv = _signed_view(src.dtype)
    if sv is not None:
        return torch.bitwise_not(src.to(sv)).to(src.dtype)
    if src.dtype.is_floating_point:
        idt = _int_view_dtype(int(src.element_size()))
        return torch.bitwise_not(src.view(idt)).view(src.dtype)
    return torch.bitwise_not(src)


def _bitwise_binary(op: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    fn = torch.bitwise_and if op == "and" else torch.bitwise_or
    sv = _signed_view(a.dtype)
    if sv is not None:
        return fn(a.view(sv), b.view(sv)).view(a.dtype)
    if a.dtype.is_floating_point or b.dtype.is_floating_point:
        idt = _int_view_dtype(int(a.element_size()))
        return fn(a.view(idt), b.view(idt)).view(a.dtype)
    return fn(a, b)


def _shift_bits(op: str, src: torch.Tensor, shift: int, round_en: bool) -> torch.Tensor:
    """A2 ShiftLeft / ShiftRight on 16- and 32-bit integers: the full fixed-width bit pattern moves (a signed left
    shift may flip the sign), a right shift is arithmetic for signed and logical for unsigned tensors."""
    bits = int(src.element_size()) * 8
    if shift < 0 or shift > bits:
        raise ValueError(f"{op}: shift must be in [0, {bits}], got {shift}")
    bit_mask, sign_mask = (1 << bits) - 1, 1 << (bits - 1)
    raw = torch.bitwise_and(src.to(torch.int64), bit_mask)
    unsigned = _signed_view(src.dtype) is not None
    if op == "shiftls":
        result = torch.bitwise_and(torch.bitwise_left_shift(raw, shift), bit_mask)
        if not unsigned:
            result = torch.where(result >= sign_mask, result - (1 << bits), result)
        return result.to(src.dtype)
    if unsigned:
        return torch.bitwise_right_shift(raw, shift).to(src.dtype)
    result = torch.bitwise_right_shift(src.to(torch.int64), shift)
    if round_en and shift > 0:
        result = result + torch.bitwise_and(torch.bitwise_right_shift(src.to(torch.int64), shift - 1), 1)
    return result.to(src.dtype)


def _pack_bits(bools: torch.Tensor, nbytes: int) -> torch.Tensor:
    bits = bools.to(torch.uint8).reshape(-1)
    padded = torch.zeros(nbytes * 8, dtype=torch.uint8)
    padded[: bits.numel()] = bits
    weights = (1 << torch.arange(8, dtype=torch.int32)).to(torch.uint8)
    return (padded.reshape(nbytes, 8) * weights).sum(dim=1).to(torch.uint8)


def _unpack_bits(bytes_: torch.Tensor, nbits: int) -> torch.Tensor:
    b = bytes_.reshape(-1).to(torch.int32)
    shifts = torch.arange(8, dtype=torch.int32)
    bits = ((b.unsqueeze(1) >> shifts) & 1).reshape(-1)
    return bits[:nbits].bool()


def _compare_ne(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    result = a != b
    if a.dtype.is_floating_point or b.dtype.is_floating_point:  # the a2 vector unit compares NaN unordered-false
        nan = torch.isnan(a) | torch.isnan(b)
        if bool(nan.any().item()):
            result = result & ~nan
    return result


_BINARY: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
    "max": torch.maximum, "min": torch.minimum,
    "and": lambda a, b: _bitwise_binary("and", a, b), "or": lambda a, b: _bitwise_binary("or", a, b),
}
_UNARY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "abs": torch.abs, "relu": lambda x: torch.clamp(x, min=0), "exp": torch.exp, "ln": torch.log, "rec": lambda x: 1.0 / x,
    "sqrt": torch.sqrt, "rsqrt": torch.rsqrt,
}
_SCALAR: dict[str, Callable[[torch.Tensor, float], torch.Tensor]] = {
    "adds": lambda x, s: x + s, "muls": lambda x, s: x * s, "maxs": lambda x, s: torch.clamp(x, min=s),
    "mins": lambda x, s: torch.clamp(x, max=s), "lrelu": lambda x, s: torch.where(x > 0, x, x * s),
}
_CMP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "eq": lambda a, b: a == b, "ne": _compare_ne, "lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}
MAX_REPEAT = 255


def _strided_touch(mask: torch.Tensor, repeat: int, rep: int, blk: int, epb: int) -> int:
    """Highest element index + 1 a masked strided op addresses (0 when nothing is written)."""
    if repeat <= 0:
        return 0
    active = torch.nonzero(mask[: 8 * epb].view(8, epb).any(dim=1), as_tuple=False)
    if not active.numel():
        return 0
    last_blk = int(active.max().item())
    lanes = int(torch.nonzero(mask[last_blk * epb:(last_blk + 1) * epb], as_tuple=False).max().item()) + 1
    return ((repeat - 1) * rep + last_blk * blk) * epb + lanes


def _block_touch(repeat: int, rep: int, blk: int, epb: int) -> int:
    return ((repeat - 1) * rep + 7 * blk + 1) * epb if repeat > 0 else 0


# -- the mixin --------------------------------------------------------------------------------------------------------


class VecOps:
    """``vec.*`` handlers of :class:`Interp` (a2 family)."""

    # -- lane state ---------------------------------------------------------------------------------------------------

    def _vmask(self, op: Op | None = None) -> torch.Tensor:
        if op is not None and op.attrs.get("count_per_rep") is not None:  # the op's own prefix mask (D-062)
            n = int(self.attr(op, "count_per_rep"))  # type: ignore[attr-defined]
            m = torch.zeros(256, dtype=torch.bool)
            m[:n] = True
            return m
        m = self.lane.spr_mask  # type: ignore[attr-defined]
        if m is None:
            m = torch.ones(256, dtype=torch.bool)
            self.lane.spr_mask = m  # type: ignore[attr-defined]
        return m

    def op_vec_set_mask(self, op: Op) -> None:
        high, low = int(self.val(op.attrs["high"])), int(self.val(op.attrs["low"]))  # type: ignore[attr-defined]
        m = self._vmask().clone()
        for i in range(64):
            m[i] = bool((low >> i) & 1)
            m[64 + i] = bool((high >> i) & 1)
        self.lane.spr_mask = m  # type: ignore[attr-defined]

    def op_vec_set_mask_by_count(self, op: Op) -> None:
        count = int(self.val(op.attrs["count"]))  # type: ignore[attr-defined]
        if not 0 <= count <= 256:
            raise self._err(f"set_mask_by_count count must be in [0, 256], got {count}", op)  # type: ignore[attr-defined]
        m = torch.zeros(256, dtype=torch.bool)
        m[:count] = True
        self.lane.spr_mask = m  # type: ignore[attr-defined]

    def op_vec_reset_mask(self, op: Op) -> None:
        self.lane.spr_mask = torch.ones(256, dtype=torch.bool)  # type: ignore[attr-defined]

    def op_vec_set_mask_count(self, op: Op) -> None:
        """A materialised mode switch (the c220 backend's, D-062): the IR ops carry their own count — nothing to do."""

    def op_vec_set_mask_normal(self, op: Op) -> None:
        """See op_vec_set_mask_count."""

    def op_vec_set_mask_counter(self, op: Op) -> None:
        """See op_vec_set_mask_count."""

    def _vcounting(self, op: Op) -> int | None:
        """The op's own counter mode (D-062: ``count`` is an attribute, not machine state), None in repeat mode."""
        count = self.attr(op, "count", None)  # type: ignore[attr-defined]
        if count is None:
            return None
        count = int(self.val(count) if hasattr(count, "name") else count)  # type: ignore[attr-defined]
        if count <= 0:
            raise self._err(f"{op.opcode}: count mode needs count > 0, got {count}", op)  # type: ignore[attr-defined]
        return count

    def _vmode_end(self, op: Op) -> None:
        """After a counted or count_per_rep op the mask returns to all ones — the bracket CANN's own dav_c220
        Level-2 calls emit (``set_mask_norm``; ``set_vector_mask(-1, -1)``), which the c220 backend materialises."""
        if op.attrs.get("count") is not None or op.attrs.get("count_per_rep") is not None:
            self.lane.spr_mask = torch.ones(256, dtype=torch.bool)  # type: ignore[attr-defined]

    # -- operands -----------------------------------------------------------------------------------------------------

    def _vlin(self, x: Any, what: str, op: Op) -> tuple[torch.Tensor, str]:
        """The flat storage from the window's origin to the end of its allocation (the old linear view), named.

        Every operand passes through here, so this is where the base half of the UB instruction rule
        (D-017, M10-065) is enforced: the footprint half is `_vguard`'s, and an operand whose own start
        is not a 32-byte boundary is rejected before the access, whatever the mask does to the lanes."""
        ref = self.raw(x)  # type: ignore[attr-defined]
        flat, origin = self.m.storage(ref, self.lane)  # type: ignore[attr-defined]
        start = int(origin) * int(flat.element_size())
        if ref.space == "ub" and start % 32:
            raise self._err(f"{op.opcode}: {what} %{ref.base} starts at byte {start} of its allocation: not 32-byte "  # type: ignore[attr-defined]
                            "aligned (an all-off mask does not exempt an executed instruction's base -- rebase the "
                            "window on a block or pad the tensor to one)", op)
        return flat[origin:], f"{what} {ref.base}"

    def _vguard(self, op: Op, name: str, lin: torch.Tensor, need: int) -> None:
        if need > lin.numel():
            raise self._err(f"{op.opcode}: {name} overruns its UB allocation: addresses {need} elements but only {lin.numel()} remain "  # type: ignore[attr-defined]
                            "to the allocation's end (widen the tensor or narrow the vector mask)", op)

    def _vepb(self, t: torch.Tensor) -> int:
        return 32 // int(t.element_size())

    def _vint(self, op: Op, name: str, default: int) -> int:
        return int(self.attr(op, name, default))  # type: ignore[attr-defined]

    def _vrepeat(self, op: Op) -> int:
        repeat = self._vint(op, "repeat", 0)
        if repeat < 0 or repeat > MAX_REPEAT:
            raise self._err(f"{op.opcode}: repeat must be in [0, {MAX_REPEAT}], got {repeat}", op)  # type: ignore[attr-defined]
        return repeat

    def _vmasked_write(self, dst: torch.Tensor, offset: int, result: torch.Tensor, blk: int, epb: int, m: torch.Tensor) -> None:
        n = int(result.numel())
        block = dst[offset:offset + n]
        if block.numel() < n:  # a tail past the view: the guard proved every lane out there is masked off
            n = block.numel()
            result = result[:n]
        if n == 0:
            return
        mask = m[blk * epb:(blk + 1) * epb][:n]
        if bool(mask.all().item()):
            block.copy_(result)
        else:
            block.copy_(_masked_blend(mask, result, block.clone()))

    def _vwrite(self, dst: torch.Tensor, offset: int, result: torch.Tensor) -> None:
        dst[offset:offset + result.numel()].copy_(result)

    # -- element-wise families ---------------------------------------------------------------------------------------

    def _vbracketed(self, op: Op, body) -> None:
        """Run one instruction and apply the end-of-op mask bracket of a counted / count_per_rep op (D-062)."""
        try:
            body()
        finally:
            self._vmode_end(op)

    def _vbinary(self, op: Op, kind: str) -> None:
        self._vbracketed(op, lambda: self._vbinary_body(op, kind))

    def _vbinary_body(self, op: Op, kind: str) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        s1, n1 = self._vlin(op.operands[1], "src1", op)
        s2, n2 = self._vlin(op.operands[2], "src2", op)
        fn, cdt = _BINARY[kind], _compute_dtype(dst.dtype)
        bitwise = kind in ("and", "or")
        count = self._vcounting(op)
        if count is not None:
            for n, t in ((dn, dst), (n1, s1), (n2, s2)):
                self._vguard(op, n, t, count)
            r = fn(s1[:count], s2[:count]) if bitwise else fn(s1[:count].to(cdt), s2[:count].to(cdt)).to(dst.dtype)
            self._vwrite(dst, 0, r)
            return
        repeat, epb, m = self._vrepeat(op), self._vepb(dst), self._vmask(op)
        dblk, drep = self._vint(op, "dst_blk_stride", 1), self._vint(op, "dst_rep_stride", 8)
        b1, r1 = self._vint(op, "src1_blk_stride", 1), self._vint(op, "src1_rep_stride", 8)
        b2, r2 = self._vint(op, "src2_blk_stride", 1), self._vint(op, "src2_rep_stride", 8)
        self._vguard(op, dn, dst, _strided_touch(m, repeat, drep, dblk, epb))
        self._vguard(op, n1, s1, _strided_touch(m, repeat, r1, b1, self._vepb(s1)))
        self._vguard(op, n2, s2, _strided_touch(m, repeat, r2, b2, self._vepb(s2)))
        for r in range(repeat):
            for b in range(8):
                di, i1, i2 = (r * drep + b * dblk) * epb, (r * r1 + b * b1) * epb, (r * r2 + b * b2) * epb
                if bitwise:
                    res = fn(self._vsrc_block(s1, i1, epb), self._vsrc_block(s2, i2, epb))
                else:
                    res = fn(self._vsrc_block(s1, i1, epb, cdt), self._vsrc_block(s2, i2, epb, cdt)).to(dst.dtype)
                self._vmasked_write(dst, di, res, b, epb, m)

    def _vunary(self, op: Op, kind: str) -> None:
        self._vbracketed(op, lambda: self._vunary_body(op, kind))

    def _vunary_body(self, op: Op, kind: str) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        cdt = _compute_dtype(dst.dtype)

        def f(x: torch.Tensor) -> torch.Tensor:
            return _bitwise_not(x).to(dst.dtype) if kind == "not" else _UNARY[kind](x.to(cdt)).to(dst.dtype)

        count = self._vcounting(op)
        if count is not None:
            self._vguard(op, sn, src, count)
            self._vguard(op, dn, dst, count)
            self._vwrite(dst, 0, f(src[:count]))
            return
        repeat, epb, m = self._vrepeat(op), self._vepb(dst), self._vmask(op)
        dblk, drep = self._vint(op, "dst_blk_stride", 1), self._vint(op, "dst_rep_stride", 8)
        sblk, srep = self._vint(op, "src_blk_stride", 1), self._vint(op, "src_rep_stride", 8)
        self._vguard(op, dn, dst, _strided_touch(m, repeat, drep, dblk, epb))
        self._vguard(op, sn, src, _strided_touch(m, repeat, srep, sblk, self._vepb(src)))
        for r in range(repeat):
            for b in range(8):
                di, si = (r * drep + b * dblk) * epb, (r * srep + b * sblk) * epb
                self._vmasked_write(dst, di, f(self._vsrc_block(src, si, epb)), b, epb, m)

    def _vunary_scalar(self, op: Op, kind: str) -> None:
        self._vbracketed(op, lambda: self._vunary_scalar_body(op, kind))

    def _vunary_scalar_body(self, op: Op, kind: str) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        raw = self.val(op.operands[2])  # type: ignore[attr-defined]
        shift = kind in ("shiftls", "shiftrs")
        val: Any = int(raw) if shift else float(raw)
        round_en = bool(self.attr(op, "round_en", False))  # type: ignore[attr-defined]
        cdt = _compute_dtype(dst.dtype)

        def f(x: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
            if kind == "axpy":
                return (old.to(cdt) + x.to(cdt) * val).to(dst.dtype)
            if shift:
                return _shift_bits(kind, x, val, round_en)
            return _SCALAR[kind](x.to(cdt), val).to(dst.dtype)

        count = self._vcounting(op)
        if count is not None:
            self._vguard(op, sn, src, count)
            self._vguard(op, dn, dst, count)
            self._vwrite(dst, 0, f(src[:count], dst[:count].clone()))
            return
        repeat, epb, m = self._vrepeat(op), self._vepb(dst), self._vmask(op)
        dblk, drep = self._vint(op, "dst_blk_stride", 1), self._vint(op, "dst_rep_stride", 8)
        sblk, srep = self._vint(op, "src_blk_stride", 1), self._vint(op, "src_rep_stride", 8)
        self._vguard(op, dn, dst, _strided_touch(m, repeat, drep, dblk, epb))
        self._vguard(op, sn, src, _strided_touch(m, repeat, srep, sblk, self._vepb(src)))
        for r in range(repeat):
            for b in range(8):
                di, si = (r * drep + b * dblk) * epb, (r * srep + b * sblk) * epb
                self._vmasked_write(dst, di, f(self._vsrc_block(src, si, epb), self._vsrc_block(dst, di, epb)), b, epb, m)

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

    def op_vec_adds(self, op: Op) -> None:
        self._vunary_scalar(op, "adds")

    def op_vec_muls(self, op: Op) -> None:
        self._vunary_scalar(op, "muls")

    def op_vec_maxs(self, op: Op) -> None:
        self._vunary_scalar(op, "maxs")

    def op_vec_mins(self, op: Op) -> None:
        self._vunary_scalar(op, "mins")

    def op_vec_lrelu(self, op: Op) -> None:
        self._vunary_scalar(op, "lrelu")

    def op_vec_axpy(self, op: Op) -> None:
        self._vunary_scalar(op, "axpy")

    def op_vec_shiftls(self, op: Op) -> None:
        self._vunary_scalar(op, "shiftls")

    def op_vec_shiftrs(self, op: Op) -> None:
        self._vunary_scalar(op, "shiftrs")

    def op_vec_muladddst(self, op: Op) -> None:
        self._vbracketed(op, lambda: self.op_vec_muladddst_body(op))

    def op_vec_muladddst_body(self, op: Op) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        s1, n1 = self._vlin(op.operands[1], "src1", op)
        s2, n2 = self._vlin(op.operands[2], "src2", op)
        cdt = _compute_dtype(dst.dtype)
        count = self._vcounting(op)
        if count is not None:
            for n, t in ((dn, dst), (n1, s1), (n2, s2)):
                self._vguard(op, n, t, count)
            snap = dst[:count].clone()
            self._vwrite(dst, 0, (s1[:count].to(cdt) * s2[:count].to(cdt) + snap.to(cdt)).to(dst.dtype))
            return
        repeat, epb, m = self._vrepeat(op), self._vepb(dst), self._vmask(op)
        dblk, drep = self._vint(op, "dst_blk_stride", 1), self._vint(op, "dst_rep_stride", 8)
        b1, r1 = self._vint(op, "src1_blk_stride", 1), self._vint(op, "src1_rep_stride", 8)
        b2, r2 = self._vint(op, "src2_blk_stride", 1), self._vint(op, "src2_rep_stride", 8)
        epb1, epb2 = self._vepb(s1), self._vepb(s2)
        mixed = epb1 != epb or epb2 != epb
        if mixed:
            m = m.clone()
            m[8 * epb:] = False  # one logical lane per destination element, including half sources
        self._vguard(op, dn, dst, _strided_touch(m, repeat, drep, dblk, epb))
        self._vguard(op, n1, s1, _strided_touch(m, repeat, r1, b1, epb1))
        self._vguard(op, n2, s2, _strided_touch(m, repeat, r2, b2, epb2))
        if mixed and dblk == b1 == b2 == 1 and bool(m[:8 * epb].all()):
            per_rep, nd = 8 * epb, int(dst.numel())
            for r in range(repeat):
                d0 = r * drep * epb
                if d0 >= nd:
                    break
                seg = min(per_rep, nd - d0)
                s10, s20 = r * r1 * epb1, r * r2 * epb2
                snap = dst[d0:d0 + seg].clone()
                self._vwrite(dst, d0, (s1[s10:s10 + seg].to(cdt) * s2[s20:s20 + seg].to(cdt) + snap.to(cdt)).to(dst.dtype))
            return
        for r in range(repeat):
            for b in range(8):
                di = (r * drep + b * dblk) * epb
                lane = b * epb
                i1 = (r * r1 + (lane // epb1) * b1) * epb1 + lane % epb1
                i2 = (r * r2 + (lane // epb2) * b2) * epb2 + lane % epb2
                snap = self._vsrc_block(dst, di, epb, cdt)
                res = (self._vsrc_block(s1, i1, epb, cdt) * self._vsrc_block(s2, i2, epb, cdt) + snap).to(dst.dtype)
                self._vmasked_write(dst, di, res, b, epb, m)

    # -- fills, casts, broadcasts ------------------------------------------------------------------------------------

    def op_vec_dup(self, op: Op) -> None:
        self._vbracketed(op, lambda: self.op_vec_dup_body(op))

    def op_vec_dup_body(self, op: Op) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        scalar = float(self.val(op.operands[1]))  # type: ignore[attr-defined]
        count = self._vcounting(op)
        if count is not None:
            self._vguard(op, dn, dst, count)
            dst[:count].fill_(scalar)
            return
        repeat, epb, m = self._vrepeat(op), self._vepb(dst), self._vmask(op)
        dblk, drep = self._vint(op, "dst_blk_stride", 1), self._vint(op, "dst_rep_stride", 8)
        self._vguard(op, dn, dst, _strided_touch(m, repeat, drep, dblk, epb))
        for r in range(repeat):
            for b in range(8):
                self._vmasked_write(dst, (r * drep + b * dblk) * epb, torch.full((epb,), scalar, dtype=dst.dtype), b, epb, m)

    def _vcast_chunk(self, src: torch.Tensor, dst_dtype: torch.dtype, mode: str) -> torch.Tensor:
        if src.dtype == dst_dtype:
            return apply_cast_dtype(src, dst_dtype, mode) if src.dtype.is_floating_point else src
        if src.dtype.is_floating_point and not dst_dtype.is_floating_point:
            return apply_cast_dtype(src, dst_dtype, mode, saturate=True)
        if src.dtype.is_floating_point or dst_dtype.is_floating_point:
            return apply_cast_dtype(src, dst_dtype, mode)
        return src.to(dst_dtype)

    def op_vec_cast(self, op: Op) -> None:
        self._vbracketed(op, lambda: self.op_vec_cast_body(op))

    def op_vec_cast_body(self, op: Op) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        mode = str(self.attr(op, "mode", "none") or "none")  # type: ignore[attr-defined]
        mode = getattr(mode, "name", mode)
        count = self._vcounting(op)
        if count is not None:
            self._vguard(op, sn, src, count)
            self._vguard(op, dn, dst, count)
            self._vwrite(dst, 0, self._vcast_chunk(src[:count].clone(), dst.dtype, mode))
            return
        repeat = self._vrepeat(op)
        srep, drep = self._vint(op, "src_rep_stride", 8), self._vint(op, "dst_rep_stride", 8)
        dc0, sc0 = self._vepb(dst), self._vepb(src)
        # a repeat converts 256 / max(sizeof) logical elements; the repeat strides are addresses between repeats
        per_rep = 256 // max(int(dst.element_size()), int(src.element_size()))
        if repeat:
            self._vguard(op, sn, src, (repeat - 1) * srep * sc0 + per_rep)
            self._vguard(op, dn, dst, (repeat - 1) * drep * dc0 + per_rep)
        mask = self._vmask(op)[:per_rep]
        mask_all = bool(mask.all().item())
        for r in range(repeat):
            s0, d0 = r * srep * sc0, r * drep * dc0
            res = self._vcast_chunk(src[s0:s0 + per_rep].clone(), dst.dtype, mode)
            chunk = dst[d0:d0 + per_rep]
            chunk.copy_(res if mask_all else _masked_blend(mask, res, chunk.clone()))

    def op_vec_brcb(self, op: Op) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        repeat, epb = self._vrepeat(op), self._vepb(dst)
        dblk, drep = self._vint(op, "dst_blk_stride", 1), self._vint(op, "dst_rep_stride", 8)
        self._vguard(op, sn, src, repeat * 8)
        self._vguard(op, dn, dst, _block_touch(repeat, drep, dblk, epb))
        for r in range(repeat):
            for b in range(8):
                di = (r * drep + b * dblk) * epb
                dst[di:di + epb].fill_(src[r * 8 + b].item())

    # -- reductions --------------------------------------------------------------------------------------------------

    def _vsrc_block(self, src: torch.Tensor, si: int, epb: int, cdt: torch.dtype | None = None) -> torch.Tensor:
        """One source block as an ``epb``-length tensor (``cdt`` converts, ``None`` keeps the dtype). A tail
        past the view is zero-filled: the footprint guard has already proven every mask-on lane there is off,
        so the fill never survives the masked write / identity blend that follows (the hardware does not read
        masked-off blocks at all)."""
        dt = src.dtype if cdt is None else cdt
        block = src[si:si + epb]
        if block.numel() == epb:
            return block.clone().to(dt)
        out = torch.zeros(epb, dtype=dt)
        if block.numel():
            out[: block.numel()] = block.to(dt)
        return out

    def _vreduce(self, op: Op, kind: str) -> None:
        self._vbracketed(op, lambda: self._vreduce_body(op, kind))

    def _vreduce_body(self, op: Op, kind: str) -> None:
        if self._vcounting(op) is not None:
            raise self._err(f"{op.opcode}: a group reduction has no counter-mode form", op)  # type: ignore[attr-defined]
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        whole = kind in ("cmax", "cmin", "cadd")
        repeat, epb, m = self._vrepeat(op), self._vepb(src), self._vmask(op)
        sblk, srep = self._vint(op, "src_blk_stride", 1), self._vint(op, "src_rep_stride", 8)
        drep = self._vint(op, "dst_rep_stride", 1)
        cdt = _compute_dtype(src.dtype)
        if repeat > 0:
            self._vguard(op, sn, src, _strided_touch(m, repeat, srep, sblk, epb))
            if whole:
                self._vguard(op, dn, dst, (repeat - 1) * drep + 1)
            else:
                active = torch.nonzero(m[: 8 * epb].view(8, epb).any(dim=1), as_tuple=False)
                if active.numel():
                    self._vguard(op, dn, dst, (repeat - 1) * drep * 8 + int(active.max().item()) + 1)
        for r in range(repeat):
            base = r * srep * epb
            blocks, masks = [], []
            for b in range(8):
                si = base + b * sblk * epb
                block = self._vsrc_block(src, si, epb, cdt)
                mb = m[b * epb:(b + 1) * epb]
                if kind in ("cadd", "cgadd"):
                    block = torch.where(mb, block, torch.zeros_like(block))
                elif kind in ("cmax", "cgmax"):
                    block = torch.where(mb, block, torch.full_like(block, float("-inf")))
                else:
                    block = torch.where(mb, block, torch.full_like(block, float("inf")))
                blocks.append(block)
                masks.append(mb)
            if whole:
                if bool(torch.cat(masks).any().item()):
                    combined = torch.cat(blocks)
                    red = torch.max(combined) if kind == "cmax" else torch.min(combined) if kind == "cmin" else torch.sum(combined)
                    dst[r * drep] = red.to(dst.dtype)
            else:
                dbase = r * drep * 8
                for b, block in enumerate(blocks):
                    if bool(masks[b].any().item()):
                        red = torch.max(block) if kind == "cgmax" else torch.min(block) if kind == "cgmin" else torch.sum(block)
                        dst[dbase + b] = red.to(dst.dtype)

    def op_vec_cmax(self, op: Op) -> None:
        self._vreduce(op, "cmax")

    def op_vec_cmin(self, op: Op) -> None:
        self._vreduce(op, "cmin")

    def op_vec_cadd(self, op: Op) -> None:
        self._vreduce(op, "cadd")

    def op_vec_cgmax(self, op: Op) -> None:
        self._vreduce(op, "cgmax")

    def op_vec_cgmin(self, op: Op) -> None:
        self._vreduce(op, "cgmin")

    def op_vec_cgadd(self, op: Op) -> None:
        self._vreduce(op, "cgadd")

    def op_vec_cpadd(self, op: Op) -> None:
        self._vbracketed(op, lambda: self.op_vec_cpadd_body(op))

    def op_vec_cpadd_body(self, op: Op) -> None:
        if self._vcounting(op) is not None:
            raise self._err("cpadd has no counter-mode form", op)  # type: ignore[attr-defined]
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        repeat, epb, m = self._vrepeat(op), self._vepb(src), self._vmask(op)
        sblk, srep = self._vint(op, "src_blk_stride", 1), self._vint(op, "src_rep_stride", 8)
        drep = self._vint(op, "dst_rep_stride", 1)
        per_rep = epb * 8
        out = per_rep // 2
        cdt = _compute_dtype(src.dtype)
        if repeat > 0:
            self._vguard(op, sn, src, _strided_touch(m, repeat, srep, sblk, epb))
            self._vguard(op, dn, dst, (repeat - 1) * drep * out + out)
        for r in range(repeat):
            base = r * srep * epb
            gathered = []
            for b in range(8):
                si = base + b * sblk * epb
                block = self._vsrc_block(src, si, epb, cdt)
                gathered.append(torch.where(m[b * epb:(b + 1) * epb], block, torch.zeros_like(block)))
            pair = torch.cat(gathered).reshape(-1, 2).sum(dim=1)
            self._vwrite(dst, r * drep * out, pair.to(dst.dtype))

    # -- compares and selects ----------------------------------------------------------------------------------------

    def _vmode(self, op: Op) -> str:
        mode = self.attr(op, "mode", "")  # type: ignore[attr-defined]
        return str(getattr(mode, "name", mode)).lower()

    def _vcompare(self, op: Op, scalar: bool) -> None:
        if self._vcounting(op) is not None:
            raise self._err(f"{op.opcode} has no counter-mode form", op)  # type: ignore[attr-defined]
        dst, dn = self._vlin(op.operands[0], "dst", op)
        s1, n1 = self._vlin(op.operands[1], "src1", op)
        if dst.dtype not in (torch.uint8, torch.int8):
            raise self._err(f"compare dst must be uint8 / int8, got {dst.dtype}", op)  # type: ignore[attr-defined]
        repeat = self._vrepeat(op)
        if repeat == 0:
            return
        cmp = _CMP.get(self._vmode(op))
        if cmp is None:
            raise self._err(f"unsupported vec compare mode {self._vmode(op)!r}", op)  # type: ignore[attr-defined]
        b1, r1 = self._vint(op, "src1_blk_stride", 1), self._vint(op, "src1_rep_stride", 8)
        epb = self._vepb(s1)
        per_rep = epb * 8
        nbytes = per_rep // 8
        cdt = _compute_dtype(s1.dtype)
        self._vguard(op, dn, dst, nbytes * repeat)
        self._vguard(op, n1, s1, _block_touch(repeat, r1, b1, epb))
        if scalar:
            value = float(self.val(op.operands[2]))  # type: ignore[attr-defined]
        else:
            s2, n2 = self._vlin(op.operands[2], "src2", op)
            b2, r2 = self._vint(op, "src2_blk_stride", 1), self._vint(op, "src2_rep_stride", 8)
            self._vguard(op, n2, s2, _block_touch(repeat, r2, b2, epb))
        for r in range(repeat):
            lhs = torch.cat([s1[(r * r1 + b * b1) * epb:(r * r1 + b * b1) * epb + epb].clone() for b in range(8)]).to(cdt)
            if scalar:
                rhs = torch.full_like(lhs, value)
            else:
                rhs = torch.cat([s2[(r * r2 + b * b2) * epb:(r * r2 + b * b2) * epb + epb].clone() for b in range(8)]).to(cdt)
            self._vwrite(dst, r * nbytes, _pack_bits(cmp(lhs, rhs), nbytes).to(dst.dtype))

    def op_vec_compare(self, op: Op) -> None:
        self._vcompare(op, scalar=False)

    def op_vec_compare_scalar(self, op: Op) -> None:
        self._vcompare(op, scalar=True)

    def op_vec_select(self, op: Op) -> None:
        if self._vcounting(op) is not None:
            raise self._err("select has no counter-mode form", op)  # type: ignore[attr-defined]
        dst, dn = self._vlin(op.operands[0], "dst", op)
        sel, sname = self._vlin(op.operands[1], "selmask", op)
        s1, n1 = self._vlin(op.operands[2], "src1", op)
        s2, n2 = self._vlin(op.operands[3], "src2", op)
        if sel.dtype != torch.uint8:
            raise self._err(f"select requires a uint8 selmask, got {sel.dtype}", op)  # type: ignore[attr-defined]
        mode = self._vmode(op)
        if mode not in ("tensor_tensor", "tensor_scalar"):
            raise self._err(f"select mode must be TENSOR_TENSOR or TENSOR_SCALAR, got {mode}", op)  # type: ignore[attr-defined]
        repeat = self._vrepeat(op)
        if repeat == 0:
            return
        dblk, drep = self._vint(op, "dst_blk_stride", 1), self._vint(op, "dst_rep_stride", 8)
        b1, r1 = self._vint(op, "src1_blk_stride", 1), self._vint(op, "src1_rep_stride", 8)
        b2, r2 = self._vint(op, "src2_blk_stride", 1), self._vint(op, "src2_rep_stride", 8)
        epb = self._vepb(s1)
        per_rep = epb * 8
        nbytes = per_rep // 8
        self._vguard(op, sname, sel, nbytes * repeat)
        self._vguard(op, dn, dst, _block_touch(repeat, drep, dblk, epb))
        self._vguard(op, n1, s1, _block_touch(repeat, r1, b1, epb))
        if mode == "tensor_tensor":
            self._vguard(op, n2, s2, _block_touch(repeat, r2, b2, epb))
        scalar_rhs = float(s2[0].item()) if mode == "tensor_scalar" else None
        for r in range(repeat):
            bits = _unpack_bits(sel[r * nbytes:(r + 1) * nbytes], per_rep)
            for b in range(8):
                mb = bits[b * epb:(b + 1) * epb]
                i1 = (r * r1 + b * b1) * epb
                a = s1[i1:i1 + epb].clone()
                if scalar_rhs is not None:
                    rhs = torch.full((epb,), scalar_rhs, dtype=s1.dtype)
                else:
                    i2 = (r * r2 + b * b2) * epb
                    rhs = s2[i2:i2 + epb].clone()
                self._vwrite(dst, (r * drep + b * dblk) * epb, _masked_blend(mb, a, rhs.to(s1.dtype)))

    # -- gathers, scatters, transposes -------------------------------------------------------------------------------

    def _voffsets(self, op: Op, x: Any) -> tuple[torch.Tensor, str]:
        off, name = self._vlin(x, "offset", op)
        if off.dtype not in (torch.int32, getattr(torch, "uint32", torch.int32)):
            raise self._err(f"{op.opcode} requires a uint32 / int32 offset tensor, got {off.dtype}", op)  # type: ignore[attr-defined]
        return off.to(torch.int64) if off.dtype != torch.int64 else off, name

    def op_vec_gather(self, op: Op) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        off, on = self._voffsets(op, op.operands[2])
        esize, epb = int(src.element_size()), self._vepb(src)
        start = self._vint(op, "start_idx", 0)
        count = self._vcounting(op)
        if count is not None:
            self._vguard(op, on, off, count)
            self._vguard(op, dn, dst, count)
            self._vgather_elems(op, dst, 0, src, off[:count] + start, esize)
            return
        repeat, drep = self._vrepeat(op), self._vint(op, "dst_rep_stride", 8)
        per_rep = epb * 8
        if repeat == 0:
            return
        self._vguard(op, on, off, repeat * per_rep)
        self._vguard(op, dn, dst, (repeat - 1) * drep * epb + per_rep)
        for r in range(repeat):
            self._vgather_elems(op, dst, r * drep * epb, src, off[r * per_rep:(r + 1) * per_rep] + start, esize)

    def _vgather_elems(self, op: Op, dst: torch.Tensor, d0: int, src: torch.Tensor, byte_addr: torch.Tensor, esize: int) -> None:
        if bool((byte_addr % esize).any().item()):
            raise self._err(f"gather offset byte address must align with the element size {esize}", op)  # type: ignore[attr-defined]
        idx = byte_addr // esize
        if bool((idx < 0).any().item()) or bool((idx >= src.numel()).any().item()):
            raise self._err("gather source index out of range", op)  # type: ignore[attr-defined]
        dst[d0:d0 + idx.numel()].copy_(src[idx])

    def op_vec_gather_block(self, op: Op) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        off, on = self._voffsets(op, op.operands[2])
        esize, epb = int(src.element_size()), self._vepb(src)
        repeat = self._vrepeat(op)
        dblk, drep = self._vint(op, "dst_blk_stride", 1), self._vint(op, "dst_rep_stride", 8)
        if repeat == 0:
            return
        self._vguard(op, on, off, repeat * 8)
        self._vguard(op, dn, dst, ((repeat - 1) * drep + 7 * dblk + 1) * epb)
        for r in range(repeat):
            for k in range(8):
                addr = int(off[r * 8 + k].item())
                if addr % 32:
                    raise self._err(f"gather_block offset must be 32B block-aligned, got {addr}", op)  # type: ignore[attr-defined]
                si = addr // esize
                if si < 0 or si + epb > src.numel():
                    raise self._err("gather_block source block out of range", op)  # type: ignore[attr-defined]
                di = (r * drep + k * dblk) * epb
                dst[di:di + epb].copy_(src[si:si + epb])

    def op_vec_scatter(self, op: Op) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        off, on = self._voffsets(op, op.operands[2])
        esize, epb = int(src.element_size()), self._vepb(src)
        start = self._vint(op, "start_idx", 0)
        count = self._vcounting(op)
        if count is not None:
            self._vguard(op, on, off, count)
            self._vguard(op, sn, src, count)
            self._vscatter_elems(op, dst, src[:count], off[:count] + start, esize)
            return
        repeat, srep = self._vrepeat(op), self._vint(op, "src_rep_stride", 8)
        per_rep = epb * 8
        if repeat == 0:
            return
        self._vguard(op, on, off, repeat * per_rep)
        self._vguard(op, sn, src, (repeat - 1) * srep * epb + per_rep)
        for r in range(repeat):
            s0 = r * srep * epb
            self._vscatter_elems(op, dst, src[s0:s0 + per_rep], off[r * per_rep:(r + 1) * per_rep] + start, esize)

    def _vscatter_elems(self, op: Op, dst: torch.Tensor, values: torch.Tensor, byte_addr: torch.Tensor, esize: int) -> None:
        if bool((byte_addr % esize).any().item()):
            raise self._err(f"scatter offset byte address must align with the element size {esize}", op)  # type: ignore[attr-defined]
        idx = byte_addr // esize
        if bool((idx < 0).any().item()) or bool((idx >= dst.numel()).any().item()):
            raise self._err("scatter destination index out of range", op)  # type: ignore[attr-defined]
        for i in range(idx.numel()):  # in order: a repeated index keeps the last write
            dst[int(idx[i].item())] = values[i]

    def op_vec_transdata5hd(self, op: Op) -> None:
        dst, dn = self._vlin(op.operands[0], "dst", op)
        src, sn = self._vlin(op.operands[1], "src", op)
        if dst.element_size() != 2 or src.element_size() != 2:
            raise self._err("transdata5hd supports b16 dtypes only", op)  # type: ignore[attr-defined]
        repeat = self._vint(op, "repeat", 1)
        srs, drs = self._vint(op, "src_row_stride", 0), self._vint(op, "dst_row_stride", 16)
        srep, drep = self._vint(op, "src_rep_stride", 1), self._vint(op, "dst_rep_stride", 16)
        if not 1 <= repeat <= MAX_REPEAT or min(srs, drs, srep, drep) < 0 or srs % 16 or drs % 16:
            raise self._err("transdata5hd: repeat in [1, 255], non-negative strides, row strides multiples of 16", op)  # type: ignore[attr-defined]
        span = (repeat - 1) * 16
        self._vguard(op, sn, src, 15 * srs + span * srep + 16)
        self._vguard(op, dn, dst, 15 * drs + span * drep + 16)
        u_src, u_dst = src.view(torch.int16), dst.view(torch.int16)
        for r in range(repeat):
            tile = torch.stack([u_src[i * srs + r * srep * 16:i * srs + r * srep * 16 + 16] for i in range(16)]).t().contiguous()
            for i in range(16):
                u_dst[i * drs + r * drep * 16:i * drs + r * drep * 16 + 16] = tile[i]


__all__ = ["VecOps", "apply_cast_dtype"]
