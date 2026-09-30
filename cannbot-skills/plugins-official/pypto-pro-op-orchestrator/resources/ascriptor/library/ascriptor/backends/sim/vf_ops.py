# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Register-level (``vf.*``) op handlers of the reference interpreter beyond the M2 core set.

Semantics follow the old ``simulator/pipe_micro.py`` handler by handler (see the notes in
``docs/rfc/0002`` §9): registers are 256 bytes viewed as the register dtype, masks are one bit
per lane, inactive lanes of a compute op are zeroed, mask ops keep the old bits of inactive
lanes, and integer arithmetic runs natively in the register width (unsigned dtypes through the
same-width signed view, as the old simulator did on CPU).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ...ir.core import Op
from ...ir.types import DType

REG_BYTES = 256
SIGNED_VIEW = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
CMP_FNS = {
    "lt": torch.lt, "le": torch.le, "gt": torch.gt, "ge": torch.ge, "eq": torch.eq, "ne": torch.ne,
}
#: Register gathers whose A5 vselr reads indices modulo the source lanes, as measured (RFC-0001 §4.4).
WRAPPED_GATHERS = frozenset({("f32", "i32"), ("f32", "u32"), ("i32", "i32"), ("i32", "u32"), ("f16", "u16")})


def _esize(dt: DType) -> int:
    return max(dt.bits, 8) // 8


def _is_unsigned(dt: DType) -> bool:
    return dt.name.startswith("u")


def _signed_view(t: torch.Tensor) -> torch.Tensor:
    """Bit-identical signed integer view of an integer register (torch CPU lacks many unsigned kernels)."""
    if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return t
    return t.view(SIGNED_VIEW[t.element_size()])


_INDEX_VIEW = {torch.uint16: torch.int16, torch.uint32: torch.int32, torch.uint64: torch.int64}


def _index_values(t: torch.Tensor) -> torch.Tensor:
    """An integer register's lanes as int64 indices, honouring the register's signedness: an unsigned register
    (``arange << 2`` reinterpreted as u8 to gather every fourth byte of a 256-lane hif8 register) reads 0..255,
    not a signed view of the same bits."""
    if t.dtype == torch.uint8:
        return t.to(torch.int64)
    if t.dtype in (torch.uint16, torch.uint32):
        return _signed_view(t).to(torch.int64) & ((1 << (8 * t.element_size())) - 1)
    return _signed_view(t).to(torch.int64)
for _n in ("float8_e4m3fn", "float8_e5m2"):
    if hasattr(torch, _n):
        _INDEX_VIEW[getattr(torch, _n)] = torch.uint8


def _indexable(t: torch.Tensor) -> torch.Tensor:
    """A bit-identical view torch can index (float8 and the wide unsigned dtypes cannot)."""
    return t.view(_INDEX_VIEW[t.dtype]) if t.dtype in _INDEX_VIEW else t


def _where(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.where`` that also accepts the wide unsigned dtypes torch CPU cannot select on (through signed views)."""
    if a.dtype in (torch.uint16, torch.uint32, torch.uint64):
        sv = SIGNED_VIEW[a.element_size()]
        return torch.where(mask, a.view(sv), b.to(a.dtype).view(sv)).view(a.dtype)
    return torch.where(mask, a, b)


def _fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``a * b + c`` on binary32 lanes with one nearest-even rounding, as A5 vaxpy/vmula/vmadd execute.

    binary64 holds the <=48-bit product exactly and TwoSum recovers the exact sum ``s + e``. A binary32
    conversion only needs ``e`` when ``s`` lies exactly on a binary32 midpoint: the exact value is then on
    the side of ``e``, whereas rounding ``s`` alone would tie to even (see docs/defects/I008). The overflow
    threshold 2**128 - 2**103 is such a midpoint too: an error toward zero keeps the largest finite value.
    """
    p, c = a.double() * b.double(), c.double()
    s = p + c
    v = s - p
    e = (p - (s - v)) + (c - v)
    r = s.float()
    up = torch.nextafter(r, torch.full_like(r, math.inf))
    down = torch.nextafter(r, torch.full_like(r, -math.inf))
    r = torch.where(((r.double() + up.double()) / 2 == s) & (e > 0), up, r)
    r = torch.where(((r.double() + down.double()) / 2 == s) & (e < 0), down, r)
    below = (s.abs() == 2.0**128 - 2.0**103) & (e * s < 0)  # I021
    return torch.where(below, torch.copysign(torch.full_like(r, torch.finfo(torch.float32).max), s.float()), r)


def _round_half_away(values: torch.Tensor, mantissa: int, exponent: int) -> torch.Tensor:
    """Round exact binary64 values once to a binary format, ties away from zero (A5 vmulscvt, I011).

    Dividing by the power-of-two quantum is exact and the <=48-bit products leave room for ``+ 0.5``.
    """
    bias = (1 << (exponent - 1)) - 1
    magnitude = values.abs()
    _, power = torch.frexp(magnitude)
    quantum = torch.pow(2.0, torch.clamp(power - 1 - mantissa, min=1 - bias - mantissa).double())
    rounded = torch.floor(magnitude / quantum + 0.5) * quantum
    rounded = torch.where(rounded >= 2.0 ** (bias + 1), torch.full_like(rounded, math.inf), rounded)
    return torch.copysign(rounded, values)


def _flip_key(t: torch.Tensor) -> torch.Tensor:
    """Signed view with the sign bit flipped: orders unsigned values correctly under signed compares."""
    sv = _signed_view(t)
    return sv ^ torch.tensor(-(1 << (8 * sv.element_size() - 1)), dtype=sv.dtype)


class VfOps:
    """Mixin of :class:`Interp`."""

    # -- helpers ----------------------------------------------------------------------------------

    def _reg(self, x: Any) -> Any:
        return self.raw(x)

    def _num_view(self, reg: Any) -> torch.Tensor:
        """A tensor torch can compute on: floats as fp32, ints as the (signed) native width."""
        t = reg.tensor()
        if reg.dtype.is_float:
            if reg.dtype.name == "hif8":
                from ...dtypes.hif8_codec import hif8_to_fp32

                return hif8_to_fp32(t)
            return t.float()
        return _signed_view(t)

    def _store_num(self, op: Op, dst: Any, values: torch.Tensor) -> None:
        """Write computed values (fp32 or signed-int) into a register, zeroing inactive lanes."""
        reg = dst.tensor()
        if dst.dtype.is_float:
            if dst.dtype.name == "hif8":
                from ...dtypes.hif8_codec import fp32_to_hif8

                out = fp32_to_hif8(values.float())
            else:
                out = values.to(reg.dtype)
        else:
            out = values.to(_signed_view(reg).dtype).view(reg.dtype)
        mask = self._mask(op, reg.numel())
        if mask is not None:
            out = _where(mask, out, torch.zeros_like(out))
        reg.copy_(out)
        dst.valid = dst.lanes

    def _scalar(self, x: Any, dt: DType) -> Any:
        v = self.val(x)
        if dt.is_float and dt.name in ("f16", "bf16"):
            v = torch.tensor(float(v), dtype=torch.float16 if dt.name == "f16" else torch.bfloat16).item()
        return v

    def _unsigned_minmax(self, op: Op, *, maximum: bool, scalar: bool = False) -> bool:
        """Compare unsigned lane keys while selecting the original payload bits.

        The generic arithmetic view is signed, which is useful for modular arithmetic
        but orders high-bit unsigned values incorrectly. Do not change that shared view.
        """
        dst, source = self._reg(op.operands[0]), self._reg(op.operands[1])
        if source.dtype.name not in ("u16", "u32", "u64"):
            return False
        left = _signed_view(source.tensor())
        if scalar:
            width = 8 * left.element_size()
            bits = int(self.val(op.operands[2])) & ((1 << width) - 1)
            signed = bits - (1 << width) if bits >= 1 << (width - 1) else bits
            right = torch.full_like(left, signed)
        else:
            right = _signed_view(self._reg(op.operands[2]).tensor())
        left_key, right_key = _flip_key(left), _flip_key(right)
        choose_left = left_key >= right_key if maximum else left_key <= right_key
        # Materialize before the destination write: dst may alias either operand.
        self._store_num(op, dst, torch.where(choose_left, left, right))
        return True

    # -- arithmetic ---------------------------------------------------------------------------------

    def op_vf_mod(self, op: Op) -> None:
        dst, a, b = (self._reg(x) for x in op.operands[:3])
        # Python integers preserve unsigned high bits and avoid the CPU's INT64_MIN / -1 trap.
        zero = -1 if dst.dtype.name == "i64" else (1 << 64) - 1
        values = [x % y if y else zero for x, y in zip(a.tensor().tolist(), b.tensor().tolist(), strict=True)]
        out = torch.tensor(values, dtype=dst.tensor().dtype)
        mask = self._mask(op, dst.lanes)
        if mask is not None:
            out = _where(mask, out, torch.zeros_like(out))
        dst.tensor().copy_(out)
        dst.valid = dst.lanes

    def op_vf_log(self, op: Op) -> None:
        self._unary(op, torch.log)

    def op_vf_log2(self, op: Op) -> None:
        self._unary(op, torch.log2)

    def op_vf_log10(self, op: Op) -> None:
        self._unary(op, torch.log10)

    def op_vf_not(self, op: Op) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        self._store_num(op, dst, ~_signed_view(src.tensor()))

    def _bitwise(self, op: Op, fn: Any) -> None:
        dst, a, b = (self._reg(x) for x in op.operands[:3])
        self._store_num(op, dst, fn(_signed_view(a.tensor()), _signed_view(b.tensor())))

    def op_vf_and(self, op: Op) -> None:
        self._bitwise(op, torch.bitwise_and)

    def op_vf_or(self, op: Op) -> None:
        self._bitwise(op, torch.bitwise_or)

    def op_vf_xor(self, op: Op) -> None:
        self._bitwise(op, torch.bitwise_xor)

    def op_vf_lrelu(self, op: Op) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        s = self._scalar(op.operands[2], src.dtype)
        x = self._num_view(src)
        self._store_num(op, dst, torch.where(x > 0, x, x * s))

    def op_vf_prelu(self, op: Op) -> None:
        dst, a, b = (self._reg(x) for x in op.operands[:3])
        x, y = self._num_view(a), self._num_view(b)
        self._store_num(op, dst, torch.where(x > 0, x, x * y))

    def op_vf_axpy(self, op: Op) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        s = self._scalar(op.operands[2], src.dtype)
        if src.dtype.name == "f32":
            x = self._num_view(src)
            result = _fma32(x, torch.full_like(x, float(s)), self._num_view(dst))  # the scalar is a C float argument
        elif src.dtype.is_float:
            result = (self._num_view(dst).double() + self._num_view(src).double() * float(s)).float()
        else:
            result = self._num_view(dst) + self._num_view(src) * int(s)
        self._store_num(op, dst, result)

    def op_vf_abssub(self, op: Op) -> None:
        dst, a, b = (self._reg(x) for x in op.operands[:3])
        if a.dtype.name == "u64":
            left, right = _signed_view(a.tensor()), _signed_view(b.tensor())
            # Unsigned max-minus-min; materialize before a possible aliased write.
            left_is_larger = _flip_key(left) >= _flip_key(right)
            result = torch.where(left_is_larger, left - right, right - left)
        else:
            result = torch.abs(self._num_view(a) - self._num_view(b))
        self._store_num(op, dst, result)

    def op_vf_muldstadd(self, op: Op) -> None:  # dst = dst * src0 + src1
        dst, a, b = (self._reg(x) for x in op.operands[:3])
        if dst.dtype.name == "f32":
            result = _fma32(self._num_view(dst), self._num_view(a), self._num_view(b))
        else:
            result = self._num_view(dst) * self._num_view(a) + self._num_view(b)
        self._store_num(op, dst, result)

    def op_vf_muladddst(self, op: Op) -> None:  # dst = src0 * src1 + dst
        dst, a, b = (self._reg(x) for x in op.operands[:3])
        if dst.dtype.name == "f32":
            result = _fma32(self._num_view(a), self._num_view(b), self._num_view(dst))
        else:
            result = self._num_view(a) * self._num_view(b) + self._num_view(dst)
        self._store_num(op, dst, result)

    # -- shifts ------------------------------------------------------------------------------------------

    @staticmethod
    def _shift_bits(src: torch.Tensor, shift: int, left: bool, unsigned: bool) -> torch.Tensor:
        if shift < 0:
            return VfOps._shift_bits(src, -shift, not left, unsigned)
        sv = _signed_view(src)
        width = 8 * sv.element_size()
        if left:
            out = sv << shift
        elif not unsigned:
            out = sv >> shift
        elif shift == 0:
            out = sv
        elif shift >= width:
            out = torch.zeros_like(sv)
        else:
            out = (sv >> shift) & ((1 << (width - shift)) - 1)
        return out

    def op_vf_shiftls(self, op: Op) -> None:
        self._shift_scalar(op, True)

    def op_vf_shiftrs(self, op: Op) -> None:
        self._shift_scalar(op, False)

    def _shift_scalar(self, op: Op, left: bool) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        shift = int(self.val(op.operands[2]))
        self._store_num(op, dst, self._shift_bits(src.tensor(), shift, left, _is_unsigned(src.dtype)))

    def op_vf_shiftl(self, op: Op) -> None:
        self._shift_reg(op, True)

    def op_vf_shiftr(self, op: Op) -> None:
        self._shift_reg(op, False)

    def _shift_reg(self, op: Op, left: bool) -> None:
        dst, a, b = (self._reg(x) for x in op.operands[:3])
        data = a.tensor()
        sh = _signed_view(b.tensor()).to(torch.int64)
        out = _signed_view(data).clone()
        for s in torch.unique(sh).tolist():
            shifted = self._shift_bits(data, int(s), left, _is_unsigned(a.dtype))
            out = torch.where(sh == s, shifted, out)
        self._store_num(op, dst, out)

    # -- reductions -------------------------------------------------------------------------------------

    def _masked_src(self, op: Op, src: Any) -> torch.Tensor:
        v = self._num_view(src).clone()
        mask = self._mask(op, v.numel())
        return v[mask] if mask is not None else v

    @staticmethod
    def _extremum(kind: str, v: torch.Tensor, unsigned: bool) -> tuple[Any, int]:
        """vcmax/vcmin over the active values ``v`` (RFC-0001): (value, first position holding it).

        An empty selection gives (None, 0) and an active NaN (math.nan, its first position); -0 orders below +0."""
        if v.numel() == 0:
            return None, 0
        if v.is_floating_point() and bool(torch.isnan(v).any()):
            return math.nan, int(torch.isnan(v).nonzero()[0])
        key = _flip_key(v) if unsigned and not v.is_floating_point() else v
        same = key == (key.max() if kind == "max" else key.min())
        if v.is_floating_point() and float(v[same][0]) == 0:
            negative, positive = same & torch.signbit(v), same & ~torch.signbit(v)
            same = (positive if bool(positive.any()) else negative) if kind == "max" else \
                (negative if bool(negative.any()) else positive)
        position = int(same.nonzero()[0])
        return v[position], position

    @staticmethod
    def _put_extremum(dst: Any, lane: int, kind: str, value: Any) -> None:
        """Write an extremum into ``lane``: None is the dtype's lowest (max) or highest (min) value and NaN the
        all-ones positive NaN of an IEEE lane (A5 writes 0x7FFFFFFF for FP32)."""
        reg = dst.tensor()
        if not dst.dtype.is_float:
            view = _signed_view(reg)
            if value is None:
                info = torch.iinfo(view.dtype)
                value = (0 if kind == "max" else -1) if _is_unsigned(dst.dtype) else info.min if kind == "max" else info.max
            view[lane] = value if isinstance(value, int) else value.to(view.dtype)
            return
        if value is None:
            value = torch.tensor(-math.inf if kind == "max" else math.inf)
        elif isinstance(value, float) and dst.dtype.name in ("f16", "bf16", "f32"):
            reg.view(SIGNED_VIEW[reg.element_size()])[lane] = (1 << (8 * reg.element_size() - 1)) - 1
            return
        elif isinstance(value, float):
            value = torch.tensor(value)
        reg[lane] = value.to(reg.dtype)

    def _whole_reduce(self, op: Op, kind: str) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        if kind == "add":
            v = self._masked_src(op, src)
            reg = dst.tensor()
            reg.zero_()
            if v.numel():
                out = reg if dst.dtype.is_float else _signed_view(reg)
                out[0] = v.sum().to(out.dtype)
            dst.valid = dst.lanes
            return
        values = self._num_view(src).clone()
        mask = self._mask(op, values.numel())
        lanes = torch.arange(values.numel()) if mask is None else mask.nonzero().flatten()
        value, position = self._extremum(kind, values[lanes], _is_unsigned(src.dtype))
        reg = dst.tensor()
        reg.zero_()
        self._put_extremum(dst, 0, kind, value)
        if self.attr(op, "index", False) and value is not None:  # the first active lane holding it, as lane bits
            bits = 8 * reg.element_size()
            index = int(lanes[position])
            reg.view(SIGNED_VIEW[reg.element_size()])[1] = index - (1 << bits) if index >= 1 << (bits - 1) else index
        dst.valid = dst.lanes

    def op_vf_cadd(self, op: Op) -> None:
        self._whole_reduce(op, "add")

    def op_vf_cmax(self, op: Op) -> None:
        self._whole_reduce(op, "max")

    def op_vf_cmin(self, op: Op) -> None:
        self._whole_reduce(op, "min")

    def _group_reduce(self, op: Op, kind: str) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        v = self._num_view(src).clone()
        mask = self._mask(op, v.numel())
        c0 = 32 // _esize(dst.dtype)
        reg = dst.tensor()
        reg.zero_()
        out = _signed_view(reg) if not dst.dtype.is_float else reg
        for bi in range(8):
            block = v[bi * c0:(bi + 1) * c0]
            if mask is not None:
                block = block[mask[bi * c0:(bi + 1) * c0][: block.numel()]]
            if kind != "add":  # an empty group writes the dtype extreme (RFC-0001)
                self._put_extremum(dst, bi, kind, self._extremum(kind, block, _is_unsigned(src.dtype))[0])
            elif block.numel():
                out[bi] = block.sum().to(out.dtype)
        dst.valid = dst.lanes

    def op_vf_cgadd(self, op: Op) -> None:
        self._group_reduce(op, "add")

    def op_vf_cgmax(self, op: Op) -> None:
        self._group_reduce(op, "max")

    def op_vf_cgmin(self, op: Op) -> None:
        self._group_reduce(op, "min")

    def op_vf_cpadd(self, op: Op) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        v = self._num_view(src).clone()
        n = v.numel()
        mask = self._mask(op, n)
        if mask is not None:
            v = _where(mask, v, torch.zeros_like(v))
        pairs = n // 2
        sums = v[: pairs * 2].reshape(pairs, 2).sum(dim=1)
        reg = dst.tensor()
        reg.zero_()
        out = reg if dst.dtype.is_float else _signed_view(reg)
        out[:pairs] = sums.to(out.dtype)
        dst.valid = dst.lanes

    # -- compare / select / masks --------------------------------------------------------------------

    def _compare(self, op: Op, lhs: torch.Tensor, rhs: torch.Tensor, unsigned: bool) -> None:
        dst = self.raw(op.operands[0])
        mode = str(self.attr(op, "mode", "lt"))
        if unsigned and not lhs.is_floating_point():
            lhs, rhs = _flip_key(lhs), _flip_key(rhs)
        result = CMP_FNS[mode](lhs, rhs)
        n = min(result.numel(), dst.bits.numel())
        mask = self._mask(op, n)
        # Compare generates a new predicate. Inactive lanes are false, even
        # when the destination is also the execution mask (M10-051).
        bits = result[:n] if mask is None else result[:n] & mask
        dst.write_lanes(bits)

    def op_vf_cmp(self, op: Op) -> None:
        a, b = self._reg(op.operands[1]), self._reg(op.operands[2])
        self._compare(op, self._num_view(a), self._num_view(b), _is_unsigned(a.dtype))

    def op_vf_cmps(self, op: Op) -> None:
        a = self._reg(op.operands[1])
        v = self._num_view(a)
        s = self._scalar(op.operands[2], a.dtype)
        if v.is_floating_point():
            rhs = torch.full_like(v, s)
        else:  # integer immediates are compared through the register's bit pattern (signed view of any width)
            bits = 8 * v.element_size()
            wrapped = int(s) & ((1 << bits) - 1)
            rhs = torch.full_like(v, wrapped - (1 << bits) if wrapped >= 1 << (bits - 1) else wrapped)
        self._compare(op, v, rhs, _is_unsigned(a.dtype))

    def op_vf_select(self, op: Op) -> None:
        dst, a, b = (self._reg(x) for x in op.operands[:3])
        ta, tb = a.tensor(), b.tensor()
        n = dst.lanes
        mask = self._mask(op, n)
        sel = mask if mask is not None else torch.ones(n, dtype=torch.bool)
        va, vb = _signed_view(ta) if not a.dtype.is_float else ta, _signed_view(tb) if not b.dtype.is_float else tb
        out = _where(sel, va[:n], vb[:n])
        reg = dst.tensor()
        if dst.dtype.is_float:
            reg.copy_(out.to(reg.dtype))
        else:
            _signed_view(reg).copy_(out)
        dst.valid = dst.lanes

    def _mask_blend(self, op: Op, dst: Any, result: torch.Tensor) -> None:
        mref = op.attrs.get("mask")
        if mref is not None:
            result = result & self.raw(mref).physical()
        dst.write_physical(result)

    def op_vf_mask_not(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        self._mask_blend(op, dst, ~src.physical())

    def op_vf_mask_mov(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        self._mask_blend(op, dst, src.physical())

    def _mask_binary(self, op: Op, fn: Any) -> None:
        dst, a, b = (self.raw(x) for x in op.operands[:3])
        self._mask_blend(op, dst, fn(a.physical(), b.physical()))

    def op_vf_mask_and(self, op: Op) -> None:
        self._mask_binary(op, torch.logical_and)

    def op_vf_mask_or(self, op: Op) -> None:
        self._mask_binary(op, torch.logical_or)

    def op_vf_mask_xor(self, op: Op) -> None:
        self._mask_binary(op, torch.logical_xor)

    def op_vf_mask_sel(self, op: Op) -> None:
        dst, a, b = (self.raw(x) for x in op.operands[:3])
        mref = op.attrs.get("mask")
        result = a.physical() if mref is None else torch.where(self.raw(mref).physical(), a.physical(), b.physical())
        dst.write_physical(result)

    def op_vf_mask_pack(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        even = src.physical()[::2]
        out = torch.zeros(256, dtype=torch.bool)
        if str(self.attr(op, "mode", "lowest")) == "lowest":
            out[:128] = even
        else:
            out[128:] = even
        dst.write_physical(out)

    def op_vf_mask_unpack(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        payload = src.physical()
        packed = payload[:128] if str(self.attr(op, "mode", "lowest")) == "lowest" else payload[128:]
        out = torch.zeros(256, dtype=torch.bool)
        out[::2] = packed
        dst.write_physical(out)

    def op_vf_mask_interleave(self, op: Op) -> None:
        d0, d1, s0, s1 = (self.raw(x) for x in op.operands[:4])
        a, b = s0.physical().reshape(-1, d0.bit_stride), s1.physical().reshape(-1, d0.bit_stride)
        half = a.shape[0] // 2
        o0, o1 = torch.empty_like(a), torch.empty_like(a)
        o0[0::2], o0[1::2] = a[:half], b[:half]
        o1[0::2], o1[1::2] = a[half:], b[half:]
        d0.write_physical(o0.flatten())
        d1.write_physical(o1.flatten())

    def op_vf_mask_deinterleave(self, op: Op) -> None:
        d0, d1, s0, s1 = (self.raw(x) for x in op.operands[:4])
        a, b = s0.physical().reshape(-1, d0.bit_stride), s1.physical().reshape(-1, d0.bit_stride)
        half = a.shape[0] // 2
        o0, o1 = torch.empty_like(a), torch.empty_like(a)
        o0[:half], o1[:half] = a[0::2], a[1::2]
        o0[half:], o1[half:] = b[0::2], b[1::2]
        d0.write_physical(o0.flatten())
        d1.write_physical(o1.flatten())

    def op_vf_mask_update(self, op: Op) -> None:
        dst = self.raw(op.operands[0])
        cnt_ref = op.attrs.get("cnt")
        cnt = int(self.val(cnt_ref))
        lanes = dst.bits.numel()
        active = max(0, min(cnt, lanes))
        bits = torch.zeros(lanes, dtype=torch.bool)
        bits[:active] = True
        dst.write_lanes(bits)
        cell = self.raw(cnt_ref) if hasattr(cnt_ref, "name") else None
        if cell is not None and hasattr(cell, "value"):
            cell.value = max(cnt - lanes, 0)  # the counter is decremented in place (UpdateMask semantics)

    def op_vf_mask_from_spr(self, op: Op) -> None:
        dst = self.raw(op.operands[0])
        spr = getattr(self.lane, "spr_mask", None)
        if spr is None:
            raise self._error("move_mask_spr needs a preceding set_mask / set_mask_by_count on this lane", op)
        n = min(dst.bits.numel(), spr.numel())
        bits = torch.zeros_like(dst.bits)
        bits[:n] = spr[:n]
        # movp_b16/b32 replicates each SPR bit into its two/four-bit group.
        if dst.bit_stride in (2, 4):
            dst.write_physical(bits.repeat_interleave(dst.bit_stride))
        else:
            dst.write_lanes(bits)  # retain the logical convention outside the defined movp forms

    def op_vf_ub_to_mask(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        flat, origin = self.m.storage(src, self.lane)
        offset = int(self.attr(op, "offset", 0))
        self._check_aligned(op, "mask load", origin + offset, flat.element_size())
        self._check_ub_ranges(op, src, (((origin + offset) * flat.element_size(), 32),))
        raw = flat.view(torch.uint8)[(origin + offset) * flat.element_size():][:32]
        bits = torch.tensor([(int(raw[i // 8]) >> (i % 8)) & 1 for i in range(256)], dtype=torch.bool)
        dst.write_physical(bits)

    def op_vf_mask_to_ub(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        flat, origin = self.m.storage(dst, self.lane)
        offset = int(self.attr(op, "offset", 0))
        self._check_aligned(op, "mask store", origin + offset, flat.element_size())
        self._check_ub_ranges(op, dst, (((origin + offset) * flat.element_size(), 32),))
        bits = src.physical()
        raw = torch.zeros(32, dtype=torch.uint8)
        for i in range(256):
            if bits[i]:
                raw[i // 8] |= 1 << (i % 8)
        flat.view(torch.uint8)[(origin + offset) * flat.element_size():][:32] = raw

    # -- data movement inside the vector unit ------------------------------------------------------

    def op_vf_barrier(self, op: Op) -> None:
        return None  # ordering only

    def op_vf_clear_spr(self, op: Op) -> None:
        return None

    def op_vf_arange(self, op: Op) -> None:
        dst = self._reg(op.operands[0])
        start = self.val(op.attrs.get("v", 0))
        increase = str(self.attr(op, "mode", "increase")) == "increase"
        n = dst.lanes
        reg = dst.tensor()
        if dst.dtype.is_float:
            idx = torch.arange(n, dtype=torch.float64)
            values = (start + idx) if increase else (start - idx)
            reg.copy_(values.to(reg.dtype))
        else:
            bits = dst.dtype.bits
            values = [(int(start) + (i if increase else -i)) & ((1 << bits) - 1) for i in range(n)]
            signed = [v if v < 1 << (bits - 1) else v - (1 << bits) for v in values]
            _signed_view(reg).copy_(torch.tensor(signed, dtype=_signed_view(reg).dtype))
        dst.valid = dst.lanes

    def op_vf_interleave(self, op: Op) -> None:
        d0, d1, s0, s1 = (self._reg(x) for x in op.operands[:4])
        a, b = s0.tensor().clone(), s1.tensor().clone()
        half = a.numel() // 2
        o0, o1 = torch.zeros_like(a), torch.zeros_like(a)
        o0[0::2], o0[1::2] = a[:half], b[:half]
        o1[0::2], o1[1::2] = a[half:], b[half:]
        d0.tensor().copy_(o0)
        d1.tensor().copy_(o1)
        d0.valid = d1.valid = d0.lanes

    def op_vf_deinterleave(self, op: Op) -> None:
        d0, d1, s0, s1 = (self._reg(x) for x in op.operands[:4])
        a, b = s0.tensor().clone(), s1.tensor().clone()
        half = a.numel() // 2
        o0, o1 = torch.zeros_like(a), torch.zeros_like(a)
        o0[:half], o1[:half] = a[0::2], a[1::2]
        o0[half:], o1[half:] = b[0::2], b[1::2]
        d0.tensor().copy_(o0)
        d1.tensor().copy_(o1)
        d0.valid = d1.valid = d0.lanes

    def op_vf_gather(self, op: Op) -> None:
        dst, src, index = (self._reg(x) for x in op.operands[:3])
        idx = _index_values(index.tensor())
        s = src.tensor()
        if (src.dtype.name, index.dtype.name) in WRAPPED_GATHERS:
            idx = idx % s.numel()
        elif bool((idx < 0).any()) or bool((idx >= s.numel()).any()):
            raise self._error("gather index out of range", op)
        values = _indexable(s)[idx].view(s.dtype)
        n = min(idx.numel(), dst.lanes)
        reg = dst.tensor()
        reg.zero_()
        reg[:n] = values[:n]
        dst.valid = dst.lanes

    def op_vf_gathermask(self, op: Op) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        mask = self._mask(op, src.lanes)
        s = src.tensor()
        # Selection reads the original lanes even when src and dst alias.
        selected = _indexable(s)[mask].view(s.dtype) if mask is not None else s.clone()
        reg = dst.tensor()
        reg.zero_()
        reg[: selected.numel()] = selected
        dst.valid = dst.lanes

    def op_vf_squeeze(self, op: Op) -> None:
        self.op_vf_gathermask(op)

    def op_vf_unsqueeze(self, op: Op) -> None:
        dst = self._reg(op.operands[0])
        mask = self._mask(op, dst.lanes)
        bits = mask if mask is not None else torch.ones(dst.lanes, dtype=torch.bool)
        excl = torch.zeros(dst.lanes, dtype=torch.int64)
        excl[1:] = torch.cumsum(bits.to(torch.int64), 0)[:-1]
        reg = dst.tensor()
        if dst.dtype.is_float:
            reg.copy_(excl.to(reg.dtype))
        else:
            _signed_view(reg).copy_(excl.to(_signed_view(reg).dtype))
        dst.valid = dst.lanes

    def _indexed_error(self, message: str, op: Op) -> Exception:
        return self._error(f"{message} at {op.loc}" if op.loc else message, op)

    def op_vf_gather_copy(self, op: Op) -> None:
        dst, src_ref, index = self._reg(op.operands[0]), self.raw(op.operands[1]), self._reg(op.operands[2])
        flat, origin = self.m.storage(src_ref, self.lane)
        start = origin + int(self.attr(op, "offset", 0))
        src = flat[start:]
        idx = _index_values(index.tensor())
        n = dst.lanes
        mask = self._mask(op, n)
        active = torch.arange(n) if mask is None else torch.nonzero(mask).reshape(-1)
        self._check_aligned(op, "gather load", start, flat.element_size())
        reg = dst.tensor()
        reg.zero_()
        if active.numel() == 0:
            dst.valid = dst.lanes
            return
        if int(active.max()) >= idx.numel():
            raise self._indexed_error("ub_to_reg_gather index register is too short", op)
        take = idx[active]
        if not 0 <= start <= flat.numel() or bool((take < 0).any()) or bool((take >= src.numel()).any()):
            raise self._indexed_error("ub_to_reg_gather index out of range", op)
        src_i, reg_i = _indexable(src), _indexable(reg)
        if src.element_size() == 1 and reg.element_size() == 2:
            values = src.view(torch.uint8)[take].to(torch.int16).view(reg_i.dtype)  # b8 -> b16 zero-extend of the bit pattern
        else:
            values = src_i[take].view(reg_i.dtype) if src.element_size() == reg.element_size() else src_i[take].to(reg_i.dtype)
        reg_i[active] = values
        dst.valid = dst.lanes

    def op_vf_gatherb(self, op: Op) -> None:
        dst, src_ref, index = self._reg(op.operands[0]), self.raw(op.operands[1]), self._reg(op.operands[2])
        flat, origin = self.m.storage(src_ref, self.lane)
        start = origin + int(self.attr(op, "offset", 0))
        src = flat[start:]
        esize = dst.tensor().element_size()
        be = 32 // esize
        n_block = dst.lanes // be
        idx = _index_values(index.tensor())
        mask = self._mask(op, dst.lanes)
        # vgatherb samples its predicate at UINT32 index lanes: physical bit 4*b selects whole block b (I014).
        live = None if mask is None else self.env[op.attrs["mask"].name].physical()[::4]
        out = torch.zeros(dst.lanes, dtype=dst.tensor().dtype)
        self._check_aligned(op, "gather block load", start, flat.element_size())
        for i in range(n_block):
            if live is not None and not bool(live[i]):
                continue
            if i >= idx.numel():
                raise self._indexed_error("gatherb index register is too short", op)
            byte_off = int(idx[i])
            if byte_off < 0:
                raise self._indexed_error("gatherb byte offset out of range", op)
            if byte_off % 32:
                raise self._indexed_error(f"gatherb byte offset {byte_off} is not 32-byte aligned", op)
            e0 = byte_off // esize
            if not 0 <= start <= flat.numel() or e0 + be > src.numel():
                raise self._indexed_error("gatherb block outside the source", op)
            out[i * be:(i + 1) * be] = src[e0:e0 + be].view(out.dtype) if src.element_size() == esize else src[e0:e0 + be].to(out.dtype)
        dst.tensor().copy_(out)  # selected blocks keep every lane, whatever their own predicate bits
        dst.valid = dst.lanes

    def op_vf_scatter_copy(self, op: Op) -> None:
        dst_ref, src, index = self.raw(op.operands[0]), self._reg(op.operands[1]), self._reg(op.operands[2])
        flat, origin = self.m.storage(dst_ref, self.lane)
        start = origin + int(self.attr(op, "offset", 0))
        target = flat[start:]
        idx = _index_values(index.tensor())
        s = src.tensor()
        lanes = src.lanes
        mask = self._mask(op, lanes)
        # Byte scatter index k consumes source/predicate 2*k, including partial masks.
        active = torch.arange(min(idx.numel(), (lanes + 1) // 2)) * 2 if s.element_size() == 1 else torch.arange(lanes)
        if mask is not None:
            active = active[mask[active]]
        self._check_aligned(op, "scatter store", start, flat.element_size())
        if active.numel() == 0:
            return
        index_lanes = active // 2 if s.element_size() == 1 else active
        if int(index_lanes.max()) >= idx.numel():
            raise self._indexed_error("reg_to_ub_scatter index register is too short", op)
        take = idx[index_lanes]
        # Validate the entire selection before the first write; Python negative indexing is not an address mode.
        if not 0 <= start <= flat.numel() or bool((take < 0).any()) or bool((take >= target.numel()).any()):
            raise self._indexed_error("reg_to_ub_scatter index out of range", op)
        writes = list(zip(active.tolist(), take.tolist(), strict=True))
        if take.unique().numel() != take.numel():
            # A5 leaves the surviving duplicate writer unspecified, so only identical payload bits are defined (I015).
            first: dict[int, tuple[int, bytes]] = {}
            for lane, destination in writes:
                payload = bytes(s[lane].to(target.dtype).reshape(1).view(torch.uint8).tolist())
                seen, bits = first.setdefault(destination, (lane, payload))
                if bits != payload:
                    raise self._indexed_error(f"reg_to_ub_scatter lanes {seen} and {lane} write different payloads to index {destination}", op)
        for lane, destination in writes:
            target[destination] = s[lane].to(target.dtype)

    def op_vf_ub_cursor(self, op: Op) -> None:
        src = self.raw(op.operands[0])
        self.set_result(op, src)

    def _advance(self, ref: Any, n: int) -> Any:
        """A window moved ``n`` elements forward in its flat storage (unaligned-access cursors).

        The leading dimension carries instead of wrapping: a cursor that has just written the last element of
        its storage stands one past the end, which is where its Post finds it (I043). Wrapping that position
        back to the origin turned the last row of a packed walk into a cursor mismatch.
        """
        flat, origin = self.m.storage(ref, self.lane)
        base = self.m.resolve_base(ref, self.lane)
        pos = origin - ref.view_offset + n  # within the window's (possibly rebased) coordinate system
        offsets = []
        for size in reversed(tuple(base.shape)[1:]):
            offsets.append(pos % size)
            pos //= size
        offsets.append(pos)
        return type(ref)(ref.space, ref.dtype, ref.base, ref.slot, tuple(reversed(offsets)), ref.extents, ref.shape,
                         view_offset=ref.view_offset)

    def op_vf_load_unalign_pre(self, op: Op) -> None:
        ureg, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        ureg.armed = True
        ureg.anchor = self.m.storage(src, self.lane)[1] + int(self.attr(op, "offset", 0))

    def op_vf_load_unalign(self, op: Op) -> None:
        dst, src, ureg = self._reg(op.operands[0]), self.raw(op.operands[1]), self.raw(op.operands[2])
        if not getattr(ureg, "armed", False):
            raise self._error("LoadUnAlign before LoadUnAlignPre", op)
        flat, origin = self.m.storage(src, self.lane)
        start = origin + int(self.attr(op, "offset", 0))
        n = dst.lanes
        reg = dst.tensor()
        reg.zero_()
        take = flat[start: start + n]
        reg[: take.numel()] = take.to(reg.dtype)
        dst.valid = dst.lanes
        stride = self.attr(op, "stride")
        if stride is not None:
            self.env[op.operands[1].name] = self._advance(src, int(self.val(stride)))

    def op_vf_store_unalign(self, op: Op) -> None:
        dst, src, ureg = self.raw(op.operands[0]), self._reg(op.operands[1]), self.raw(op.operands[2])
        count = int(self.val(op.attrs["count"]))
        flat, origin = self.m.storage(dst, self.lane)
        start = origin + int(self.attr(op, "offset", 0))
        if count < 0 or count > src.lanes or start < 0 or start + count > flat.numel():
            raise self._error("StoreUnAlign range exceeds register or destination capacity", op)
        pending = getattr(ureg, "pending", None)
        values = src.tensor()[:count].to(flat.dtype).clone()
        if pending is not None:
            previous, begin, suffix = pending
            if previous.data_ptr() != flat.data_ptr() or begin + suffix.numel() != start:
                raise self._error("StoreUnAlign must continue its pending cursor or flush with Post", op)
            start = begin
            values = torch.cat((suffix, values))
        end = start + values.numel()
        boundary = end * flat.element_size() // 32 * 32 // flat.element_size()
        committed = max(0, boundary - start)
        flat[start:start + committed] = values[:committed]
        ureg.pending = (flat, start + committed, values[committed:].clone()) if committed < values.numel() else None
        ureg.pending_at = op if ureg.pending is not None else None  # the store a missing Post would strand
        ureg.armed = True
        self.env[op.operands[0].name] = self._advance(dst, count)

    def op_vf_store_unalign_post(self, op: Op) -> None:
        dst, ureg = self.raw(op.operands[0]), self.raw(op.operands[1])
        pending = getattr(ureg, "pending", None)
        if pending is not None:
            flat, begin, suffix = pending
            current, origin = self.m.storage(dst, self.lane)
            if current.data_ptr() != flat.data_ptr() or origin != begin + suffix.numel():
                raise self._error("StoreUnAlignPost must use the pending store cursor", op)
            flat[begin:begin + suffix.numel()] = suffix
        ureg.pending = ureg.pending_at = None
        ureg.armed = False
        stride = int(self.val(self.attr(op, "stride", 0)))
        if stride:
            self.env[op.operands[0].name] = self._advance(dst, stride)

    def check_store_states(self, func: Any, env: dict[str, Any]) -> None:
        """Reject a vf function that returns while one of its store states still holds bytes (RFC-0001 §6.11).

        The suffix is not a property of the program's state object: it stays in the hardware's unaligned
        register, which outlives both the state and the launch. Measured on A5 (job 361): the bytes are never
        written at their own address, and the next store that reuses the register commits from the stranded
        run's start lane, so those stale bytes land before that store's own start and its leading lanes are
        dropped. Which state reuses which register is an allocation decision the model cannot see, so the
        model rejects the program instead of inventing one reading of it.
        """
        for op in func.walk():
            if op.opcode != "vf.unalign":
                continue
            ureg = env.get(op.results[0].name)
            pending = getattr(ureg, "pending", None)
            if pending is None:
                continue
            suffix = pending[2]
            raise self._indexed_error(
                f"vf {func.name} returns with {suffix.numel()} element(s) pending in store state "
                f"{op.results[0].name!r}; every unaligned store state must be flushed with StoreUnAlignPost "
                "before its vf returns, or the bytes are stranded in the hardware register and corrupt the "
                "next store that reuses it", getattr(ureg, "pending_at", None) or op)

    # -- fused / conversions --------------------------------------------------------------------------

    def op_vf_expsub(self, op: Op) -> None:
        dst, a, b = (self._reg(x) for x in op.operands[:3])
        ratio = _esize(dst.dtype) // _esize(a.dtype)
        slot = {"zero": 0, "one": 1}.get(str(self.attr(op, "layout", "zero")), 0)
        va, vb = self._num_view(a), self._num_view(b)
        n = dst.lanes
        src_idx = torch.arange(n) * ratio + slot
        mask = self._mask(op, a.lanes)
        result = torch.exp(va[src_idx] - vb[src_idx])
        if mask is not None:
            # A5 tests the predicate at the source position of each result, as for widening casts (I011).
            result = _where(mask[src_idx], result, torch.zeros_like(result))
        reg = dst.tensor()
        reg.copy_(result.to(reg.dtype))
        dst.valid = dst.lanes

    def op_vf_mulscast(self, op: Op) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        s = float(self.val(op.operands[2]))
        slot = {"zero": 0, "one": 1}.get(str(self.attr(op, "layout", "zero")), 0)
        v = self._num_view(src)
        n = src.lanes
        mask = self._mask(op, n)
        active = torch.arange(n) if mask is None else torch.nonzero(mask).reshape(-1)
        out = torch.zeros(dst.lanes, dtype=dst.tensor().dtype)
        out[active * 2 + slot] = _round_half_away(v[active].double() * s, 10, 5).to(out.dtype)
        dst.tensor().copy_(out)
        dst.valid = dst.lanes

    def op_vf_histograms(self, op: Op) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        mask = self._mask(op, src.lanes)
        values = src.tensor().to(torch.int64)
        if mask is not None:
            values = values[mask]
        base = int(self.attr(op, "bin_group", 0)) * 128
        counts = torch.zeros(128, dtype=torch.int64)
        accumulate = str(self.attr(op, "mode", "frequency")) == "accumulate"
        for j in range(128):
            counts[j] = int(((values <= base + j) if accumulate else (values == base + j)).sum())
        reg = _signed_view(dst.tensor())
        reg.copy_(((reg.to(torch.int64) + counts) & 0xFFFF).to(reg.dtype))
        dst.valid = dst.lanes

    def op_vf_pack(self, op: Op) -> None:
        dst, src = self._reg(op.operands[0]), self._reg(op.operands[1])
        d_size, s_size = _esize(dst.dtype), _esize(src.dtype)
        raw = src.bytes.reshape(-1, s_size)[:, :d_size].reshape(-1)  # the low bytes of every source element
        part = str(self.attr(op, "part", "lowest"))
        out = torch.zeros(REG_BYTES, dtype=torch.uint8)
        if part == "highest":
            out[REG_BYTES // 2:] = raw
        else:
            out[: REG_BYTES // 2] = raw
        dst.bytes.copy_(out)
        dst.valid = dst.lanes

    def op_debug_print_reg(self, op: Op) -> None:
        reg = self._reg(op.operands[0])
        lanes = int(self.attr(op, "lanes", 8))
        print(f"[print_reg] {self.attr(op, 'label', '')} {op.operands[0].name} = {self._num_view(reg)[:lanes].tolist()}")

    def _error(self, msg: str, op: Op) -> Exception:
        from .interp import SimError

        return SimError(f"{msg} (#{op.id})")


__all__ = ["VfOps", "math"]
