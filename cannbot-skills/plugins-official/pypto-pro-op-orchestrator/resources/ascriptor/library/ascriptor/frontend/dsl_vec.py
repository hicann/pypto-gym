# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The a2 family's tensor-vector instructions (``vec.*``): the DSL names the old ``easyasc.a2`` exported, bound to the
registry's ``vec.*`` ops (RFC-0008 §3). ``ascriptor.a2`` imports this module after :mod:`.dsl`, so on a2 the names
``cast`` / ``dup`` / ``gather`` / ``compare`` / ``select`` / ``muladddst`` mean the UB instructions, not the a5
register forms. The signatures are the old stubs': positional ``repeat`` and strides, the keyword-only ``count=`` /
``count_per_rep=`` modes (mutually exclusive); a missing ``repeat`` or stride is inferred from the view
(:mod:`.rules_vec`)."""

from __future__ import annotations

from .dsl import EnumValue, _fn

_STRIDES3 = "repeat=None, dst_blk_stride=None, src1_blk_stride=None, src2_blk_stride=None, dst_rep_stride=None, src1_rep_stride=None, src2_rep_stride=None, *, count=None, count_per_rep=None"
_STRIDES2 = "repeat=None, dst_blk_stride=None, src_blk_stride=None, dst_rep_stride=None, src_rep_stride=None, *, count=None, count_per_rep=None"
_GROUP = "repeat=None, dst_rep_stride=1, src_blk_stride=None, src_rep_stride=None, *, count_per_rep=None"


class SelectMode:
    TENSOR_TENSOR = EnumValue("SelectMode", "tensor_tensor")
    TENSOR_SCALAR = EnumValue("SelectMode", "tensor_scalar")


# binary: dst = src1 op src2
add = _fn("vec.add", "add", f"add(dst, src1, src2, {_STRIDES3})")
sub = _fn("vec.sub", "sub", f"sub(dst, src1, src2, {_STRIDES3})")
mul = _fn("vec.mul", "mul", f"mul(dst, src1, src2, {_STRIDES3})")
div = _fn("vec.div", "div", f"div(dst, src1, src2, {_STRIDES3})")
vmax = _fn("vec.max", "vmax", f"vmax(dst, src1, src2, {_STRIDES3})")
vmin = _fn("vec.min", "vmin", f"vmin(dst, src1, src2, {_STRIDES3})")
vand = _fn("vec.and", "vand", f"vand(dst, src1, src2, {_STRIDES3}): int16 / uint16")
vor = _fn("vec.or", "vor", f"vor(dst, src1, src2, {_STRIDES3}): int16 / uint16")
muladddst = _fn("vec.muladddst", "muladddst", f"muladddst(dst, src1, src2, {_STRIDES3}): dst = dst + src1 * src2")
# Unary operator form: dst = f(src)
exp = _fn("vec.exp", "exp", f"exp(dst, src, {_STRIDES2})")
ln = _fn("vec.ln", "ln", f"ln(dst, src, {_STRIDES2})")
abs = _fn("vec.abs", "abs", f"abs(dst, src, {_STRIDES2}); use builtins.abs for host scalars")  # noqa: A001
rec = _fn("vec.rec", "rec", f"rec(dst, src, {_STRIDES2}): 1 / src")
sqrt = _fn("vec.sqrt", "sqrt", f"sqrt(dst, src, {_STRIDES2})")
rsqrt = _fn("vec.rsqrt", "rsqrt", f"rsqrt(dst, src, {_STRIDES2})")
vnot = _fn("vec.not", "vnot", f"vnot(dst, src, {_STRIDES2}): int16 / uint16 bitwise not")
relu = _fn("vec.relu", "relu", f"relu(dst, src, {_STRIDES2})")
# unary with a scalar: dst = f(src, val)
adds = _fn("vec.adds", "adds", f"adds(dst, src, val, {_STRIDES2})")
muls = _fn("vec.muls", "muls", f"muls(dst, src, val, {_STRIDES2})")
shiftls = _fn("vec.shiftls", "shiftls", f"shiftls(dst, src, val, {_STRIDES2}): 16 / 32-bit integers")
shiftrs = _fn("vec.shiftrs", "shiftrs", f"shiftrs(dst, src, val, {_STRIDES2}, round_en=False): arithmetic for signed, logical for unsigned")
vmaxs = _fn("vec.maxs", "vmaxs", f"vmaxs(dst, src, val, {_STRIDES2})")
vmins = _fn("vec.mins", "vmins", f"vmins(dst, src, val, {_STRIDES2})")
lrelu = _fn("vec.lrelu", "lrelu", f"lrelu(dst, src, val, {_STRIDES2})")
axpy = _fn("vec.axpy", "axpy", f"axpy(dst, src, val, {_STRIDES2}): dst = dst + src * val")
# reductions: one value per repeat (c*) or per block (cg*), pair sums (cpadd)
cmax = _fn("vec.cmax", "cmax", f"cmax(dst, src, {_GROUP})")
cmin = _fn("vec.cmin", "cmin", f"cmin(dst, src, {_GROUP})")
cadd = _fn("vec.cadd", "cadd", f"cadd(dst, src, {_GROUP})")
cgmax = _fn("vec.cgmax", "cgmax", f"cgmax(dst, src, {_GROUP})")
cgmin = _fn("vec.cgmin", "cgmin", f"cgmin(dst, src, {_GROUP})")
cgadd = _fn("vec.cgadd", "cgadd", f"cgadd(dst, src, {_GROUP})")
cpadd = _fn("vec.cpadd", "cpadd", f"cpadd(dst, src, {_GROUP})")
# fills, broadcasts, conversions, compares, selects, gathers, transposes
dup = _fn("vec.dup", "dup", "dup(dst, value, repeat=None, dst_blk_stride=None, dst_rep_stride=None, *, count=None, count_per_rep=None)")
brcb = _fn("vec.brcb", "brcb", "brcb(dst, src, dst_blk_stride=None, dst_rep_stride=None, repeat=None): each src element fills one block of dst")
cast = _fn("vec.cast", "cast", "cast(dst, src, repeat=None, dst_blk_stride=None, src_blk_stride=None, dst_rep_stride=None, src_rep_stride=None, "
                              "round_mode=RoundMode.AWAY_FROM_ZERO, *, count=None, count_per_rep=None)")
compare = _fn("vec.compare", "compare", "compare(dst, src1, src2, mode: CompareMode, repeat=None, ...strides): packed bit mask into a uint8 tensor")
compare_scalar = _fn("vec.compare_scalar", "compare_scalar", "compare_scalar(dst, src1, scalar, mode: CompareMode, repeat=None, ...strides)")
select = _fn("vec.select", "select", "select(dst, selmask, src1, src2, mode: SelectMode, repeat=None, ...strides, tmp_addr_buf=None)")
gather = _fn("vec.gather", "gather", "gather(dst, src, offset, start_idx=0, repeat=None, dst_rep_stride=None): byte offsets, one element each")
gather_block = _fn("vec.gather_block", "gather_block", "gather_block(dst, src, offset, repeat=None, dst_blk_stride=None, dst_rep_stride=None): 32-byte blocks")
scatter = _fn("vec.scatter", "scatter", "scatter(dst, src, offset, start_idx=0, repeat=None, src_rep_stride=None)")
transdata5hd = _fn("vec.transdata5hd", "transdata5hd", "transdata5hd(dst, src, repeat=None, src_row_stride=None, dst_row_stride=None, src_rep_stride=1, "
                                                       "dst_rep_stride=16): 16x16 b16 block transposes")

__all__ = [
    'SelectMode', 'abs', 'add', 'adds', 'axpy', 'brcb',
    'cadd', 'cast', 'cgadd', 'cgmax', 'cgmin', 'cmax',
    'cmin', 'compare', 'compare_scalar', 'cpadd', 'div', 'dup',
    'exp', 'gather', 'gather_block', 'ln', 'lrelu', 'mul',
    'muladddst', 'muls', 'rec', 'relu', 'rsqrt', 'scatter',
    'select', 'shiftls', 'shiftrs', 'sqrt', 'sub', 'transdata5hd',
    'vand', 'vmax', 'vmaxs', 'vmin', 'vmins', 'vnot',
    'vor',
]
