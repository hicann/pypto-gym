# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F403, F405, F841
"""Reviewed production closures; host inputs/references are local and independent.

Pure-vector teaching units explicitly launch one vector participant. Original
mode=vec bodies, masks, byte footprints and overlapping-write barriers remain.
"""

from ascriptor.a5 import *


@vf()
def groups_i64_vf(x: Tensor, y: Tensor, out: Tensor):
    """Rows 0..8: add, sub, mul, bitwise and, floor remainder, masked fused op,
    maximum via compare/select, increasing ramp, decreasing ramp.
    """
    a = Reg(DT.int64, reg_num=2)
    b = Reg(DT.int64, reg_num=2)
    d = Reg(DT.int64, reg_num=2)
    m = MaskReg(DT.int64, reg_num=2)
    tail = MaskReg(DT.int64, reg_num=2)
    pred = MaskReg(DT.int64, reg_num=2)
    count = Var(47, dtype=DT.uint32)
    tail.update(count)  # Active lanes [0, 47), crossing the one-register boundary at 32.
    a <<= x[0]
    b <<= y[0]
    d <<= a + b
    out[0] <<= d
    d <<= a - b
    out[64] <<= d
    d <<= a * b
    out[128] <<= d
    vand(d, a, b, m)
    out[192] <<= d
    vmod(d, a, b, m)  # Floor remainder (the register expression is a % b).
    out[256] <<= d
    d <<= a
    muldstadd(d, b, a, tail)  # d = old_d * b + a; inactive lanes become zero.
    out[320] <<= d
    pred <<= a > b
    d <<= pred.select(a, b)
    out[384] <<= d
    d.arange(18014398509481991)  # start + lane; this base is too large for exact float64.
    out[448] <<= d
    d.arange(-18014398509481991, increase=False)
    out[512] <<= d


@kernel(mode="vec", block_dim=1)
def groups_i64(x: GM[i64, (1, 64)], y: GM[i64, (1, 64)], out: GM[i64, (9, 64)]):
    """First demo: stage two rows, compute over all 64 int64 lanes, publish nine rows."""
    ux = Tensor(DT.int64, [1, 64], Position.UB)
    uy = Tensor(DT.int64, [1, 64], Position.UB)
    uo = Tensor(DT.int64, [9, 64], Position.UB)
    with auto_sync():
        ux <<= x
        uy <<= y
        groups_i64_vf(ux, uy, uo)
        out <<= uo
    return out


@vf()
def groups_reduce_vf(x: Tensor, y: Tensor, out: Tensor):
    """Rows 0..2 hold sum/max/min in lane 0, with the other lanes zero.
    Rows 3..4 hold interleaved x/y; rows 5..6 recover x/y with aliased outputs.
    """
    a = Reg(DT.int64, reg_num=2)
    b = Reg(DT.int64, reg_num=2)
    d = Reg(DT.int64, reg_num=2)
    e = Reg(DT.int64, reg_num=2)
    m = MaskReg(DT.int64, reg_num=2)
    a <<= x[0]
    b <<= y[0]
    cadd(d, a, m)
    out[0] <<= d
    cmax(d, a, m)
    out[64] <<= d
    cmin(d, a, m)
    out[128] <<= d
    interleave(d, e, a, b)
    out[192] <<= d
    out[256] <<= e
    deinterleave(d, e, d, e)  # Outputs may alias inputs; both original halves are read.
    out[320] <<= d
    out[384] <<= e


@kernel(mode="vec", block_dim=1)
def groups_reduce(x: GM[i64, (1, 64)], y: GM[i64, (1, 64)], out: GM[i64, (7, 64)]):
    """Full-width reduction and reversible lane rearrangement of two int64 groups."""
    ux = Tensor(DT.int64, [1, 64], Position.UB)
    uy = Tensor(DT.int64, [1, 64], Position.UB)
    uo = Tensor(DT.int64, [7, 64], Position.UB)
    with auto_sync():
        ux <<= x
        uy <<= y
        groups_reduce_vf(ux, uy, uo)
        out <<= uo
    return out


@vf()
def groups_cast_vf(x: Tensor, y: Tensor, wide: Tensor, narrow: Tensor, floats: Tensor, back: Tensor):
    """Preserve 64 logical lanes while the physical width changes from 256 to 512 bytes.
    Outputs: x as i64, x narrowed back to i32, x as f32, y truncated toward zero to i64.
    """
    a = Reg(DT.int32)  # 64 lanes already fit in one register.
    b = Reg(DT.int64, reg_num=2)
    c = Reg(DT.int32)
    f = Reg(DT.float)
    g = Reg(DT.int64, reg_num=2)
    m = MaskReg(DT.int64, reg_num=2)
    cfg = CastConfig(round_mode=RoundMode.TRUNC)
    a <<= x[0]
    cast(b, a, cfg, m)
    wide[0] <<= b
    cast(c, b, cfg, m)
    narrow[0] <<= c
    cast(f, b, cfg, m)
    floats[0] <<= f
    f <<= y[0]
    cast(g, f, cfg, m)
    back[0] <<= g


@kernel(mode="vec", block_dim=1)
def groups_cast(x: GM[i32, (1, 64)], y: GM[f32, (1, 64)], wide: GM[i64, (1, 64)], narrow: GM[i32, (1, 64)],
    floats: GM[f32, (1, 64)], back: GM[i64, (1, 64)]):
    """Return four named conversion outputs: wide, narrow, floats, back."""
    ux = Tensor(DT.int32, [1, 64], Position.UB)
    uy = Tensor(DT.float, [1, 64], Position.UB)
    uw = Tensor(DT.int64, [1, 64], Position.UB)
    un = Tensor(DT.int32, [1, 64], Position.UB)
    uf = Tensor(DT.float, [1, 64], Position.UB)
    ub = Tensor(DT.int64, [1, 64], Position.UB)
    with auto_sync():
        ux <<= x
        uy <<= y
        groups_cast_vf(ux, uy, uw, un, uf, ub)
        wide <<= uw
        narrow <<= un
        floats <<= uf
        back <<= ub
    return wide, narrow, floats, back


@vf()
def groups_memory_vf(x: Tensor, out: Tensor):
    """Row 0 gathers reversed x into lanes [0, 47), zero elsewhere.
    Row 1 scatters to reversed destinations [17, 64), retaining the zero fill
    elsewhere. Row 2 is an unsigned right shift; inputs exercise the high bit.
    """
    a = Reg(DT.uint64, reg_num=2)
    d = Reg(DT.uint64, reg_num=2)
    positions = Reg(DT.int32)
    m = MaskReg(DT.uint64, reg_num=2)
    count = Var(47, dtype=DT.uint32)
    m.update(count)
    a <<= x[0]
    positions.arange(63, increase=False)  # Indices 63..0, counted in uint64 elements.
    idx = positions.reinterpret(DT.uint32)
    ub_to_reg_gather(d, x, idx, m)
    out[0] <<= d
    d.fill(0)  # Scatter does not zero untouched addresses: initialise the destination.
    out[64] <<= d
    reg_to_ub_scatter(out[64], a, idx, m)
    shiftrs(d, a, 17)
    out[128] <<= d


@kernel(mode="vec", block_dim=1)
def groups_memory(x: GM[u64, (1, 64)], out: GM[u64, (3, 64)]):
    """Gather/scatter 64 uint64 values using a single register of 64 uint32 indices."""
    ux = Tensor(DT.uint64, [1, 64], Position.UB)
    uo = Tensor(DT.uint64, [3, 64], Position.UB)
    with auto_sync():
        ux <<= x
        groups_memory_vf(ux, uo)
        out <<= uo
    return out


@vf()
def groups_complex_vf(x: Tensor, y: Tensor, out: Tensor):
    """Rows: x+y, x*y masked to [0, 95), x+(1+2j), x*y with an interleaved mask.
    The last mask selects [0, 64) and even lanes 64..126; other lanes become zero.
    """
    a = Reg(DT.complex32, reg_num=2)
    b = Reg(DT.complex32, reg_num=2)
    d = Reg(DT.complex32, reg_num=2)
    m = MaskReg(DT.complex32, reg_num=2)
    count = Var(95, dtype=DT.uint32)  # Cross the one-register complex32 boundary at 64.
    m.update(count)
    a <<= x[0]
    b <<= y[0]
    d <<= a + b
    out[0] <<= d
    d <<= (a * b) * m
    out[128] <<= d
    d <<= a + (1 + 2j)
    out[256] <<= d
    short = MaskReg(DT.complex32, init_mode=MaskType.LOWEST32, reg_num=2)
    woven = MaskReg(DT.complex32, reg_num=2)
    rest = MaskReg(DT.complex32, reg_num=2)
    mask_interleave(woven, rest, m, short)  # Weave two predicates; use the first result.
    d <<= (a * b) * woven
    out[384] <<= d


@kernel(mode="vec", block_dim=1)
def groups_complex(x: GM[DT.complex32, (1, 128)], y: GM[DT.complex32, (1, 128)],
    out: GM[DT.complex32, (4, 128)]):
    """Compute over 128 complex-half values; hardware backend support is CCE only."""
    ux = Tensor(DT.complex32, [1, 128], Position.UB)
    uy = Tensor(DT.complex32, [1, 128], Position.UB)
    uo = Tensor(DT.complex32, [4, 128], Position.UB)
    with auto_sync():
        ux <<= x
        uy <<= y
        groups_complex_vf(ux, uy, uo)
        out <<= uo
    return out


@vf()
def groups_complex64_vf(x: Tensor, y: Tensor, out: Tensor):
    """Row 0 is x+y; row 1 is x*y in lanes [0, 47), zero in the remaining lanes."""
    a = Reg(DT.complex64, reg_num=2)
    b = Reg(DT.complex64, reg_num=2)
    d = Reg(DT.complex64, reg_num=2)
    m = MaskReg(DT.complex64, reg_num=2)
    count = Var(47, dtype=DT.uint32)
    m.update(count)
    a <<= x[0]
    b <<= y[0]
    d <<= a + b
    out[0] <<= d
    d <<= (a * b) * m
    out[64] <<= d


@kernel(mode="vec", block_dim=1)
def groups_complex64(x: GM[DT.complex64, (1, 64)], y: GM[DT.complex64, (1, 64)],
    out: GM[DT.complex64, (2, 64)]):
    """Compute over 64 complex-float values; hardware backend support is CCE only."""
    ux = Tensor(DT.complex64, [1, 64], Position.UB)
    uy = Tensor(DT.complex64, [1, 64], Position.UB)
    uo = Tensor(DT.complex64, [2, 64], Position.UB)
    with auto_sync():
        ux <<= x
        uy <<= y
        groups_complex64_vf(ux, uy, uo)
        out <<= uo
    return out
