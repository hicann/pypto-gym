# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouped CVC residual with P retained through its final vector reader."""
import ascriptor.a5 as api

TILE = 32
DIM = 128
HALF = 16
PITCH = 17

@api.vf()
def activation(p: api.Tensor, packed_h: api.Tensor):
    lo = api.Reg(api.f32)
    hi = api.Reg(api.f32)
    lo_half = api.Reg(api.f16)
    hi_half = api.Reg(api.f16)
    joined = api.Reg(api.f16)
    unused = api.Reg(api.f16)
    for row in range(HALF):
        lo <<= p[row:row + 1, 0:64]
        hi <<= p[row:row + 1, 64:DIM]
        lo <<= lo * 0.25
        hi <<= hi * 0.25
        lo <<= lo + 0.5
        hi <<= hi + 0.5
        lo <<= lo.relu()
        hi <<= hi.relu()
        lo_half <<= lo.astype(api.f16, api.CastConfig(round_mode=api.RoundMode.TO_EVEN))
        hi_half <<= hi.astype(api.f16, api.CastConfig(round_mode=api.RoundMode.TO_EVEN))
        api.deinterleave(joined, unused, lo_half, hi_half)
        api.reg_to_ub(packed_h[row * 16], joined, PITCH)
    api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

@api.vf()
def residual(p: api.Tensor, u: api.Tensor, y: api.Tensor):
    pr = api.Reg(api.f32)
    ur = api.Reg(api.f32)
    yr = api.Reg(api.f32)
    for row in range(HALF):
        for part in range(2):
            pr <<= p[row:row + 1, part * 64:part * 64 + 64]
            ur <<= u[row:row + 1, part * 64:part * 64 + 64]
            yr <<= ur + pr
            api.reg_to_ub_normal(y[row:row + 1, part * 64:part * 64 + 64], yr)


@api.kernel(mode="mix", block_dim=1)
def late_reader_serial(x: api.GM[api.f16, ("rows", DIM)],
    w1: api.GM[api.f16, (DIM, DIM)], w2: api.GM[api.f16, (DIM, DIM)],
    y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    x_l1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1, name="x_l1")
    w1_l1 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1, name="w1_l1")
    w2_l1 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1, name="w2_l1")
    p_l0c = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C, name="p_l0c")
    u_l0c = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C, name="u_l0c")
    p_ub = api.TBuff(api.f32, [HALF, DIM], api.Position.UB, name="p_ub")
    packed_h = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB, name="packed_h")
    h_l1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1, name="h_l1")
    u_ub = api.DBuff(api.f32, [HALF, DIM], api.Position.UB, name="u_ub")
    y_ub = api.DBuff(api.f32, [HALF, DIM], api.Position.UB, name="y_ub")
    p_owner = api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    h_owner = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    u_owner = api.CvMutex(2, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    count = api.Var(rows // TILE)
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        w1_l1 <<= w1
        w2_l1 <<= w2
        for tick in range(count + 0):
            # Stage 0: C1; item = tick - 0.
            if tick >= 0:
                if tick < count + 0:
                    i0 = api.Var(tick - 0)
                    row0 = api.Var(i0 * TILE)
                    p_owner.lock()
                    x_l1[i0] <<= x[row0:row0 + TILE, :]
                    api.matmul(p_l0c[i0], x_l1[i0], w1_l1.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    p_ub[i0] <<= p_l0c[i0]
                    p_owner.ready()
            # Stage 1: V1; item = tick - 0.
            if tick >= 0:
                if tick < count + 0:
                    i1 = api.Var(tick - 0)
                    row1 = api.Var(i1 * TILE)
                    p_owner.wait()
                    activation(p_ub[i1], packed_h[i1])
                    # P remains owned by vector until the delayed residual has read it.
                    h_owner.lock()
                    h_l1[i1][half_begin:half_end, :] <<= packed_h[i1][0:HALF, :].nz()
                    h_owner.ready()
            # Stage 2: C2; item = tick - 0.
            if tick >= 0:
                if tick < count + 0:
                    i2 = api.Var(tick - 0)
                    row2 = api.Var(i2 * TILE)
                    h_owner.wait()
                    api.matmul(u_l0c[i2], h_l1[i2], w2_l1.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    h_owner.free()
                    u_owner.lock()
                    u_ub[i2] <<= u_l0c[i2]
                    u_owner.ready()
            # Stage 3: V2; item = tick - 0.
            if tick >= 0:
                if tick < count + 0:
                    i3 = api.Var(tick - 0)
                    row3 = api.Var(i3 * TILE)
                    u_owner.wait()
                    residual(p_ub[i3], u_ub[i3], y_ub[i3])
                    p_owner.free()
                    u_owner.free()
                    y[row3 + half_begin:row3 + half_end, :] <<= y_ub[i3]
    return y

@api.kernel(mode="mix", block_dim=1)
def late_reader_pipeline(x: api.GM[api.f16, ("rows", DIM)],
    w1: api.GM[api.f16, (DIM, DIM)], w2: api.GM[api.f16, (DIM, DIM)],
    y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    x_l1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1, name="x_l1")
    w1_l1 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1, name="w1_l1")
    w2_l1 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1, name="w2_l1")
    p_l0c = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C, name="p_l0c")
    u_l0c = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C, name="u_l0c")
    p_ub = api.TBuff(api.f32, [HALF, DIM], api.Position.UB, name="p_ub")
    packed_h = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB, name="packed_h")
    h_l1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1, name="h_l1")
    u_ub = api.DBuff(api.f32, [HALF, DIM], api.Position.UB, name="u_ub")
    y_ub = api.DBuff(api.f32, [HALF, DIM], api.Position.UB, name="y_ub")
    p_owner = api.CvMutex(0, depth=3, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    h_owner = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    u_owner = api.CvMutex(2, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    count = api.Var(rows // TILE)
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        w1_l1 <<= w1
        w2_l1 <<= w2
        for tick in range(count + 1):
            # Stage 0: C1; item = tick - 0.
            if tick >= 0:
                if tick < count + 0:
                    i0 = api.Var(tick - 0)
                    row0 = api.Var(i0 * TILE)
                    p_owner.lock()
                    x_l1[i0] <<= x[row0:row0 + TILE, :]
                    api.matmul(p_l0c[i0], x_l1[i0], w1_l1.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    p_ub[i0] <<= p_l0c[i0]
                    p_owner.ready()
                    # Stage 1: V1; item = tick.
                    i1 = api.Var(tick)
                    row1 = api.Var(i1 * TILE)
                    p_owner.wait()
                    activation(p_ub[i1], packed_h[i1])
                    # P remains owned by vector until the delayed residual has read it.
                    h_owner.lock()
                    h_l1[i1][half_begin:half_end, :] <<= packed_h[i1][0:HALF, :].nz()
                    h_owner.ready()
            # Stage 2: C2; item = tick - 1.
            if tick >= 1:
                if tick < count + 1:
                    i2 = api.Var(tick - 1)
                    row2 = api.Var(i2 * TILE)
                    h_owner.wait()
                    api.matmul(u_l0c[i2], h_l1[i2], w2_l1.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    h_owner.free()
                    u_owner.lock()
                    u_ub[i2] <<= u_l0c[i2]
                    u_owner.ready()
                    # Stage 3: V2; item = tick - 1.
                    i3 = api.Var(tick - 1)
                    row3 = api.Var(i3 * TILE)
                    u_owner.wait()
                    residual(p_ub[i3], u_ub[i3], y_ub[i3])
                    p_owner.free()
                    u_owner.free()
                    y[row3 + half_begin:row3 + half_end, :] <<= y_ub[i3]
    return y
