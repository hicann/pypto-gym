# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Twenty explicit mixed C/V pipelines: five stage graphs times four schedules.

Each kernel is written out in full rather than built by a Python loop over stages. That is the
point: a stage delay, a slot index and an edge's owner are all visible in the source of the
kernel that has them, and nothing is decided at device-loop time. The cost is repetition, and
the repetition is what makes the four schedules of one graph comparable line by line:

  serial           one item finishes before the next starts
  pipeline         the next item's first stage overlaps this item's later stages
  resident_serial  operands stay on chip across items, serially
  resident         both: on-chip residency and a delayed schedule
"""
import ascriptor.a5 as api

TILE = 32
DIM = 128
HALF = 16
PITCH = 17

@api.vf()
def pre_vector(source: api.Tensor, packed: api.Tensor):
    value = api.Reg(api.f16)
    for row in range(HALF):
        value <<= source[row:row + 1, 0:DIM]
        value <<= value * 0.5
        value <<= value + 0.25
        value <<= value.relu()
        api.reg_to_ub(packed[row * 16], value, PITCH)
    api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

@api.vf()
def middle_vector(source: api.Tensor, packed: api.Tensor):
    low = api.Reg(api.f32)
    high = api.Reg(api.f32)
    low_half = api.Reg(api.f16)
    high_half = api.Reg(api.f16)
    joined = api.Reg(api.f16)
    unused = api.Reg(api.f16)
    for row in range(HALF):
        low <<= source[row:row + 1, 0:64]
        high <<= source[row:row + 1, 64:DIM]
        low <<= low * 0.5
        high <<= high * 0.5
        low <<= low + 0.25
        high <<= high + 0.25
        low <<= low.relu()
        high <<= high.relu()
        low_half <<= low.astype(api.f16)
        high_half <<= high.astype(api.f16)
        api.deinterleave(joined, unused, low_half, high_half)
        api.reg_to_ub(packed[row * 16], joined, PITCH)
    api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

@api.vf()
def final_vector(source: api.Tensor, output: api.Tensor):
    value = api.Reg(api.f32)
    for row in range(HALF):
        for part in range(2):
            value <<= source[row:row + 1, part * 64:part * 64 + 64]
            value <<= value * 0.5
            value <<= value + 0.25
            value <<= value.relu()
            api.reg_to_ub_normal(output[row:row + 1, part * 64:part * 64 + 64], value)


@api.kernel(mode="mix")
def cvc_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 0):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    weight_0[item_0] <<= w0
                    api.matmul(product_0[item_0], initial[item_0], weight_0[item_0].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    weight_2[item_2] <<= w1
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2[item_2].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    y[row_2:row_2 + TILE, :] <<= product_2[item_2]
    return y


@api.kernel(mode="mix")
def cvc_pipeline(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 1):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    weight_0[item_0] <<= w0
                    api.matmul(product_0[item_0], initial[item_0], weight_0[item_0].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    weight_2[item_2] <<= w1
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2[item_2].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    y[row_2:row_2 + TILE, :] <<= product_2[item_2]
    return y


@api.kernel(mode="mix")
def cvc_resident_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_0 <<= w0
            weight_2 <<= w1
        for tick in range(count + 0):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    api.matmul(product_0[item_0], initial[item_0], weight_0.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    y[row_2:row_2 + TILE, :] <<= product_2[item_2]
    return y


@api.kernel(mode="mix")
def cvc_resident(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_0 <<= w0
            weight_2 <<= w1
        for tick in range(count + 1):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    api.matmul(product_0[item_0], initial[item_0], weight_0.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    y[row_2:row_2 + TILE, :] <<= product_2[item_2]
    return y


@api.kernel(mode="mix")
def vcv_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [HALF, DIM], api.Position.UB)
    packed_0 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_0 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_0 = api.VcMutex(0, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_1 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_1 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_1 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_1 = api.CvMutex(1, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    final = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 0):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (V): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0 + half_begin:row_0 + half_end, :]
                    pre_vector(initial[item_0], packed_0[item_0])
                    event_0.lock()
                    edge_0[item_0][half_begin:half_end, :] <<= packed_0[item_0][0:HALF, :].nz()
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (C): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    weight_1[item_1] <<= w0
                    api.matmul(product_1[item_1], edge_0[item_1], weight_1[item_1].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1] <<= product_1[item_1]
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (V): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    final_vector(edge_1[item_2], final[item_2])
                    y[row_2 + half_begin:row_2 + half_end, :] <<= final[item_2]
                    event_1.free()
    return y


@api.kernel(mode="mix")
def vcv_pipeline(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [HALF, DIM], api.Position.UB)
    packed_0 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_0 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_0 = api.VcMutex(0, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_1 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_1 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_1 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_1 = api.CvMutex(1, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    final = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 1):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (V): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0 + half_begin:row_0 + half_end, :]
                    pre_vector(initial[item_0], packed_0[item_0])
                    event_0.lock()
                    edge_0[item_0][half_begin:half_end, :] <<= packed_0[item_0][0:HALF, :].nz()
                    event_0.ready()
                    # Stage 1 (C): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    weight_1[item_1] <<= w0
                    api.matmul(product_1[item_1], edge_0[item_1], weight_1[item_1].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1] <<= product_1[item_1]
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (V): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    final_vector(edge_1[item_2], final[item_2])
                    y[row_2 + half_begin:row_2 + half_end, :] <<= final[item_2]
                    event_1.free()
    return y


@api.kernel(mode="mix")
def vcv_resident_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [HALF, DIM], api.Position.UB)
    packed_0 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_0 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_0 = api.VcMutex(0, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_1 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_1 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_1 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_1 = api.CvMutex(1, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    final = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_1 <<= w0
        for tick in range(count + 0):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (V): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0 + half_begin:row_0 + half_end, :]
                    pre_vector(initial[item_0], packed_0[item_0])
                    event_0.lock()
                    edge_0[item_0][half_begin:half_end, :] <<= packed_0[item_0][0:HALF, :].nz()
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (C): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    api.matmul(product_1[item_1], edge_0[item_1], weight_1.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1] <<= product_1[item_1]
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (V): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    final_vector(edge_1[item_2], final[item_2])
                    y[row_2 + half_begin:row_2 + half_end, :] <<= final[item_2]
                    event_1.free()
    return y


@api.kernel(mode="mix")
def vcv_resident(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [HALF, DIM], api.Position.UB)
    packed_0 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_0 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_0 = api.VcMutex(0, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_1 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_1 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_1 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_1 = api.CvMutex(1, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    final = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_1 <<= w0
        for tick in range(count + 1):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (V): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0 + half_begin:row_0 + half_end, :]
                    pre_vector(initial[item_0], packed_0[item_0])
                    event_0.lock()
                    edge_0[item_0][half_begin:half_end, :] <<= packed_0[item_0][0:HALF, :].nz()
                    event_0.ready()
                    # Stage 1 (C): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    api.matmul(product_1[item_1], edge_0[item_1], weight_1.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1] <<= product_1[item_1]
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (V): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    final_vector(edge_1[item_2], final[item_2])
                    y[row_2 + half_begin:row_2 + half_end, :] <<= final[item_2]
                    event_1.free()
    return y


@api.kernel(mode="mix")
def cvcv_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_2 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_2 = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    final = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 0):
            # Serial control closes the last handoff before starting the next item.
            event_2.lock()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    weight_0[item_0] <<= w0
                    api.matmul(product_0[item_0], initial[item_0], weight_0[item_0].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    weight_2[item_2] <<= w1
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2[item_2].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    edge_2[item_2] <<= product_2[item_2]
                    event_2.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 3 (V): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 0)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    final_vector(edge_2[item_3], final[item_3])
                    y[row_3 + half_begin:row_3 + half_end, :] <<= final[item_3]
                    event_2.free()
    return y


@api.kernel(mode="mix")
def cvcv_pipeline(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_2 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_2 = api.CvMutex(2, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    final = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 1):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    weight_0[item_0] <<= w0
                    api.matmul(product_0[item_0], initial[item_0], weight_0[item_0].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    weight_2[item_2] <<= w1
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2[item_2].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    event_2.lock()
                    edge_2[item_2] <<= product_2[item_2]
                    event_2.ready()
                    # Stage 3 (V): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 1)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    final_vector(edge_2[item_3], final[item_3])
                    y[row_3 + half_begin:row_3 + half_end, :] <<= final[item_3]
                    event_2.free()
    return y


@api.kernel(mode="mix")
def cvcv_resident_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_2 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_2 = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    final = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_0 <<= w0
            weight_2 <<= w1
        for tick in range(count + 0):
            # Serial control closes the last handoff before starting the next item.
            event_2.lock()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    api.matmul(product_0[item_0], initial[item_0], weight_0.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    edge_2[item_2] <<= product_2[item_2]
                    event_2.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 3 (V): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 0)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    final_vector(edge_2[item_3], final[item_3])
                    y[row_3 + half_begin:row_3 + half_end, :] <<= final[item_3]
                    event_2.free()
    return y


@api.kernel(mode="mix")
def cvcv_resident(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_2 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_2 = api.CvMutex(2, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    final = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_0 <<= w0
            weight_2 <<= w1
        for tick in range(count + 1):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    api.matmul(product_0[item_0], initial[item_0], weight_0.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    event_2.lock()
                    edge_2[item_2] <<= product_2[item_2]
                    event_2.ready()
                    # Stage 3 (V): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 1)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    final_vector(edge_2[item_3], final[item_3])
                    y[row_3 + half_begin:row_3 + half_end, :] <<= final[item_3]
                    event_2.free()
    return y


@api.kernel(mode="mix")
def vcvc_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [HALF, DIM], api.Position.UB)
    packed_0 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_0 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_0 = api.VcMutex(0, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_1 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_1 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_1 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_1 = api.CvMutex(1, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_2 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_2 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_2 = api.VcMutex(2, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.FIX)
    weight_3 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_3 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 0):
            # Serial control closes the last handoff before starting the next item.
            event_2.lock()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (V): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0 + half_begin:row_0 + half_end, :]
                    pre_vector(initial[item_0], packed_0[item_0])
                    event_0.lock()
                    edge_0[item_0][half_begin:half_end, :] <<= packed_0[item_0][0:HALF, :].nz()
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (C): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    weight_1[item_1] <<= w0
                    api.matmul(product_1[item_1], edge_0[item_1], weight_1[item_1].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1] <<= product_1[item_1]
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (V): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    middle_vector(edge_1[item_2], packed_2[item_2])
                    event_1.free()
                    edge_2[item_2][half_begin:half_end, :] <<= packed_2[item_2][0:HALF, :].nz()
                    event_2.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 3 (C): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 0)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    weight_3[item_3] <<= w1
                    api.matmul(product_3[item_3], edge_2[item_3], weight_3[item_3].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_2.free()
                    y[row_3:row_3 + TILE, :] <<= product_3[item_3]
    return y


@api.kernel(mode="mix")
def vcvc_pipeline(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [HALF, DIM], api.Position.UB)
    packed_0 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_0 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_0 = api.VcMutex(0, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_1 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_1 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_1 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_1 = api.CvMutex(1, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_2 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_2 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_2 = api.VcMutex(2, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_3 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_3 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 1):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (V): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0 + half_begin:row_0 + half_end, :]
                    pre_vector(initial[item_0], packed_0[item_0])
                    event_0.lock()
                    edge_0[item_0][half_begin:half_end, :] <<= packed_0[item_0][0:HALF, :].nz()
                    event_0.ready()
                    # Stage 1 (C): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    weight_1[item_1] <<= w0
                    api.matmul(product_1[item_1], edge_0[item_1], weight_1[item_1].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1] <<= product_1[item_1]
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (V): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    middle_vector(edge_1[item_2], packed_2[item_2])
                    event_1.free()
                    event_2.lock()
                    edge_2[item_2][half_begin:half_end, :] <<= packed_2[item_2][0:HALF, :].nz()
                    event_2.ready()
                    # Stage 3 (C): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 1)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    weight_3[item_3] <<= w1
                    api.matmul(product_3[item_3], edge_2[item_3], weight_3[item_3].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_2.free()
                    y[row_3:row_3 + TILE, :] <<= product_3[item_3]
    return y


@api.kernel(mode="mix")
def vcvc_resident_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [HALF, DIM], api.Position.UB)
    packed_0 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_0 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_0 = api.VcMutex(0, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_1 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_1 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_1 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_1 = api.CvMutex(1, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_2 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_2 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_2 = api.VcMutex(2, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.FIX)
    weight_3 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_3 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_1 <<= w0
            weight_3 <<= w1
        for tick in range(count + 0):
            # Serial control closes the last handoff before starting the next item.
            event_2.lock()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (V): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0 + half_begin:row_0 + half_end, :]
                    pre_vector(initial[item_0], packed_0[item_0])
                    event_0.lock()
                    edge_0[item_0][half_begin:half_end, :] <<= packed_0[item_0][0:HALF, :].nz()
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (C): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    api.matmul(product_1[item_1], edge_0[item_1], weight_1.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1] <<= product_1[item_1]
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (V): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    middle_vector(edge_1[item_2], packed_2[item_2])
                    event_1.free()
                    edge_2[item_2][half_begin:half_end, :] <<= packed_2[item_2][0:HALF, :].nz()
                    event_2.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 3 (C): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 0)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    api.matmul(product_3[item_3], edge_2[item_3], weight_3.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_2.free()
                    y[row_3:row_3 + TILE, :] <<= product_3[item_3]
    return y


@api.kernel(mode="mix")
def vcvc_resident(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [HALF, DIM], api.Position.UB)
    packed_0 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_0 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_0 = api.VcMutex(0, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_1 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_1 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_1 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_1 = api.CvMutex(1, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_2 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_2 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_2 = api.VcMutex(2, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_3 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_3 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_1 <<= w0
            weight_3 <<= w1
        for tick in range(count + 1):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (V): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0 + half_begin:row_0 + half_end, :]
                    pre_vector(initial[item_0], packed_0[item_0])
                    event_0.lock()
                    edge_0[item_0][half_begin:half_end, :] <<= packed_0[item_0][0:HALF, :].nz()
                    event_0.ready()
                    # Stage 1 (C): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    api.matmul(product_1[item_1], edge_0[item_1], weight_1.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1] <<= product_1[item_1]
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (V): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    middle_vector(edge_1[item_2], packed_2[item_2])
                    event_1.free()
                    event_2.lock()
                    edge_2[item_2][half_begin:half_end, :] <<= packed_2[item_2][0:HALF, :].nz()
                    event_2.ready()
                    # Stage 3 (C): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 1)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    api.matmul(product_3[item_3], edge_2[item_3], weight_3.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_2.free()
                    y[row_3:row_3 + TILE, :] <<= product_3[item_3]
    return y


@api.kernel(mode="mix")
def cvcvc_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_2 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_2 = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_3 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_3 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_3 = api.VcMutex(3, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_4 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_4 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 0):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    weight_0[item_0] <<= w0
                    api.matmul(product_0[item_0], initial[item_0], weight_0[item_0].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    weight_2[item_2] <<= w1
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2[item_2].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    event_2.lock()
                    edge_2[item_2] <<= product_2[item_2]
                    event_2.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 3 (V): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 0)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    middle_vector(edge_2[item_3], packed_3[item_3])
                    event_2.free()
                    event_3.lock()
                    edge_3[item_3][half_begin:half_end, :] <<= packed_3[item_3][0:HALF, :].nz()
                    event_3.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 4 (C): its own work item, never the current producer index.
                    item_4 = api.Var(tick - 0)
                    row_4 = api.Var((first + item_4) * TILE)
                    event_3.wait()
                    weight_4[item_4] <<= w2
                    api.matmul(product_4[item_4], edge_3[item_4], weight_4[item_4].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_3.free()
                    y[row_4:row_4 + TILE, :] <<= product_4[item_4]
    return y


@api.kernel(mode="mix")
def cvcvc_pipeline(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_2 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_2 = api.CvMutex(2, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_3 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_3 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_3 = api.VcMutex(3, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_4 = api.DBuff(api.f16, [DIM, DIM], api.Position.L1)
    product_4 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        for tick in range(count + 2):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    weight_0[item_0] <<= w0
                    api.matmul(product_0[item_0], initial[item_0], weight_0[item_0].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    weight_2[item_2] <<= w1
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2[item_2].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    event_2.lock()
                    edge_2[item_2] <<= product_2[item_2]
                    event_2.ready()
                    # Stage 3 (V): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 1)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    middle_vector(edge_2[item_3], packed_3[item_3])
                    event_2.free()
                    event_3.lock()
                    edge_3[item_3][half_begin:half_end, :] <<= packed_3[item_3][0:HALF, :].nz()
                    event_3.ready()
            if tick >= 2:
                if tick < count + 2:
                    # Stage 4 (C): its own work item, never the current producer index.
                    item_4 = api.Var(tick - 2)
                    row_4 = api.Var((first + item_4) * TILE)
                    event_3.wait()
                    weight_4[item_4] <<= w2
                    api.matmul(product_4[item_4], edge_3[item_4], weight_4[item_4].T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_3.free()
                    y[row_4:row_4 + TILE, :] <<= product_4[item_4]
    return y


@api.kernel(mode="mix")
def cvcvc_resident_serial(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_2 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_2 = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_3 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_3 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_3 = api.VcMutex(3, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_4 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_4 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_0 <<= w0
            weight_2 <<= w1
            weight_4 <<= w2
        for tick in range(count + 0):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    api.matmul(product_0[item_0], initial[item_0], weight_0.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 0)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    event_2.lock()
                    edge_2[item_2] <<= product_2[item_2]
                    event_2.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 3 (V): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 0)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    middle_vector(edge_2[item_3], packed_3[item_3])
                    event_2.free()
                    event_3.lock()
                    edge_3[item_3][half_begin:half_end, :] <<= packed_3[item_3][0:HALF, :].nz()
                    event_3.ready()
            if tick >= 0:
                if tick < count + 0:
                    # Stage 4 (C): its own work item, never the current producer index.
                    item_4 = api.Var(tick - 0)
                    row_4 = api.Var((first + item_4) * TILE)
                    event_3.wait()
                    api.matmul(product_4[item_4], edge_3[item_4], weight_4.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_3.free()
                    y[row_4:row_4 + TILE, :] <<= product_4[item_4]
    return y


@api.kernel(mode="mix")
def cvcvc_resident(x: api.GM[api.f16, ("rows", DIM)],
    w0: api.GM[api.f16, (DIM, DIM)], w1: api.GM[api.f16, (DIM, DIM)],
    w2: api.GM[api.f16, (DIM, DIM)], y: api.GM[api.f32, ("rows", DIM)], rows: api.i32):
    initial = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    weight_0 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_0 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_0 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_0 = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_1 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_1 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_1 = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_2 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_2 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    edge_2 = api.DBuff(api.f32, [HALF, DIM], api.Position.UB)
    event_2 = api.CvMutex(2, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    packed_3 = api.DBuff(api.f16, [PITCH, DIM], api.Position.UB)
    edge_3 = api.DBuff(api.f16, [TILE, DIM], api.Position.L1)
    event_3 = api.VcMutex(3, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
    weight_4 = api.Tensor(api.f16, [DIM, DIM], api.Position.L1)
    product_4 = api.DBuff(api.f32, [TILE, DIM], api.Position.L0C)
    items = api.Var(rows // TILE)
    per_core = api.Var(api.CeilDiv(items, api.GetCubeNum()))
    first = api.Var(per_core * api.GetCubeIdx())
    last = api.Var(api.Min(first + per_core, items))
    count = api.Var(api.Max(last - first, 0))
    half_begin = api.Var(api.GetSubBlockIdx() * HALF)
    half_end = api.Var(half_begin + HALF)
    with api.auto_sync():
        if count > 0:
            weight_0 <<= w0
            weight_2 <<= w1
            weight_4 <<= w2
        for tick in range(count + 2):
            if tick >= 0:
                if tick < count + 0:
                    # Stage 0 (C): its own work item, never the current producer index.
                    item_0 = api.Var(tick - 0)
                    row_0 = api.Var((first + item_0) * TILE)
                    initial[item_0] <<= x[row_0:row_0 + TILE, :]
                    api.matmul(product_0[item_0], initial[item_0], weight_0.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_0.lock()
                    edge_0[item_0] <<= product_0[item_0]
                    event_0.ready()
                    # Stage 1 (V): its own work item, never the current producer index.
                    item_1 = api.Var(tick - 0)
                    row_1 = api.Var((first + item_1) * TILE)
                    event_0.wait()
                    middle_vector(edge_0[item_1], packed_1[item_1])
                    event_0.free()
                    event_1.lock()
                    edge_1[item_1][half_begin:half_end, :] <<= packed_1[item_1][0:HALF, :].nz()
                    event_1.ready()
            if tick >= 1:
                if tick < count + 1:
                    # Stage 2 (C): its own work item, never the current producer index.
                    item_2 = api.Var(tick - 1)
                    row_2 = api.Var((first + item_2) * TILE)
                    event_1.wait()
                    api.matmul(product_2[item_2], edge_1[item_2], weight_2.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_1.free()
                    event_2.lock()
                    edge_2[item_2] <<= product_2[item_2]
                    event_2.ready()
                    # Stage 3 (V): its own work item, never the current producer index.
                    item_3 = api.Var(tick - 1)
                    row_3 = api.Var((first + item_3) * TILE)
                    event_2.wait()
                    middle_vector(edge_2[item_3], packed_3[item_3])
                    event_2.free()
                    event_3.lock()
                    edge_3[item_3][half_begin:half_end, :] <<= packed_3[item_3][0:HALF, :].nz()
                    event_3.ready()
            if tick >= 2:
                if tick < count + 2:
                    # Stage 4 (C): its own work item, never the current producer index.
                    item_4 = api.Var(tick - 2)
                    row_4 = api.Var((first + item_4) * TILE)
                    event_3.wait()
                    api.matmul(product_4[item_4], edge_3[item_4], weight_4.T, m=TILE, n=DIM, k=DIM, splitn=DIM)
                    event_3.free()
                    y[row_4:row_4 + TILE, :] <<= product_4[item_4]
    return y
