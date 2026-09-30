# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three standard MHA schedules with independent per-head keys and values."""

import math
from functools import lru_cache

import ascriptor.a5 as api

# ----------------------------------------------------------------------------------------------------
# partition.py
# Host-only deterministic continuous partition from declared source work.
#
# The private helper uses no profile input. Its table is elaborated as literals;
# it is neither a GM allocation nor a kernel argument.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def weighted_partition(batch, queries, keys, heads, row_block, key_tile, causal, grid=28):
    groups = (queries + row_block - 1) // row_block
    items = batch * heads * groups
    if items < grid:
        return None
    assert row_block > 0 and key_tile > 0 and key_tile % 2 == 0 and keys % key_tile == 0
    quantum = key_tile // 2
    per_head = tuple(((keys - queries + min((group + 1) * row_block, queries) + quantum - 1) // quantum)
                     if causal else keys // quantum for group in range(groups))
    weights = per_head * (batch * heads)
    prefix = [0]
    for value in weights:
        assert value > 0
        prefix.append(prefix[-1] + value)
    total = prefix[-1]
    max_items = (items + grid - 1) // grid + 1

    @lru_cache(maxsize=None)
    def cache_misses(begin, end):
        # K and V independently visit this same head-tagged sequence. Every
        # miss requests a complete key_tile x dim rectangle; the factor of two
        # for separate K/V inputs does not affect this tie-break.
        tags, misses = [None] * 3, 0
        for item in range(begin, end):
            owner = item // groups
            for tile in range((weights[item] + 1) // 2):
                tag = (owner, tile)
                slot = tile % 3
                if tags[slot] != tag:
                    misses += 1
                    tags[slot] = tag
        return misses

    def solve(minimum, maximum, optimize=False):
        states = {0: ((0, 0, 0), (0,))}
        for used in range(1, grid + 1):
            remaining, next_states = grid - used, {}
            for begin, (score, cuts) in states.items():
                for end in range(begin + 1, min(items - remaining, begin + max_items) + 1):
                    work = prefix[end] - prefix[begin]
                    if work > maximum:
                        break
                    if work < minimum or not remaining * minimum <= total - prefix[end] <= remaining * maximum:
                        continue
                    extra = ((grid * work - total) ** 2, cache_misses(begin, end), (end - begin) ** 2) if optimize else (0, 0, 0)
                    candidate = (tuple(a + b for a, b in zip(score, extra)), (*cuts, end))
                    if end not in next_states or candidate < next_states[end]:
                        next_states[end] = candidate
            states = next_states
            if not states:
                return None
        return states.get(items)

    floor = [items * core // grid for core in range(grid + 1)]
    low = max(max(weights), (total + grid - 1) // grid)
    high = max(prefix[end] - prefix[begin] for begin, end in zip(floor, floor[1:]))
    while low < high:
        middle = (low + high) // 2
        if solve(1, middle) is None:
            low = middle + 1
        else:
            high = middle
    maximum = low
    low, high, minimum = 1, total // grid, 1
    while low <= high:
        middle = (low + high) // 2
        if solve(middle, maximum) is None:
            high = middle - 1
        else:
            minimum, low = middle, middle + 1
    result = solve(minimum, maximum, optimize=True)
    assert result is not None
    cuts = result[1]
    assert cuts[0] == 0 and cuts[-1] == items and len(cuts) == grid + 1
    assert all(0 < end - begin <= max_items for begin, end in zip(cuts, cuts[1:]))
    return cuts

# ----------------------------------------------------------------------------------------------------
# resident.py
# Head-local single-KV MHA with a two-item probability pipeline.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def make_resident_kernel(dtype_name, layout, batch, queries, keys, heads, dim, causal, scale_value,
                row_block=128):
    if dtype_name not in ("float16", "bfloat16") or layout != "BSND" or dim != 128 or keys != 128:
        raise ValueError("Resident MHA requires FP16/BF16 BSND with D128 and KV128")
    if min(batch, queries, heads) <= 0 or (causal and queries > keys) or row_block not in (16, 32, 64, 128):
        raise ValueError("Positive shapes, visible causal rows and M16/32/64/128 are required")
    dtype = api.f16 if dtype_name == "float16" else api.bf16
    scale = scale_value if scale_value > 0 else 1 / math.sqrt(dim)
    half, pitch = row_block // 2, row_block // 2 + 1
    groups = (queries + row_block - 1) // row_block
    items = batch * heads * groups

    @api.vf()
    def softmax(score: api.Tensor, probability: api.Tensor, denominator: api.Tensor,
                rows: api.Var, first_query: api.Var):
        first = api.Reg(api.f32)
        second = api.Reg(api.f32)
        maximum = api.Reg(api.f32)
        partial = api.Reg(api.f32)
        total = api.Reg(api.f32)
        first_half = api.Reg(dtype)
        second_half = api.Reg(dtype)
        packed = api.Reg(dtype)
        discarded = api.Reg(dtype)
        if causal:
            index = api.Reg(api.i32)
            valid = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
            negative = api.Reg(api.f32)
            zero = api.Reg(api.f32)
            index.arange(0)
            negative <<= -99999.0
            zero <<= 0.0
        for row in range(rows):
            first <<= score[row * keys]
            second <<= score[row * keys + 64]
            first <<= first * scale
            second <<= second * scale
            if causal:
                allowed = api.Var(keys - queries + first_query + row + 1)
                api.compare(valid, index, allowed, api.CompareMode.LT)
                api.select(first, first, negative, mask=valid)
                api.compare(valid, index, allowed - 64, api.CompareMode.LT)
                api.select(second, second, negative, mask=valid)
            maximum <<= first.cmax()
            partial <<= second.cmax()
            maximum <<= maximum.vmax(partial)
            maximum <<= maximum.dup()
            first <<= first - maximum
            second <<= second - maximum
            first <<= first.exp()
            second <<= second.exp()
            if causal:
                api.compare(valid, index, allowed, api.CompareMode.LT)
                api.select(first, first, zero, mask=valid)
                api.compare(valid, index, allowed - 64, api.CompareMode.LT)
                api.select(second, second, zero, mask=valid)
            total <<= first.cadd()
            partial <<= second.cadd()
            total <<= total + partial
            denominator[row] <<= total.single_value()
            first_half <<= first.astype(dtype)
            second_half <<= second.astype(dtype)
            api.deinterleave(packed, discarded, first_half, second_half)
            api.reg_to_ub(probability[row * 16], packed, pitch)
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def finish(product: api.Tensor, denominator: api.Tensor, output: api.Tensor, rows: api.Var):
        value = api.Reg(api.f32)
        total = api.Reg(api.f32)
        narrowed = api.Reg(dtype)
        for row in range(rows):
            total <<= denominator[row].single()
            for chunk in api.unroll(dim // 64):
                value <<= product[row * dim + chunk * 64]
                value <<= value / total
                narrowed <<= value.astype(dtype)
                api.reg_to_ub_downsample(output[row * dim + chunk * 64], narrowed)

    @api.kernel(mode="mix")
    def mha_resident_pipeline(query: api.GM[dtype, (batch, queries, heads, dim)],
                              key: api.GM[dtype, (batch, keys, heads, dim)],
                              value: api.GM[dtype, (batch, keys, heads, dim)],
                              output: api.GM[dtype, (batch, queries, heads, dim)]):
        q = api.Tensor(dtype, [row_block, dim], api.Position.L1)
        k = api.Tensor(dtype, [keys, dim], api.Position.L1)
        values = api.DBuff(dtype, [keys, dim], api.Position.L1)
        probability = api.DBuff(dtype, [row_block, keys], api.Position.L1)
        score = api.DBuff(api.f32, [row_block, keys], api.Position.L0C)
        product = api.Tensor(api.f32, [row_block, dim], api.Position.L0C)
        ub_score = api.DBuff(api.f32, [half, keys], api.Position.UB)
        ub_product = api.Tensor(api.f32, [half, dim], api.Position.UB)
        ub_probability = api.DBuff(dtype, [pitch, keys], api.Position.UB)
        denominator = api.DBuff(api.f32, [1, 64], api.Position.UB)
        ub_output = api.Tensor(dtype, [half, dim], api.Position.UB)
        qk = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        publish = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
        pv = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        begin = api.Var(items * api.GetCubeIdx() // api.GetCubeNum())
        end = api.Var(items * (api.GetCubeIdx() + 1) // api.GetCubeNum())
        lane_begin = api.Var(api.GetSubBlockIdx() * half)
        k_owner = api.Var(-1)
        v_owners = [api.Var(-1) for _ in range(2)]
        with api.auto_sync():
            for tick in range(end - begin + 1):
                # The drain computes scalar metadata only; all current-item
                # memory accesses remain under the explicit producer guard.
                producer = api.Var(tick)
                load_item = api.Var(begin + tick)
                load_owner = api.Var(load_item // groups)
                load_batch = api.Var(load_owner // heads)
                load_head = api.Var(load_owner % heads)
                first_query = api.Var((load_item % groups) * row_block)
                rows = api.Var(api.Min(row_block, queries - first_query))
                local_rows = api.Var(api.Max(0, api.Min(half, rows - lane_begin)))
                if tick < end - begin:
                    if k_owner != load_owner:
                        k <<= key.view([keys, dim], [heads * dim, 1],
                                       (load_batch * keys * heads + load_head) * dim)
                        k_owner.set(load_owner)
                    for value_slot in api.unroll(2):
                        if producer % 2 == value_slot:
                            if v_owners[value_slot] != load_owner:
                                values[value_slot] <<= value.view([keys, dim], [heads * dim, 1],
                                                                  (load_batch * keys * heads + load_head) * dim)
                                v_owners[value_slot].set(load_owner)
                    if rows < row_block:
                        api.set_constant_to_l1(q, 0)
                        api.set_constant_to_l1(probability[producer], 0)
                        api.bar_all()
                    q[:rows, :] <<= query.view([rows, dim], [heads * dim, 1],
                                              ((load_batch * queries + first_query) * heads + load_head) * dim)
                if tick < end - begin:
                    api.matmul(score[producer], q, k, m=row_block, n=keys, k=dim)
                    qk.lock()
                    ub_score[producer] <<= score[producer]
                    qk.ready()
                    qk.wait()
                    if local_rows > 0:
                        softmax(ub_score[producer], ub_probability[producer], denominator[producer],
                                local_rows, first_query + lane_begin)
                    qk.free()
                    publish.lock()
                    if local_rows > 0:
                        probability[producer][lane_begin:lane_begin + local_rows, :] <<= ub_probability[producer][:local_rows, :].nz()
                    publish.ready()
                if tick > 0:
                    consumer = api.Var(tick - 1)
                    store_item = api.Var(begin + tick - 1)
                    store_owner = api.Var(store_item // groups)
                    store_batch = api.Var(store_owner // heads)
                    store_head = api.Var(store_owner % heads)
                    store_query = api.Var((store_item % groups) * row_block)
                    store_rows = api.Var(api.Min(row_block, queries - store_query))
                    store_local_rows = api.Var(api.Max(0, api.Min(half, store_rows - lane_begin)))
                    publish.wait()
                    api.matmul(product, probability[consumer], values[consumer].T,
                               m=row_block, n=dim, k=keys)
                    publish.free()
                    pv.lock()
                    ub_product <<= product
                    pv.ready()
                    pv.wait()
                    if store_local_rows > 0:
                        finish(ub_product, denominator[consumer], ub_output, store_local_rows)
                        destination = output.view([store_local_rows, dim], [heads * dim, 1],
                                                  ((store_batch * queries + store_query + lane_begin) * heads + store_head) * dim)
                        destination <<= ub_output[:store_local_rows, :]
                    # Product is read by V; the separate output UB allocation
                    # retains its own MTE3-to-next-V overwrite dependency.
                    pv.free()
        return output

    return mha_resident_pipeline

# ----------------------------------------------------------------------------------------------------
# preload.py
# M128/N256 query carry with half last tiles only when all later keys are masked.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def make_preload_kernel(dtype_name, layout, batch, queries, keys, heads, dim, causal, scale_value,
                row_block=128, key_tile=256):
    if dtype_name not in ("float16", "bfloat16") or layout != "BSND" or dim != 128:
        raise ValueError("Preload MHA requires FP16/BF16 BSND D128")
    if min(batch, queries, keys, heads) <= 0 or causal and queries > keys:
        raise ValueError("Positive dimensions and visible causal rows are required")
    if row_block != 128 or key_tile != 256 or keys % key_tile:
        raise ValueError("M128 and complete KV256 tiles are required")
    if row_block * (key_tile + dim) * 4 > 256 * 1024:
        raise ValueError("Score and product exceed L0C capacity")
    dtype = api.f16 if dtype_name == "float16" else api.bf16
    scale = scale_value if scale_value > 0 else 1 / math.sqrt(dim)
    half, pitch = row_block // 2, key_tile + 1
    state_columns = ((half + 63) // 64) * 64
    chunks = key_tile // 64
    groups = (queries + row_block - 1) // row_block
    items = batch * heads * groups
    operand_split = 32768 // (key_tile * 2)
    pv_split = 32768 // (row_block * 2)
    k_cache_slots = 3
    v_cache_slots = 3
    key_tiles = keys // key_tile
    planned_cuts = weighted_partition(batch, queries, keys, heads, row_block, key_tile, causal)

    @api.vf()
    def initialize(total: api.Tensor, maximum: api.Tensor):
        value = api.Reg(api.f32)
        value <<= 0.0
        for chunk in api.unroll(state_columns // 64):
            total[chunk * 64] <<= value
        value <<= -99999.0
        for chunk in api.unroll(state_columns // 64):
            maximum[chunk * 64] <<= value
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    def make_softmax(first_tile, apply_mask):
        @api.vf()
        def softmax(score: api.Tensor, total: api.Tensor, maximum: api.Tensor,
                    rescale: api.Tensor, probability: api.Tensor, rows: api.Var,
                    first_query: api.Var, first_key: api.Var, key_rows: api.Var):
            value = api.RegList(api.f32, 2)
            old_max = api.Reg(api.f32)
            old_sum = api.Reg(api.f32)
            new_max = api.Reg(api.f32)
            new_sum = api.Reg(api.f32)
            correction = api.Reg(api.f32)
            narrowed = api.Reg(dtype)
            zero_half = api.Reg(dtype)
            packed = api.Reg(dtype)
            discarded = api.Reg(dtype)
            output_half = api.MaskReg(dtype, init_mode=api.MaskType.LOWHALF)
            zero_half <<= 0.0
            if causal and apply_mask or queries % row_block:
                indices = api.Reg(api.i32)
                indices.arange(0)
                valid = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
                negative = api.Reg(api.f32)
                zero = api.Reg(api.f32)
                negative <<= -99999.0
                zero <<= 0.0
            if causal and apply_mask:
                visible = api.Reg(api.i32)
                visible <<= indices + (keys - queries + first_query)
            if queries % row_block:
                query_valid = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
                api.compare(query_valid, indices, rows, api.CompareMode.LT)
            new_max <<= -99999.0
            for key_pair in range(key_rows // 2):
                for bank in api.unroll(2):
                    value[bank] <<= score[(key_pair * 2 + bank) * half]
                for bank in api.unroll(2):
                    value[bank] <<= value[bank] * scale
                for bank in api.unroll(2):
                    if causal and apply_mask:
                        api.compare(valid, visible, first_key + key_pair * 2 + bank, api.CompareMode.GE)
                        api.select(value[bank], value[bank], negative, mask=valid)
                    new_max <<= new_max.vmax(value[bank])
            if first_tile:
                correction <<= 0.0
            else:
                old_max <<= maximum[0]
                new_max <<= new_max.vmax(old_max)
                correction <<= old_max - new_max
                correction <<= correction.exp()
            new_sum <<= 0.0
            for key_pair in range(key_rows // 2):
                for bank in api.unroll(2):
                    value[bank] <<= score[(key_pair * 2 + bank) * half]
                for bank in api.unroll(2):
                    value[bank] <<= value[bank] * scale
                for bank in api.unroll(2):
                    value[bank] <<= value[bank] - new_max
                for bank in api.unroll(2):
                    value[bank] <<= value[bank].exp()
                for bank in api.unroll(2):
                    if causal and apply_mask:
                        api.compare(valid, visible, first_key + key_pair * 2 + bank, api.CompareMode.GE)
                        api.select(value[bank], value[bank], zero, mask=valid)
                    if queries % row_block:
                        api.select(value[bank], value[bank], zero, mask=query_valid)
                    new_sum <<= new_sum + value[bank]
                    narrowed <<= value[bank].astype(dtype)
                    api.deinterleave(packed, discarded, narrowed, zero_half)
                    api.reg_to_ub(probability[(key_pair * 2 + bank) * 16], packed, pitch, mask=output_half)
            if not first_tile:
                old_sum <<= total[0]
                new_sum <<= new_sum + old_sum * correction
            maximum[0] <<= new_max
            total[0] <<= new_sum
            rescale[0] <<= correction
            api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)
        return softmax

    first_masked = make_softmax(True, True)
    next_masked = make_softmax(False, True)
    first_unmasked = make_softmax(True, False)
    next_unmasked = make_softmax(False, False)

    def make_accumulate(first_tile):
        @api.vf()
        def accumulate(accumulator: api.Tensor, product: api.Tensor, rescale: api.Tensor, rows: api.Var):
            acc = api.Reg(api.f32)
            current = api.Reg(api.f32)
            correction = api.Reg(api.f32)
            for row in range(rows):
                if not first_tile:
                    correction <<= rescale[row].single()
                for chunk in api.unroll(dim // 64):
                    current <<= product[row * dim + chunk * 64]
                    if first_tile:
                        acc <<= current
                    else:
                        acc <<= accumulator[row * dim + chunk * 64]
                        acc <<= acc * correction + current
                    accumulator[row * dim + chunk * 64] <<= acc
            api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)
        return accumulate

    first_product = make_accumulate(True)
    next_product = make_accumulate(False)

    @api.vf()
    def finish(accumulator: api.Tensor, total: api.Tensor, output: api.Tensor, rows: api.Var):
        value = api.Reg(api.f32)
        denominator = api.Reg(api.f32)
        narrowed = api.Reg(dtype)
        for row in range(rows):
            denominator <<= total[row].single()
            for chunk in api.unroll(dim // 64):
                value <<= accumulator[row * dim + chunk * 64]
                value <<= value / denominator
                narrowed <<= value.astype(dtype)
                # The independent b16 output has its own dense row stride.
                api.reg_to_ub_downsample(output[row * dim + chunk * 64], narrowed)

    @api.kernel(mode="mix")
    def mha_preload(query: api.GM[dtype, (batch, queries, heads, dim)],
                    key: api.GM[dtype, (batch, keys, heads, dim)],
                    value: api.GM[dtype, (batch, keys, heads, dim)],
                    output: api.GM[dtype, (batch, queries, heads, dim)]):
        q = api.DBuff(dtype, [row_block, dim], api.Position.L1)
        k = api.TBuff(dtype, [key_tile, dim], api.Position.L1)
        values = api.TBuff(dtype, [key_tile, dim], api.Position.L1)
        p = api.Tensor(dtype, [key_tile, row_block], api.Position.L1)
        score = api.Tensor(api.f32, [key_tile, row_block], api.Position.L0C)
        score_half = api.Tensor(api.f32, [key_tile // 2, row_block], api.Position.L0C)
        product = api.Tensor(api.f32, [row_block, dim], api.Position.L0C)
        ub_score = api.DBuff(api.f32, [key_tile, half], api.Position.UB)
        ub_product = api.Tensor(api.f32, [half, dim], api.Position.UB)
        ub_p = api.Tensor(dtype, [pitch, half], api.Position.UB)
        total = api.DBuff(api.f32, [1, state_columns], api.Position.UB)
        maximum = api.DBuff(api.f32, [1, state_columns], api.Position.UB)
        rescale = api.DBuff(api.f32, [1, state_columns], api.Position.UB)
        # Consumers finish each query in order, so only one numerator is live.
        accumulator = api.Tensor(api.f32, [half, dim], api.Position.UB)
        ub_output = api.Tensor(dtype, [half, dim], api.Position.UB)
        qk = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        publish = api.VcMutex(1, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
        pv = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        q_available = api.SEvent(api.Pipe.MTE1, api.Pipe.MTE2)
        q_ready = api.SEvent(api.Pipe.MTE2, api.Pipe.MTE1)
        k_ready = api.SEvent(api.Pipe.MTE2, api.Pipe.MTE1)
        k_available = api.SEvent(api.Pipe.MTE1, api.Pipe.MTE2)
        v_ready = api.SEvent(api.Pipe.MTE2, api.Pipe.MTE1, preset=True)
        v_available = api.SEvent(api.Pipe.MTE1, api.Pipe.MTE2)
        begin = api.Var(items * api.GetCubeIdx() // api.GetCubeNum())
        end = api.Var(items * (api.GetCubeIdx() + 1) // api.GetCubeNum())
        if planned_cuts is not None:
            if api.GetCubeNum() == 28:
                for dispatch_core in api.unroll(28):
                    if api.GetCubeIdx() == dispatch_core:
                        begin.set(planned_cuts[dispatch_core])
                        end.set(planned_cuts[dispatch_core + 1])
        lane_begin = api.Var(api.GetSubBlockIdx() * half)
        k_cache_tags = [api.Var(-1) for _ in range(k_cache_slots)]
        v_cache_tags = [api.Var(-1) for _ in range(v_cache_slots)]
        # Count the complete causal producer stream without touching any data.
        total_steps = api.Var(0)
        for count_item in range(begin, end):
            count_query = api.Var((count_item % groups) * row_block)
            count_rows = api.Var(api.Min(row_block, queries - count_query))
            if causal:
                count_visible = api.Var(keys - queries + count_query + count_rows)
            else:
                count_visible = api.Var(keys)
            total_steps += api.CeilDiv(count_visible, key_tile)
        current_item = api.Var(begin)
        current_tile = api.Var(0)
        previous_item = api.Var(begin)
        previous_tile = api.Var(0)
        previous_tiles = api.Var(0)
        with api.auto_sync():
            # The first K has no previous reader. Idle cores publish only a
            # dummy ready token and perform no GM access.
            if begin < end:
                initial_owner = api.Var(begin // groups)
                initial_batch = api.Var(initial_owner // heads)
                initial_head = api.Var(initial_owner % heads)
                initial_query = api.Var((begin % groups) * row_block)
                initial_rows = api.Var(api.Min(row_block, queries - initial_query))
                initial_slot = api.Var(begin % 2)
                # PyPTO requires constant tile addresses before the main
                # loop. Select the same runtime slot through two static views.
                for initial_q_slot in api.unroll(2):
                    if initial_slot == initial_q_slot:
                        if initial_rows < row_block:
                            api.set_constant_to_l1(q[initial_q_slot], 0)
                            api.bar_all()
                        q[initial_q_slot][:initial_rows, :] <<= query.view(
                            [initial_rows, dim], [heads * dim, 1],
                            ((initial_batch * queries + initial_query) * heads + initial_head) * dim)
                k[0] <<= key.view([key_tile, dim], [heads * dim, 1],
                                  ((initial_batch * keys * heads) + initial_head) * dim)
                k_cache_tags[0].set(initial_owner * key_tiles)
            q_ready.set()
            k_ready.set()
            # Every microstep consumes/replenishes one K and V token. The
            # initial V and the final K/V are dummy tokens, never operand reads.
            for microtick in range(total_steps + 1):
                q_ready.wait()
                k_ready.wait()
                remaining = api.Var(total_steps - microtick)
                query_slot = api.Var(current_item % 2)
                owner = api.Var(current_item // groups)
                bi = api.Var(owner // heads)
                head = api.Var(owner % heads)
                first_query = api.Var((current_item % groups) * row_block)
                rows = api.Var(api.Min(row_block, queries - first_query))
                local_rows = api.Var(api.Max(0, api.Min(half, rows - lane_begin)))
                if causal:
                    visible = api.Var(keys - queries + first_query + rows)
                else:
                    visible = api.Var(keys)
                tiles = api.Var(api.CeilDiv(visible, key_tile))
                producer = api.Var(microtick)
                first_key = api.Var(current_tile * key_tile)
                requested_tag = api.Var(owner * key_tiles + current_tile)
                active_keys = api.Var(key_tile)
                # The largest real query decides for both Vector participants.
                # Keys at or above this boundary are masked for every real row.
                if causal and visible <= first_key + key_tile // 2:
                    active_keys.set(key_tile // 2)
                if remaining > 0:
                    if current_tile == 0:
                        initialize(total[query_slot], maximum[query_slot])
                    # Keep both physical QK and FIX carrier shapes static.
                    # The separate half score has a fixed compact M128 pitch.
                    # Full score, half score and product use256KiB L0C total.
                    if active_keys == key_tile // 2:
                        api.matmul(score_half, k[current_tile], q[query_slot],
                                   m=key_tile // 2, n=row_block, k=dim, splitk=operand_split)
                    else:
                        api.matmul(score, k[current_tile], q[query_slot],
                                   m=key_tile, n=row_block, k=dim, splitk=operand_split)
                    qk.lock()
                    if active_keys == key_tile // 2:
                        api.l0c_to_ub(ub_score[producer][:key_tile // 2, :], score_half,
                                      M=key_tile // 2, N=row_block, N_dst=half, M_src=key_tile // 2,
                                      dual_mode=api.DualMode.SPLITN)
                    else:
                        api.l0c_to_ub(ub_score[producer], score, M=key_tile, N=row_block,
                                      N_dst=half, M_src=key_tile, dual_mode=api.DualMode.SPLITN)
                    qk.ready()
                    qk.wait()
                    # Both Vector participants publish their complete 64-column
                    # P half, including zero-masked padded query lanes. This
                    # fully initializes P without overwriting an old L1 P tile
                    # before its delayed PV reader retires.
                    if causal and keys - queries + first_query + lane_begin + 1 < first_key + active_keys:
                        if current_tile == 0:
                            first_masked(ub_score[producer], total[query_slot], maximum[query_slot], rescale[producer], ub_p, local_rows,
                                         first_query + lane_begin, first_key, active_keys)
                        else:
                            next_masked(ub_score[producer], total[query_slot], maximum[query_slot], rescale[producer], ub_p, local_rows,
                                        first_query + lane_begin, first_key, active_keys)
                    else:
                        if current_tile == 0:
                            first_unmasked(ub_score[producer], total[query_slot], maximum[query_slot], rescale[producer], ub_p, local_rows,
                                           first_query + lane_begin, first_key, active_keys)
                        else:
                            next_unmasked(ub_score[producer], total[query_slot], maximum[query_slot], rescale[producer], ub_p, local_rows,
                                          first_query + lane_begin, first_key, active_keys)
                    qk.free()
                    # V1 can prepare UB P while old L1 P is still being read.
                    # The sole L1 credit gates only the MTE3 publication.
                    publish.lock()
                    p[:active_keys, lane_begin:lane_begin + half] <<= ub_p[:active_keys, :].nz()
                    publish.ready()
                # Q and K's current MTE1 readers precede any future writes
                # in this same lexical microstep, including query wraps.
                q_available.set()
                q_available.wait()
                k_available.set()
                k_available.wait()
                if remaining > 1:
                    next_item = api.Var(current_item)
                    next_tile = api.Var(current_tile + 1)
                    if next_tile == tiles:
                        next_item += 1
                        next_tile.set(0)
                    next_owner = api.Var(next_item // groups)
                    next_batch = api.Var(next_owner // heads)
                    next_head = api.Var(next_owner % heads)
                    next_key = api.Var(next_tile * key_tile)
                    next_tag = api.Var(next_owner * key_tiles + next_tile)
                    if next_tile == 0:
                        next_query = api.Var((next_item % groups) * row_block)
                        next_rows = api.Var(api.Min(row_block, queries - next_query))
                        next_slot = api.Var(next_item % 2)
                        if next_rows < row_block:
                            api.set_constant_to_l1(q[next_slot], 0)
                            api.bar_all()
                        q[next_slot][:next_rows, :] <<= query.view(
                            [next_rows, dim], [heads * dim, 1],
                            ((next_batch * queries + next_query) * heads + next_head) * dim)
                    # This credit follows the current K's last MTE1 read.
                    # It also protects KV3 -> next-query KV0 slot reuse.
                    # Issue next K before the older PV's V-retirement wait.
                    for next_cache_slot in api.unroll(k_cache_slots):
                        if next_tile % k_cache_slots == next_cache_slot:
                            if k_cache_tags[next_cache_slot] != next_tag:
                                k[next_cache_slot] <<= key.view([key_tile, dim], [heads * dim, 1],
                                                               ((next_batch * keys + next_key) * heads + next_head) * dim)
                                k_cache_tags[next_cache_slot].set(next_tag)
                q_ready.set()
                k_ready.set()
                v_ready.wait()
                if microtick > 0:
                    consumer = api.Var(microtick - 1)
                    previous_slot = api.Var(previous_item % 2)
                    previous_owner = api.Var(previous_item // groups)
                    previous_batch = api.Var(previous_owner // heads)
                    previous_head = api.Var(previous_owner % heads)
                    previous_query = api.Var((previous_item % groups) * row_block)
                    previous_rows = api.Var(api.Min(row_block, queries - previous_query))
                    previous_local_rows = api.Var(api.Max(0, api.Min(half, previous_rows - lane_begin)))
                    publish.wait()
                    previous_visible = api.Var(keys - queries + previous_query + previous_rows)
                    if causal and previous_visible <= previous_tile * key_tile + key_tile // 2:
                        # A single K128 fragment consumes only the published
                        # lower half. Retain all inferred cross-call guards:
                        # this odd fragment count is outside even-trip pruning.
                        api.matmul(product, p.T, values[previous_tile].T, m=row_block, n=dim,
                                   k=key_tile // 2)
                    else:
                        api.matmul(product, p.T, values[previous_tile].T, m=row_block, n=dim,
                                   k=key_tile, splitk=pv_split)
                    # This MTE1 endpoint is P's last reader, not the later
                    # MMAD/FIX/V2 use of the staged product.
                    publish.free()
                    pv.lock()
                    ub_product <<= product
                    pv.ready()
                    pv.wait()
                    if previous_local_rows > 0:
                        if previous_tile == 0:
                            first_product(accumulator, ub_product, rescale[consumer], previous_local_rows)
                        else:
                            next_product(accumulator, ub_product, rescale[consumer], previous_local_rows)
                    if previous_tile + 1 == previous_tiles:
                        if previous_local_rows > 0:
                            finish(accumulator, total[previous_slot], ub_output, previous_local_rows)
                            destination = output.view([previous_local_rows, dim], [heads * dim, 1],
                                                      ((previous_batch * queries + previous_query + lane_begin) * heads + previous_head) * dim)
                            destination <<= ub_output[:previous_local_rows, :dim]
                        # Output has independent storage. Autosync retains its
                        # MTE3 reader before a subsequent finish overwrites it.
                    pv.free()
                # A real consumer retires its V MTE1 reader here. The initial
                # and idle microsteps acknowledge only their dummy V token.
                v_available.set()
                v_available.wait()
                if remaining > 0:
                    # In particular old KV3 and new-query KV0 both occupy V slot zero.
                    # Loading new V before the delayed old PV would corrupt that PV.
                    for value_slot in api.unroll(v_cache_slots):
                        if current_tile % v_cache_slots == value_slot:
                            if v_cache_tags[value_slot] != requested_tag:
                                values[value_slot] <<= value.view([key_tile, dim], [heads * dim, 1],
                                                                ((bi * keys + first_key) * heads + head) * dim)
                                v_cache_tags[value_slot].set(requested_tag)
                    # Save identities only after the previous consumer has used them.
                    previous_item.set(current_item)
                    previous_tile.set(current_tile)
                    previous_tiles.set(tiles)
                    current_tile += 1
                    if current_tile == tiles:
                        current_item += 1
                        current_tile.set(0)
                v_ready.set()
            q_ready.wait()
            k_ready.wait()
        return output


    return mha_preload

# ----------------------------------------------------------------------------------------------------
# packed_decode.py
# Pack two independent decode heads with an explicit block-diagonal P matrix.
#
# QK computes cross-head columns, but each real query row consumes only its own
# 128-key segment. All other P columns and physical M16 padding remain zero.
# Independent K and V are staged separately; no head shares another head's values.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def make_packed_decode_kernel(dtype_name, layout, batch, queries, keys, heads, dim, causal, scale_value):
    if dtype_name != "float16" or layout != "BSND" or queries != 1 or keys != 128 or dim != 128 or causal:
        raise ValueError("Packed decode requires FP16 BSND SQ1/KV128/D128 noncausal MHA")
    if batch <= 0 or heads <= 0 or heads % 4:
        raise ValueError("Packed decode retains declared head counts divisible by four")
    dtype = api.f16
    head_pack, physical_m, total_keys = 2, 16, 256
    half, pitch = 8, 9
    scale = scale_value if scale_value > 0 else 1 / math.sqrt(dim)
    groups = heads // head_pack
    items = batch * groups

    @api.vf()
    def initialize_probability(probability: api.Tensor):
        zero = api.Reg(dtype)
        zero <<= 0.0
        for chunk in range(pitch * total_keys // 128):
            probability[chunk * 128] <<= zero
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def softmax(score: api.Tensor, probability: api.Tensor, denominator: api.Tensor,
                rows: api.Var, first_row: api.Var):
        first = api.Reg(api.f32)
        second = api.Reg(api.f32)
        maximum = api.Reg(api.f32)
        partial = api.Reg(api.f32)
        total = api.Reg(api.f32)
        first_half = api.Reg(dtype)
        second_half = api.Reg(dtype)
        packed = api.Reg(dtype)
        discarded = api.Reg(dtype)
        for row in range(rows):
            own_head = api.Var(first_row + row)
            first <<= score[row * total_keys + own_head * keys]
            second <<= score[row * total_keys + own_head * keys + 64]
            first <<= first * scale
            second <<= second * scale
            maximum <<= first.cmax()
            partial <<= second.cmax()
            maximum <<= maximum.vmax(partial)
            maximum <<= maximum.dup()
            first <<= first - maximum
            second <<= second - maximum
            first <<= first.exp()
            second <<= second.exp()
            total <<= first.cadd()
            partial <<= second.cadd()
            total <<= total + partial
            denominator[row] <<= total.single_value()
            first_half <<= first.astype(dtype)
            second_half <<= second.astype(dtype)
            api.deinterleave(packed, discarded, first_half, second_half)
            # The head block and local row determine the compact-NZ address.
            # Zero initialization owns every other head block throughout reuse.
            api.reg_to_ub(probability[(own_head * 8 * pitch + row) * 16], packed, pitch)
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def finish(product: api.Tensor, denominator: api.Tensor, output: api.Tensor, rows: api.Var):
        value = api.Reg(api.f32)
        total = api.Reg(api.f32)
        narrowed = api.Reg(dtype)
        for row in range(rows):
            total <<= denominator[row].single()
            for chunk in api.unroll(dim // 64):
                value <<= product[row * dim + chunk * 64]
                value <<= value / total
                narrowed <<= value.astype(dtype)
                api.reg_to_ub_downsample(output[row * dim + chunk * 64], narrowed)

    @api.kernel(mode="mix")
    def mha_packed_decode(query: api.GM[dtype, (batch, queries, heads, dim)],
                          key: api.GM[dtype, (batch, keys, heads, dim)],
                          value: api.GM[dtype, (batch, keys, heads, dim)],
                          output: api.GM[dtype, (batch, queries, heads, dim)]):
        q = api.Tensor(dtype, [physical_m, dim], api.Position.L1)
        # Two adjacent heads per row, with two temporal K/V versions.
        # Cube consumers select one independent head through aligned columns.
        k = api.DBuff(dtype, [keys, head_pack * dim], api.Position.L1)
        values = api.DBuff(dtype, [keys, head_pack * dim], api.Position.L1)
        probability = api.DBuff(dtype, [physical_m, total_keys], api.Position.L1)
        score = api.DBuff(api.f32, [physical_m, total_keys], api.Position.L0C)
        product = api.Tensor(api.f32, [physical_m, dim], api.Position.L0C)
        ub_score = api.DBuff(api.f32, [half, total_keys], api.Position.UB)
        ub_product = api.Tensor(api.f32, [half, dim], api.Position.UB)
        ub_probability = api.DBuff(dtype, [pitch, total_keys], api.Position.UB)
        denominator = api.DBuff(api.f32, [1, 64], api.Position.UB)
        ub_output = api.Tensor(dtype, [half, dim], api.Position.UB)
        qk = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        publish = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
        pv = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        begin = api.Var(items * api.GetCubeIdx() // api.GetCubeNum())
        end = api.Var(items * (api.GetCubeIdx() + 1) // api.GetCubeNum())
        lane_begin = api.Var(api.GetSubBlockIdx() * half)
        local_rows = api.Var(api.Max(0, api.Min(half, head_pack - lane_begin)))
        with api.auto_sync():
            api.set_constant_to_l1(q, 0)
            for slot in api.unroll(2):
                api.set_constant_to_l1(probability[slot], 0)
                initialize_probability(ub_probability[slot])
            api.bar_all()
            counter = api.Var(0)
            for iteration in range(end - begin):
                producer = api.Var(counter)
                load_item = api.Var(begin + counter)
                load_batch = api.Var(load_item // groups)
                load_head = api.Var((load_item % groups) * head_pack)
                k[counter] <<= key.view([keys, head_pack * dim], [heads * dim, 1],
                               (load_batch * keys * heads + load_head) * dim)
                values[counter] <<= value.view([keys, head_pack * dim], [heads * dim, 1],
                                                (load_batch * keys * heads + load_head) * dim)
                q[:head_pack, :] <<= query.view([head_pack, dim], [dim, 1],
                                               (load_batch * heads + load_head) * dim)
                score_slot = api.Var(counter)
                for score_head in api.unroll(head_pack):
                    api.matmul(score[counter][:, score_head * keys:(score_head + 1) * keys],
                               q, k[counter][:, score_head * dim:(score_head + 1) * dim], m=physical_m, n=keys, k=dim)
                qk.lock()
                ub_score[counter] <<= score[counter]
                qk.ready()
                qk.wait()
                if local_rows > 0:
                    softmax(ub_score[counter], ub_probability[counter], denominator[counter], local_rows, lane_begin)
                qk.free()
                publish.lock()
                if local_rows > 0:
                    probability[counter][lane_begin:lane_begin + local_rows, :] <<= ub_probability[counter][:local_rows, :].nz()
                publish.ready()
                if counter > 0:
                    consumer = api.Var(counter - 1)
                    store_item = api.Var(begin + counter - 1)
                    store_batch = api.Var(store_item // groups)
                    store_head = api.Var((store_item % groups) * head_pack)
                    publish.wait()
                    for value_head in api.unroll(head_pack):
                        api.matmul(product, probability[consumer][:, value_head * keys:(value_head + 1) * keys],
                                   values[consumer][:, value_head * dim:(value_head + 1) * dim].T, m=physical_m, n=dim, k=keys,
                                   is_init=value_head == 0)
                    publish.free()
                    pv.lock()
                    ub_product <<= product
                    pv.ready()
                    pv.wait()
                    if local_rows > 0:
                        finish(ub_product, denominator[consumer], ub_output, local_rows)
                        destination = output.view([local_rows, dim], [dim, 1],
                                                  (store_batch * heads + store_head + lane_begin) * dim)
                        destination <<= ub_output[:local_rows, :]
                    pv.free()
                counter += 1
            if end > begin:
                for final_slot in api.unroll(2):
                    if (end - begin - 1) % 2 == final_slot:
                        store_item = api.Var(end - 1)
                        store_batch = api.Var(store_item // groups)
                        store_head = api.Var((store_item % groups) * head_pack)
                        publish.wait()
                        for value_head in api.unroll(head_pack):
                            api.matmul(product, probability[final_slot][:, value_head * keys:(value_head + 1) * keys],
                                       values[final_slot][:, value_head * dim:(value_head + 1) * dim].T, m=physical_m, n=dim, k=keys,
                                       is_init=value_head == 0)
                        publish.free()
                        pv.lock()
                        ub_product <<= product
                        pv.ready()
                        pv.wait()
                        if local_rows > 0:
                            finish(ub_product, denominator[final_slot], ub_output, local_rows)
                            destination = output.view([local_rows, dim], [dim, 1],
                                                      (store_batch * heads + store_head + lane_begin) * dim)
                            destination <<= ub_output[:local_rows, :]
                        pv.free()
        return output

    return mha_packed_decode

# ----------------------------------------------------------------------------------------------------
# mha.py
# Select one of the three declared schedules. They share a kernel signature and differ
# only in how K/V move on chip, so a case picks a schedule by name and nothing else changes.
# ----------------------------------------------------------------------------------------------------

SCHEDULES = {"head_resident": "make_resident_kernel",
             "head_preload": "make_preload_kernel",
             "packed_decode": "make_packed_decode_kernel"}


def make_kernel(dtype, layout, batch, queries, keys, heads, dim, causal, scale_value,
                variant="head_resident"):
    if variant not in SCHEDULES:
        raise ValueError(f"Unknown MHA schedule: {variant}; declared: {', '.join(SCHEDULES)}")
    factory = globals()[SCHEDULES[variant]]
    return factory(dtype, layout, batch, queries, keys, heads, dim, causal, scale_value)
