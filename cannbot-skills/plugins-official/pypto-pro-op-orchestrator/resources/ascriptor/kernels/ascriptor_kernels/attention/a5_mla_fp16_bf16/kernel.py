# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Six standard MLA schedules behind one 4-D ABI.

Each module is one schedule; mla.py at the end selects between them by name. They differ in how
the KV cache moves on chip and where the online softmax state lives, not in what they compute."""

import math
from functools import lru_cache

import ascriptor.a5 as api

# ----------------------------------------------------------------------------------------------------
# serial.py
# Serial QK -> online softmax -> PV teaching kernel with direct 4-D access.
#
# Eight logical query heads share one physical M16 cube tile. Every Q/P row
# read by M16 is initialized, including the unused second vector participant.
# This deliberately simple schedule is a correctness baseline, not a speed claim.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def make_serial_nd_kernel(dtype_name, layout, batch, queries, keys, heads, nope, causal):
    dtype = api.f16 if dtype_name == "float16" else api.bf16
    bnsd = layout == "BNSD"
    q_shape = (batch, heads, queries, nope) if bnsd else (batch, queries, heads, nope)
    qr_shape = (*q_shape[:-1], 64)
    k_shape = (batch, 1, keys, nope) if bnsd else (batch, keys, 1, nope)
    kr_shape = (*k_shape[:-1], 64)
    scale = 1.0 / math.sqrt(nope + 64)
    groups = (heads + 7) // 8
    items = batch * queries * groups

    @api.vf()
    def initialize(total: api.Tensor, maximum: api.Tensor, accumulator: api.Tensor):
        value = api.Reg(api.f32)
        value <<= 0.0
        total <<= value
        for chunk in range(8 * nope // 64):
            accumulator[chunk * 64] <<= value
        value <<= -99999.0
        maximum <<= value

    @api.vf()
    def softmax(score: api.Tensor, total: api.Tensor, maximum: api.Tensor,
                rescale: api.Tensor, probability: api.Tensor, rows: api.Var, valid_keys: api.Var):
        value = api.Reg(api.f32)
        old_max = api.Reg(api.f32)
        old_sum = api.Reg(api.f32)
        new_max = api.Reg(api.f32)
        new_sum = api.Reg(api.f32)
        correction = api.Reg(api.f32)
        negative = api.Reg(api.f32)
        packed = api.Reg(dtype)
        indices = api.Reg(api.i32)
        valid = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
        negative <<= -99999.0
        indices.arange(0)
        api.compare(valid, indices, valid_keys, api.CompareMode.LT)
        for row in range(rows):
            value <<= score[row * 64]
            value <<= value * scale
            api.select(value, value, negative, mask=valid)
            old_max <<= maximum[row].single()
            old_sum <<= total[row].single()
            new_max <<= value.cmax()
            new_max <<= new_max.dup()
            new_max <<= new_max.vmax(old_max)
            correction <<= old_max - new_max
            correction <<= correction.exp()
            value <<= value - new_max
            value <<= value.exp()
            # Mask after exp as well: a wholly masked later tile contributes zero.
            negative <<= 0.0
            api.select(value, value, negative, mask=valid)
            negative <<= -99999.0
            new_sum <<= value.cadd()
            new_sum <<= new_sum.dup()
            new_sum <<= new_sum + old_sum * correction
            packed <<= value.astype(dtype)
            # FP32 -> b16 cast has ZERO lanes: PACK_B32 stores 64 values from
            # 128 physical b16 lanes. LOWHALF here would truncate to 32 values.
            api.reg_to_ub_downsample(probability[row * 64], packed)
            maximum[row] <<= new_max.single_value()
            total[row] <<= new_sum.single_value()
            rescale[row] <<= correction.single_value()
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def accumulate(accumulator: api.Tensor, product: api.Tensor,
                   rescale: api.Tensor, rows: api.Var):
        acc = api.Reg(api.f32)
        current = api.Reg(api.f32)
        correction = api.Reg(api.f32)
        for row in range(rows):
            correction <<= rescale[row].single()
            for chunk in range(nope // 64):
                acc <<= accumulator[row * nope + chunk * 64]
                current <<= product[row * nope + chunk * 64]
                acc <<= acc * correction + current
                accumulator[row * nope + chunk * 64] <<= acc
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def finish(accumulator: api.Tensor, total: api.Tensor, output: api.Tensor, rows: api.Var):
        value = api.Reg(api.f32)
        denominator = api.Reg(api.f32)
        packed = api.Reg(dtype)
        for row in range(rows):
            denominator <<= total[row].single()
            for chunk in range(nope // 64):
                value <<= accumulator[row * nope + chunk * 64]
                value <<= value / denominator
                packed <<= value.astype(dtype)
                api.reg_to_ub_downsample(output[row * nope + chunk * 64], packed)

    @api.kernel(mode="mix")
    def mla(q_nope: api.GM[dtype, q_shape], q_rope: api.GM[dtype, qr_shape],
            k_nope: api.GM[dtype, k_shape], k_rope: api.GM[dtype, kr_shape],
            v: api.GM[dtype, k_shape], output: api.GM[dtype, q_shape]):
        qn = api.Tensor(dtype, [16, nope], api.Position.L1)
        qr = api.Tensor(dtype, [16, 64], api.Position.L1)
        kn = api.Tensor(dtype, [64, nope], api.Position.L1)
        kr = api.Tensor(dtype, [64, 64], api.Position.L1)
        p = api.Tensor(dtype, [16, 64], api.Position.L1)
        score = api.Tensor(api.f32, [16, 64], api.Position.L0C)
        product = api.Tensor(api.f32, [16, nope], api.Position.L0C)
        ub_score = api.Tensor(api.f32, [8, 64], api.Position.UB)
        ub_product = api.Tensor(api.f32, [8, nope], api.Position.UB)
        ub_p = api.Tensor(dtype, [8, 64], api.Position.UB)
        total = api.Tensor(api.f32, [1, 64], api.Position.UB)
        maximum = api.Tensor(api.f32, [1, 64], api.Position.UB)
        rescale = api.Tensor(api.f32, [1, 64], api.Position.UB)
        accumulator = api.Tensor(api.f32, [8, nope], api.Position.UB)
        ub_output = api.Tensor(dtype, [8, nope], api.Position.UB)
        qk_handoff = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        p_handoff = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
        pv_handoff = api.CvMutex(2, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        per_core = api.CeilDiv(items, api.GetCubeNum())
        begin = api.Var(per_core * api.GetCubeIdx())
        end = api.Min(begin + per_core, items)
        with api.auto_sync():
            for item in range(begin, end):
                bi = api.Var(item // (queries * groups))
                qi = api.Var((item // groups) % queries)
                head = api.Var((item % groups) * 8)
                rows = api.Var(api.Min(8, heads - head))
                # The full physical operands are written before any M16 read.
                api.set_constant_to_l1(qn, 0)
                api.set_constant_to_l1(qr, 0)
                api.bar_all()
                if bnsd:
                    qn[:rows, :] <<= q_nope.view([rows, nope], [queries * nope, 1],
                                                ((bi * heads + head) * queries + qi) * nope)
                    qr[:rows, :] <<= q_rope.view([rows, 64], [queries * 64, 1],
                                                ((bi * heads + head) * queries + qi) * 64)
                else:
                    qn[:rows, :] <<= q_nope.view([rows, nope], [nope, 1],
                                                ((bi * queries + qi) * heads + head) * nope)
                    qr[:rows, :] <<= q_rope.view([rows, 64], [64, 1],
                                                ((bi * queries + qi) * heads + head) * 64)
                initialize(total, maximum, accumulator)
                for key in range(0, keys, 64):
                    # Reset all P rows before the QK handoff permits vector publication.
                    api.set_constant_to_l1(p, 0)
                    # Nkv=1 gives the same contiguous physical KV address in both layouts.
                    kn <<= k_nope.view([64, nope], [nope, 1], (bi * keys + key) * nope)
                    kr <<= k_rope.view([64, 64], [64, 1], (bi * keys + key) * 64)
                    api.matmul(score, qn, kn, m=16, n=64, k=nope, splitk=64, is_init=True)
                    api.matmul(score, qr, kr, m=16, n=64, k=64, is_init=False)
                    api.bar_all()
                    qk_handoff.lock()
                    ub_score <<= score
                    qk_handoff.ready()
                    qk_handoff.wait()
                    if api.GetSubBlockIdx() == 0:
                        allowed = api.Var(64)
                        if causal:
                            allowed <<= api.Min(api.Max(keys - queries + qi + 1 - key, 0), 64)
                        softmax(ub_score, total, maximum, rescale, ub_p, rows, allowed)
                    qk_handoff.free()
                    p_handoff.lock()
                    if api.GetSubBlockIdx() == 0:
                        p[:rows, :] <<= ub_p[:rows, :]
                    p_handoff.ready()
                    p_handoff.wait()
                    # V equals K_nope by contract; reuse the already staged values.
                    api.matmul(product, p, kn.T, m=16, n=nope, k=64, splitn=64)
                    p_handoff.free()
                    pv_handoff.lock()
                    ub_product <<= product
                    pv_handoff.ready()
                    pv_handoff.wait()
                    if api.GetSubBlockIdx() == 0:
                        accumulate(accumulator, ub_product, rescale, rows)
                    pv_handoff.free()
                if api.GetSubBlockIdx() == 0:
                    finish(accumulator, total, ub_output, rows)
                    if bnsd:
                        destination = output.view([rows, nope], [queries * nope, 1],
                                                  ((bi * heads + head) * queries + qi) * nope)
                        destination <<= ub_output[:rows, :]
                    else:
                        destination = output.view([rows, nope], [nope, 1],
                                                  ((bi * queries + qi) * heads + head) * nope)
                        destination <<= ub_output[:rows, :]
                api.bar_all()
        return output

    return mla

# ----------------------------------------------------------------------------------------------------
# prefill.py
# Full-KV prefill: resident KV and two Vector participants per head block.
#
# This source preserves the original 4-D BSND ABI and unnormalized P conversion
# boundary. One KV tile eliminates repeated online-state updates.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def make_prefill_resident_kernel(dtype_name, layout, batch, queries, keys, heads, nope, causal,
                row_block=None, compact_nz=True):
    if row_block is None:
        row_block = min(heads, 128)
    if (dtype_name, nope) not in (("float16", 512), ("bfloat16", 448)):
        raise ValueError("Prefill covers the two declared dtype/feature pairs")
    if layout != "BSND" or keys != 128 or not causal or row_block not in (16, 32, 64, 128):
        raise ValueError("Prefill requires BSND, KV128, causal and M16/32/64/128")
    if batch <= 0 or not 1 <= queries <= keys or heads % row_block:
        raise ValueError("Positive batch, SQ<=KV and complete row blocks are required")
    dtype = api.f16 if dtype_name == "float16" else api.bf16
    rows_per_vector = row_block // 2
    pitch = rows_per_vector + 1
    groups = heads // row_block
    items = queries * groups
    split_k = 128 if nope % 128 == 0 else 64
    scale = 1.0 / math.sqrt(nope + 64)

    @api.vf()
    def softmax_full(score: api.Tensor, probability: api.Tensor,
                     denominator: api.Tensor, valid_keys: api.Var):
        first = api.Reg(api.f32)
        second = api.Reg(api.f32)
        first_max = api.Reg(api.f32)
        second_max = api.Reg(api.f32)
        total = api.Reg(api.f32)
        partial = api.Reg(api.f32)
        negative = api.Reg(api.f32)
        index0 = api.Reg(api.i32)
        index1 = api.Reg(api.i32)
        valid0 = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
        valid1 = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
        half0 = api.Reg(dtype)
        half1 = api.Reg(dtype)
        packed = api.Reg(dtype)
        discarded = api.Reg(dtype)
        negative <<= -99999.0
        index0.arange(0)
        index1.arange(64)
        api.compare(valid0, index0, valid_keys, api.CompareMode.LT)
        api.compare(valid1, index1, valid_keys, api.CompareMode.LT)
        for row in range(rows_per_vector):
            first <<= score[row * 128]
            second <<= score[row * 128 + 64]
            first <<= first * scale
            second <<= second * scale
            api.select(first, first, negative, mask=valid0)
            api.select(second, second, negative, mask=valid1)
            first_max <<= first.cmax()
            second_max <<= second.cmax()
            first_max <<= first_max.vmax(second_max)
            first_max <<= first_max.dup()
            first <<= first - first_max
            second <<= second - first_max
            first <<= first.exp()
            second <<= second.exp()
            total <<= first.cadd()
            partial <<= second.cadd()
            total <<= total + partial
            denominator[row] <<= total.single_value()
            half0 <<= first.astype(dtype)
            half1 <<= second.astype(dtype)
            # Both sparse FP32->b16 cast carriers become one compact 128-lane register.
            api.deinterleave(packed, discarded, half0, half1)
            if compact_nz:
                api.reg_to_ub(probability[row * 16], packed, pitch)
            else:
                api.reg_to_ub_normal(probability[row * 128], packed)

    @api.vf()
    def normalize(product: api.Tensor, denominator: api.Tensor, output: api.Tensor):
        value = api.Reg(api.f32)
        total = api.Reg(api.f32)
        narrowed = api.Reg(dtype)
        for row in range(rows_per_vector):
            total <<= denominator[row].single()
            total <<= total.dup()
            for chunk in range(nope // 64):
                value <<= product[row * nope + chunk * 64]
                value <<= value / total
                narrowed <<= value.astype(dtype)
                api.reg_to_ub_downsample(output[row * nope + chunk * 64], narrowed)

    @api.kernel(mode="mix")
    def mla_prefill(q_nope: api.GM[dtype, (batch, queries, heads, nope)],
                    q_rope: api.GM[dtype, (batch, queries, heads, 64)],
                    k_nope: api.GM[dtype, (batch, 128, 1, nope)],
                    k_rope: api.GM[dtype, (batch, 128, 1, 64)],
                    v: api.GM[dtype, (batch, 128, 1, nope)],
                    output: api.GM[dtype, (batch, queries, heads, nope)]):
        query_nope = api.Tensor(dtype, [row_block, nope], api.Position.L1)
        query_rope = api.Tensor(dtype, [row_block, 64], api.Position.L1)
        key_nope = api.Tensor(dtype, [128, nope], api.Position.L1)
        key_rope = api.Tensor(dtype, [128, 64], api.Position.L1)
        probability = api.Tensor(dtype, [row_block, 128], api.Position.L1)
        # QK and PV are sequential consumers of the same L0C allocation.
        accumulation = api.Tensor(api.f32, [row_block, nope], api.Position.L0C)
        score = accumulation[:, :128]
        ub_score = api.Tensor(api.f32, [rows_per_vector, 128], api.Position.UB)
        ub_product = api.Tensor(api.f32, [rows_per_vector, nope], api.Position.UB)
        ub_probability = api.Tensor(dtype, [pitch, 128], api.Position.UB)
        denominator = api.Tensor(api.f32, [1, 64], api.Position.UB)
        ub_output = api.Tensor(dtype, [rows_per_vector, nope], api.Position.UB)
        qk = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        pv = api.CvMutex(1, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        publish = api.VcMutex(2, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
        per_core = api.CeilDiv(items, api.GetCubeNum())
        first = api.Var(per_core * api.GetCubeIdx())
        last = api.Min(first + per_core, items)
        vector_begin = api.GetSubBlockIdx() * rows_per_vector
        with api.auto_sync():
            for bi in range(batch):
                if first < last:
                    key_nope <<= k_nope.view([128, nope], [nope, 1], bi * 128 * nope)
                    key_rope <<= k_rope.view([128, 64], [64, 1], bi * 128 * 64)
                for item in range(first, last):
                    qi = api.Var(item // groups)
                    head = api.Var((item % groups) * row_block)
                    base_row = api.Var((bi * queries + qi) * heads + head)
                    query_nope <<= q_nope.view([row_block, nope], [nope, 1], base_row * nope)
                    query_rope <<= q_rope.view([row_block, 64], [64, 1], base_row * 64)
                    api.matmul(score, query_nope, key_nope, m=row_block, n=128, k=nope,
                               splitk=split_k, is_init=True)
                    api.matmul(score, query_rope, key_rope, m=row_block, n=128, k=64, is_init=False)
                    qk.lock()
                    ub_score <<= score
                    qk.ready()
                    qk.wait()
                    allowed = api.Var(keys - queries + qi + 1)
                    softmax_full(ub_score, ub_probability, denominator, allowed)
                    qk.free()
                    publish.lock()
                    if compact_nz:
                        probability[vector_begin:vector_begin + rows_per_vector, :] <<= ub_probability[:rows_per_vector, :].nz()
                    else:
                        probability[vector_begin:vector_begin + rows_per_vector, :] <<= ub_probability[:rows_per_vector, :]
                    publish.ready()
                    publish.wait()
                    api.matmul(accumulation, probability, key_nope.T, m=row_block, n=nope,
                               k=128, splitn=64)
                    publish.free()
                    pv.lock()
                    ub_product <<= accumulation
                    pv.ready()
                    pv.wait()
                    normalize(ub_product, denominator, ub_output)
                    destination = output.view([rows_per_vector, nope], [nope, 1],
                                              (base_row + vector_begin) * nope)
                    destination <<= ub_output
                    pv.free()
                api.bar_all()
        return output

    return mla_prefill

# ----------------------------------------------------------------------------------------------------
# split_decode.py
# Split-KV decode with direct single-tile partials and four-row merge owners.
#
# FP32 online state remains in UB. Inputs preserve their original 4-D ABI and
# layout; noncausal rows may be grouped in physical order within each batch.
# FP32 partials use private GM workspace and an all-Vector barrier, then stable
# log-sum-exp merge in the same actual runtime kernel; inputs remain read-only.
# Producer M64 groups and merge row groups are independently distributed.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _build_split_decode(dtype_name, layout, batch, queries, keys, heads, nope, causal,
          tile_m=64, tile_n=256, split_count=8, merge_rows=4):
    if causal or dtype_name != "float16" or nope != 512:
        raise ValueError("This schedule requires FP16 noncausal Dn512 decode")
    if tile_m not in (8, 16, 32, 64) or tile_n not in (128, 256) or keys % tile_n:
        raise ValueError("Aligned KV128/256 and M8/16/32/64 are required")
    if layout != "BSND" or queries != 1 or batch <= 0 or heads <= 0 or keys <= 0:
        raise ValueError("This schedule requires positive BSND single-query decode")
    dtype = api.f16
    physical_m = max(16, tile_m)
    half = physical_m // 2
    if merge_rows not in (4, 8) or half % merge_rows or tile_m % merge_rows:
        raise ValueError("Merge rows must be 4/8 and divide the producer row partitions")
    pitch = half + 1
    chunks = tile_n // 64
    bnsd = layout == "BNSD"
    q_shape = (batch, heads, queries, nope) if bnsd else (batch, queries, heads, nope)
    qr_shape = (*q_shape[:-1], 64)
    k_shape = (batch, 1, keys, nope) if bnsd else (batch, keys, 1, nope)
    kr_shape = (*k_shape[:-1], 64)
    scale = 1.0 / math.sqrt(nope + 64)
    batch_rows = queries * heads
    groups = (batch_rows + tile_m - 1) // tile_m
    group_items = batch * groups
    merges_per_batch = (batch_rows + merge_rows - 1) // merge_rows
    merge_items = batch * merges_per_batch
    items = group_items * split_count
    tiles = keys // tile_n
    tiles_per_split = (tiles + split_count - 1) // split_count
    single_tile = tiles_per_split == 1
    may_pad_rows = tile_m < physical_m or batch_rows % tile_m != 0
    qk_split = 128 if tile_n == 128 else 64
    pv_split = 128 if tile_n == 128 else 64

    @api.vf()
    def initialize_state(total: api.Tensor, maximum: api.Tensor):
        value = api.Reg(api.f32)
        value <<= 0.0
        total <<= value
        value <<= -99999.0
        maximum <<= value
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def zero_partial(accumulator: api.Tensor):
        value = api.Reg(api.f32)
        value <<= 0.0
        for chunk in range(half * nope // 64):
            accumulator[chunk * 64] <<= value
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def initialize(total: api.Tensor, maximum: api.Tensor, accumulator: api.Tensor):
        value = api.Reg(api.f32)
        value <<= 0.0
        total <<= value
        for chunk in range(half * nope // 64):
            accumulator[chunk * 64] <<= value
        value <<= -99999.0
        maximum <<= value
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def softmax(score: api.Tensor, total: api.Tensor, maximum: api.Tensor,
                rescale: api.Tensor, probability: api.Tensor, rows: api.Var):
        values = api.RegList(api.f32, chunks)
        halves = api.RegList(dtype, chunks)
        part = api.Reg(api.f32)
        old_max = api.Reg(api.f32)
        old_sum = api.Reg(api.f32)
        new_max = api.Reg(api.f32)
        new_sum = api.Reg(api.f32)
        correction = api.Reg(api.f32)
        packed = api.Reg(dtype)
        unused = api.Reg(dtype)
        for row in range(rows):
            new_max <<= -99999.0
            for chunk in api.unroll(chunks):
                values[chunk] <<= score[row * tile_n + chunk * 64]
                values[chunk] <<= values[chunk] * scale
                part <<= values[chunk].cmax()
                new_max <<= new_max.vmax(part)
            new_max <<= new_max.dup()
            # A single-tile partition has no previous mass or numerator.
            if not single_tile:
                old_max <<= maximum[row].single()
                old_sum <<= total[row].single()
                new_max <<= new_max.vmax(old_max)
                correction <<= old_max - new_max
                correction <<= correction.exp()
            new_sum <<= 0.0
            for chunk in api.unroll(chunks):
                values[chunk] <<= values[chunk] - new_max
                values[chunk] <<= values[chunk].exp()
                part <<= values[chunk].cadd()
                new_sum <<= new_sum + part
                halves[chunk] <<= values[chunk].astype(dtype)
            new_sum <<= new_sum.dup()
            if not single_tile:
                new_sum <<= new_sum + old_sum * correction
            for pair in api.unroll(chunks // 2):
                api.deinterleave(packed, unused, halves[pair * 2], halves[pair * 2 + 1])
                api.reg_to_ub(probability[(pair * 8 * pitch + row) * 16], packed, pitch)
            maximum[row] <<= new_max.single_value()
            total[row] <<= new_sum.single_value()
            if not single_tile:
                rescale[row] <<= correction.single_value()
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def accumulate(accumulator: api.Tensor, product: api.Tensor,
                   rescale: api.Tensor, rows: api.Var):
        acc = api.Reg(api.f32)
        current = api.Reg(api.f32)
        correction = api.Reg(api.f32)
        for row in range(rows):
            correction <<= rescale[row].single()
            for chunk in range(nope // 64):
                acc <<= accumulator[row * nope + chunk * 64]
                current <<= product[row * nope + chunk * 64]
                acc <<= acc * correction + current
                accumulator[row * nope + chunk * 64] <<= acc
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def finish(accumulator: api.Tensor, total: api.Tensor, output: api.Tensor, rows: api.Var):
        value = api.Reg(api.f32)
        denominator = api.Reg(api.f32)
        packed = api.Reg(dtype)
        for row in range(rows):
            denominator <<= total[row].single()
            for chunk in range(nope // 64):
                value <<= accumulator[row * nope + chunk * 64]
                value <<= value / denominator
                packed <<= value.astype(dtype)
                api.reg_to_ub_downsample(output[row * nope + chunk * 64], packed)

    @api.vf()
    def initialize_merge(accumulator: api.Tensor, part_max: api.Tensor, part_sum: api.Tensor):
        value = api.Reg(api.f32)
        value <<= 0.0
        for chunk in range(merge_rows * nope // 64):
            accumulator[chunk * 64] <<= value
        for part in range(split_count):
            part_sum[part * 64] <<= value
        value <<= -99999.0
        for part in range(split_count):
            part_max[part * 64] <<= value
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def merge_weights(part_max: api.Tensor, part_sum: api.Tensor,
                      weights: api.Tensor, final_sum: api.Tensor):
        maximum_value = api.Reg(api.f32)
        maximum_part = api.Reg(api.f32)
        denominator = api.Reg(api.f32)
        denominator_part = api.Reg(api.f32)
        weight = api.Reg(api.f32)
        maximum_value <<= -99999.0
        denominator <<= 0.0
        for part in range(split_count):
            maximum_part <<= part_max[part * 64]
            maximum_value <<= maximum_value.vmax(maximum_part)
        for part in range(split_count):
            maximum_part <<= part_max[part * 64]
            denominator_part <<= part_sum[part * 64]
            weight <<= maximum_part - maximum_value
            weight <<= weight.exp()
            denominator <<= denominator + denominator_part * weight
            weights[part * 64] <<= weight
        final_sum <<= denominator
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def merge_product(accumulator: api.Tensor, partial: api.Tensor,
                      weights: api.Tensor, rows: api.Var):
        acc = api.Reg(api.f32)
        product_value = api.Reg(api.f32)
        weight = api.Reg(api.f32)
        for row in range(rows):
            weight <<= weights[row].single()
            for chunk in range(nope // 64):
                acc <<= accumulator[row * nope + chunk * 64]
                product_value <<= partial[row * nope + chunk * 64]
                acc <<= acc + product_value * weight
                accumulator[row * nope + chunk * 64] <<= acc
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.kernel(mode="mix")
    def mla_decode_splitkv(q_nope: api.GM[dtype, q_shape], q_rope: api.GM[dtype, qr_shape],
                   k_nope: api.GM[dtype, k_shape], k_rope: api.GM[dtype, kr_shape],
                   v: api.GM[dtype, k_shape], output: api.GM[dtype, q_shape]):
        qn = api.Tensor(dtype, [physical_m, nope], api.Position.L1)
        qr = api.Tensor(dtype, [physical_m, 64], api.Position.L1)
        kn = api.Tensor(dtype, [tile_n, nope], api.Position.L1)
        kr = api.Tensor(dtype, [tile_n, 64], api.Position.L1)
        p = api.Tensor(dtype, [physical_m, tile_n], api.Position.L1)
        score = api.Tensor(api.f32, [physical_m, tile_n], api.Position.L0C)
        product = api.Tensor(api.f32, [physical_m, nope], api.Position.L0C)
        ub_score = api.Tensor(api.f32, [half, tile_n], api.Position.UB)
        ub_product = api.Tensor(api.f32, [half, nope], api.Position.UB)
        ub_p = api.Tensor(dtype, [pitch, tile_n], api.Position.UB)
        total = api.Tensor(api.f32, [1, 64], api.Position.UB)
        maximum = api.Tensor(api.f32, [1, 64], api.Position.UB)
        rescale = api.Tensor(api.f32, [1, 64], api.Position.UB)
        accumulator = api.Tensor(api.f32, [half, nope], api.Position.UB)
        ub_output = api.Tensor(dtype, [half, nope], api.Position.UB)
        qk_handoff = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        p_handoff = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
        # Direct partial publication's last UB reader is MTE3, not the VF pipe.
        pv_handoff = api.CvMutex(2, depth=2, src_end_pipe=api.Pipe.FIX,
                                dst_end_pipe=api.Pipe.MTE3 if single_tile else api.Pipe.V)
        partial_accumulator = api.split_workspace(api.f32, [items, physical_m, nope], name="partial_accumulator")
        partial_maximum = api.split_workspace(api.f32, [items * 2, 1, 64], name="partial_maximum")
        partial_total = api.split_workspace(api.f32, [items * 2, 1, 64], name="partial_total")
        merge_maximum = api.Tensor(api.f32, [split_count, 64], api.Position.UB)
        merge_total = api.Tensor(api.f32, [split_count, 64], api.Position.UB)
        merge_weight = api.Tensor(api.f32, [split_count, 64], api.Position.UB)
        begin = api.Var(items * api.GetCubeIdx() // api.GetCubeNum())
        end = api.Var(items * (api.GetCubeIdx() + 1) // api.GetCubeNum())
        lane_begin = api.Var(api.GetSubBlockIdx() * half)
        with api.auto_sync():
            for item in range(begin, end):
                bi = api.Var((item // split_count) // groups)
                first_row = api.Var(((item // split_count) % groups) * tile_m)
                rows = api.Var(api.Min(tile_m, batch_rows - first_row))
                local_rows = api.Var(api.Min(half, api.Max(rows - lane_begin, 0)))
                if may_pad_rows:
                    if rows < physical_m:
                        api.set_constant_to_l1(qn, 0)
                        api.set_constant_to_l1(qr, 0)
                        api.bar_all()
                qn[:rows, :] <<= q_nope.view([rows, nope], [nope, 1],
                                           (bi * batch_rows + first_row) * nope)
                qr[:rows, :] <<= q_rope.view([rows, 64], [64, 1],
                                           (bi * batch_rows + first_row) * 64)
                if single_tile:
                    initialize_state(total, maximum)
                else:
                    initialize(total, maximum, accumulator)
                first_key = api.Var((item % split_count) * tiles_per_split * tile_n)
                last_key = api.Var(api.Min(first_key + tiles_per_split * tile_n, keys))
                for key in range(first_key, last_key, tile_n):
                    if may_pad_rows:
                        if rows < physical_m:
                            api.set_constant_to_l1(p, 0)
                    kn <<= k_nope.view([tile_n, nope], [nope, 1], (bi * keys + key) * nope)
                    kr <<= k_rope.view([tile_n, 64], [64, 1], (bi * keys + key) * 64)
                    api.matmul(score, qn, kn, m=physical_m, n=tile_n, k=nope,
                               splitk=qk_split, is_init=True)
                    api.matmul(score, qr, kr, m=physical_m, n=tile_n, k=64, is_init=False)
                    api.bar_all()
                    qk_handoff.lock()
                    ub_score <<= score
                    qk_handoff.ready()
                    qk_handoff.wait()
                    if local_rows > 0:
                        softmax(ub_score, total, maximum, rescale, ub_p, local_rows)
                    qk_handoff.free()
                    p_handoff.lock()
                    if local_rows > 0:
                        p[lane_begin:lane_begin + local_rows, :] <<= ub_p[:local_rows, :].nz()
                    p_handoff.ready()
                    p_handoff.wait()
                    api.matmul(product, p, kn.T, m=physical_m, n=nope, k=tile_n, splitn=pv_split)
                    p_handoff.free()
                    pv_handoff.lock()
                    ub_product <<= product
                    pv_handoff.ready()
                    pv_handoff.wait()
                    if single_tile:
                        partial_accumulator[item, lane_begin:lane_begin + half, :] <<= ub_product
                    else:
                        if local_rows > 0:
                            accumulate(accumulator, ub_product, rescale, local_rows)
                    pv_handoff.free()
                if single_tile:
                    if first_key >= last_key:
                        zero_partial(accumulator)
                        partial_accumulator[item, lane_begin:lane_begin + half, :] <<= accumulator
                else:
                    partial_accumulator[item, lane_begin:lane_begin + half, :] <<= accumulator
                partial_maximum[item * 2 + api.GetSubBlockIdx(), 0:1, :] <<= maximum
                partial_total[item * 2 + api.GetSubBlockIdx(), 0:1, :] <<= total
                api.bar_all()

            # Every Vector participant, including idle owners, publishes and waits.
            api.allvec_ready(6, api.Pipe.MTE3)
            api.allvec_wait(6, api.Pipe.MTE2)
            # Each Vector owns a small independent output row group. A group
            # cannot cross its producer's Vector-state or M-tile boundary.
            for merge_item in range(api.GetVecIdx(), merge_items, api.GetVecNum()):
                merge_batch = api.Var(merge_item // merges_per_batch)
                merge_row = api.Var((merge_item % merges_per_batch) * merge_rows)
                merge_local_rows = api.Var(api.Min(merge_rows, batch_rows - merge_row))
                source_group = api.Var(merge_batch * groups + merge_row // tile_m)
                source_row = api.Var(merge_row % tile_m)
                state_owner = api.Var(source_row // half)
                state_row = api.Var(source_row % half)
                initialize_merge(accumulator, merge_maximum, merge_total)
                state_base = api.Var((source_group * split_count * 2 + state_owner) * 64 + state_row)
                merge_maximum[:, :merge_rows] <<= partial_maximum.view(
                    [split_count, merge_rows], [128, 1], state_base)
                merge_total[:, :merge_rows] <<= partial_total.view(
                    [split_count, merge_rows], [128, 1], state_base)
                merge_weights(merge_maximum, merge_total, merge_weight, total)
                for part in range(split_count):
                    ub_product[:merge_rows, :] <<= partial_accumulator[source_group * split_count + part, source_row:source_row + merge_rows, :]
                    merge_product(accumulator, ub_product, merge_weight[part:part + 1, :], merge_local_rows)
                finish(accumulator, total, ub_output, merge_local_rows)
                if merge_local_rows > 0:
                    destination = output.view([merge_local_rows, nope], [nope, 1],
                                              (merge_batch * batch_rows + merge_row) * nope)
                    destination <<= ub_output[:merge_local_rows, :]
                api.bar_all()
        return output

    return mla_decode_splitkv


def make_decode_splitkv_kernel(*args):
    return _build_split_decode(*args, tile_m=64, tile_n=256, split_count=8, merge_rows=4)

# ----------------------------------------------------------------------------------------------------
# decode.py
# Decode one-step CVCV pipeline: contiguous physical M, dual Vector, compact NZ P.
#
# FP32 online state remains in UB. Inputs preserve their original 4-D ABI and
# layout; noncausal rows may be grouped in physical order within each batch.
# No host packing, GM workspace, extra launch, or integer rescaling is used.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _build_decode(dtype_name, layout, batch, queries, keys, heads, nope, causal,
          tile_m=64, tile_n=128):
    if causal or dtype_name != "float16" or nope != 512:
        raise ValueError("This schedule requires FP16 noncausal Dn512 decode")
    if tile_m not in (8, 16, 32, 64) or tile_n != 128 or keys % tile_n:
        raise ValueError("Aligned KV128/256 and M8/16/32/64 are required")
    dtype = api.f16
    physical_m = max(16, tile_m)
    half = physical_m // 2
    pitch = half + 1
    chunks = tile_n // 64
    bnsd = layout == "BNSD"
    q_shape = (batch, heads, queries, nope) if bnsd else (batch, queries, heads, nope)
    qr_shape = (*q_shape[:-1], 64)
    k_shape = (batch, 1, keys, nope) if bnsd else (batch, keys, 1, nope)
    kr_shape = (*k_shape[:-1], 64)
    scale = 1.0 / math.sqrt(nope + 64)
    batch_rows = queries * heads
    groups = (batch_rows + tile_m - 1) // tile_m
    items = batch * groups
    qk_split = 128 if tile_n == 128 else 64
    pv_split = 128 if tile_n == 128 else 64

    @api.vf()
    def initialize(total: api.Tensor, maximum: api.Tensor, accumulator: api.Tensor):
        value = api.Reg(api.f32)
        value <<= 0.0
        total <<= value
        for chunk in range(half * nope // 64):
            accumulator[chunk * 64] <<= value
        value <<= -99999.0
        maximum <<= value
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def softmax(score: api.Tensor, total: api.Tensor, maximum: api.Tensor,
                rescale: api.Tensor, probability: api.Tensor, rows: api.Var):
        values = api.RegList(api.f32, chunks)
        halves = api.RegList(dtype, chunks)
        part = api.Reg(api.f32)
        old_max = api.Reg(api.f32)
        old_sum = api.Reg(api.f32)
        new_max = api.Reg(api.f32)
        new_sum = api.Reg(api.f32)
        correction = api.Reg(api.f32)
        packed = api.Reg(dtype)
        unused = api.Reg(dtype)
        for row in range(rows):
            new_max <<= -99999.0
            for chunk in api.unroll(chunks):
                values[chunk] <<= score[row * tile_n + chunk * 64]
                values[chunk] <<= values[chunk] * scale
                part <<= values[chunk].cmax()
                new_max <<= new_max.vmax(part)
            new_max <<= new_max.dup()
            old_max <<= maximum[row].single()
            old_sum <<= total[row].single()
            new_max <<= new_max.vmax(old_max)
            correction <<= old_max - new_max
            correction <<= correction.exp()
            new_sum <<= 0.0
            for chunk in api.unroll(chunks):
                values[chunk] <<= values[chunk] - new_max
                values[chunk] <<= values[chunk].exp()
                part <<= values[chunk].cadd()
                new_sum <<= new_sum + part
                halves[chunk] <<= values[chunk].astype(dtype)
            new_sum <<= new_sum.dup()
            new_sum <<= new_sum + old_sum * correction
            for pair in api.unroll(chunks // 2):
                api.deinterleave(packed, unused, halves[pair * 2], halves[pair * 2 + 1])
                api.reg_to_ub(probability[(pair * 8 * pitch + row) * 16], packed, pitch)
            maximum[row] <<= new_max.single_value()
            total[row] <<= new_sum.single_value()
            rescale[row] <<= correction.single_value()
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def accumulate(accumulator: api.Tensor, product: api.Tensor,
                   rescale: api.Tensor, rows: api.Var):
        acc = api.Reg(api.f32)
        current = api.Reg(api.f32)
        correction = api.Reg(api.f32)
        for row in range(rows):
            correction <<= rescale[row].single()
            for chunk in range(nope // 64):
                acc <<= accumulator[row * nope + chunk * 64]
                current <<= product[row * nope + chunk * 64]
                acc <<= acc * correction + current
                accumulator[row * nope + chunk * 64] <<= acc
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    @api.vf()
    def finish(accumulator: api.Tensor, total: api.Tensor, output: api.Tensor, rows: api.Var):
        value = api.Reg(api.f32)
        denominator = api.Reg(api.f32)
        packed = api.Reg(dtype)
        for row in range(rows):
            denominator <<= total[row].single()
            for chunk in range(nope // 64):
                value <<= accumulator[row * nope + chunk * 64]
                value <<= value / denominator
                packed <<= value.astype(dtype)
                api.reg_to_ub_downsample(output[row * nope + chunk * 64], packed)

    @api.kernel(mode="mix")
    def mla_decode_pipeline(q_nope: api.GM[dtype, q_shape], q_rope: api.GM[dtype, qr_shape],
                           k_nope: api.GM[dtype, k_shape], k_rope: api.GM[dtype, kr_shape],
                           v: api.GM[dtype, k_shape], output: api.GM[dtype, q_shape]):
        qn = api.Tensor(dtype, [physical_m, nope], api.Position.L1)
        qr = api.Tensor(dtype, [physical_m, 64], api.Position.L1)
        kn = api.DBuff(dtype, [tile_n, nope], api.Position.L1)
        kr = api.DBuff(dtype, [tile_n, 64], api.Position.L1)
        p = api.DBuff(dtype, [physical_m, tile_n], api.Position.L1)
        score = api.DBuff(api.f32, [physical_m, tile_n], api.Position.L0C)
        product = api.Tensor(api.f32, [physical_m, nope], api.Position.L0C)
        ub_score = api.DBuff(api.f32, [half, tile_n], api.Position.UB)
        ub_product = api.Tensor(api.f32, [half, nope], api.Position.UB)
        ub_p = api.DBuff(dtype, [pitch, tile_n], api.Position.UB)
        total = api.Tensor(api.f32, [1, 64], api.Position.UB)
        maximum = api.Tensor(api.f32, [1, 64], api.Position.UB)
        rescale = api.DBuff(api.f32, [1, 64], api.Position.UB)
        accumulator = api.Tensor(api.f32, [half, nope], api.Position.UB)
        ub_output = api.Tensor(dtype, [half, nope], api.Position.UB)
        qk_handoff = api.CvMutex(0, depth=2, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        p_handoff = api.VcMutex(1, depth=2, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
        pv_handoff = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        begin = api.Var(items * api.GetCubeIdx() // api.GetCubeNum())
        end = api.Var(items * (api.GetCubeIdx() + 1) // api.GetCubeNum())
        lane_begin = api.Var(api.GetSubBlockIdx() * half)
        with api.auto_sync():
            for item in range(begin, end):
                bi = api.Var(item // groups)
                first_row = api.Var((item % groups) * tile_m)
                rows = api.Var(api.Min(tile_m, batch_rows - first_row))
                local_rows = api.Var(api.Min(half, api.Max(rows - lane_begin, 0)))
                if rows < physical_m:
                    api.set_constant_to_l1(qn, 0)
                    api.set_constant_to_l1(qr, 0)
                    api.bar_all()
                qn[:rows, :] <<= q_nope.view([rows, nope], [nope, 1],
                                           (bi * batch_rows + first_row) * nope)
                qr[:rows, :] <<= q_rope.view([rows, 64], [64, 1],
                                           (bi * batch_rows + first_row) * 64)
                initialize(total, maximum, accumulator)
                for tick in range(keys // tile_n + 1):
                    if tick < keys // tile_n:
                        producer = api.Var(tick)
                        key = api.Var(producer * tile_n)
                        if rows < physical_m:
                            api.set_constant_to_l1(p[producer], 0)
                        kn[producer] <<= k_nope.view([tile_n, nope], [nope, 1], (bi * keys + key) * nope)
                        kr[producer] <<= k_rope.view([tile_n, 64], [64, 1], (bi * keys + key) * 64)
                        api.matmul(score[producer], qn, kn[producer], m=physical_m, n=tile_n, k=nope,
                                   splitk=qk_split, is_init=True)
                        api.matmul(score[producer], qr, kr[producer], m=physical_m, n=tile_n, k=64, is_init=False)
                        qk_handoff.lock()
                        ub_score[producer] <<= score[producer]
                        qk_handoff.ready()
                        qk_handoff.wait()
                        if local_rows > 0:
                            softmax(ub_score[producer], total, maximum, rescale[producer], ub_p[producer], local_rows)
                        qk_handoff.free()
                        p_handoff.lock()
                        if local_rows > 0:
                            p[producer][lane_begin:lane_begin + local_rows, :] <<= ub_p[producer][:local_rows, :].nz()
                        p_handoff.ready()
                    if tick > 0:
                        consumer = api.Var(tick - 1)
                        p_handoff.wait()
                        api.matmul(product, p[consumer], kn[consumer].T, m=physical_m, n=nope,
                                   k=tile_n, splitn=pv_split)
                        p_handoff.free()
                        pv_handoff.lock()
                        ub_product <<= product
                        pv_handoff.ready()
                        pv_handoff.wait()
                        if local_rows > 0:
                            accumulate(accumulator, ub_product, rescale[consumer], local_rows)
                        pv_handoff.free()
                if local_rows > 0:
                    finish(accumulator, total, ub_output, local_rows)
                    destination = output.view([local_rows, nope], [nope, 1],
                                              (bi * batch_rows + first_row + lane_begin) * nope)
                    destination <<= ub_output[:local_rows, :]
        return output

    return mla_decode_pipeline


def make_decode_preload_kernel(*args):
    return _build_decode(*args, tile_m=64, tile_n=128)

# ----------------------------------------------------------------------------------------------------
# online_prefetch.py
# M128 online MLA with KV prefetch and streamed PV feature fragments.
#
# FP32 online state remains in UB. Inputs preserve their original 4-D ABI and
# layout; causal query positions are recovered from each physical row.
# No host packing, GM workspace, extra launch, or integer rescaling is used.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _build_online_prefetch(dtype_name, layout, batch, queries, keys, heads, nope, causal,
          tile_m=128, tile_n=128):
    if dtype_name not in ("float16", "bfloat16") or nope != 512:
        raise ValueError("This schedule requires FP16/BF16 Dn512")
    if layout not in ("BSND", "BNSD") or min(batch, queries, keys, heads) <= 0:
        raise ValueError("Positive BSND/BNSD dimensions are required")
    if causal and queries > keys:
        raise ValueError("Causal rows require at least one visible key")
    if tile_m != 128 or tile_n != 128 or keys % tile_n:
        raise ValueError("This schedule requires M128 and aligned KV128")
    dtype = api.f16 if dtype_name == "float16" else api.bf16
    physical_m = max(16, tile_m)
    half = physical_m // 2
    pitch = half + 1
    chunks = tile_n // 64
    bnsd = layout == "BNSD"
    q_shape = (batch, heads, queries, nope) if bnsd else (batch, queries, heads, nope)
    qr_shape = (*q_shape[:-1], 64)
    k_shape = (batch, 1, keys, nope) if bnsd else (batch, keys, 1, nope)
    kr_shape = (*k_shape[:-1], 64)
    scale = 1.0 / math.sqrt(nope + 64)
    batch_rows = queries * heads
    groups = (batch_rows + tile_m - 1) // tile_m
    items = batch * groups
    qk_split = 128 if tile_n == 128 else 64
    pv_split = 128 if tile_n == 128 else 64

    @api.vf()
    def initialize(total: api.Tensor, maximum: api.Tensor):
        value = api.Reg(api.f32)
        value <<= 0.0
        total <<= value
        value <<= -99999.0
        maximum <<= value
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    def make_softmax(first_tile, apply_mask):
        @api.vf()
        def softmax(score: api.Tensor, total: api.Tensor, maximum: api.Tensor,
                    rescale: api.Tensor, probability: api.Tensor, rows: api.Var,
                    first_row: api.Var, first_key: api.Var):
            values = api.RegList(api.f32, chunks)
            halves = api.RegList(dtype, chunks)
            part = api.Reg(api.f32)
            old_max = api.Reg(api.f32)
            old_sum = api.Reg(api.f32)
            new_max = api.Reg(api.f32)
            new_sum = api.Reg(api.f32)
            correction = api.Reg(api.f32)
            packed = api.Reg(dtype)
            unused = api.Reg(dtype)
            if causal and apply_mask:
                indices = api.Reg(api.i32)
                valid = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
                negative = api.Reg(api.f32)
                zero = api.Reg(api.f32)
                indices.arange(0)
                negative <<= -99999.0
                zero <<= 0.0
            for row in range(rows):
                if causal and apply_mask:
                    if bnsd:
                        # A5-UP-047: a floor remainder inside a vf is spelled through its
                        # quotient. `%` reaches the vendor compiler as its own remainder added
                        # back to a selected divisor, and bisheng aborts selecting instructions
                        # for that shape; the quotient form is the same value and compiles.
                        # The backend refuses the `%` form outright (pypto-pro-mapping #34).
                        flat = api.Var(first_row + row)
                        query = api.Var(flat - queries * (flat // queries))
                    else:
                        query = api.Var((first_row + row) // heads)
                    allowed = api.Var(keys - queries + query + 1 - first_key)
                new_max <<= -99999.0
                for chunk in api.unroll(chunks):
                    values[chunk] <<= score[row * tile_n + chunk * 64]
                    values[chunk] <<= values[chunk] * scale
                    if causal and apply_mask:
                        api.compare(valid, indices, allowed - chunk * 64, api.CompareMode.LT)
                        api.select(values[chunk], values[chunk], negative, mask=valid)
                    part <<= values[chunk].cmax()
                    new_max <<= new_max.vmax(part)
                new_max <<= new_max.dup()
                if first_tile:
                    correction <<= 0.0
                else:
                    old_max <<= maximum[row].single()
                    new_max <<= new_max.vmax(old_max)
                    correction <<= old_max - new_max
                    correction <<= correction.exp()
                new_sum <<= 0.0
                for chunk in api.unroll(chunks):
                    values[chunk] <<= values[chunk] - new_max
                    values[chunk] <<= values[chunk].exp()
                    if causal and apply_mask:
                        api.compare(valid, indices, allowed - chunk * 64, api.CompareMode.LT)
                        api.select(values[chunk], values[chunk], zero, mask=valid)
                    part <<= values[chunk].cadd()
                    new_sum <<= new_sum + part
                    halves[chunk] <<= values[chunk].astype(dtype)
                new_sum <<= new_sum.dup()
                if not first_tile:
                    old_sum <<= total[row].single()
                    new_sum <<= new_sum + old_sum * correction
                for pair in api.unroll(chunks // 2):
                    api.deinterleave(packed, unused, halves[pair * 2], halves[pair * 2 + 1])
                    api.reg_to_ub(probability[(pair * 8 * pitch + row) * 16], packed, pitch)
                maximum[row] <<= new_max.single_value()
                total[row] <<= new_sum.single_value()
                rescale[row] <<= correction.single_value()
            api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)
        return softmax

    softmax_first = make_softmax(True, True)
    softmax_following = make_softmax(False, True)

    softmax_first_unmasked = make_softmax(True, False)
    softmax_following_unmasked = make_softmax(False, False)

    def make_accumulate(first_tile):
        @api.vf()
        def accumulate(accumulator: api.Tensor, product: api.Tensor,
                       rescale: api.Tensor, rows: api.Var, column: api.Var):
            acc = api.Reg(api.f32)
            current = api.Reg(api.f32)
            correction = api.Reg(api.f32)
            for row in range(rows):
                if not first_tile:
                    correction <<= rescale[row].single()
                for chunk in api.unroll(256 // 64):
                    current <<= product[row * 256 + chunk * 64]
                    if first_tile:
                        acc <<= current
                    else:
                        acc <<= accumulator[row * nope + column + chunk * 64]
                        acc <<= acc * correction + current
                    accumulator[row * nope + column + chunk * 64] <<= acc
            api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)
        return accumulate

    accumulate_first = make_accumulate(True)
    accumulate_following = make_accumulate(False)

    @api.vf()
    def finish(accumulator: api.Tensor, total: api.Tensor, output: api.Tensor, rows: api.Var):
        value = api.Reg(api.f32)
        denominator = api.Reg(api.f32)
        packed = api.Reg(dtype)
        for row in range(rows):
            denominator <<= total[row].single()
            for chunk in api.unroll(nope // 64):
                value <<= accumulator[row * nope + chunk * 64]
                value <<= value / denominator
                packed <<= value.astype(dtype)
                api.reg_to_ub_downsample(output[row * nope + chunk * 64], packed)

    @api.kernel(mode="mix")
    def mla_online_m128(q_nope: api.GM[dtype, q_shape], q_rope: api.GM[dtype, qr_shape],
                           k_nope: api.GM[dtype, k_shape], k_rope: api.GM[dtype, kr_shape],
                           v: api.GM[dtype, k_shape], output: api.GM[dtype, q_shape]):
        qn = api.Tensor(dtype, [physical_m, nope], api.Position.L1)
        qr = api.Tensor(dtype, [physical_m, 64], api.Position.L1)
        kn = api.DBuff(dtype, [tile_n, nope], api.Position.L1)
        kr = api.DBuff(dtype, [tile_n, 64], api.Position.L1)
        p = api.Tensor(dtype, [physical_m, tile_n], api.Position.L1)
        score = api.Tensor(api.f32, [physical_m, tile_n], api.Position.L0C)
        product = api.Tensor(api.f32, [physical_m, 256], api.Position.L0C)
        ub_score = api.Tensor(api.f32, [half, tile_n], api.Position.UB)
        ub_product = api.Tensor(api.f32, [half, 256], api.Position.UB)
        ub_p = api.Tensor(dtype, [pitch, tile_n], api.Position.UB)
        total = api.Tensor(api.f32, [1, 64], api.Position.UB)
        maximum = api.Tensor(api.f32, [1, 64], api.Position.UB)
        rescale = api.Tensor(api.f32, [1, 64], api.Position.UB)
        accumulator = api.Tensor(api.f32, [half, nope], api.Position.UB)
        ub_output = ub_product.reinterpret(dtype)
        qk_handoff = api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        p_handoff = api.VcMutex(1, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)
        pv_handoff = api.CvMutex(2, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
        begin = api.Var(items * api.GetCubeIdx() // api.GetCubeNum())
        end = api.Var(items * (api.GetCubeIdx() + 1) // api.GetCubeNum())
        lane_begin = api.Var(api.GetSubBlockIdx() * half)
        with api.auto_sync():
            for item in range(begin, end):
                bi = api.Var(item // groups)
                first_row = api.Var((item % groups) * tile_m)
                rows = api.Var(api.Min(tile_m, batch_rows - first_row))
                local_rows = api.Var(api.Min(half, api.Max(rows - lane_begin, 0)))
                if rows < physical_m:
                    api.set_constant_to_l1(qn, 0)
                    api.set_constant_to_l1(qr, 0)
                    api.bar_all()
                qn[:rows, :] <<= q_nope.view([rows, nope], [nope, 1],
                                           (bi * batch_rows + first_row) * nope)
                qr[:rows, :] <<= q_rope.view([rows, 64], [64, 1],
                                           (bi * batch_rows + first_row) * 64)
                initialize(total, maximum)
                tile_count = api.Var(keys // tile_n)
                if causal:
                    if bnsd:
                        last_query = api.Var(api.Min(queries - 1, first_row % queries + rows - 1))
                        minimum_query = api.Var((first_row + lane_begin) % queries)
                        if minimum_query + local_rows > queries:
                            minimum_query <<= 0
                    else:
                        last_query = api.Var((first_row + rows - 1) // heads)
                        minimum_query = api.Var((first_row + lane_begin) // heads)
                    tile_count <<= api.CeilDiv(keys - queries + last_query + 1, tile_n)
                for tick in range(tile_count + 1):
                    producer = api.Var(tick)
                    key = api.Var(producer * tile_n)
                    if tick < tile_count:
                        kn[producer] <<= k_nope.view([tile_n, nope], [nope, 1], (bi * keys + key) * nope)
                        kr[producer] <<= k_rope.view([tile_n, 64], [64, 1], (bi * keys + key) * 64)
                    if tick > 0:
                        consumer = api.Var(tick - 1)
                        p_handoff.wait()
                        for part in api.unroll(2):
                            api.matmul(product, p, kn[consumer][:, part * 256:(part + 1) * 256].T,
                                       m=physical_m, n=256, k=tile_n, splitn=128)
                            pv_handoff.lock()
                            ub_product <<= product
                            pv_handoff.ready()
                            pv_handoff.wait()
                            if local_rows > 0:
                                if consumer == 0:
                                    accumulate_first(accumulator, ub_product, rescale, local_rows, part * 256)
                                else:
                                    accumulate_following(accumulator, ub_product, rescale, local_rows, part * 256)
                            if part == 0:
                                pv_handoff.free()
                            else:
                                if tick < tile_count:
                                    pv_handoff.free()
                        p_handoff.free()
                    if tick < tile_count:
                        if rows < physical_m:
                            api.set_constant_to_l1(p, 0)
                            api.bar_all()
                        api.matmul(score, qn, kn[producer], m=physical_m, n=tile_n, k=nope,
                                   splitk=qk_split, is_init=True)
                        api.matmul(score, qr, kr[producer], m=physical_m, n=tile_n, k=64, is_init=False)
                        qk_handoff.lock()
                        ub_score <<= score
                        qk_handoff.ready()
                        qk_handoff.wait()
                        if local_rows > 0:
                            if causal:
                                if keys - queries + minimum_query + 1 >= key + tile_n:
                                    if key == 0:
                                        softmax_first_unmasked(ub_score, total, maximum, rescale,
                                                              ub_p, local_rows, first_row + lane_begin, key)
                                    else:
                                        softmax_following_unmasked(ub_score, total, maximum, rescale,
                                                                  ub_p, local_rows, first_row + lane_begin, key)
                                else:
                                    if key == 0:
                                        softmax_first(ub_score, total, maximum, rescale,
                                                      ub_p, local_rows, first_row + lane_begin, key)
                                    else:
                                        softmax_following(ub_score, total, maximum, rescale,
                                                          ub_p, local_rows, first_row + lane_begin, key)
                            else:
                                if key == 0:
                                    softmax_first(ub_score, total, maximum, rescale,
                                                  ub_p, local_rows, first_row + lane_begin, key)
                                else:
                                    softmax_following(ub_score, total, maximum, rescale,
                                                      ub_p, local_rows, first_row + lane_begin, key)
                        qk_handoff.free()
                        p_handoff.lock()
                        if local_rows > 0:
                            p[lane_begin:lane_begin + local_rows, :] <<= ub_p[:local_rows, :].nz()
                        p_handoff.ready()
                if local_rows > 0:
                    finish(accumulator, total, ub_output, local_rows)
                    destination = output.view([local_rows, nope], [nope, 1],
                                              (bi * batch_rows + first_row + lane_begin) * nope)
                    destination <<= ub_output[:local_rows, :]
                # The final product lease also owns its reinterpreted output.
                # Join its actual MTE3 reader before allowing the next FIX.
                with api.vec_scope():
                    api.bar_all()
                pv_handoff.free()
        return output

    return mla_online_m128


def make_online_prefetch_kernel(*args):
    return _build_online_prefetch(*args, tile_m=128, tile_n=128)

# ----------------------------------------------------------------------------------------------------
# online_paired.py
# Paired M64 MLA with core intervals grouped by even item ceilings.
#
# FP32 online state remains in UB. Inputs preserve their original 4-D ABI and
# layout; causal query positions are recovered from each physical row.
# No host packing, GM workspace, extra launch, or integer rescaling is used.
# ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _build_online_paired(dtype_name, layout, batch, queries, keys, heads, nope, causal,
          tile_m=64, tile_n=128):
    if dtype_name not in ("float16", "bfloat16") or nope != 512:
        raise ValueError("This schedule requires FP16/BF16 Dn512")
    if layout not in ("BSND", "BNSD") or min(batch, queries, keys, heads) <= 0:
        raise ValueError("Positive BSND/BNSD dimensions are required")
    if causal and queries > keys:
        raise ValueError("Causal rows require at least one visible key")
    if tile_m != 64 or tile_n != 128 or keys % tile_n:
        raise ValueError("This schedule requires M64 and aligned KV128")
    dtype = api.f16 if dtype_name == "float16" else api.bf16
    physical_m = max(16, tile_m)
    half = physical_m // 2
    pitch = half + 1
    chunks = tile_n // 64
    bnsd = layout == "BNSD"
    q_shape = (batch, heads, queries, nope) if bnsd else (batch, queries, heads, nope)
    qr_shape = (*q_shape[:-1], 64)
    k_shape = (batch, 1, keys, nope) if bnsd else (batch, keys, 1, nope)
    kr_shape = (*k_shape[:-1], 64)
    scale = 1.0 / math.sqrt(nope + 64)
    batch_rows = queries * heads
    groups = (batch_rows + tile_m - 1) // tile_m
    items = batch * groups
    qk_split = 128 if tile_n == 128 else 64
    pv_split = 128 if tile_n == 128 else 64

    @api.vf()
    def initialize(total: api.Tensor, maximum: api.Tensor):
        value = api.Reg(api.f32)
        value <<= 0.0
        total <<= value
        value <<= -99999.0
        maximum <<= value
        api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)

    def make_softmax(first_tile, apply_mask):
        @api.vf()
        def softmax(score: api.Tensor, total: api.Tensor, maximum: api.Tensor,
                    rescale: api.Tensor, probability: api.Tensor, rows: api.Var,
                    first_row: api.Var, first_key: api.Var):
            values = api.RegList(api.f32, chunks)
            halves = api.RegList(dtype, chunks)
            part = api.Reg(api.f32)
            old_max = api.Reg(api.f32)
            old_sum = api.Reg(api.f32)
            new_max = api.Reg(api.f32)
            new_sum = api.Reg(api.f32)
            correction = api.Reg(api.f32)
            packed = api.Reg(dtype)
            unused = api.Reg(dtype)
            if causal and apply_mask:
                indices = api.Reg(api.i32)
                valid = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
                negative = api.Reg(api.f32)
                zero = api.Reg(api.f32)
                indices.arange(0)
                negative <<= -99999.0
                zero <<= 0.0
            for row in range(rows):
                if causal and apply_mask:
                    if bnsd:
                        # A5-UP-047: a floor remainder inside a vf is spelled through its
                        # quotient. `%` reaches the vendor compiler as its own remainder added
                        # back to a selected divisor, and bisheng aborts selecting instructions
                        # for that shape; the quotient form is the same value and compiles.
                        # The backend refuses the `%` form outright (pypto-pro-mapping #34).
                        flat = api.Var(first_row + row)
                        query = api.Var(flat - queries * (flat // queries))
                    else:
                        query = api.Var((first_row + row) // heads)
                    allowed = api.Var(keys - queries + query + 1 - first_key)
                new_max <<= -99999.0
                for chunk in api.unroll(chunks):
                    values[chunk] <<= score[row * tile_n + chunk * 64]
                    values[chunk] <<= values[chunk] * scale
                    if causal and apply_mask:
                        api.compare(valid, indices, allowed - chunk * 64, api.CompareMode.LT)
                        api.select(values[chunk], values[chunk], negative, mask=valid)
                    part <<= values[chunk].cmax()
                    new_max <<= new_max.vmax(part)
                new_max <<= new_max.dup()
                if first_tile:
                    correction <<= 0.0
                else:
                    old_max <<= maximum[row].single()
                    new_max <<= new_max.vmax(old_max)
                    correction <<= old_max - new_max
                    correction <<= correction.exp()
                new_sum <<= 0.0
                for chunk in api.unroll(chunks):
                    values[chunk] <<= values[chunk] - new_max
                    values[chunk] <<= values[chunk].exp()
                    if causal and apply_mask:
                        api.compare(valid, indices, allowed - chunk * 64, api.CompareMode.LT)
                        api.select(values[chunk], values[chunk], zero, mask=valid)
                    part <<= values[chunk].cadd()
                    new_sum <<= new_sum + part
                    halves[chunk] <<= values[chunk].astype(dtype)
                new_sum <<= new_sum.dup()
                if not first_tile:
                    old_sum <<= total[row].single()
                    new_sum <<= new_sum + old_sum * correction
                for pair in api.unroll(chunks // 2):
                    api.deinterleave(packed, unused, halves[pair * 2], halves[pair * 2 + 1])
                    api.reg_to_ub(probability[(pair * 8 * pitch + row) * 16], packed, pitch)
                maximum[row] <<= new_max.single_value()
                total[row] <<= new_sum.single_value()
                rescale[row] <<= correction.single_value()
            api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)
        return softmax

    softmax_first = make_softmax(True, True)
    softmax_following = make_softmax(False, True)

    softmax_first_unmasked = make_softmax(True, False)
    softmax_following_unmasked = make_softmax(False, False)

    def make_accumulate(first_tile):
        @api.vf()
        def accumulate(accumulator: api.Tensor, product: api.Tensor,
                       rescale: api.Tensor, rows: api.Var, column: api.Var):
            acc = api.Reg(api.f32)
            current = api.Reg(api.f32)
            correction = api.Reg(api.f32)
            for row in range(rows):
                if not first_tile:
                    correction <<= rescale[row].single()
                for chunk in api.unroll(256 // 64):
                    current <<= product[row * 256 + chunk * 64]
                    if first_tile:
                        acc <<= current
                    else:
                        acc <<= accumulator[row * nope + column + chunk * 64]
                        acc <<= acc * correction + current
                    accumulator[row * nope + column + chunk * 64] <<= acc
            api.vf_barrier(api.VfPipe.STORE, api.VfPipe.LOAD)
        return accumulate

    accumulate_first = make_accumulate(True)
    accumulate_following = make_accumulate(False)

    @api.vf()
    def finish(accumulator: api.Tensor, total: api.Tensor, output: api.Tensor, rows: api.Var):
        value = api.Reg(api.f32)
        denominator = api.Reg(api.f32)
        packed = api.Reg(dtype)
        for row in range(rows):
            denominator <<= total[row].single()
            for chunk in api.unroll(nope // 64):
                value <<= accumulator[row * nope + chunk * 64]
                value <<= value / denominator
                packed <<= value.astype(dtype)
                api.reg_to_ub_downsample(output[row * nope + chunk * 64], packed)

    @api.kernel(mode="mix")
    def mla_online_paired(q_nope: api.GM[dtype, q_shape], q_rope: api.GM[dtype, qr_shape],
                          k_nope: api.GM[dtype, k_shape], k_rope: api.GM[dtype, kr_shape],
                          v: api.GM[dtype, k_shape], output: api.GM[dtype, q_shape]):
        qn = api.DBuff(dtype, [physical_m, nope], api.Position.L1)
        qr = api.DBuff(dtype, [physical_m, 64], api.Position.L1)
        kn = api.DBuff(dtype, [tile_n, nope], api.Position.L1)
        kr = api.DBuff(dtype, [tile_n, 64], api.Position.L1)
        p = api.DBuff(dtype, [physical_m, tile_n], api.Position.L1)
        a_l0 = api.DBuff(dtype, [physical_m, 128], api.Position.L0A)
        b_l0 = api.DBuff(dtype, [128, 128], api.Position.L0B)
        score = api.DBuff(api.f32, [physical_m, tile_n], api.Position.L0C)
        product = api.DBuff(api.f32, [physical_m, 256], api.Position.L0C)
        ub_score = api.DBuff(api.f32, [half, tile_n], api.Position.UB)
        ub_product = api.DBuff(api.f32, [half, 256], api.Position.UB)
        ub_p = api.DBuff(dtype, [pitch, tile_n], api.Position.UB)
        total = api.DBuff(api.f32, [1, 64], api.Position.UB)
        maximum = api.DBuff(api.f32, [1, 64], api.Position.UB)
        rescale = api.DBuff(api.f32, [1, 64], api.Position.UB)
        accumulator = api.DBuff(api.f32, [half, nope], api.Position.UB)
        qk_handoffs = [api.CvMutex(0, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V),
                       api.CvMutex(1, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)]
        p_handoffs = [api.VcMutex(2, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1),
                      api.VcMutex(3, depth=1, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.MTE1)]
        pv_handoffs = [api.CvMutex(4, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V),
                       api.CvMutex(5, depth=1, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)]
        a_available = [api.SEvent(api.Pipe.M, api.Pipe.MTE1, preset=True) for _ in range(2)]
        b_available = [api.SEvent(api.Pipe.M, api.Pipe.MTE1, preset=True) for _ in range(2)]
        b_ready = [api.SEvent(api.Pipe.MTE1, api.Pipe.M) for _ in range(2)]
        product_ready = [api.SEvent(api.Pipe.M, api.Pipe.FIX) for _ in range(2)]
        core_count = api.GetCubeNum()
        core_index = api.GetCubeIdx()
        ceil_items = api.CeilDiv(items, core_count)
        quantum = 2 - ceil_items % 2
        group_count = api.CeilDiv(items, quantum)
        begin = api.Var(quantum * (group_count * core_index // core_count))
        end = api.Var(api.Min(items, quantum * (group_count * (core_index + 1) // core_count)))
        lane_begin = api.Var(api.GetSubBlockIdx() * half)
        next_item = api.Var(begin)
        with api.auto_sync():
            for item in range(begin, end):
                if item == next_item:
                    bi = api.Var(item // groups)
                    group = api.Var(item % groups)
                    contexts = api.Var(api.Min(2, api.Min(end - item, groups - group)))
                    second_context = contexts > 1
                    present = [True, second_context]
                    next_item <<= item + contexts
                    starts = [api.Var((group + context) * tile_m) for context in range(2)]
                    rows = [api.Var(api.Min(tile_m, api.Max(batch_rows - starts[context], 0)))
                            for context in range(2)]
                    local_rows = [api.Var(api.Min(half, api.Max(rows[context] - lane_begin, 0)))
                                  for context in range(2)]
                    context_tiles = [api.Var(keys // tile_n) for _ in range(2)]
                    minimum_queries = [api.Var(0) for _ in range(2)]
                    for context in api.unroll(2):
                        q_load_context = present[context]
                        if q_load_context:
                            if rows[context] < physical_m:
                                api.set_constant_to_l1(qn[context], 0)
                                api.set_constant_to_l1(qr[context], 0)
                                api.bar_all()
                            # A context with no rows reaches its load with a zero extent, and
                            # PyPTO refuses a zero-extent pl.load (A5-UP-048) where cce spells
                            # it as a zero-burst DMA. The guard is exact rather than defensive:
                            # rows == 0 takes the fill above, so both tiles are already zero and
                            # the skipped transfer would have moved nothing.
                            if rows[context] > 0:
                                qn[context][:rows[context], :] <<= q_nope.view(
                                    [rows[context], nope], [nope, 1],
                                    (bi * batch_rows + starts[context]) * nope)
                                qr[context][:rows[context], :] <<= q_rope.view(
                                    [rows[context], 64], [64, 1],
                                    (bi * batch_rows + starts[context]) * 64)
                            initialize(total[context], maximum[context])
                            if causal:
                                if bnsd:
                                    last_query = api.Var(api.Min(queries - 1, starts[context] % queries + rows[context] - 1))
                                    minimum_queries[context] <<= (starts[context] + lane_begin) % queries
                                    if minimum_queries[context] + local_rows[context] > queries:
                                        minimum_queries[context] <<= 0
                                else:
                                    last_query = api.Var((starts[context] + rows[context] - 1) // heads)
                                    minimum_queries[context] <<= (starts[context] + lane_begin) // heads
                                context_tiles[context] <<= api.CeilDiv(keys - queries + last_query + 1, tile_n)
                    tile_count = api.Var(context_tiles[0])
                    if contexts == 2:
                        tile_count <<= api.Max(tile_count, context_tiles[1])
                    for tick in range(tile_count + 1):
                        producer = api.Var(tick)
                        key = api.Var(producer * tile_n)
                        if causal:
                            qk_active = [((True if context == 0 else second_context)
                                                and producer < context_tiles[context]) for context in range(2)]
                        else:
                            qk_active = present
                        if producer < tile_count:
                            kn[producer] <<= k_nope.view([tile_n, nope], [nope, 1], (bi * keys + key) * nope)
                            kr[producer] <<= k_rope.view([tile_n, 64], [64, 1], (bi * keys + key) * 64)
                        if tick > 0:
                            consumer = api.Var(tick - 1)
                            if causal:
                                pv_active = [((True if context == 0 else second_context)
                                                    and consumer < context_tiles[context]) for context in range(2)]
                            else:
                                pv_active = present
                            for context in api.unroll(2):
                                pv_load_context = pv_active[context]
                                if pv_load_context:
                                    p_handoffs[context].wait()
                                    a_available[context].wait()
                                    api.l1_to_l0(a_l0[context], p[context])
                                    p_handoffs[context].free()
                            for part in api.unroll(2):
                                for column in api.unroll(2):
                                    b_available[column].wait()
                                    api.l1_to_l0(
                                        b_l0[part * 2 + column],
                                        kn[consumer][:, part * 256 + column * 128:part * 256 + (column + 1) * 128].T)
                                    b_ready[column].set()
                                    b_ready[column].wait()
                                    for context in api.unroll(2):
                                        pv_mmad_context = pv_active[context]
                                        if pv_mmad_context:
                                            api.mmad(product[context][:, column * 128:(column + 1) * 128],
                                                     a_l0[context], b_l0[part * 2 + column],
                                                     M=physical_m, N=128, K=tile_n, is_init=True)
                                            if column == 1:
                                                product_ready[context].set()
                                                if part == 1:
                                                    a_available[context].set()
                                    b_available[column].set()
                                for context in api.unroll(2):
                                    pv_publish_context = pv_active[context]
                                    if pv_publish_context:
                                        pv_handoffs[context].lock()
                                        product_ready[context].wait()
                                        ub_product[context] <<= product[context]
                                        pv_handoffs[context].ready()
                                        pv_handoffs[context].wait()
                                        if local_rows[context] > 0:
                                            if consumer == 0:
                                                accumulate_first(accumulator[context], ub_product[context], rescale[context],
                                                                 local_rows[context], part * 256)
                                            else:
                                                accumulate_following(accumulator[context], ub_product[context], rescale[context],
                                                                     local_rows[context], part * 256)
                                        if part == 0:
                                            pv_handoffs[context].free()
                                        else:
                                            if consumer + 1 < context_tiles[context]:
                                                pv_handoffs[context].free()
                        if producer < tile_count:
                            for context in api.unroll(2):
                                p_fill_context = qk_active[context]
                                if p_fill_context:
                                    if rows[context] < physical_m:
                                        api.set_constant_to_l1(p[context], 0)
                                        api.bar_all()
                            for dimension in api.unroll(4):
                                b_available[dimension % 2].wait()
                                api.l1_to_l0(b_l0[dimension], kn[producer][:, dimension * 128:(dimension + 1) * 128])
                                b_ready[dimension % 2].set()
                                b_ready[dimension % 2].wait()
                                for context in api.unroll(2):
                                    qk_mmad_context = qk_active[context]
                                    if qk_mmad_context:
                                        a_available[context].wait()
                                        api.l1_to_l0(a_l0[context], qn[context][:, dimension * 128:(dimension + 1) * 128])
                                        api.mmad(score[context], a_l0[context], b_l0[dimension],
                                                 M=physical_m, N=tile_n, K=128, is_init=dimension == 0)
                                        a_available[context].set()
                                b_available[dimension % 2].set()
                            b_available[0].wait()
                            api.l1_to_l0(b_l0[0][:, :64], kr[producer])
                            b_ready[0].set()
                            b_ready[0].wait()
                            for context in api.unroll(2):
                                rope_mmad_context = qk_active[context]
                                if rope_mmad_context:
                                    a_available[context].wait()
                                    api.l1_to_l0(a_l0[context][:, :64], qr[context])
                                    api.mmad(score[context], a_l0[context][:, :64], b_l0[0][:, :64],
                                             M=physical_m, N=tile_n, K=64, is_init=False)
                                    a_available[context].set()
                            b_available[0].set()
                            for context in api.unroll(2):
                                qk_publish_context = qk_active[context]
                                if qk_publish_context:
                                    qk_handoffs[context].lock()
                                    ub_score[context] <<= score[context]
                                    qk_handoffs[context].ready()
                                    qk_handoffs[context].wait()
                                    if local_rows[context] > 0:
                                        if causal:
                                            if keys - queries + minimum_queries[context] + 1 >= key + tile_n:
                                                if key == 0:
                                                    softmax_first_unmasked(ub_score[context], total[context], maximum[context],
                                                                          rescale[context], ub_p[context], local_rows[context],
                                                                          starts[context] + lane_begin, key)
                                                else:
                                                    softmax_following_unmasked(ub_score[context], total[context], maximum[context],
                                                                              rescale[context], ub_p[context], local_rows[context],
                                                                              starts[context] + lane_begin, key)
                                            else:
                                                if key == 0:
                                                    softmax_first(ub_score[context], total[context], maximum[context],
                                                                  rescale[context], ub_p[context], local_rows[context],
                                                                  starts[context] + lane_begin, key)
                                                else:
                                                    softmax_following(ub_score[context], total[context], maximum[context],
                                                                      rescale[context], ub_p[context], local_rows[context],
                                                                      starts[context] + lane_begin, key)
                                        else:
                                            if key == 0:
                                                softmax_first(ub_score[context], total[context], maximum[context],
                                                              rescale[context], ub_p[context], local_rows[context],
                                                              starts[context] + lane_begin, key)
                                            else:
                                                softmax_following(ub_score[context], total[context], maximum[context],
                                                                  rescale[context], ub_p[context], local_rows[context],
                                                                  starts[context] + lane_begin, key)
                                    qk_handoffs[context].free()
                                    p_handoffs[context].lock()
                                    if local_rows[context] > 0:
                                        p[context][lane_begin:lane_begin + local_rows[context], :] <<= ub_p[context][:local_rows[context], :].nz()
                                    p_handoffs[context].ready()
                    for context in api.unroll(2):
                        output_context = present[context]
                        if output_context:
                            if local_rows[context] > 0:
                                alias = ub_product[context].reinterpret(dtype)
                                finish(accumulator[context], total[context], alias, local_rows[context])
                                destination = output.view([local_rows[context], nope], [nope, 1],
                                                          (bi * batch_rows + starts[context] + lane_begin) * nope)
                                destination <<= alias[:local_rows[context], :]
                    # Both independent product leases also own their output
                    # aliases. Retire every MTE3 reader before returning them.
                    with api.vec_scope():
                        api.bar_all()
                    for context in api.unroll(2):
                        release_context = present[context]
                        if release_context:
                            pv_handoffs[context].free()
        return output

    return mla_online_paired


def make_online_paired_kernel(*args):
    return _build_online_paired(*args, tile_m=64, tile_n=128)

# ----------------------------------------------------------------------------------------------------
# mla.py
# Select one declared attention schedule. Every schedule preserves the same 4-D ABI, so a
# case picks one by name and nothing else about the call changes.
# ----------------------------------------------------------------------------------------------------

SCHEDULES = {"serial_nd": "make_serial_nd_kernel",
             "prefill_resident": "make_prefill_resident_kernel",
             "decode_splitkv": "make_decode_splitkv_kernel",
             "decode_preload": "make_decode_preload_kernel",
             "online_prefetch": "make_online_prefetch_kernel",
             "online_paired": "make_online_paired_kernel"}


def make_kernel(dtype_name, layout, batch, queries, keys, heads, nope, causal,
                variant="serial_nd"):
    if variant not in SCHEDULES:
        raise ValueError(f"Unknown MLA schedule: {variant}; declared: {', '.join(SCHEDULES)}")
    return globals()[SCHEDULES[variant]](dtype_name, layout, batch, queries, keys, heads,
                                         nope, causal)
