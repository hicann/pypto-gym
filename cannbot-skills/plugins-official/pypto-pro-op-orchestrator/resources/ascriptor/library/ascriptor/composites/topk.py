# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 radix selection composed entirely from ordinary register/VF operations.

RFC-0014 defines the public domain. The kernel repository owns complete launch
units and independent references; this module has no tensor-library dependency.
"""

from ..frontend.dsl import (
    DT,
    CeilDiv,
    CompareMode,
    HighLowPart,
    HistBin,
    HistMode,
    MaskReg,
    MaskType,
    Reg,
    Var,
    VfPipe,
    add,
    adds,
    arange,
    cadd,
    cmax,
    cmin,
    compare,
    dup,
    gather,
    histograms,
    mask_and,
    pack,
    reg_to_ub_scatter,
    select,
    shiftls,
    shiftrs,
    sub,
    ub_to_reg,
    unsqueeze,
    update_mask,
    vand,
    vf_barrier,
    vmin,
    vor,
    vxor,
)
from ..frontend.dsl import VfFn as _vf
from ..frontend.dsl import marker as _marker

_GROUP = 64  # FP32/UINT32 lanes per register
_LEVELS = ((24, 0x00000000), (16, 0xFF000000),
           (8, 0xFFFF0000), (0, 0xFFFFFF00))


@_vf
def _radix_select_vf(keys, cand, cidx, count: Var, k: Var):
    """Select the ``k`` largest of ``count`` float32 keys into ``cand``/``cidx``."""
    fkey = Reg(DT.float, name="rs_fkey")
    ukey = Reg(DT.uint32, name="rs_ukey")
    sbit = Reg(DT.int, name="rs_sbit")
    flip = Reg(DT.uint32, name="rs_flip")
    hi = Reg(DT.uint32, name="rs_hi")
    byte32 = Reg(DT.uint32, name="rs_byte32")
    halfword = Reg(DT.uint16, name="rs_halfword")
    byte = Reg(DT.uint8, name="rs_byte")
    zero32 = Reg(DT.uint32, name="rs_zero32")
    prefix = Reg(DT.uint32, name="rs_prefix")
    bwide = Reg(DT.uint32, name="rs_bwide")
    bidx32 = Reg(DT.uint32, name="rs_bidx32")
    signbit = Reg(DT.uint32, name="rs_signbit")
    bytemask = Reg(DT.uint32, name="rs_bytemask")
    himask = Reg(DT.uint32, name="rs_himask")
    lowmask = Reg(DT.uint32, name="rs_lowmask")
    bidx16 = Reg(DT.uint16, name="rs_bidx16")
    h0 = Reg(DT.uint16, name="rs_h0")
    h1 = Reg(DT.uint16, name="rs_h1")
    total = Reg(DT.uint16, name="rs_total")
    limit = Reg(DT.uint16, name="rs_limit")
    need = Reg(DT.uint16, name="rs_need")
    lane16 = Reg(DT.int16, name="rs_lane16")
    zeros16 = Reg(DT.uint16, name="rs_zeros16")
    big16 = Reg(DT.uint16, name="rs_big16")
    tmp0 = Reg(DT.uint16, name="rs_tmp0")
    tmp1 = Reg(DT.uint16, name="rs_tmp1")
    red0 = Reg(DT.uint16, name="rs_red0")
    red1 = Reg(DT.uint16, name="rs_red1")
    acc16 = Reg(DT.uint16, name="rs_acc16")
    bcount = Reg(DT.uint16, name="rs_bcount")
    inclb = Reg(DT.uint16, name="rs_inclb")
    lo0 = MaskReg(DT.uint16, init_mode=MaskType.NONE, name="rs_lo0")
    lo1 = MaskReg(DT.uint16, init_mode=MaskType.NONE, name="rs_lo1")
    live8 = MaskReg(DT.uint8, init_mode=MaskType.NONE, name="rs_live8")
    keep = MaskReg(DT.uint32, init_mode=MaskType.NONE, name="rs_keep")
    groups = CeilDiv(count, _GROUP)
    dup(zero32, 0)
    dup(zeros16, 0)
    dup(bidx32, 0)
    dup(signbit, 0x80000000)
    dup(bytemask, 0xFF)
    dup(lowmask, 0xFFFF)
    dup(bidx16, 0)
    dup(big16, 0xFFFF)
    arange(lane16, 1)
    lanep1 = lane16.reinterpret(DT.uint16, name="rs_lanep1")
    dup(prefix, 0)
    dup(need, Var(k, dtype=DT.uint16, name="rs_k16"))

    # The levels are unrolled at trace time so the byte offset and the
    # high-bit mask are immediates rather than scalar registers.
    for shift, hi_mask in _LEVELS:
        dup(h0, 0)
        dup(h1, 0)
        # Level 0 has no high bits: its mask is 0, so the bucket test below is
        # vacuously true and the byte survives.  Emitting it unconditionally
        # keeps each byte level expressed by the same register operations.
        dup(himask, hi_mask)
        for grp in range(groups):
            off = Var(grp * _GROUP, name=f"rs_off_{shift}")
            lanes = Var(_GROUP, dtype=DT.uint32, name=f"rs_lanes_{shift}")
            update_mask(live8, lanes)
            ub_to_reg(fkey, keys[off])
            # float order -> unsigned integer order
            ukey <<= fkey.reinterpret(DT.uint32, name=f"rs_fbits_{shift}")
            sbit <<= fkey.reinterpret(DT.int, name=f"rs_ibits_{shift}")
            shiftrs(sbit, sbit, 31)
            flip <<= sbit.reinterpret(DT.uint32, name=f"rs_sbitu_{shift}")
            vor(flip, flip, signbit)
            vxor(ukey, ukey, flip)
            shiftrs(byte32, ukey, shift)
            vand(byte32, byte32, bytemask)
            # outside the running bucket -> bin 0, which the search ignores
            vand(hi, ukey, himask)
            compare(keep, hi, prefix, CompareMode.EQ)
            select(byte32, byte32, zero32, mask=keep)
            pack(halfword, byte32, HighLowPart.LOWEST)
            pack(byte, halfword, HighLowPart.LOWEST)
            histograms(h0, byte, HistBin.BIN0, HistMode.ACCUMULATE, mask=live8)
            histograms(h1, byte, HistBin.BIN1, HistMode.ACCUMULATE, mask=live8)

        # total = incl[255]: the prefix count never decreases, so its max is
        # its last lane.
        cmax(red1, h1)
        gather(total, red1, bidx16)
        # the largest bin b with #(byte >= b) >= need is popcount(incl <= L)
        sub(limit, total, need)
        compare(lo0, h0, limit, CompareMode.LE)
        compare(lo1, h1, limit, CompareMode.LE)
        # popcount(lo) without a sum-reduce: `incl` never decreases, so the
        # selected lanes are a prefix and the count is the last selected lane
        # index + 1.  A u16 ReduceSum does not exist on dav_c310 (vcadd is
        # u16 -> u32 only), but ReduceMax on u16 does.
        select(tmp0, lanep1, zeros16, mask=lo0)
        select(tmp1, lanep1, zeros16, mask=lo1)
        cmax(red0, tmp0)
        cmax(red1, tmp1)
        # a selected lane in h1 implies all 128 of h0 are selected, so the two
        # window counts simply add.
        add(acc16, red0, red1)
        gather(bcount, acc16, bidx16)
        # incl[b] is the smallest prefix count that is above L
        select(tmp0, big16, h0, mask=lo0)
        select(tmp1, big16, h1, mask=lo1)
        cmin(red0, tmp0)
        cmin(red1, tmp1)
        vmin(acc16, red0, red1)
        gather(inclb, acc16, bidx16)
        # fold the bin into the prefix and drop what it accounted for.
        # The bin count is broadcast across every u16 lane, so a u32 view of
        # the register holds `b | b << 16` in each lane and one AND recovers it
        # -- there is no u16 -> u32 Cast on this part.
        vand(bwide, bcount.reinterpret(DT.uint32, name=f"rs_bcnt32_{shift}"),
             lowmask)
        shiftls(bwide, bwide, shift)
        vor(prefix, prefix, bwide)
        sub(tmp0, total, inclb)
        sub(need, need, tmp0)

    # `prefix` is now the k-th largest key; compact around it.
    lane = Reg(DT.int, name="rs_lane")
    idx = Reg(DT.int, name="rs_idx")
    rank = Reg(DT.uint32, name="rs_rank")
    pos = Reg(DT.uint32, name="rs_pos")
    base_gt = Reg(DT.uint32, name="rs_basegt")
    base_eq = Reg(DT.uint32, name="rs_baseeq")
    cnt = Reg(DT.uint32, name="rs_cnt")
    red32 = Reg(DT.uint32, name="rs_red32")
    one32 = Reg(DT.uint32, name="rs_one32")
    needw = Reg(DT.uint32, name="rs_needw")
    gt = MaskReg(DT.uint32, init_mode=MaskType.NONE, name="rs_gt")
    eq = MaskReg(DT.uint32, init_mode=MaskType.NONE, name="rs_eq")
    room = MaskReg(DT.uint32, init_mode=MaskType.NONE, name="rs_room")
    take = MaskReg(DT.uint32, init_mode=MaskType.NONE, name="rs_take")

    dup(one32, 1)
    dup(base_gt, 0)
    # elements strictly above the threshold occupy the first k-need slots
    vand(needw, need.reinterpret(DT.uint32, name="rs_need32"), lowmask)
    sub(base_eq, zero32, needw)
    adds(base_eq, base_eq, k)

    for grp2 in range(groups):
        off2 = Var(grp2 * _GROUP, name="rs_off2")
        ub_to_reg(fkey, keys[off2])
        ukey <<= fkey.reinterpret(DT.uint32, name="rs_fbits2")
        sbit <<= fkey.reinterpret(DT.int, name="rs_ibits2")
        shiftrs(sbit, sbit, 31)
        flip <<= sbit.reinterpret(DT.uint32, name="rs_sbitu2")
        vor(flip, flip, signbit)
        vxor(ukey, ukey, flip)
        arange(lane, 0)
        adds(idx, lane, off2)
        # Scatter index payloads as UINT32 bits. PyPTO's signed-data scatter
        # casts offsets to INT32 on some CANN versions (A5-UP-037).
        index_bits = idx.reinterpret(DT.uint32)

        # No predicate: a predicated compare PRESERVES its inactive lanes, so a
        # tail group would inherit the previous group's bits.  The trailing
        # lanes are padded to -inf, whose key is the smallest possible, so an
        # unpredicated compare excludes them anyway.
        compare(gt, ukey, prefix, CompareMode.GT)
        unsqueeze(rank, gt)
        add(pos, rank, base_gt)
        reg_to_ub_scatter(cand, fkey, pos, mask=gt)
        reg_to_ub_scatter(cidx, index_bits, pos, mask=gt)
        select(cnt, one32, zero32, mask=gt)
        cadd(red32, cnt)
        gather(cnt, red32, bidx32)
        add(base_gt, base_gt, cnt)

        compare(eq, ukey, prefix, CompareMode.EQ)
        unsqueeze(rank, eq)
        add(pos, rank, base_eq)
        compare(room, pos, k, CompareMode.LT)
        mask_and(take, room, eq)
        reg_to_ub_scatter(cand, fkey, pos, mask=take)
        reg_to_ub_scatter(cidx, index_bits, pos, mask=take)
        select(cnt, one32, zero32, mask=eq)
        cadd(red32, cnt)
        gather(cnt, red32, bidx32)
        add(base_eq, base_eq, cnt)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@_marker("composite.radix_topk")
def radix_topk(dst_values, dst_indices, src, count, k, *, largest=True, sorted=False):
    """Select k largest finite FP32 values and INT32 input indices, unordered.

    Call from an A5 kernel with distinct, complete UB buffers: src[1,4096],
    dst_values[1,512], dst_indices[1,512]. Require 1 <= count <= 4096 and
    1 <= k <= min(count,512); pad src[count:] with negative infinity. Only
    the first k outputs are defined. Input bits are preserved; tied selection
    and output order are unspecified. Only largest=True, sorted=False is
    supported. The caller owns DMA ordering and runtime input validation.
    """
    # Carry the original output address into VF; do not create a callee-local
    # reinterpret of a potentially indexed tile-group element (D-126).
    index_bits = dst_indices.reinterpret(DT.uint32)
    _radix_select_vf(src, dst_values, index_bits, count, k)
