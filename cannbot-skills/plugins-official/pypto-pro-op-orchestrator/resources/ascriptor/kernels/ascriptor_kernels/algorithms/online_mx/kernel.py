# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two online FP32 -> MX (E4M3 payload, E8M0 scale) quantize-and-multiply paths, plus the
two leaf kernels that publish the raw payload and scale bytes of the same VFs."""

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# nd.py
# ----------------------------------------------------------------------------------------------------

# Mechanically retained through tools/port_kernel.Source; complete corrected VF and V-to-C bodies.
# Ported from the old repository's kernels/a5/matmul/float_to_mxfp8_online_cast_matmul.py by tools/port_kernel.py; hand edits welcome.
# Hand edit 2026-08-28: the packed scale read is a masked block load of the 128-byte staging tile (the hardware lint of D-051).
asc_range = range  # the old `range as asc_range`: loops over dynamic bounds compile to cf.for
ND_M = 16
ND_N = 16
K = 64
GROUP = 32
GROUPS_PER_ROW = K // GROUP
EXP_MASK = 0x7F800000


@vf()
def online_cast_f32_to_mxfp8_e4m3_k64(
    src_f32: Tensor,
    src_u32: Tensor,
    dst_e4m3: Tensor,
    dst_scale_u8: Tensor,
    scale_bits_u32: Tensor,
    scale_bits_f32: Tensor,
    scale_bits_u8: Tensor,
    rows: Var,
):
    mask_f32_g32 = MaskReg(DT.float, init_mode=MaskType.LOWEST32)
    mask_u32_g32 = MaskReg(DT.uint32, init_mode=MaskType.LOWEST32)
    mask_e4m3_pack32 = MaskReg(DT.e4m3, init_mode=MaskType.LOWEST128)
    mask_u8_pack32 = MaskReg(DT.uint8, init_mode=MaskType.LOWEST128)

    src_reg = Reg(DT.float)
    norm_reg = Reg(DT.float)
    scale_reg = Reg(DT.float)
    scale_dup_reg = Reg(DT.float)
    fp8_reg = Reg(DT.e4m3)

    src_bits_reg = Reg(DT.uint32)
    exp_mask_reg = Reg(DT.uint32)
    reciprocal_base_reg = Reg(DT.uint32)
    reciprocal_bits_reg = Reg(DT.uint32)
    exp_bits_reg = Reg(DT.uint32)
    max_exp_bits_reg = Reg(DT.uint32)
    scale_byte_bits_reg = Reg(DT.uint32)
    packed_scale_reg = Reg(DT.uint8)

    cfg_zero = CastConfig(reg_layout=RegLayout.ZERO, name="cfg_online_e4m3")
    exp_mask_reg <<= EXP_MASK
    reciprocal_base_reg <<= 0x7F000000

    for r in asc_range(rows):
        for g in asc_range(GROUPS_PER_ROW):
            src_off = Var(r * K + g * GROUP)
            scale_off = Var(r * GROUPS_PER_ROW + g)

            ub_to_reg(src_reg, src_f32[src_off], mask=mask_f32_g32)
            ub_to_reg(src_bits_reg, src_u32[src_off], mask=mask_u32_g32)
            vand(exp_bits_reg, src_bits_reg, exp_mask_reg, mask=mask_u32_g32)
            cmax(max_exp_bits_reg, exp_bits_reg, mask=mask_u32_g32)

            # A5 native vdiv does not preserve this source's IEEE subnormal
            # normalization. Keep the group exponent E, but construct the exact
            # NORMAL reciprocal 2^(127-E) from (254-E)<<23. E is0..167 in
            # the unchanged finite input domain, so the reciprocal is finite.
            sub(reciprocal_bits_reg, reciprocal_base_reg, max_exp_bits_reg, mask=mask_u32_g32)
            scale_bits_u32[scale_off] <<= reciprocal_bits_reg.single_value()
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            scale_reg <<= scale_bits_f32[scale_off].single()
            dup(scale_dup_reg, scale_reg)
            mul(norm_reg, src_reg, scale_dup_reg, mask=mask_f32_g32)
            cast(fp8_reg, norm_reg, cfg_zero, mask=mask_e4m3_pack32)
            dst_e4m3[src_off] <<= mask_e4m3_pack32 * fp8_reg.pack4()

            vf_barrier(VfPipe.STORE, VfPipe.STORE)
            shiftrs(scale_byte_bits_reg, max_exp_bits_reg, 23)
            scale_bits_u32[scale_off] <<= scale_byte_bits_reg.single_value()

    # The packed read below aliases the staging slots written above; without a
    # store->load barrier the youngest store (the last row's second group) is
    # still in flight on hardware and the read returns the stale byte. The
    # functional model and lowered pipeline checks observe different scheduling details.
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    # The staging tile is 128 bytes (rows x groups of u32); a whole-register read
    # would take the next tile's 128 bytes into the upper lanes, so read only the
    # four 32-byte blocks that exist (the same mask packs them below; D-051).
    ub_to_reg(packed_scale_reg, scale_bits_u8[0], mask=mask_u8_pack32)
    dst_scale_u8[0] <<= mask_u8_pack32 * packed_scale_reg.pack4()


def make_nd_matmul_kernel(items=1, *, batched=False):
    """Keep the original ABI; batched controls exercise the same rotating body."""
    if items < 1 or (not batched and items != 1):
        raise ValueError("Multiple work items require batched inputs and outputs")
    input_shape = (items, 16, 64) if batched else (16, 64)
    output_shape = (items, 16, 16) if batched else (16, 16)

    @kernel()
    def float_to_mxfp8_online_cast_matmul_kernel(x: GM[f32, input_shape], y: GM[f32, input_shape], z: GM[f32, output_shape], dummy: i32):
        l1x_buf = DBuff(DT.e4m3, [ND_M, K], Position.L1)
        l1y_buf = DBuff(DT.e4m3, [ND_N, K], Position.L1)
        scale_x_buf = DBuff(DT.uint8, [ND_M, GROUPS_PER_ROW], Position.L1)
        scale_y_buf = DBuff(DT.uint8, [ND_N, GROUPS_PER_ROW], Position.L1)
        vcmutex = VcMutex(0, guards=l1x_buf, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
        l0z = Tensor(DT.float, [ND_M, ND_N], Position.L0C)

        ub_x_f32 = Tensor(DT.float, [ND_M, K], Position.UB)
        ub_y_f32 = Tensor(DT.float, [ND_N, K], Position.UB)
        ub_x_u32 = ub_x_f32.reinterpret(DT.uint32)
        ub_y_u32 = ub_y_f32.reinterpret(DT.uint32)
        ub_x_fp8 = Tensor(DT.e4m3, [ND_M, K], Position.UB)
        ub_y_fp8 = Tensor(DT.e4m3, [ND_N, K], Position.UB)

        scale_bits_x_u32 = Tensor(DT.uint32, [ND_M, GROUPS_PER_ROW], Position.UB)
        scale_bits_y_u32 = Tensor(DT.uint32, [ND_N, GROUPS_PER_ROW], Position.UB)
        scale_bits_x_f32 = scale_bits_x_u32.reinterpret(DT.float)
        scale_bits_y_f32 = scale_bits_y_u32.reinterpret(DT.float)
        scale_bits_x_u8 = scale_bits_x_u32.reinterpret(DT.uint8)
        scale_bits_y_u8 = scale_bits_y_u32.reinterpret(DT.uint8)
        ub_scale_x = Tensor(DT.uint8, [ND_M, GROUPS_PER_ROW], Position.UB)
        ub_scale_y = Tensor(DT.uint8, [ND_N, GROUPS_PER_ROW], Position.UB)

        work_items = Var(items)
        slot = Var(0)
        items_per_core = CeilDiv(work_items, GetCubeNum())
        item_begin = Var(items_per_core * GetCubeIdx())
        item_end = Min(item_begin + items_per_core, work_items)

        with auto_sync():
            for _item in asc_range(item_begin, item_end):
                l1x = l1x_buf[slot]
                l1y = l1y_buf[slot]
                scale_x = scale_x_buf[slot]
                scale_y = scale_y_buf[slot]
                item_x = x[_item, :, :] if batched else x
                item_y = y[_item, :, :] if batched else y
                item_z = z[_item, :, :] if batched else z
                vcmutex.lock()
                ub_x_f32[:, :] <<= item_x[:, :]
                ub_y_f32[:, :] <<= item_y[:, :]
                online_cast_f32_to_mxfp8_e4m3_k64(
                    ub_x_f32, ub_x_u32, ub_x_fp8, ub_scale_x,
                    scale_bits_x_u32, scale_bits_x_f32, scale_bits_x_u8, ND_M,
                )
                online_cast_f32_to_mxfp8_e4m3_k64(
                    ub_y_f32, ub_y_u32, ub_y_fp8, ub_scale_y,
                    scale_bits_y_u32, scale_bits_y_f32, scale_bits_y_u8, ND_N,
                )
                # Both sub-blocks compute identical tiles; only sub 0 publishes them so the L1
                # stores have a single writer (the ready() below stays outside: the cube waits
                # for BOTH subs' ready, and sub 1's fires on an empty MTE3 queue).
                if GetSubBlockIdx() == 0:
                    ub_to_l1_nd2nz(l1x, ub_x_fp8, m_dst=ND_M, n_dst=K, m_src=ND_M, n_src=K, N_src=K)
                    ub_to_l1_nd2nz(l1y, ub_y_fp8, m_dst=ND_N, n_dst=K, m_src=ND_N, n_src=K, N_src=K)
                    ub_to_l1(scale_x, ub_scale_x, n_burst=1, burst_len=1, src_stride=0, dst_stride=0)
                    ub_to_l1(scale_y, ub_scale_y, n_burst=1, burst_len=1, src_stride=0, dst_stride=0)
                vcmutex.ready()

                vcmutex.wait()
                matmul_mx(l0z, l1x, l1y, scale_x, scale_y, m=ND_M, n=ND_N, k=K, is_init=True)
                item_z[:, :] <<= l0z
                vcmutex.free()
                slot += 1

        return z

    return float_to_mxfp8_online_cast_matmul_kernel


float_to_mxfp8_online_cast_matmul_kernel = make_nd_matmul_kernel()

# ----------------------------------------------------------------------------------------------------
# transposed.py
# ----------------------------------------------------------------------------------------------------

# Mechanically retained through tools/port_kernel.Source; complete corrected VF and V-to-C bodies.
# Ported from the old repository's kernels/a5/matmul/float_to_mxfp8_online_cast_transpose_matmul.py by tools/port_kernel.py; hand edits welcome.
TRANSPOSED_M = 32
TRANSPOSED_N = 32


@vf()
def online_cast_f32_krows_to_mxfp8_e4m3_k64(
    src_f32: Tensor,
    src_u32: Tensor,
    dst_e4m3_u16: Tensor,
    dst_scale_u8: Tensor,
    scale_bits_u32: Tensor,
    scale_bits_f32: Tensor,
    scale_bits_u8: Tensor,
    index_bits_u32: Tensor,
    index_bits_i32: Tensor,
    index_bits_u16: Tensor,
    index_bits_i16: Tensor,
    packed_fp8: Tensor,
    packed_fp8_u8: Tensor,
    pair_bytes_u8: Tensor,
    pair_words_u16: Tensor,
    rows: Var,
    rows_half: Var,
):
    mask_f32_g32 = MaskReg(DT.float, init_mode=MaskType.LOWEST32)
    mask_u32_g32 = MaskReg(DT.uint32, init_mode=MaskType.LOWEST32)
    mask_e4m3_g32 = MaskReg(DT.e4m3, init_mode=MaskType.LOWEST32)
    mask_e4m3_pack32 = MaskReg(DT.e4m3, init_mode=MaskType.LOWEST128)
    mask_u8_g32 = MaskReg(DT.uint8, init_mode=MaskType.LOWEST32)
    mask_u8_pair128 = MaskReg(DT.uint8, init_mode=MaskType.LOWEST128)
    mask_u16_g32 = MaskReg(DT.uint16, init_mode=MaskType.LOWEST32)
    mask_u8_pack64 = MaskReg(DT.uint8, init_mode=MaskType.ALL)

    src_reg = Reg(DT.float)
    norm_reg = Reg(DT.float)
    scale_reg = Reg(DT.float)
    scale_dup_reg = Reg(DT.float)
    fp8_reg = Reg(DT.e4m3)

    src_bits_reg = Reg(DT.uint32)
    exp_mask_reg = Reg(DT.uint32)
    reciprocal_base_reg = Reg(DT.uint32)
    reciprocal_bits_reg = Reg(DT.uint32)
    exp_bits_reg = Reg(DT.uint32)
    max_exp_bits_reg = Reg(DT.uint32)
    scale_byte_bits_reg = Reg(DT.uint32)
    packed_scale_reg = Reg(DT.uint8)

    data_index_i32 = Reg(DT.int)
    data_index_u32 = Reg(DT.uint32)
    pair_index_i16 = Reg(DT.int16)
    pair_index_u16 = Reg(DT.uint16)
    packed_even_u8 = Reg(DT.uint8)
    packed_odd_u8 = Reg(DT.uint8)
    pair_lo_u8 = Reg(DT.uint8)
    pair_hi_u8 = Reg(DT.uint8)
    pair_u16_reg = Reg(DT.uint16)

    cfg_zero = CastConfig(reg_layout=RegLayout.ZERO, name="cfg_online_transpose_e4m3")
    exp_mask_reg <<= EXP_MASK
    reciprocal_base_reg <<= 0x7F000000

    for p in asc_range(rows_half):
        for g in asc_range(GROUPS_PER_ROW):
            row_even = Var(p * 2)
            row_odd = Var(p * 2 + 1)
            group_base_even = Var(g * GROUP * rows + row_even)
            group_base_odd = Var(g * GROUP * rows + row_odd)
            scale_off_even = Var(row_even * GROUPS_PER_ROW + g)
            scale_off_odd = Var(row_odd * GROUPS_PER_ROW + g)

            data_index_i32.arange(0)
            data_index_i32 <<= data_index_i32 * rows + group_base_even
            reg_to_ub(index_bits_i32, data_index_i32)
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            ub_to_reg(data_index_u32, index_bits_u32)

            ub_to_reg_gather(src_reg, src_f32, data_index_u32, mask=mask_f32_g32)
            ub_to_reg_gather(src_bits_reg, src_u32, data_index_u32, mask=mask_u32_g32)
            vand(exp_bits_reg, src_bits_reg, exp_mask_reg, mask=mask_u32_g32)
            cmax(max_exp_bits_reg, exp_bits_reg, mask=mask_u32_g32)

            # A5 native vdiv does not preserve this source's IEEE subnormal
            # normalization. Keep the group exponent E, but construct the exact
            # NORMAL reciprocal 2^(127-E) from (254-E)<<23. E is0..167 in
            # the unchanged finite input domain, so the reciprocal is finite.
            sub(reciprocal_bits_reg, reciprocal_base_reg, max_exp_bits_reg, mask=mask_u32_g32)
            scale_bits_u32[scale_off_even] <<= reciprocal_bits_reg.single_value()
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            scale_reg <<= scale_bits_f32[scale_off_even].single()
            dup(scale_dup_reg, scale_reg)
            mul(norm_reg, src_reg, scale_dup_reg, mask=mask_f32_g32)
            cast(fp8_reg, norm_reg, cfg_zero, mask=mask_e4m3_pack32)
            reg_to_ub_pack4(packed_fp8, fp8_reg, mask=mask_e4m3_pack32)
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            ub_to_reg(packed_even_u8, packed_fp8_u8, mask=mask_u8_g32)

            vf_barrier(VfPipe.STORE, VfPipe.STORE)
            shiftrs(scale_byte_bits_reg, max_exp_bits_reg, 23)
            scale_bits_u32[scale_off_even] <<= scale_byte_bits_reg.single_value()

            data_index_i32.arange(0)
            data_index_i32 <<= data_index_i32 * rows + group_base_odd
            reg_to_ub(index_bits_i32, data_index_i32)
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            ub_to_reg(data_index_u32, index_bits_u32)

            ub_to_reg_gather(src_reg, src_f32, data_index_u32, mask=mask_f32_g32)
            ub_to_reg_gather(src_bits_reg, src_u32, data_index_u32, mask=mask_u32_g32)
            vand(exp_bits_reg, src_bits_reg, exp_mask_reg, mask=mask_u32_g32)
            cmax(max_exp_bits_reg, exp_bits_reg, mask=mask_u32_g32)

            # A5 native vdiv does not preserve this source's IEEE subnormal
            # normalization. Keep the group exponent E, but construct the exact
            # NORMAL reciprocal 2^(127-E) from (254-E)<<23. E is0..167 in
            # the unchanged finite input domain, so the reciprocal is finite.
            sub(reciprocal_bits_reg, reciprocal_base_reg, max_exp_bits_reg, mask=mask_u32_g32)
            scale_bits_u32[scale_off_odd] <<= reciprocal_bits_reg.single_value()
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            scale_reg <<= scale_bits_f32[scale_off_odd].single()
            dup(scale_dup_reg, scale_reg)
            mul(norm_reg, src_reg, scale_dup_reg, mask=mask_f32_g32)
            cast(fp8_reg, norm_reg, cfg_zero, mask=mask_e4m3_pack32)
            reg_to_ub_pack4(packed_fp8, fp8_reg, mask=mask_e4m3_pack32)
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            ub_to_reg(packed_odd_u8, packed_fp8_u8, mask=mask_u8_g32)

            vf_barrier(VfPipe.STORE, VfPipe.STORE)
            shiftrs(scale_byte_bits_reg, max_exp_bits_reg, 23)
            scale_bits_u32[scale_off_odd] <<= scale_byte_bits_reg.single_value()

            interleave(pair_lo_u8, pair_hi_u8, packed_even_u8, packed_odd_u8)
            reg_to_ub(pair_bytes_u8, pair_lo_u8, mask=mask_u8_pair128)
            pair_base = Var(g * GROUP * rows_half + p)
            pair_index_i16.arange(0)
            pair_index_i16 <<= pair_index_i16 * rows_half + pair_base
            reg_to_ub(index_bits_i16, pair_index_i16)
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            ub_to_reg(pair_u16_reg, pair_words_u16, mask=mask_u16_g32)
            ub_to_reg(pair_index_u16, index_bits_u16, mask=mask_u16_g32)
            reg_to_ub_scatter(dst_e4m3_u16, pair_u16_reg, pair_index_u16, mask=mask_u16_g32)

    packed_scale_reg <<= scale_bits_u8[0]
    dst_scale_u8[0] <<= mask_u8_pack64 * packed_scale_reg.pack4()


def make_transposed_matmul_kernel(items=1, *, batched=False):
    """Keep the original ABI; batched controls exercise the same rotating body."""
    if items < 1 or (not batched and items != 1):
        raise ValueError("Multiple work items require batched inputs and outputs")
    input_shape = (items, 64, 32) if batched else (64, 32)
    output_shape = (items, 32, 32) if batched else (32, 32)

    @kernel()
    def float_to_mxfp8_online_cast_transpose_matmul_kernel(x: GM[f32, input_shape], y: GM[f32, input_shape], z: GM[f32, output_shape], dummy: i32):
        l1x_buf = DBuff(DT.e4m3, [K, TRANSPOSED_M], Position.L1)
        l1y_buf = DBuff(DT.e4m3, [K, TRANSPOSED_N], Position.L1)
        scale_x_buf = DBuff(DT.uint8, [TRANSPOSED_M, GROUPS_PER_ROW], Position.L1)
        scale_y_buf = DBuff(DT.uint8, [TRANSPOSED_N, GROUPS_PER_ROW], Position.L1)
        vcmutex = VcMutex(0, guards=l1x_buf, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
        l0z = Tensor(DT.float, [TRANSPOSED_M, TRANSPOSED_N], Position.L0C)

        ub_x_f32 = Tensor(DT.float, [K, TRANSPOSED_M], Position.UB)
        ub_y_f32 = Tensor(DT.float, [K, TRANSPOSED_N], Position.UB)
        ub_x_u32 = ub_x_f32.reinterpret(DT.uint32)
        ub_y_u32 = ub_y_f32.reinterpret(DT.uint32)
        ub_x_fp8 = Tensor(DT.e4m3, [K, TRANSPOSED_M], Position.UB)
        ub_y_fp8 = Tensor(DT.e4m3, [K, TRANSPOSED_N], Position.UB)
        ub_x_fp8_u16 = ub_x_fp8.reinterpret(DT.uint16)
        ub_y_fp8_u16 = ub_y_fp8.reinterpret(DT.uint16)

        scale_bits_x_u32 = Tensor(DT.uint32, [TRANSPOSED_M, GROUPS_PER_ROW], Position.UB)
        scale_bits_y_u32 = Tensor(DT.uint32, [TRANSPOSED_N, GROUPS_PER_ROW], Position.UB)
        scale_bits_x_f32 = scale_bits_x_u32.reinterpret(DT.float)
        scale_bits_y_f32 = scale_bits_y_u32.reinterpret(DT.float)
        scale_bits_x_u8 = scale_bits_x_u32.reinterpret(DT.uint8)
        scale_bits_y_u8 = scale_bits_y_u32.reinterpret(DT.uint8)
        ub_scale_x = Tensor(DT.uint8, [TRANSPOSED_M, GROUPS_PER_ROW], Position.UB)
        ub_scale_y = Tensor(DT.uint8, [TRANSPOSED_N, GROUPS_PER_ROW], Position.UB)
        index_bits_u32 = Tensor(DT.uint32, [1, 64], Position.UB)
        index_bits_u16 = Tensor(DT.uint16, [1, 128], Position.UB)
        packed_fp8 = Tensor(DT.e4m3, [1, 64], Position.UB)
        packed_fp8_u8 = packed_fp8.reinterpret(DT.uint8)
        pair_bytes_u8 = Tensor(DT.uint8, [1, 128], Position.UB)
        pair_words_u16 = pair_bytes_u8.reinterpret(DT.uint16)
        index_bits_i32 = index_bits_u32.reinterpret(DT.int)
        index_bits_i16 = index_bits_u16.reinterpret(DT.int16)

        work_items = Var(items)
        slot = Var(0)
        items_per_core = CeilDiv(work_items, GetCubeNum())
        item_begin = Var(items_per_core * GetCubeIdx())
        item_end = Min(item_begin + items_per_core, work_items)

        with auto_sync():
            for _item in asc_range(item_begin, item_end):
                l1x = l1x_buf[slot]
                l1y = l1y_buf[slot]
                scale_x = scale_x_buf[slot]
                scale_y = scale_y_buf[slot]
                item_x = x[_item, :, :] if batched else x
                item_y = y[_item, :, :] if batched else y
                item_z = z[_item, :, :] if batched else z
                vcmutex.lock()
                ub_x_f32[:, :] <<= item_x[:, :]
                ub_y_f32[:, :] <<= item_y[:, :]
                online_cast_f32_krows_to_mxfp8_e4m3_k64(
                    ub_x_f32, ub_x_u32, ub_x_fp8_u16, ub_scale_x,
                    scale_bits_x_u32, scale_bits_x_f32, scale_bits_x_u8,
                    index_bits_u32, index_bits_i32, index_bits_u16, index_bits_i16,
                    packed_fp8, packed_fp8_u8, pair_bytes_u8, pair_words_u16, TRANSPOSED_M, TRANSPOSED_M // 2,
                )
                online_cast_f32_krows_to_mxfp8_e4m3_k64(
                    ub_y_f32, ub_y_u32, ub_y_fp8_u16, ub_scale_y,
                    scale_bits_y_u32, scale_bits_y_f32, scale_bits_y_u8,
                    index_bits_u32, index_bits_i32, index_bits_u16, index_bits_i16,
                    packed_fp8, packed_fp8_u8, pair_bytes_u8, pair_words_u16, TRANSPOSED_N, TRANSPOSED_N // 2,
                )
                # Both sub-blocks compute identical tiles; only sub 0 publishes them so the L1
                # stores have a single writer (the ready() below stays outside: the cube waits
                # for BOTH subs' ready, and sub 1's fires on an empty MTE3 queue).
                if GetSubBlockIdx() == 0:
                    ub_to_l1_nd2nz(l1x, ub_x_fp8, m_dst=K, n_dst=TRANSPOSED_M, m_src=K, n_src=TRANSPOSED_M, N_src=TRANSPOSED_M)
                    ub_to_l1_nd2nz(l1y, ub_y_fp8, m_dst=K, n_dst=TRANSPOSED_N, m_src=K, n_src=TRANSPOSED_N, N_src=TRANSPOSED_N)
                    ub_to_l1(scale_x, ub_scale_x, n_burst=1, burst_len=2, src_stride=0, dst_stride=0)
                    ub_to_l1(scale_y, ub_scale_y, n_burst=1, burst_len=2, src_stride=0, dst_stride=0)
                vcmutex.ready()

                vcmutex.wait()
                matmul_mx(l0z, l1x.T, l1y.T, scale_x, scale_y, m=TRANSPOSED_M, n=TRANSPOSED_N, k=K, is_init=True)
                item_z[:, :] <<= l0z
                vcmutex.free()
                slot += 1

        return z

    return float_to_mxfp8_online_cast_transpose_matmul_kernel


float_to_mxfp8_online_cast_transpose_matmul_kernel = make_transposed_matmul_kernel()

# ----------------------------------------------------------------------------------------------------
# leaves.py
# Observable payload/scale boundaries of the exact local source VFs.
# ----------------------------------------------------------------------------------------------------

@kernel(mode='vec', block_dim=1)
def quantize_nd_leaf(x: GM[f32, (16, 64)], payload: GM[u8, (16, 64)], scales: GM[u8, (1, 32)], dummy: i32):
    src = Tensor(DT.float, [16, 64], Position.UB)
    codes = Tensor(DT.e4m3, [16, 64], Position.UB)
    scale = Tensor(DT.uint8, [16, 2], Position.UB)
    bits = Tensor(DT.uint32, [16, 2], Position.UB)
    with auto_sync():
        src <<= x
        online_cast_f32_to_mxfp8_e4m3_k64(src, src.reinterpret(DT.uint32), codes, scale,
                                        bits, bits.reinterpret(DT.float), bits.reinterpret(DT.uint8), 16)
        payload <<= codes.reinterpret(DT.uint8)
        ub_to_gm_pad(scales, scale, n_burst=1, burst_len_element=32, src_stride=0, dst_stride_element=0)
    return payload, scales


@kernel(mode='vec', block_dim=1)
def quantize_transposed_leaf(x: GM[f32, (64, 32)], payload: GM[u8, (64, 32)], scales: GM[u8, (1, 64)], dummy: i32):
    src = Tensor(DT.float, [64, 32], Position.UB)
    codes = Tensor(DT.e4m3, [64, 32], Position.UB)
    scale = Tensor(DT.uint8, [32, 2], Position.UB)
    bits = Tensor(DT.uint32, [32, 2], Position.UB)
    index32 = Tensor(DT.uint32, [1, 64], Position.UB)
    index16 = Tensor(DT.uint16, [1, 128], Position.UB)
    packed = Tensor(DT.e4m3, [1, 64], Position.UB)
    pairs = Tensor(DT.uint8, [1, 128], Position.UB)
    with auto_sync():
        src <<= x
        online_cast_f32_krows_to_mxfp8_e4m3_k64(
            src, src.reinterpret(DT.uint32), codes.reinterpret(DT.uint16), scale,
            bits, bits.reinterpret(DT.float), bits.reinterpret(DT.uint8),
            index32, index32.reinterpret(DT.int), index16, index16.reinterpret(DT.int16),
            packed, packed.reinterpret(DT.uint8), pairs, pairs.reinterpret(DT.uint16), 32, 16,
        )
        payload <<= codes.reinterpret(DT.uint8)
        ub_to_gm_pad(scales, scale, n_burst=1, burst_len_element=64, src_stride=0, dst_stride_element=0)
    return payload, scales
