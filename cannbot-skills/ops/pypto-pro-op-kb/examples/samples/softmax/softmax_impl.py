# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# SAMPLE PROVENANCE -- describes THIS reference implementation only.
# Do not copy this header into generated code: a generated kernel inherits
# no validation from the sample it was modelled on.
# STATUS: VALIDATED on Ascend a5 (950) NPU — 2026-07-17. pl tile DSL (pypto_pro.language).
#   rows=64 cols=64 -> max_abs_diff = 2.98e-08 vs torch.softmax. PASS. Multicore (block_dim=4).
#   The vector-engine core of attention's online-softmax pattern (patterns/online-softmax-tail.md).
#   Modeled on python/tests/ut/block/frontend/a5/tile_vector/test_softmax.py.
#   Fully dynamic rows AND cols; row-tiles spread across vector cores; tail handled by
#   pl.set_validshape. pl has NO RunMode.SIM -> validated ONLY on Tier-3 NPU.
# VALIDATED-CODE-SHA256: c3238d2b21879ab4e34a4dfbd570661eee7bed91e406d3eae0f750fb9cae3bb8
import pypto_pro.language as pl
import torch
import torch_npu  # noqa: F401

MAX_N = 512  # max columns == compile-time UB tile width
TILE_ROWS = 16  # rows per tile-group slot (row count is dynamic)
SLOT_BYTES = TILE_ROWS * MAX_N * 4
RED_BYTES = 512
VA_IN0 = 0
VA_IN1 = VA_IN0 + SLOT_BYTES
VA_OUT0 = VA_IN1 + SLOT_BYTES
VA_OUT1 = VA_OUT0 + SLOT_BYTES
VA_TMP0 = VA_OUT1 + SLOT_BYTES
VA_TMP1 = VA_TMP0 + SLOT_BYTES
VA_RED0 = VA_TMP1 + SLOT_BYTES
VA_RED1 = VA_RED0 + RED_BYTES


@pl.jit(auto_mutex=True)
def softmax_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tile_type = pl.TileType(
        shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    red_type = pl.TileType(
        shape=[TILE_ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    in_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT0, VA_OUT1], mutex_ids=[2, 3])
    tmp_group = pl.make_tile_group(type=tile_type, addrs=[VA_TMP0, VA_TMP1], mutex_ids=[4, 5])
    red_group = pl.make_tile_group(type=red_type, addrs=[VA_RED0, VA_RED1], mutex_ids=[6, 7])

    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS
        for tile_id in pl.range(core_id, num_tiles, num_cores):  # row-tiles across vec cores
            row_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - row_off)  # tail row-tile is partial
            in_slot = in_group.next()
            pl.set_validshape(in_slot, [valid_rows, cols])
            pl.load(in_slot, x, [row_off, 0])
            out_slot = out_group.next()
            tmp_slot = tmp_group.next()
            red_slot = red_group.next()
            pl.set_validshape(out_slot, [valid_rows, cols])
            pl.set_validshape(tmp_slot, [valid_rows, cols])
            pl.set_validshape(red_slot, [valid_rows, 1])
            pl.row_max(red_slot, in_slot, tmp_slot)  # m = max over N
            pl.row_expand_sub(out_slot, in_slot, red_slot)  # x - m (row broadcast)
            pl.exp(out_slot, out_slot)  # exp(x - m)
            pl.row_sum(red_slot, out_slot, tmp_slot)  # s = sum over N
            pl.row_expand_div(out_slot, out_slot, red_slot)  # / s
            pl.store(y, out_slot, [row_off, 0])


def softmax_wrapper(x):
    """x:[rows, cols] fp32 on npu -> row-softmax. cols <= MAX_N."""
    rows = x.shape[0]
    y = torch.empty_like(x)
    num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS
    softmax_kernel[None, min(32, num_tiles)](x, y)
    torch.npu.synchronize()
    return y
