# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F403, F405
"""Whole rows moved one at a time, addressed by a value read out of memory.

A row gather is a scalar read followed by a copy: the index lands in a cell, the cell becomes the
row subscript of a GM view, and one DMA moves that row. This unit is the addressing only. There is
no score, no mask and no reduction in it, and nothing here is an attention operator.

Three things are worth reading carefully, and each has its own paragraph below: the two tables are
gathered at their own row widths, the padding slots are copied unconditionally, and a clamped
destination row is legal and wrong.
"""

from ascriptor.a5 import *

SLOTS, S2, DK, DV = 16, 128, 64, 48
GUARD = 1024.0


@kernel(mode="vec", block_dim=1)
def indexed_row_gather(index: GM[i32, (1, SLOTS)], count: GM[i32, (1, 1)],
                       k_table: GM[f16, (S2, DK)], v_table: GM[f16, (S2, DV)],
                       k_rows: GM[f16, (SLOTS, DK)], v_rows: GM[f16, (SLOTS, DV)],
                       guarded: GM[f16, (SLOTS, DV)], clamped: GM[f16, (SLOTS, DV)]):
    slots = Tensor(DT.int, [1, SLOTS], Position.UB)
    live_ub = Tensor(DT.int, [1, 1], Position.UB)
    k_row = Tensor(f16, [1, DK], Position.UB)
    v_row = Tensor(f16, [1, DV], Position.UB)
    with auto_sync():
        # The whole index table and the live count come across once. A per-slot GM scalar read
        # would be one MTE2 round trip per row; `GetValueFrom` reads UB, so the table is staged.
        slots <<= index
        live_ub <<= count
        live = Var(0, DT.int)
        live.GetValueFrom(live_ub[0:1, 0:1])
        last = live - 1
        for s in range(SLOTS):
            # The index is a value, not a subscript, until it is in a cell. `GetValueFrom` takes
            # the first element of the view it is given, so the view is narrowed to that element -
            # a wider one would load `slots[0, 0]` every trip with no diagnostic at all.
            row = Var(0, DT.int)
            row.GetValueFrom(slots[0:1, s:s + 1])

            # One copy per table, each at its own table's row width. `k_table` and `v_table` are
            # DK and DV wide and DV is the narrower of the two, so no single width addresses both.
            # These two are separately declared, so a DK-wide read of `v_table` is rejected -
            # `tests/check_row_widths.py` records that diagnostic and the layout where the same
            # mistake is silent instead: a DV prefix of a DK-wide row, which is where `Dv < Dk`
            # usually comes from, accepts the wide read and returns the next field's columns.
            k_row[0:1, 0:DK] <<= k_table[row:row + 1, 0:DK]
            v_row[0:1, 0:DV] <<= v_table[row:row + 1, 0:DV]

            # The copies above are unconditional, so every slot's index is dereferenced - the
            # padding slots included. A padding index therefore has to be a legal row of its
            # table; what makes it padding is that nothing downstream reads the result, not that
            # the copy is skipped. These two outputs record what every slot actually fetched.
            k_rows[s:s + 1, 0:DK] <<= k_row[0:1, 0:DK]
            v_rows[s:s + 1, 0:DV] <<= v_row[0:1, 0:DV]

            # `live` rows of the destination are real and the loop runs the padded slot count, so
            # slots at and past `live` have no row to be written to. A guard is the answer.
            if s < live:
                guarded[s:s + 1, 0:DV] <<= v_row[0:1, 0:DV]

            # The trap, reproduced on purpose. Clamping the destination row keeps the address
            # inside the tensor, which is the reflex the out-of-range write invites, and every
            # padding slot then lands on row `live - 1` - so the last real row of the block holds
            # the last padding slot's data. The reference predicts that corruption, because a
            # claim this example only asserted in prose would be worth less than one it proves.
            dst = Min(s, last)
            clamped[dst:dst + 1, 0:DV] <<= v_row[0:1, 0:DV]
    return k_rows, v_rows, guarded, clamped
