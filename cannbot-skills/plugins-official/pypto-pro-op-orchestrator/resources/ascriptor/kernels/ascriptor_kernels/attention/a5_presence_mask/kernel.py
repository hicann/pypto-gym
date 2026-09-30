# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A sparse-attention stage that rebuilds its mask instead of keeping one.

The geometry is fixed and stated once, here: 64 query rows, 1024 key columns, up to 64
selected columns per query, covered by 8 presence rebuilds of 128 columns each. reference.py
restates the same numbers on the host side and main.py's cases are patterns over them."""

from ascriptor.a5 import *

QUERIES = 64        # query rows this stage owns: one b32 register covers all of them
KEYS = 1024         # key columns the tile attends over
TOPK = 64           # index slots per query row; a slot holding KEYS or more means "no key"
CHUNK = 128         # key columns one presence rebuild covers
EPOCHS = 4          # generations a marker survives before the table is really cleared
CHUNKS = KEYS // CHUNK
GUARD = 1           # one spare table row, as a hedge; the lines below say how much of a hedge.
                    # The production kernel this pattern comes from keeps a tail because scatter
                    # address generation reaches one physical vector past the LOGICAL WINDOW even
                    # for lanes the mask closed - but its window is a one-row slice of a wider
                    # tensor, and this scatter's window is the whole table. Measured here: with
                    # GUARD = 0 the board returns a bit-identical answer, so this geometry does
                    # not reproduce the over-touch and the row is not load-bearing at this size.


@vf()
def _clear_presence(table: Tensor):
    """Generation 0. Every marker below is >= 1, so a cleared row matches no epoch."""
    blank = Reg(DT.uint32)
    dup(blank, 0)
    for row in range(CHUNK + GUARD):
        reg_to_ub(table[row * QUERIES], blank)


@vf()
def _zero_denominator(total: Tensor):
    accumulator = Reg(DT.float)
    dup(accumulator, 0.0)
    reg_to_ub(total[0], accumulator)


@vf()
def _chunk(prob: Tensor, present: Tensor, total: Tensor, score: Tensor, table: Tensor,
           index_ub: Tensor, rowmax: Tensor, base: Var, epoch: Var):
    """Rebuild one chunk of the presence table, then read it back as this chunk's predicate.

    Both halves share one `@vf` entry on purpose: an entry and its tail barrier cost more
    than the work in here. Two entries would buy nothing but one fewer `vf_barrier`.
    """
    origin = Reg(DT.uint32)
    slot = Reg(DT.uint32)
    wanted = Reg(DT.uint32)
    marker = Reg(DT.uint32)
    stored = Reg(DT.uint32)
    row = Reg(DT.float)
    ceiling = Reg(DT.float)
    numerator = Reg(DT.float)
    flag = Reg(DT.float)
    accumulator = Reg(DT.float)
    live = MaskReg(DT.uint32)

    dup(origin, base)
    dup(marker, epoch)
    for query in range(QUERIES):
        ub_to_reg(wanted, index_ub[query * TOPK])
        sub(slot, wanted, origin)
        # Unsigned. A column below `base` wraps to something enormous, so the one `< CHUNK`
        # test catches it, the columns past this chunk, and the "no key" padding alike.
        compare(live, slot, CHUNK, CompareMode.LT)
        shiftls(slot, slot, 6)                      # row-major table: slot * QUERIES
        adds(slot, slot, query)
        # The predicate goes straight into the scatter. The older shape of this kernel spent
        # two `select`s rewriting rejected indices to a scratch column instead.
        reg_to_ub_scatter(table, marker, slot, mask=live)

    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    ub_to_reg(ceiling, rowmax[0])
    ub_to_reg(accumulator, total[0])
    for step in range(CHUNK):
        ub_to_reg(stored, table[step * QUERIES])
        # A residue from an earlier generation carries an earlier number, so EQ misses it.
        # That is what replaces clearing the table between chunks.
        compare(live, stored, epoch, CompareMode.EQ)
        ub_to_reg(row, score[step * QUERIES])
        # Masked writes ZERO the inactive destination lanes of a register (M10-051), so an
        # absent key leaves 0 here and needs no second pass. See the library's
        # docs/api/mask-write-semantics.md: memory is the only thing a mask can skip.
        expsub(numerator, row, ceiling, live)
        dup(flag, 1.0, live)
        reg_to_ub(prob[step * QUERIES], numerator)
        reg_to_ub(present[step * QUERIES], flag)
        add(accumulator, accumulator, numerator)
    reg_to_ub(total[0], accumulator)


@kernel(mode="vec", block_dim=1)
def presence_mask_stage(score_t: GM[f32, (KEYS, QUERIES)], index: GM[i32, (QUERIES, TOPK)],
                        rowmax: GM[f32, (1, QUERIES)], prob_t: GM[f32, (KEYS, QUERIES)],
                        present_t: GM[f32, (KEYS, QUERIES)], denom: GM[f32, (1, QUERIES)]):
    """Scores arrive transposed, the way `score^T = K @ Q^T` leaves them on the cube."""
    table = Tensor(DT.uint32, [CHUNK + GUARD, QUERIES], Position.UB)
    ub_index_signed = Tensor(DT.int, [QUERIES, TOPK], Position.UB)
    ub_index = ub_index_signed.reinterpret(DT.uint32)
    ub_score = Tensor(DT.float, [CHUNK, QUERIES], Position.UB)
    ub_prob = Tensor(DT.float, [CHUNK, QUERIES], Position.UB)
    ub_present = Tensor(DT.float, [CHUNK, QUERIES], Position.UB)
    ub_max = Tensor(DT.float, [1, QUERIES], Position.UB)
    ub_total = Tensor(DT.float, [1, QUERIES], Position.UB)
    with auto_sync():
        ub_index_signed[:, :] <<= index
        ub_max[:, :] <<= rowmax
        _zero_denominator(ub_total)
        for chunk in range(CHUNKS):
            if chunk % EPOCHS == 0:
                _clear_presence(table)
            ub_score[:, :] <<= score_t[chunk * CHUNK:(chunk + 1) * CHUNK, :]
            _chunk(ub_prob, ub_present, ub_total, ub_score, table, ub_index, ub_max,
                   Var(chunk * CHUNK), Var(chunk % EPOCHS + 1))
            prob_t[chunk * CHUNK:(chunk + 1) * CHUNK, :] <<= ub_prob
            present_t[chunk * CHUNK:(chunk + 1) * CHUNK, :] <<= ub_present
        denom[:, :] <<= ub_total
    return prob_t, present_t, denom
