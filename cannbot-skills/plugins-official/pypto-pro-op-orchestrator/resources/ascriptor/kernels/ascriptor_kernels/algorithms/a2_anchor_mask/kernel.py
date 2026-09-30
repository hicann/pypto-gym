# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dense anchor mask with optional base windows and causal synthetic blocks."""

import importlib

from ascriptor.a2 import *
from functools import lru_cache

TILE_Q = 64
TILE_N = 64
C0F = 8

def anchor_mask(kv_doc: GM[f32, (1, 'T')], kv_iota: GM[f32, (1, 'T')], q_doc: GM[f32, (1, 'Q')], q_anchor: GM[f32, (1, 'Q')], blk_idx: GM[f32, (1, 'Q')], mask_out: GM[u8, ('Q', 'KV')], total_seq_len: i32, q_len: i32, block_size: i32, sliding_window: i32, synthetic_causal: i32):
    ub_qdoc = Tensor(DT.float, [1, TILE_Q], Position.UB)
    ub_qanchor = Tensor(DT.float, [1, TILE_Q], Position.UB)
    ub_qblk = Tensor(DT.float, [1, TILE_Q], Position.UB)
    ub_qdoc_b = Tensor(DT.float, [TILE_Q, C0F], Position.UB)
    ub_qanchor_b = Tensor(DT.float, [TILE_Q, C0F], Position.UB)
    ub_qblk_b = Tensor(DT.float, [TILE_Q, C0F], Position.UB)
    ub_col_a = DBuff(DT.float, [1, TILE_N], Position.UB)
    ub_col_b = DBuff(DT.float, [1, TILE_N], Position.UB)
    ones_f = Tensor(DT.float, [TILE_Q, TILE_N], Position.UB)
    t1 = Tensor(DT.float, [TILE_Q, TILE_N], Position.UB)
    t2 = Tensor(DT.float, [TILE_Q, TILE_N], Position.UB)
    h_out = Tensor(DT.half, [TILE_Q, TILE_N], Position.UB)
    u8_out = DBuff(DT.uint8, [TILE_Q, TILE_N], Position.UB)
    n_qtiles = CeilDiv(q_len, TILE_Q)
    tiles_per_core = CeilDiv(n_qtiles, GetVecNum())
    t_begin = Var(tiles_per_core * GetVecIdx())
    t_end = Min(t_begin + tiles_per_core, n_qtiles)
    n_base_chunks = CeilDiv(total_seq_len, TILE_N)
    n_syn_chunks = CeilDiv(q_len, TILE_N)
    with auto_sync():
        set_mask((1 << 64) - 1, (1 << 64) - 1)
        dup(ones_f, 1.0)
        for ti in range(t_begin, t_end):
            q0 = Var(ti * TILE_Q)
            valid_q = Min(TILE_Q, q_len - q0)
            rep_brcb = CeilDiv(valid_q, 8)
            ub_qdoc[0:1, 0:valid_q] <<= q_doc[0:1, q0:q0 + valid_q]
            ub_qanchor[0:1, 0:valid_q] <<= q_anchor[0:1, q0:q0 + valid_q]
            ub_qblk[0:1, 0:valid_q] <<= blk_idx[0:1, q0:q0 + valid_q]
            brcb(ub_qdoc_b, ub_qdoc, repeat=rep_brcb, dst_blk_stride=1, dst_rep_stride=8)
            brcb(ub_qanchor_b, ub_qanchor, repeat=rep_brcb, dst_blk_stride=1, dst_rep_stride=8)
            brcb(ub_qblk_b, ub_qblk, repeat=rep_brcb, dst_blk_stride=1, dst_rep_stride=8)
            for cb in range(n_base_chunks):
                n0 = Var(cb * TILE_N)
                col_a = ub_col_a[cb]
                col_b = ub_col_b[cb]
                u8 = u8_out[cb]
                col_a <<= kv_doc[0:1, n0:n0 + TILE_N]
                sub(t1, col_a, ub_qdoc_b, repeat=valid_q)
                abs(t1, t1, repeat=valid_q)
                vmins(t1, t1, 1.0, repeat=valid_q)
                sub(t1, ones_f, t1, repeat=valid_q)
                col_b <<= kv_iota[0:1, n0:n0 + TILE_N]
                sub(t2, ub_qanchor_b, col_b, repeat=valid_q)
                vmaxs(t2, t2, 0.0, repeat=valid_q)
                vmins(t2, t2, 1.0, repeat=valid_q)
                mul(t1, t1, t2, repeat=valid_q)
                if sliding_window >= 0:
                    sub(t2, col_b, ub_qanchor_b, repeat=valid_q)
                    adds(t2, t2, sliding_window, repeat=valid_q)
                    adds(t2, t2, 1.0, repeat=valid_q)
                    vmaxs(t2, t2, 0.0, repeat=valid_q)
                    vmins(t2, t2, 1.0, repeat=valid_q)
                    mul(t1, t1, t2, repeat=valid_q)
                cast(h_out, t1)
                cast(u8, h_out)
                mask_out[q0:q0 + valid_q, n0:n0 + TILE_N] <<= u8[0:valid_q, 0:TILE_N]
            bar_all()
            # The base readers have drained. Reuse their row buffers for q indices;
            # the validated first 64 base indices supply the same exact local ramp.
            if synthetic_causal != 0:
                ub_qdoc <<= kv_iota[0:1, 0:TILE_Q]
                adds(ub_qdoc, ub_qdoc, q0)
                brcb(ub_qdoc_b, ub_qdoc, repeat=rep_brcb, dst_blk_stride=1, dst_rep_stride=8)
            for cs in range(n_syn_chunks):
                n0 = Var(cs * TILE_N)
                col_a = ub_col_a[cs]
                u8 = u8_out[cs]
                col_a <<= blk_idx[0:1, n0:n0 + TILE_N]
                sub(t1, col_a, ub_qblk_b, repeat=valid_q)
                abs(t1, t1, repeat=valid_q)
                vmins(t1, t1, 1.0, repeat=valid_q)
                sub(t1, ones_f, t1, repeat=valid_q)
                if synthetic_causal != 0:
                    col_b = ub_col_b[cs]
                    col_b <<= kv_iota[0:1, 0:TILE_N]
                    adds(col_b, col_b, n0)
                    sub(t2, ub_qdoc_b, col_b, repeat=valid_q)
                    adds(t2, t2, 1.0, repeat=valid_q)
                    vmaxs(t2, t2, 0.0, repeat=valid_q)
                    vmins(t2, t2, 1.0, repeat=valid_q)
                    mul(t1, t1, t2, repeat=valid_q)
                cast(h_out, t1)
                cast(u8, h_out)
                kv0 = Var(total_seq_len + n0)
                mask_out[q0:q0 + valid_q, kv0:kv0 + TILE_N] <<= u8[0:valid_q, 0:TILE_N]
            bar_all()
    return mask_out

@lru_cache(maxsize=2)
def kernel_for(device):
    if device not in ("a2", "a3"):
        raise ValueError("Anchor mask supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec", block_dim=1)(anchor_mask)
