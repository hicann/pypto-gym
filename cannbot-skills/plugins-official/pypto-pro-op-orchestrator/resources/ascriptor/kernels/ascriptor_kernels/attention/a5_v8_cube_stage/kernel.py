# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserved V8 Cube stage: QK, both drain modes of one L0C tile, and PV.

v8_helpers holds the exact local helper closure from the corrected V8 and V6 sources; stage.py
is the kernel that uses it."""

import math

from ascriptor.a5 import *

# ----------------------------------------------------------------------------------------------------
# v8_helpers.py
# Exact local helper closure from corrected V8 and V6 sources.
# ----------------------------------------------------------------------------------------------------

TILE_M = 128
ROWS_PER_SB = 64
TILE_N = 128
SLAB = TILE_N // 4
KEYS_PER_GRP = 4

@func()
def _qk_bridge_splitn_group(ub_dst, l0c_src):
    l0c_to_ub(ub_dst, l0c_src[0:TILE_N, 0:TILE_M],
              M=TILE_N, N=TILE_M, N_dst=ROWS_PER_SB, M_src=TILE_N,
              dual_mode=DualMode.SPLITN, sub_block_id=0, scale=1.0)

@func()
def _publish_group_p(l1p_dst, ub_p, row_begin, p_mutex):
    p_mutex.lock()
    for g in unroll(KEYS_PER_GRP):
        l1p_dst[g * SLAB:(g + 1) * SLAB, row_begin:row_begin + ROWS_PER_SB] <<= (
            ub_p.nz()[0:SLAB, g * ROWS_PER_SB:(g + 1) * ROWS_PER_SB]
        )
    p_mutex.ready()

# ----------------------------------------------------------------------------------------------------
# stage.py
# Preserved V8 stage sample; independent generated reference replaces golden recording.
# ----------------------------------------------------------------------------------------------------

TILE = 128
ROWS = 64

@kernel()
def v8_cube_path(q: GM[u8, (128, 128)], k: GM[u8, (128, 128)], v: GM[u8, (128, 128)], pbytes: GM[u8, (33, 256)],
                 pbytes2: GM[u8, (33, 256)], score: GM[f32, (128, 128)], score2: GM[f32, (256, 64)], score3: GM[f32, (128, 128)],
                 pv: GM[f32, (128, 128)], pv2: GM[f32, (128, 128)]):
    # The consumers of both bridges here are MTE3 stores (v8's are V-pipe vf functions): the wait and the free must
    # sit on MTE3, or the free fires while the store still reads the tile (a WAR hazard the pipe-level simulator reports).
    cv = CvMutex(0, depth=1, src_start_pipe=Pipe.FIX, dst_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE3)
    pvm = CvMutex(2, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE3)
    pm = VcMutex(3, depth=1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE1)
    q_h = q.reinterpret(DT.hif8, name="q_h")
    k_h = k.reinterpret(DT.hif8, name="k_h")
    v_h = v.reinterpret(DT.hif8, name="v_h")
    l1q = Tensor(DT.hif8, [TILE, TILE], Position.L1)
    l1k = Tensor(DT.hif8, [TILE, TILE], Position.L1)
    l1v = Tensor(DT.hif8, [TILE, TILE], Position.L1)
    l1p = Tensor(DT.hif8, [TILE, TILE], Position.L1)
    l1p2 = Tensor(DT.hif8, [TILE, TILE], Position.L1)
    l0c_qk = Tensor(DT.float, [TILE, TILE], Position.L0C)
    l0c_pv = Tensor(DT.float, [TILE, TILE], Position.L0C)
    l0c_pv2 = Tensor(DT.float, [TILE, TILE], Position.L0C)
    ub_score = Tensor(DT.float, [TILE, ROWS], Position.UB)
    ub_full = Tensor(DT.float, [TILE, TILE], Position.UB)
    ub_pv = Tensor(DT.float, [ROWS, TILE], Position.UB)
    ub_pv2 = Tensor(DT.float, [ROWS, TILE], Position.UB)
    ub_p = Tensor(DT.hif8, [33, 256], Position.UB)
    ub_p2 = Tensor(DT.hif8, [33, 256], Position.UB)
    row_begin = Var(GetSubBlockIdx() * ROWS)
    with auto_sync():
        l1q[0:TILE, 0:TILE] <<= q_h[0:TILE, 0:TILE]
        l1k[0:TILE, 0:TILE] <<= k_h[0:TILE, 0:TILE]
        l1v[0:TILE, 0:TILE] <<= v_h[0:TILE, 0:TILE]
        matmul(l0c_qk, l1k, l1q, m=TILE, n=TILE, k=TILE, is_init=True)
        cv.lock()
        _qk_bridge_splitn_group(ub_score, l0c_qk)
        cv.ready()
        cv.wait()
        score[0:TILE, row_begin:row_begin + ROWS] <<= ub_score
        score2[row_begin * 2:row_begin * 2 + TILE, 0:ROWS] <<= ub_score
        cv.free()
        cv.lock()
        l0c_to_ub(ub_full, l0c_qk, M=TILE, N=TILE, N_dst=TILE, M_src=TILE, dual_mode=DualMode.SINGLE, sub_block_id=0)
        cv.ready()
        cv.wait()
        if row_begin == 0:
            score3[0:TILE, 0:TILE] <<= ub_full
        ub_p_u8 = ub_p.reinterpret(DT.uint8)
        ub_p_u8 <<= pbytes
        ub_p2_u8 = ub_p2.reinterpret(DT.uint8)
        ub_p2_u8 <<= pbytes2
        cv.free()
        _publish_group_p(l1p, ub_p, row_begin, pm)
        pm.wait()
        matmul(l0c_pv, l1p.T, l1v.T, m=TILE, n=TILE, k=TILE, is_init=True)
        pm.free()
        _publish_group_p(l1p2, ub_p2, row_begin, pm)
        pm.wait()
        matmul(l0c_pv2, l1p2.T, l1v.T, m=TILE, n=TILE, k=TILE, is_init=True)
        pm.free()
        pvm.lock()
        l0c_to_ub(ub_pv, l0c_pv, M=TILE, N=TILE, N_dst=TILE, M_src=TILE, dual_mode=DualMode.SPLITM, sub_block_id=0)
        pvm.ready()
        pvm.wait()
        pv[row_begin:row_begin + ROWS, 0:TILE] <<= ub_pv
        pvm.free()
        pvm.lock()
        l0c_to_ub(ub_pv2, l0c_pv2, M=TILE, N=TILE, N_dst=TILE, M_src=TILE, dual_mode=DualMode.SPLITM, sub_block_id=0)
        pvm.ready()
        pvm.wait()
        pv2[row_begin:row_begin + ROWS, 0:TILE] <<= ub_pv2
        pvm.free()
    return score, score2, score3, pv, pv2
