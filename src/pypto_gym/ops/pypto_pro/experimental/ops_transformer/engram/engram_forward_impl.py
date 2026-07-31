#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO-Pro engram_forward kernel implementation.

Forward of the Engram Gated Memory operator (forward = engram_v4).

Single-kernel CV design: the cube section computes the shared value projection
and per-head key projections into FP32 workspaces; the vector section consumes
those workspaces and computes RMS-normalized scores, signed-sqrt gates, and the
gated value output. Per-head cross-core events pipeline the cube and vector
sections while ACK events provide backpressure when event IDs are reused.

7-step forward (per head m):
  Step1  value = embeddings @ W_v
  Step2  key   = embeddings @ W_k[m]
  Step3  compute RMS(key) and RMS(hidden_states[:, m, :])
  Step4  nKey = key * rsqrt(mean(key²) + eps) * key_gamma[m]
         nQuery = hidden_states[:, m, :] * rsqrt(mean(hidden_states²) + eps)
                  * query_gamma[m]
  Step5  score = (nKey · nQuery) * (1 / √H)
  Step6  gate = sigmoid(sign(score) * √max(abs(score), clamp))
  Step7  value_out = cast_bf16(gate * value)

BF16 inputs are cast to FP32 at the vector computation boundary. Scores,
projected keys, projected values, and gates are written as FP32 workspaces;
only the final value_out is stored as BF16. The host wrapper flattens [B, S]
to [B*S], launches the kernel, and reshapes outputs back to the model layout.
"""

import math

import torch

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


class EngramTilingKey:
    HMode = TilingKeyField(bits=1, values=[0, 1])  # 0 -> H=1280, 1 -> H=2560
    CMode = TilingKeyField(bits=1, values=[0, 1])  # 0 -> 128, 1 -> 64


# ════════════════════════════════════════════════
# Layer B: Compile-time constants
# ════════════════════════════════════════════════

TILE_M = 64           # cube TILE_M (reference; actual value bound per CMode in kernel)
TILE_M_VEC = 8        # vec TILE_M (reference; actual value bound per HMode in kernel)
TILE_K = 128
TILE_N = 128
H_CHUNK = 1280        # per-chunk hidden dim (reference; actual value bound per HMode)

# Per-HMode vector-tile split sizes. H=1280 doubles the row count vs H=2560
# (constant-product UB: [8,1280] and [4,2560] are byte-identical 40KB FP32).
TILE_M_VEC_1280 = 8
TILE_M_VEC_2560 = 4
H_CHUNK_1280 = 1280
H_CHUNK_2560 = 2560
LANES_FP32 = 64
LANES_BF16 = 128

CLAMP_VALUE = 1.0e-6
RMS_EPS = 1.0e-6

INV_H_1280 = 1.0 / 1280.0
INV_SQRT_H_1280 = 1.0 / math.sqrt(1280.0)
INV_H_2560 = 1.0 / 2560.0
INV_SQRT_H_2560 = 1.0 / math.sqrt(2560.0)

# ── 跨核事件: parity ping-pong + ack 背压 ──
# 二值事件: set 置1 / wait 消费清0; 两次 set 间无 wait 会合并. 故 event_id 一旦复用就必须有
# ack 做背压, 否则 cube 跑得比 vec 快时, 跨 M-tile 复用同一 ID 会让 set 合并 -> vec 永久 wait.
# K_READY{0,1} (cube->vec): head h 的 key_back 已落 GM;  ACK{0,1} (vec->cube): vec 已消费 head h.
# 按 head%2 分槽, 深度 2: cube 最多领先 vec 2 head, ID 复用前必先被消费 -> 不合并不死锁.
K_READY_IDS = (0, 1)
ACK_IDS = (2, 3)

# (A) H-invariant addresses (byte-constant [TILE_M_VEC, H_CHUNK] tiles).
VA_PROJ_F32 = 0x00000    # [rows,H_CHUNK] FP32  key_back load (直接 FP32, 无 cast)
VA_QUERY_F16 = 0x0A000   # [rows,H_CHUNK] BF16  hidden_states (query) load; vout_f16 别名复用
VA_VALUE_F32 = 0x0F000   # [rows,H_CHUNK] FP32  value_back load (直接 FP32, 无 cast)

# (B) HMode-specialized addresses.  H=1280 keeps the original compact layout;
#     H=2560 shifts tiles after gamma_f16 (doubled size) and uses 0x400-stride
#     accumulator tiles (halved rows).
# ---- HMode = 0 (H = 1280) ----
VA_GAMMA_F16_1280 = 0x19000  # [1,H_CHUNK] BF16 gamma
VA_QUERY_F32_1280 = 0x19A00  # [rows,H_CHUNK] FP32
VA_VOUT_F32_1280 = 0x23A00  # [rows,H_CHUNK] FP32
VA_KGAMMA_F32_1280 = 0x2DA00  # [1,H_CHUNK] FP32
VA_QGAMMA_F32_1280 = 0x2EE00  # [1,H_CHUNK] FP32
VA_SCOUT_F32_1280 = 0x30200  # [rows,64] FP32
VA_GAOUT_F32_1280 = 0x30A00  # [rows,64] FP32
VA_KEY_SQ_ACC_1280 = 0x31200  # [rows,64] FP32
VA_QUERY_SQ_ACC_1280 = 0x31A00  # [rows,64] FP32
VA_SCORE_ACC_1280 = 0x32200  # [rows,64] FP32
VA_KEY_RMS_1280 = 0x32A00  # [rows,64] FP32
VA_QUERY_RMS_1280 = 0x33200  # [rows,64] FP32

# ---- HMode = 1 (H = 2560) ----
VA_GAMMA_F16_2560 = 0x19000  # [1,H_CHUNK] BF16 gamma
VA_QUERY_F32_2560 = 0x1A400  # [rows,H_CHUNK] FP32
VA_VOUT_F32_2560 = 0x24400  # [rows,H_CHUNK] FP32
VA_KGAMMA_F32_2560 = 0x2E400  # [1,H_CHUNK] FP32
VA_QGAMMA_F32_2560 = 0x30C00  # [1,H_CHUNK] FP32
VA_SCOUT_F32_2560 = 0x33400  # [rows,64] FP32
VA_GAOUT_F32_2560 = 0x33800  # [rows,64] FP32
VA_KEY_SQ_ACC_2560 = 0x33C00  # [rows,64] FP32
VA_QUERY_SQ_ACC_2560 = 0x34000  # [rows,64] FP32
VA_SCORE_ACC_2560 = 0x34400  # [rows,64] FP32
VA_KEY_RMS_2560 = 0x34800  # [rows,64] FP32
VA_QUERY_RMS_2560 = 0x34C00  # [rows,64] FP32

LA_EMB = 0x00000     # emb_wide [128,1024] BF16 = 256KB
LA_W = 0x40000       # w_l1 [128,128] BF16 ×2 = 64KB (双缓冲)
# Total L1: 320KB / 512KB (62.5%)
DE_MAX = 1024

# L0A/L0B/L0C
L0A_BASE = 0x0000
L0B_BASE = 0x0000
L0C_BASE = 0x0000


# ════════════════════════════════════════════════
# Layer C1: VF helper — zero accumulator tile
# ════════════════════════════════════════════════

@pl.vector_function
def vf_fill_zero(tile, n_rows):
    """Fill the first element of each row in tile with 0.0 (FP32)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, n_rows):
        vf.store_align(tile + m * 64, zero, preg)


# ════════════════════════════════════════════════
# Layer C2: VF — accumulate key_sq and query_sq per H-chunk ( trimmed: 无 keyout 回写 )
# ════════════════════════════════════════════════
# 相对串行版: 去掉 keyout_f32 参数与 store (key_back 已由 cube 直出, vec 只读).
# ════════════════════════════════════════════════

@pl.vector_function
def vf_accum_sq(
    proj_f32,              # [TILE_M_VEC, H_CHUNK] FP32 (key projection from GM key_back, read-only)
    query_f32,             # [TILE_M_VEC, H_CHUNK] FP32 (query, read-only)
    key_sq_acc,            # [TILE_M_VEC, 64] FP32 in/out
    query_sq_acc,          # [TILE_M_VEC, 64] FP32 in/out
    n_rows,
    n_cols,                # H_CHUNK
    row_stride,            # H_CHUNK (FP32 words per row; = n_cols for full-H tiles)
):
    """Per-chunk: accumulate key_sq and query_sq (no key_back writeback)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32

    for m in pl.range(0, n_rows):
        base = m * row_stride
        prev_key_sq = vf.load_align(key_sq_acc, m * 64)
        prev_query_sq = vf.load_align(query_sq_acc, m * 64)
        key_sq_sum = prev_key_sq
        query_sq_sum = prev_query_sq

        for r in pl.range(0, n_regs):
            off = base + r * LANES_FP32
            kreg = vf.load_align(proj_f32, off)
            sq = vf.mul(kreg, kreg, preg)
            part = vf.reduce_sum(sq, preg, merge_mode=pl.MergeMode.ZEROING)
            key_sq_sum = vf.add(key_sq_sum, part, preg)
            qreg = vf.load_align(query_f32, off)
            sq = vf.mul(qreg, qreg, preg)
            part = vf.reduce_sum(sq, preg, merge_mode=pl.MergeMode.ZEROING)
            query_sq_sum = vf.add(query_sq_sum, part, preg)

        vf.store_align(key_sq_acc + m * 64, key_sq_sum, preg)
        vf.store_align(query_sq_acc + m * 64, query_sq_sum, preg)


# ════════════════════════════════════════════════
# Layer C3: VF — compute RMS from accumulated square sums
# ════════════════════════════════════════════════

@pl.vector_function
def vf_compute_rms(
    key_sq_acc, query_sq_acc,
    key_rms_out, query_rms_out,
    n_rows, inv_h,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    eps_reg = vf.full(RMS_EPS, preg, dtype=pl.DT_FP32)
    inv_h_reg = vf.full(inv_h, preg, dtype=pl.DT_FP32)
    one_reg = vf.full(1.0, preg, dtype=pl.DT_FP32)

    for m in pl.range(0, n_rows):
        ks = vf.load_align(key_sq_acc, m * 64)
        mean_k = vf.mul(ks, inv_h_reg, preg)
        mean_k_eps = vf.add(mean_k, eps_reg, preg)
        sqrt_k = vf.sqrt(mean_k_eps, preg)
        rms_k = vf.div(one_reg, sqrt_k, preg)
        vf.store_align(key_rms_out + m * 64, rms_k, preg)

        qs = vf.load_align(query_sq_acc, m * 64)
        mean_q = vf.mul(qs, inv_h_reg, preg)
        mean_q_eps = vf.add(mean_q, eps_reg, preg)
        sqrt_q = vf.sqrt(mean_q_eps, preg)
        rms_q = vf.div(one_reg, sqrt_q, preg)
        vf.store_align(query_rms_out + m * 64, rms_q, preg)


# ════════════════════════════════════════════════
# Layer C4: VF — score dot product per H-chunk
# ════════════════════════════════════════════════

@pl.vector_function
def vf_score_dot(
    proj_f32, query_f32, kgamma_f32, qgamma_f32,
    key_rms, query_rms, score_acc,
    n_rows, n_cols, row_stride,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32

    for m in pl.range(0, n_rows):
        base = m * row_stride
        kr = vf.load_align(key_rms, m * 64)
        qr = vf.load_align(query_rms, m * 64)
        kr_brc = vf.full(kr, preg)
        qr_brc = vf.full(qr, preg)
        prev_score = vf.load_align(score_acc, m * 64)
        score_sum = prev_score

        for r in pl.range(0, n_regs):
            off = base + r * LANES_FP32
            kreg = vf.load_align(proj_f32, off)
            gk = vf.load_align(kgamma_f32, r * LANES_FP32)
            nk = vf.mul(kreg, kr_brc, preg)
            nk = vf.mul(nk, gk, preg)
            qreg = vf.load_align(query_f32, off)
            gq = vf.load_align(qgamma_f32, r * LANES_FP32)
            nq = vf.mul(qreg, qr_brc, preg)
            nq = vf.mul(nq, gq, preg)
            dq = vf.mul(nk, nq, preg)
            part = vf.reduce_sum(dq, preg, merge_mode=pl.MergeMode.ZEROING)
            score_sum = vf.add(score_sum, part, preg)

        vf.store_align(score_acc + m * 64, score_sum, preg)


# ════════════════════════════════════════════════
# Layer C5: VF — gate finalization
# ════════════════════════════════════════════════

@pl.vector_function
def vf_compute_gate(
    score_acc, scout_f32, gaout_f32, n_rows, inv_sqrt_h,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one_reg = vf.full(1.0, preg, dtype=pl.DT_FP32)
    zero_reg = vf.full(0.0, preg, dtype=pl.DT_FP32)
    neg_one_reg = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    clamp_reg = vf.full(CLAMP_VALUE, preg, dtype=pl.DT_FP32)
    inv_sqrt_h_reg = vf.full(inv_sqrt_h, preg, dtype=pl.DT_FP32)

    for m in pl.range(0, n_rows):
        score_sum = vf.load_align(score_acc, m * 64)
        score = vf.mul(score_sum, inv_sqrt_h_reg, preg)
        vf.store_align(scout_f32 + m * 64, score, preg)

        score_brc = vf.full(score, preg)
        abs_s = vf.abs(score_brc, preg)
        clamped = vf.max(abs_s, clamp_reg, preg)
        sqrt_s = vf.sqrt(clamped, preg)
        pos_mask = vf.gt(score_brc, 0.0, preg)
        neg_mask = vf.lt(score_brc, 0.0, preg)
        sign_r = vf.select(one_reg, zero_reg, pos_mask)
        sign_r = vf.select(neg_one_reg, sign_r, neg_mask)
        raw_gate = vf.mul(sqrt_s, sign_r, preg)
        ex = vf.exp_sub(zero_reg, raw_gate, preg)
        denom = vf.adds(ex, 1.0, preg)
        gate = vf.div(one_reg, denom, preg)
        vf.store_align(gaout_f32 + m * 64, gate, preg)


# ════════════════════════════════════════════════
# Layer C5b: VF — broadcast multiply gate * value
# ════════════════════════════════════════════════

@pl.vector_function
def vf_bcast_mul(
    value_f32, gate_f32, vout_f32, n_rows, n_cols, row_stride,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32

    for m in pl.range(0, n_rows):
        base = m * row_stride
        gate = vf.load_align(gate_f32, m * 64)
        gate_brc = vf.full(gate, preg)
        for r in pl.range(0, n_regs):
            off = base + r * LANES_FP32
            vreg = vf.load_align(value_f32, off)
            vout_reg = vf.mul(vreg, gate_brc, preg)
            vf.store_align(vout_f32 + off, vout_reg, preg)


# ════════════════════════════════════════════════
# Layer D: engram_forward_kernel — CV 并行 (cube section + vec section 并发)
# ════════════════════════════════════════════════


@pl.jit(auto_mutex=True, tiling_key=EngramTilingKey)
def engram_forward_kernel(
    hidden_states: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    embeddings: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    key_proj_weights: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    value_proj_weights: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    key_gamma: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    query_gamma: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    value_out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    score_back: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
    key_back: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    value_back: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    gate_back: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 64], pl.DT_FP32],
):
    """engram CV 并行 + GM 中转 + sub_idx split kernel (v2).

    cube section 与 vec section 在同一 block 的两个核上并发执行:
      - cube: 大 TILE_M matmul -> store key_back/value_back 到 GM -> set K_READY[h]
      - vec:  wait K_READY[h] -> load key_back/value_back/hidden -> per-head 4 phase VF
    per-head 握手 (K_READY[h], h=0..15) 驱动流水; event_id 不复用 -> 无 ack.
    """
    m = embeddings.shape[0]
    de = embeddings.shape[-1]
    hidden_dim = hidden_states.shape[-1]
    m_h = hidden_states.shape[1]
    n_k = de // TILE_K
    n_n = hidden_dim // TILE_N

    # ── TilingKey: HMode selects tile_m_vec / h_chunk_size at compile time ──
    # H=1280 doubles the row count vs H=2560 (constant-product UB:
    # [8,1280] and [4,2560] are byte-identical 40KB FP32).
    # n_h_chunks = 1 in BOTH cases (full-H tile).
    if pl.constexpr(HMode == 1):  # H = 2560
        tile_m_vec = TILE_M_VEC_2560
        h_chunk_size = H_CHUNK_2560
        inv_h = INV_H_2560
        inv_sqrt_h = INV_SQRT_H_2560
    else:  # H = 1280
        tile_m_vec = TILE_M_VEC_1280  # 8
        h_chunk_size = H_CHUNK_1280  # 1280
        inv_h = INV_H_1280
        inv_sqrt_h = INV_SQRT_H_1280
    n_h_chunks = hidden_dim // h_chunk_size  # always 1 (full-H tile)

    # ── TilingKey: CMode selects tile_m at compile time ──
    if pl.constexpr(CMode == 0):
        tile_m = 128
    else:
        tile_m = 64

    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    sub_idx = pl.get_subblock_idx()
    num_subcores = 2
    n_m_tiles = (m + tile_m - 1) // tile_m

    # ═════════ CUBE SECTION ═════════
    # emb 宽 tile: [tile_m, DE_MAX] 常驻 L1 (只读, 单 buffer), 每 M-tile load 一次, offset move 复用
    tt_emb_wide = pl.TileType(shape=[tile_m, DE_MAX], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1)
    tt_w_l1 = pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1)
    emb_wide_grp = pl.make_tile_group(type=tt_emb_wide, addrs=LA_EMB, mutex_ids=[0])
    w_l1_grp = pl.make_tile_group(type=tt_w_l1, addrs=LA_W, mutex_ids=[2, 3])

    tt_left = pl.TileType(shape=[tile_m, TILE_K], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1)
    tt_right = pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1)
    tt_acc = pl.TileType(shape=[tile_m, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, fractal=1024,
                         valid_shape=[-1, -1], compact=1)
    a_left_grp = pl.make_tile_group(type=tt_left, addrs=L0A_BASE, mutex_ids=[4, 5])
    b_right_grp = pl.make_tile_group(type=tt_right, addrs=L0B_BASE, mutex_ids=[6, 7])
    acc_grp = pl.make_tile_group(type=tt_acc, addrs=L0C_BASE, mutex_ids=[8, 9])


    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)

        # 每 core 按stride取 M-tile
        for mi in pl.range(core_id, n_m_tiles, num_cores):
            row_off = mi * tile_m
            valid_m = pl.min(tile_m, m - row_off)

            # ── emb 复用: 本 tile 的 emb [valid_m, De] 一次性 GM->L1 进宽 tile, 后面 value+16head 只 offset move ──
            emb_wide = emb_wide_grp.current()
            pl.set_validshape(emb_wide, [valid_m, de])
            pl.load(emb_wide, embeddings, [row_off, 0])
            pl.set_validshape(emb_wide, [valid_m, TILE_K])   # 收窄读窗口到子块大小, 配合 offset

            # ── Value projection (共享, 每 M-tile 一次; 先于 key, 故 vec 见 K_READY[0] 时已就绪) ──
            for n_idx in pl.range(0, n_n):
                col_off_n = n_idx * TILE_N
                ac = acc_grp.next()
                pl.set_validshape(ac, [valid_m, TILE_N])
                for k_idx in pl.range(0, n_k):
                    k_off = k_idx * TILE_K
                    w_l1 = w_l1_grp.next()
                    pl.set_validshape(w_l1, [TILE_K, TILE_N])
                    pl.load(w_l1, value_proj_weights, [k_off, col_off_n])
                    al = a_left_grp.next()
                    br = b_right_grp.next()
                    pl.set_validshape(al, [valid_m, TILE_K])
                    pl.move(al, emb_wide, offset=[0, k_off])   # emb 常驻 L1, offset 搬第 k_idx 个子块
                    pl.move(br, w_l1)
                    if k_idx == 0:
                        pl.matmul(ac, al, br)
                    elif k_idx == n_k - 1:
                        pl.matmul_acc(ac, ac, al, br)
                    else:
                        pl.matmul_acc(ac, ac, al, br)
                pl.store(value_back, ac, [row_off, col_off_n])

            # ── Key projection (per head) + per-head K_READY 信号 ──
            for h in pl.range(0, m_h):
                for n_idx in pl.range(0, n_n):
                    col_off_n = n_idx * TILE_N
                    ac = acc_grp.next()
                    pl.set_validshape(ac, [valid_m, TILE_N])
                    for k_idx in pl.range(0, n_k):
                        k_off = k_idx * TILE_K
                        wk_l1 = w_l1_grp.next()
                        pl.set_validshape(wk_l1, [TILE_K, TILE_N])
                        pl.load(wk_l1, key_proj_weights,
                                [h, k_off, col_off_n], order=[1, 2])
                        al = a_left_grp.next()
                        br = b_right_grp.next()
                        pl.set_validshape(al, [valid_m, TILE_K])
                        pl.move(al, emb_wide, offset=[0, k_off])   # emb 常驻 L1, offset 搬子块
                        pl.move(br, wk_l1)
                        if k_idx == 0:
                            pl.matmul(ac, al, br)
                        elif k_idx == n_k - 1:
                            pl.matmul_acc(ac, ac, al, br)
                        else:
                            pl.matmul_acc(ac, ac, al, br)
                    pl.store(key_back, ac, [row_off, h, col_off_n],
                             order=[0, 2])

                # 背压: 除本核首个 M-tile 的 head0/head1 外, 先等 vec 消费掉同槽上一轮 (防 ID 复用合并死锁)
                # mi 从 core_id 起, 冷启动判据是 mi==core_id (本核第一轮), 不是 mi==0:
                # 否则 core>=1 的 h0 会 wait 一个永远不来的 ACK -> 死锁卡住.

                if not (mi == core_id and h < 2):
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX, event_id=ACK_IDS[h % 2],
                        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

                # head h 的 key_back 全 N-tile 已落 GM -> 通知 vec.
                # cube 核的 Acc->GM store 走 FIX pipe (与串行版 set(FIX) 一致; cube 侧无 MTE3).
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX, event_id=K_READY_IDS[h % 2],
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.system.set_mm_layout_transform(enabled=False)

    # ═════════ VECTOR SECTION (与 cube 并发) ═════════
    tt_mv_f16 = pl.TileType(shape=[tile_m_vec, h_chunk_size], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)
    tt_1d_f16 = pl.TileType(shape=[1, h_chunk_size], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)
    tt_mv_f32 = pl.TileType(shape=[tile_m_vec, h_chunk_size], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)
    tt_1d_f32 = pl.TileType(shape=[1, h_chunk_size], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)
    tt_mv64_f32 = pl.TileType(shape=[tile_m_vec, 64], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)

    # ── 保留下来的 BF16 tiles (hidden_states/gamma 仍是 BF16, vout 最终输出 BF16) ──
    query_f16_grp = pl.make_tile_group(type=tt_mv_f16, addrs=VA_QUERY_F16, mutex_ids=[11])
    gamma_f16_grp = pl.make_tile_group(type=tt_1d_f16,
                     addrs=HMode * VA_GAMMA_F16_2560 + (1 - HMode) * VA_GAMMA_F16_1280,
                     mutex_ids=[14])
    vout_f16_grp = pl.make_tile_group(type=tt_mv_f16, addrs=VA_QUERY_F16, mutex_ids=[11])  # 别名复用

    # FP32 tiles (key_back/value_back 直接 FP32 load, 无需中间 BF16 cast)
    proj_f32_grp = pl.make_tile_group(type=tt_mv_f32, addrs=VA_PROJ_F32, mutex_ids=[21])
    query_f32_grp = pl.make_tile_group(type=tt_mv_f32,
                     addrs=HMode * VA_QUERY_F32_2560 + (1 - HMode) * VA_QUERY_F32_1280,
                     mutex_ids=[22])
    value_f32_grp = pl.make_tile_group(type=tt_mv_f32, addrs=VA_VALUE_F32, mutex_ids=[23])
    vout_f32_grp = pl.make_tile_group(type=tt_mv_f32,
                    addrs=HMode * VA_VOUT_F32_2560 + (1 - HMode) * VA_VOUT_F32_1280,
                    mutex_ids=[25])
    scout_f32_grp = pl.make_tile_group(type=tt_mv64_f32,
                     addrs=HMode * VA_SCOUT_F32_2560 + (1 - HMode) * VA_SCOUT_F32_1280,
                     mutex_ids=[29])
    gaout_f32_grp = pl.make_tile_group(type=tt_mv64_f32,
                     addrs=HMode * VA_GAOUT_F32_2560 + (1 - HMode) * VA_GAOUT_F32_1280,
                     mutex_ids=[30])
    kgamma_f32_grp = pl.make_tile_group(type=tt_1d_f32,
                      addrs=HMode * VA_KGAMMA_F32_2560 + (1 - HMode) * VA_KGAMMA_F32_1280,
                      mutex_ids=[31])
    qgamma_f32_grp = pl.make_tile_group(type=tt_1d_f32,
                      addrs=HMode * VA_QGAMMA_F32_2560 + (1 - HMode) * VA_QGAMMA_F32_1280,
                      mutex_ids=[13])

    # H-split accumulator tiles
    key_sq_acc_grp = pl.make_tile_group(type=tt_mv64_f32,
                      addrs=HMode * VA_KEY_SQ_ACC_2560 + (1 - HMode) * VA_KEY_SQ_ACC_1280,
                      mutex_ids=[15])
    query_sq_acc_grp = pl.make_tile_group(type=tt_mv64_f32,
                        addrs=HMode * VA_QUERY_SQ_ACC_2560 + (1 - HMode) * VA_QUERY_SQ_ACC_1280,
                        mutex_ids=[16])
    score_acc_grp = pl.make_tile_group(type=tt_mv64_f32,
                     addrs=HMode * VA_SCORE_ACC_2560 + (1 - HMode) * VA_SCORE_ACC_1280,
                     mutex_ids=[17])
    key_rms_grp = pl.make_tile_group(type=tt_mv64_f32,
                   addrs=HMode * VA_KEY_RMS_2560 + (1 - HMode) * VA_KEY_RMS_1280,
                   mutex_ids=[26])
    query_rms_grp = pl.make_tile_group(type=tt_mv64_f32,
                     addrs=HMode * VA_QUERY_RMS_2560 + (1 - HMode) * VA_QUERY_RMS_1280,
                     mutex_ids=[27])

    with pl.section_vector():
        for mi in pl.range(core_id, n_m_tiles, num_cores):
            row_off = mi * tile_m
            # per-head 消费: wait K_READY[h] 后, 把 head h 的 [tile_m, H] 按 tile_m_vec 细粒度处理
            for h in pl.range(0, m_h):
                # 等 cube 存完 head h 的 key_back(value_back 在 K_READY[0] 前已就绪)
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE2, event_id=K_READY_IDS[h % 2],
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

                # 跨 vm_start 段复用: head h 的 gamma 只加载一次 (n_h_chunks==1, 全 H 维度在一个 chunk)
                gamma_f16 = gamma_f16_grp.current()
                pl.set_validshape(gamma_f16, [1, h_chunk_size])
                pl.load(gamma_f16, key_gamma, [h, 0], order=[0])
                kgamma_f32 = kgamma_f32_grp.current()
                pl.set_validshape(kgamma_f32, [1, h_chunk_size])
                pl.cast(kgamma_f32, gamma_f16, mode=pl.RoundMode.CAST_NONE)

                pl.load(gamma_f16, query_gamma, [h, 0], order=[0])
                qgamma_f32 = qgamma_f32_grp.current()
                pl.set_validshape(qgamma_f32, [1, h_chunk_size])
                pl.cast(qgamma_f32, gamma_f16, mode=pl.RoundMode.CAST_NONE)

                # AIV split: sub_idx 0 负责偶数 vm_start 段, sub_idx 1 负责奇数段
                for vm_start in pl.range(sub_idx * tile_m_vec, tile_m, num_subcores * tile_m_vec):
                    valid_mv = pl.min(tile_m_vec, m - row_off - vm_start)
                    if valid_mv <= 0:
                        continue
                    # ── 累加器清零 ──
                    key_sq_acc = key_sq_acc_grp.current()
                    pl.set_validshape(key_sq_acc, [valid_mv, 64])
                    vf_fill_zero(key_sq_acc, valid_mv)
                    query_sq_acc = query_sq_acc_grp.current()
                    pl.set_validshape(query_sq_acc, [valid_mv, 64])
                    vf_fill_zero(query_sq_acc, valid_mv)
                    score_acc = score_acc_grp.current()
                    pl.set_validshape(score_acc, [valid_mv, 64])
                    vf_fill_zero(score_acc, valid_mv)

                    # ── Phase 1: 跨 H-chunk 累加 key_sq / query_sq ──
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * h_chunk_size

                        proj_f32 = proj_f32_grp.current()
                        pl.set_validshape(proj_f32, [valid_mv, h_chunk_size])
                        pl.load(proj_f32, key_back,
                                [row_off + vm_start, h, h_off], order=[0, 2])

                        query_f16 = query_f16_grp.current()
                        pl.set_validshape(query_f16, [valid_mv, h_chunk_size])
                        pl.load(query_f16, hidden_states,
                                [row_off + vm_start, h, h_off], order=[0, 2])
                        query_f32 = query_f32_grp.current()
                        pl.set_validshape(query_f32, [valid_mv, h_chunk_size])
                        pl.cast(query_f32, query_f16, mode=pl.RoundMode.CAST_NONE)

                        vf_accum_sq(
                            proj_f32, query_f32,
                            key_sq_acc, query_sq_acc,
                            valid_mv, h_chunk_size, h_chunk_size,
                        )

                    # ── Phase 2: 由累加平方和算 rms ──
                    key_rms = key_rms_grp.current()
                    pl.set_validshape(key_rms, [valid_mv, 64])
                    query_rms = query_rms_grp.current()
                    pl.set_validshape(query_rms, [valid_mv, 64])
                    vf_compute_rms(
                        key_sq_acc, query_sq_acc, key_rms, query_rms,
                        valid_mv, inv_h,
                    )

                    # ── Phase 3: 跨 H-chunk score 点积 + gate ──
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * h_chunk_size

                        proj_f32 = proj_f32_grp.current()
                        pl.set_validshape(proj_f32, [valid_mv, h_chunk_size])
                        if n_h_chunks != 1:
                            pl.load(proj_f32, key_back,
                                    [row_off + vm_start, h, h_off], order=[0, 2])

                        query_f16 = query_f16_grp.current()
                        pl.set_validshape(query_f16, [valid_mv, h_chunk_size])

                        if n_h_chunks != 1:
                            pl.load(query_f16, hidden_states,
                                    [row_off + vm_start, h, h_off], order=[0, 2])

                        query_f32 = query_f32_grp.current()
                        pl.set_validshape(query_f32, [valid_mv, h_chunk_size])
                        pl.cast(query_f32, query_f16, mode=pl.RoundMode.CAST_NONE)

                        kgamma_f32 = kgamma_f32_grp.current()
                        qgamma_f32 = qgamma_f32_grp.current()

                        vf_score_dot(
                            proj_f32, query_f32, kgamma_f32, qgamma_f32,
                            key_rms, query_rms, score_acc,
                            valid_mv, h_chunk_size, h_chunk_size,
                        )

                    scout_f32 = scout_f32_grp.current()
                    pl.set_validshape(scout_f32, [valid_mv, 64])
                    gaout_f32 = gaout_f32_grp.current()
                    pl.set_validshape(gaout_f32, [valid_mv, 64])
                    vf_compute_gate(
                        score_acc, scout_f32, gaout_f32,
                        valid_mv, inv_sqrt_h,
                    )

                    # store score_back / gate_back (直接 FP32, 无需 cast)
                    pl.store(score_back, scout_f32,
                             [row_off + vm_start, h, 0], order=[0, 2])

                    gaout_f32 = gaout_f32_grp.current()
                    pl.set_validshape(gaout_f32, [valid_mv, 64])
                    pl.store(gate_back, gaout_f32,
                             [row_off + vm_start, h, 0], order=[0, 2])

                    # ── Phase 4: 跨 H-chunk gate * value 广播乘 ──
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * h_chunk_size
                        value_f32 = value_f32_grp.current()
                        pl.set_validshape(value_f32, [valid_mv, h_chunk_size])
                        pl.load(value_f32, value_back, [row_off + vm_start, h_off])
                        vout_f32 = vout_f32_grp.current()
                        pl.set_validshape(vout_f32, [valid_mv, h_chunk_size])
                        vf_bcast_mul(
                            value_f32, gaout_f32, vout_f32,
                            valid_mv, h_chunk_size, h_chunk_size,
                        )

                        vout_f16 = vout_f16_grp.current()
                        pl.set_validshape(vout_f16, [valid_mv, h_chunk_size])
                        pl.cast(vout_f16, vout_f32, mode=pl.RoundMode.CAST_ROUND)
                        pl.store(value_out, vout_f16,
                                 [row_off + vm_start, h, h_off], order=[0, 2])

                # head h 全部 vm_start 处理完 -> set ACK 释放同槽, 允许 cube 推进下一轮 (背压)
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=ACK_IDS[h % 2],
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)


def engram_forward_wrapper(
    hidden_states,        # [B, S, M, H] bf16
    embeddings,           # [B, S, De]   bf16
    key_proj_weights,     # [M, De, H]   bf16
    value_proj_weights,   # [De, H]      bf16
    key_gamma,            # [M, H]       bf16
    query_gamma,          # [M, H]       bf16
):
    b, s, m_dim, h_out = hidden_states.shape
    m = b * s
    de = embeddings.shape[-1]
    device = hidden_states.device

    # ── TilingKey HMode: only H=1280/2560 have a specialized kernel ──
    if h_out not in (1280, 2560):
        raise ValueError(
            f"engram_wrapper only supports H in {{1280, 2560}} (TilingKey "
            f"specialization), got H={h_out}"
        )
    hmode = 1 if h_out == 2560 else 0

    # ── TilingKey CMode selection: M > 2048 -> TILE_M=128, else TILE_M=64 ──
    if m > 2048:
        tile_m, cmode = 128, 0
    else:
        tile_m, cmode = 64, 1
    tiling_key = {"HMode": hmode, "CMode": cmode}

    hs_m = hidden_states.reshape(m, m_dim, h_out).contiguous()
    emb_m = embeddings.reshape(m, de).contiguous()

    value_out = torch.empty((m, m_dim, h_out), dtype=torch.bfloat16, device=device)
    score_back = torch.empty((m, m_dim, 64), dtype=torch.float32, device=device)
    key_back = torch.empty((m, m_dim, h_out), dtype=torch.float32, device=device)
    value_back = torch.empty((m, h_out), dtype=torch.float32, device=device)
    gate_back = torch.empty((m, m_dim, 64), dtype=torch.float32, device=device)

    num_cores = min(32, (m + tile_m - 1) // tile_m)
    if num_cores < 1:
        num_cores = 1

    engram_forward_kernel[None, num_cores, tiling_key](
        hs_m, emb_m, key_proj_weights, value_proj_weights,
        key_gamma, query_gamma,
        value_out, score_back, key_back, value_back, gate_back,
    )

    value_out = value_out.reshape(b, s, m_dim, h_out)
    score_back = score_back[:, :, 0:1].reshape(b, s, m_dim, 1).contiguous()
    key_back = key_back.reshape(b, s, m_dim, h_out)
    value_back = value_back.reshape(b, s, h_out)
    gate_back = gate_back[:, :, 0:1].reshape(b, s, m_dim, 1).contiguous()
    return value_out, score_back, key_back, value_back, gate_back
