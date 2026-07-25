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

import math

import torch
import torch_npu  # noqa: F401

import pypto_pro.language as pl

# ════════════════════════════════════════════════
# Layer B: Compile-time constants
# ════════════════════════════════════════════════

TILE_M = 64           # cube TILE_M (解耦后可调; P0=64 满 32 核, 大 M 可升到 128)
TILE_K = 128
TILE_N = 128
TILE_M_VEC = 8        # vec TILE_M (UB 容量决定; FP32 化后从 4 升到 8, ~206KB < 248KB)
H_CHUNK = 1280        # per-chunk hidden dim for H-split
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

# ════════════════════════════════════════════════
# UB addresses (TILE_M_VEC=8, H_CHUNK=1280)
# BF16 [8,1280]=20480=0x5000  FP32 [8,1280]=40960=0xA000
# ════════════════════════════════════════════════

# 地址布局: key_back/value_back/score_back/gate_back 均已改为 FP32, 省去 BF16 中间 tile.
# 保留 BF16 tile: query (hidden_states 仍然是 BF16), gamma (外部输入 BF16), vout (最终输出 BF16).
VA_PROJ_F32 = 0x00000    # [8,1280] FP32  key_back load (直接 FP32, 无 cast)
VA_QUERY_F16 = 0x0A000   # [8,1280] BF16  hidden_states (query) load; vout_f16 别名复用
VA_VALUE_F32 = 0x0F000   # [8,1280] FP32  value_back load (直接 FP32, 无 cast)
VA_GAMMA_F16 = 0x19000   # [1,1280] BF16  gamma load (kgamma/qgamma 复用)
VA_QUERY_F32 = 0x19A00   # [8,1280] FP32
VA_VOUT_F32 = 0x23A00    # [8,1280] FP32
VA_KGAMMA_F32 = 0x2DA00  # [1,1280] FP32
VA_QGAMMA_F32 = 0x2EE00  # [1,1280] FP32
VA_SCOUT_F32 = 0x30200   # [8,64]   FP32
VA_GAOUT_F32 = 0x30A00   # [8,64]   FP32

# ── H-split 累加器 + rms ──
VA_KEY_SQ_ACC = 0x31200   # [8,64] FP32
VA_QUERY_SQ_ACC = 0x31A00 # [8,64] FP32
VA_SCORE_ACC = 0x32200    # [8,64] FP32
VA_KEY_RMS = 0x32A00      # [8,64] FP32
VA_QUERY_RMS = 0x33200    # [8,64] FP32
# Total UB = 0x33A00 = 211,456 bytes ≈ 206 KB < 248 KB ✓

# L1
LA_EMB = 0x00000
LA_W = 0x20000
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
def engram_vf_accum_sq(
    proj_f32,              # [TILE_M_VEC, H_CHUNK] FP32 (key projection from GM key_back, read-only)
    query_f32,             # [TILE_M_VEC, H_CHUNK] FP32 (query, read-only)
    key_sq_acc,            # [TILE_M_VEC, 64] FP32 in/out
    query_sq_acc,          # [TILE_M_VEC, 64] FP32 in/out
    n_rows,
    n_cols,                # H_CHUNK (=1280)
):
    """Per-chunk: accumulate key_sq and query_sq (no key_back writeback)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    row_stride = 1280

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
def engram_vf_compute_rms(
    key_sq_acc, query_sq_acc,
    key_rms_out, query_rms_out,
    n_rows, inv_h_val,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    eps_reg = vf.full(RMS_EPS, preg, dtype=pl.DT_FP32)
    inv_h_reg = vf.full(inv_h_val, preg, dtype=pl.DT_FP32)
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
def engram_vf_score_dot(
    proj_f32, query_f32, kgamma_f32, qgamma_f32,
    key_rms, query_rms, score_acc,
    n_rows, n_cols,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    row_stride = 1280

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
def engram_vf_compute_gate(
    score_acc, scout_f32, gaout_f32, n_rows, inv_sqrt_h_val,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one_reg = vf.full(1.0, preg, dtype=pl.DT_FP32)
    zero_reg = vf.full(0.0, preg, dtype=pl.DT_FP32)
    neg_one_reg = vf.full(-1.0, preg, dtype=pl.DT_FP32)
    clamp_reg = vf.full(CLAMP_VALUE, preg, dtype=pl.DT_FP32)
    inv_sqrt_h_reg = vf.full(inv_sqrt_h_val, preg, dtype=pl.DT_FP32)

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
def engram_vf_bcast_mul(
    value_f32, gate_f32, vout_f32, n_rows, n_cols,
):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (n_cols + LANES_FP32 - 1) // LANES_FP32
    row_stride = 1280

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
# Layer D: engram_kernel — CV 并行 (cube section + vec section 并发)
# ════════════════════════════════════════════════


@pl.jit(auto_mutex=True)
def engram_kernel(
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
    M = embeddings.shape[0]
    De = embeddings.shape[-1]
    H = hidden_states.shape[-1]
    M_H = hidden_states.shape[1]
    n_k = De // TILE_K
    n_n = H // TILE_N
    n_h_chunks = H // H_CHUNK

    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    sub_idx = pl.get_subblock_idx()
    num_subcores = 2
    n_m_tiles = (M + TILE_M - 1) // TILE_M

    # ═════════ CUBE SECTION ═════════
    # emb 宽 tile: [TILE_M, DE_MAX] 常驻 L1 (只读, 单 buffer), 每 M-tile load 一次, offset move 复用
    tt_emb_wide = pl.TileType(shape=[TILE_M, DE_MAX], dtype=pl.DT_BF16,
                              target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1)
    tt_w_l1 = pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Mat, valid_shape=[-1, -1], compact=1)
    emb_wide_grp = pl.make_tile_group(type=tt_emb_wide, addrs=LA_EMB, mutex_ids=[0])
    w_l1_grp = pl.make_tile_group(type=tt_w_l1, addrs=LA_W, mutex_ids=[2, 3])

    tt_left = pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_BF16,
                          target_memory=pl.MemorySpace.Left, valid_shape=[-1, -1], compact=1)
    tt_right = pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_BF16,
                           target_memory=pl.MemorySpace.Right, valid_shape=[-1, -1], compact=1)
    tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, fractal=1024,
                         valid_shape=[-1, -1], compact=1)
    a_left_grp = pl.make_tile_group(type=tt_left, addrs=L0A_BASE, mutex_ids=[4, 5])
    b_right_grp = pl.make_tile_group(type=tt_right, addrs=L0B_BASE, mutex_ids=[6, 7])
    acc_grp = pl.make_tile_group(type=tt_acc, addrs=L0C_BASE, mutex_ids=[8, 9])


    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)

        # 每 core 按stride取 M-tile (P0 TILE_M=64, M=2048 -> 32 tile, 32核各1)
        for mi in pl.range(core_id, n_m_tiles, num_cores):
            row_off = mi * TILE_M
            valid_m = pl.min(TILE_M, M - row_off)

            # ── emb 复用: 本 tile 的 emb [valid_m, De] 一次性 GM->L1 进宽 tile, 后面 value+16head 只 offset move ──
            emb_wide = emb_wide_grp.current()
            pl.set_validshape(emb_wide, [valid_m, De])
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
            for h in pl.range(0, M_H):
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
                             tile_dims=[0, 2])

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
    tt_mv_f16 = pl.TileType(shape=[TILE_M_VEC, H_CHUNK], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)
    tt_1d_f16 = pl.TileType(shape=[1, H_CHUNK], dtype=pl.DT_BF16,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)
    tt_mv_f32 = pl.TileType(shape=[TILE_M_VEC, H_CHUNK], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)
    tt_1d_f32 = pl.TileType(shape=[1, H_CHUNK], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)
    tt_mv64_f32 = pl.TileType(shape=[TILE_M_VEC, 64], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], compact=1)

    # ── 保留下来的 BF16 tiles (hidden_states/gamma 仍是 BF16, vout 最终输出 BF16) ──
    query_f16_grp = pl.make_tile_group(type=tt_mv_f16, addrs=VA_QUERY_F16, mutex_ids=[11])
    gamma_f16_grp = pl.make_tile_group(type=tt_1d_f16, addrs=VA_GAMMA_F16, mutex_ids=[14])
    vout_f16_grp = pl.make_tile_group(type=tt_mv_f16, addrs=VA_QUERY_F16, mutex_ids=[11])  # 别名复用

    # FP32 tiles (key_back/value_back 直接 FP32 load, 无需中间 BF16 cast)
    proj_f32_grp = pl.make_tile_group(type=tt_mv_f32, addrs=VA_PROJ_F32, mutex_ids=[21])
    query_f32_grp = pl.make_tile_group(type=tt_mv_f32, addrs=VA_QUERY_F32, mutex_ids=[22])
    value_f32_grp = pl.make_tile_group(type=tt_mv_f32, addrs=VA_VALUE_F32, mutex_ids=[23])
    vout_f32_grp = pl.make_tile_group(type=tt_mv_f32, addrs=VA_VOUT_F32, mutex_ids=[25])
    scout_f32_grp = pl.make_tile_group(type=tt_mv64_f32, addrs=VA_SCOUT_F32, mutex_ids=[29])
    gaout_f32_grp = pl.make_tile_group(type=tt_mv64_f32, addrs=VA_GAOUT_F32, mutex_ids=[30])
    kgamma_f32_grp = pl.make_tile_group(type=tt_1d_f32, addrs=VA_KGAMMA_F32, mutex_ids=[31])
    qgamma_f32_grp = pl.make_tile_group(type=tt_1d_f32, addrs=VA_QGAMMA_F32, mutex_ids=[13])

    # H-split accumulator tiles
    key_sq_acc_grp = pl.make_tile_group(type=tt_mv64_f32, addrs=VA_KEY_SQ_ACC, mutex_ids=[15])
    query_sq_acc_grp = pl.make_tile_group(type=tt_mv64_f32, addrs=VA_QUERY_SQ_ACC, mutex_ids=[16])
    score_acc_grp = pl.make_tile_group(type=tt_mv64_f32, addrs=VA_SCORE_ACC, mutex_ids=[17])
    key_rms_grp = pl.make_tile_group(type=tt_mv64_f32, addrs=VA_KEY_RMS, mutex_ids=[26])
    query_rms_grp = pl.make_tile_group(type=tt_mv64_f32, addrs=VA_QUERY_RMS, mutex_ids=[27])

    with pl.section_vector():
        for mi in pl.range(core_id, n_m_tiles, num_cores):
            row_off = mi * TILE_M
            # per-head 消费: wait K_READY[h] 后, 把 head h 的 [TILE_M, H] 按 TILE_M_VEC 细粒度处理
            # num_cores==1 时: cube section 已 sync_all, value_back/key_back 全部就绪, 无需 per-head 等待
            for h in pl.range(0, M_H):
                # 等 cube 存完 head h 的 key_back(value_back 在 K_READY[0] 前已就绪)
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE2, event_id=K_READY_IDS[h % 2],
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

                # AIV split: sub_idx 0 负责偶数 vm_start 段, sub_idx 1 负责奇数段
                for vm_start in pl.range(sub_idx * TILE_M_VEC, TILE_M, num_subcores * TILE_M_VEC):
                    valid_mv = pl.min(TILE_M_VEC, M - row_off - vm_start)
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

                    if H == 1280:
                        inv_h = INV_H_1280
                        inv_sqrt_h = INV_SQRT_H_1280
                    else:
                        inv_h = INV_H_2560
                        inv_sqrt_h = INV_SQRT_H_2560

                    # ── Phase 1: 跨 H-chunk 累加 key_sq / query_sq ──
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * H_CHUNK

                        proj_f32 = proj_f32_grp.current()
                        pl.set_validshape(proj_f32, [valid_mv, H_CHUNK])
                        pl.load(proj_f32, key_back,
                                [row_off + vm_start, h, h_off], order=[0, 2])

                        query_f16 = query_f16_grp.current()
                        pl.set_validshape(query_f16, [valid_mv, H_CHUNK])
                        pl.load(query_f16, hidden_states,
                                [row_off + vm_start, h, h_off], order=[0, 2])
                        query_f32 = query_f32_grp.current()
                        pl.set_validshape(query_f32, [valid_mv, H_CHUNK])
                        pl.cast(query_f32, query_f16, mode=pl.RoundMode.CAST_NONE)

                        engram_vf_accum_sq(
                            proj_f32, query_f32,
                            key_sq_acc, query_sq_acc,
                            valid_mv, H_CHUNK,
                        )

                    # ── Phase 2: 由累加平方和算 rms ──
                    key_rms = key_rms_grp.current()
                    pl.set_validshape(key_rms, [valid_mv, 64])
                    query_rms = query_rms_grp.current()
                    pl.set_validshape(query_rms, [valid_mv, 64])
                    engram_vf_compute_rms(
                        key_sq_acc, query_sq_acc, key_rms, query_rms,
                        valid_mv, inv_h,
                    )

                    # ── Phase 3: 跨 H-chunk score 点积 + gate ──
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * H_CHUNK

                        proj_f32 = proj_f32_grp.current()
                        pl.set_validshape(proj_f32, [valid_mv, H_CHUNK])
                        if n_h_chunks != 1:
                            pl.load(proj_f32, key_back,
                                    [row_off + vm_start, h, h_off], order=[0, 2])

                        query_f16 = query_f16_grp.current()
                        pl.set_validshape(query_f16, [valid_mv, H_CHUNK])

                        if n_h_chunks != 1:
                            pl.load(query_f16, hidden_states,
                                    [row_off + vm_start, h, h_off], order=[0, 2])

                        query_f32 = query_f32_grp.current()
                        pl.set_validshape(query_f32, [valid_mv, H_CHUNK])
                        pl.cast(query_f32, query_f16, mode=pl.RoundMode.CAST_NONE)

                        gamma_f16 = gamma_f16_grp.current()
                        pl.set_validshape(gamma_f16, [1, H_CHUNK])
                        pl.load(gamma_f16, key_gamma, [h, h_off], order=[0])
                        kgamma_f32 = kgamma_f32_grp.current()
                        pl.set_validshape(kgamma_f32, [1, H_CHUNK])
                        pl.cast(kgamma_f32, gamma_f16, mode=pl.RoundMode.CAST_NONE)

                        pl.load(gamma_f16, query_gamma, [h, h_off], order=[0])
                        qgamma_f32 = qgamma_f32_grp.current()
                        pl.set_validshape(qgamma_f32, [1, H_CHUNK])
                        pl.cast(qgamma_f32, gamma_f16, mode=pl.RoundMode.CAST_NONE)

                        engram_vf_score_dot(
                            proj_f32, query_f32, kgamma_f32, qgamma_f32,
                            key_rms, query_rms, score_acc,
                            valid_mv, H_CHUNK,
                        )

                    scout_f32 = scout_f32_grp.current()
                    pl.set_validshape(scout_f32, [valid_mv, 64])
                    gaout_f32 = gaout_f32_grp.current()
                    pl.set_validshape(gaout_f32, [valid_mv, 64])
                    engram_vf_compute_gate(
                        score_acc, scout_f32, gaout_f32,
                        valid_mv, inv_sqrt_h,
                    )

                    # store score_back / gate_back (直接 FP32, 无需 cast)
                    pl.store(score_back, scout_f32,
                             [row_off + vm_start, h, 0], tile_dims=[0, 2])

                    gaout_f32 = gaout_f32_grp.current()
                    pl.set_validshape(gaout_f32, [valid_mv, 64])
                    pl.store(gate_back, gaout_f32,
                             [row_off + vm_start, h, 0], tile_dims=[0, 2])

                    # ── Phase 4: 跨 H-chunk gate * value 广播乘 ──
                    for h_chunk in pl.range(0, n_h_chunks):
                        h_off = h_chunk * H_CHUNK
                        value_f32 = value_f32_grp.current()
                        pl.set_validshape(value_f32, [valid_mv, H_CHUNK])
                        pl.load(value_f32, value_back, [row_off + vm_start, h_off])
                        vout_f32 = vout_f32_grp.current()
                        pl.set_validshape(vout_f32, [valid_mv, H_CHUNK])
                        engram_vf_bcast_mul(
                            value_f32, gaout_f32, vout_f32,
                            valid_mv, H_CHUNK,
                        )

                        vout_f16 = vout_f16_grp.current()
                        pl.set_validshape(vout_f16, [valid_mv, H_CHUNK])
                        pl.cast(vout_f16, vout_f32, mode=pl.RoundMode.CAST_ROUND)
                        pl.store(value_out, vout_f16,
                                 [row_off + vm_start, h, h_off], tile_dims=[0, 2])

                # head h 全部 vm_start 处理完 -> set ACK 释放同槽, 允许 cube 推进下一轮 (背压)
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=ACK_IDS[h % 2],
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)



# ════════════════════════════════════════════════
# Layer E: Host wrapper
# ════════════════════════════════════════════════

def engram_wrapper(
    hidden_states,        # [B, S, M, H] bf16
    embeddings,           # [B, S, De]   bf16
    key_proj_weights,     # [M, De, H]   bf16
    value_proj_weights,   # [De, H]      bf16
    key_gamma,            # [M, H]       bf16
    query_gamma,          # [M, H]       bf16
):
    B, S, M_dim, H_out = hidden_states.shape
    M = B * S
    De = embeddings.shape[-1]
    device = hidden_states.device

    hs_m = hidden_states.reshape(M, M_dim, H_out).contiguous()
    emb_m = embeddings.reshape(M, De).contiguous()

    value_out = torch.empty((M, M_dim, H_out), dtype=torch.bfloat16, device=device)
    score_back = torch.empty((M, M_dim, 64), dtype=torch.float32, device=device)
    key_back = torch.empty((M, M_dim, H_out), dtype=torch.float32, device=device)
    value_back = torch.empty((M, H_out), dtype=torch.float32, device=device)
    gate_back = torch.empty((M, M_dim, 64), dtype=torch.float32, device=device)

    num_cores = min(32, (M + TILE_M - 1) // TILE_M)
    if num_cores < 1:
        num_cores = 1

    engram_kernel[None, num_cores](
        hs_m, emb_m, key_proj_weights, value_proj_weights,
        key_gamma, query_gamma,
        value_out, score_back, key_back, value_back, gate_back,
    )

    value_out = value_out.reshape(B, S, M_dim, H_out)
    score_back = score_back[:, :, 0:1].reshape(B, S, M_dim, 1).contiguous()
    key_back = key_back.reshape(B, S, M_dim, H_out)
    value_back = value_back.reshape(B, S, H_out)
    gate_back = gate_back[:, :, 0:1].reshape(B, S, M_dim, 1).contiguous()
    return value_out, score_back, key_back, value_back, gate_back

