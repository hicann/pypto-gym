#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
BSA Forward PyPTO Kernel Implementation

Implements block sparse attention forward pass using PyPTO,
targeting Huawei Ascend NPU.
"""

import math

import torch
import pypto
from torch._dynamo import allow_in_graph

from bsa_common import (
    DEFAULT_CONFIG,
    _pad_to_block_aligned, _is_dense_mask, _build_sparse_kv,
    _make_jit_opts,
    _VEC_TILE_LOAD, _CUBE_TILE, _VEC_TILE_OUTPUT, _CUBE_TILE_LIST,
)


# ===========================================================================
# Forward Sparse Kernel (compacted KV — skip invalid blocks)
# ===========================================================================
_fwd_sparse_cache = {}


def _get_sparse_fwd_kernel(B, Hq, Hkv, Sq, D, numQB, maxSel, cfg):
    key = ("sparse_fwd", B, Hq, Hkv, Sq, D, numQB, maxSel)
    if key in _fwd_sparse_cache:
        return _fwd_sparse_cache[key]

    softmax_scale = cfg.softmax_scale
    bx = cfg.block_shape_x
    by = cfg.block_shape_y
    large_neg = cfg.large_neg
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD
    vto = _VEC_TILE_OUTPUT
    TOTAL_OUTER = B * Hq * numQB

    @pypto.frontend.jit(**_make_jit_opts(cfg, total_outer=TOTAL_OUTER))
    def kernel(
        q_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        k_compact: pypto.Tensor([B * Hq * numQB * maxSel * by, D], pypto.DT_FP16),
        v_compact: pypto.Tensor([B * Hq * numQB * maxSel * by, D], pypto.DT_FP16),
        valid_mask: pypto.Tensor([B * Hq * numQB * maxSel * bx, by], pypto.DT_FP32),
        output_3d: pypto.Tensor([B * Hq, Sq, D], pypto.DT_FP16),
        lse_2d: pypto.Tensor([B * Hq, Sq], pypto.DT_FP32),
    ):
        dtype = q_2d.dtype
        BLOCK = bx
        KV_BLOCK = by

        for outer in pypto.loop(TOTAL_OUTER, name="LOOP_fwd_s_qblk", idx_name="outer_idx"):
            u = outer % numQB
            rest = outer // numQB
            h_q_idx = rest % Hq
            b_idx = rest // Hq
            bh_ofs = b_idx * Hq + h_q_idx

            q_row_ofs = bh_ofs * Sq + u * BLOCK

            mi_update = pypto.tensor([BLOCK, 1], pypto.DT_FP32, "mi_update")
            li_update = pypto.tensor([BLOCK, 1], pypto.DT_FP32, "li_update")
            oi_update = pypto.tensor([BLOCK, D], pypto.DT_FP32, "oi_update")

            for v_idx in pypto.loop(maxSel, name="LOOP_fwd_s_kblk", idx_name="v_idx"):
                kv_row_ofs = outer * maxSel * KV_BLOCK + v_idx * KV_BLOCK

                pypto.set_vec_tile_shapes(*vtl)
                q_block = pypto.view(q_2d, [BLOCK, D], [q_row_ofs, 0])
                k_block = pypto.view(k_compact, [KV_BLOCK, D], [kv_row_ofs, 0])
                v_block = pypto.view(v_compact, [KV_BLOCK, D], [kv_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                S = pypto.matmul(q_block, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)
                S_scaled = pypto.mul(S, softmax_scale)

                mask_row_ofs = outer * maxSel * BLOCK + v_idx * BLOCK
                mask_float = pypto.view(valid_mask, [BLOCK, KV_BLOCK], [mask_row_ofs, 0])
                inv_mask = pypto.add(pypto.mul(mask_float, -1.0), 1.0)
                S_masked = pypto.add(
                    pypto.mul(S_scaled, mask_float),
                    pypto.mul(inv_mask, large_neg))

                m_ij = pypto.amax(S_masked, dim=-1, keepdim=True)
                P_ij = pypto.exp(pypto.sub(S_masked, m_ij))
                l_ij = pypto.sum(P_ij, dim=-1, keepdim=True)
                P_ij_fp16 = pypto.cast(P_ij, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                o_ij = pypto.matmul(P_ij_fp16, v_block, pypto.DT_FP32)

                if pypto.is_loop_begin(v_idx):
                    if pypto.is_loop_end(v_idx):
                        O_final = pypto.div(o_ij, l_ij)
                        pypto.set_vec_tile_shapes(*vto)
                        O_cast = pypto.cast(pypto.reshape(O_final, [1, BLOCK, D]), dtype)
                        pypto.assemble(O_cast, [bh_ofs, u * BLOCK, 0], output_3d)
                        lse_val = pypto.add(m_ij, pypto.log(l_ij))
                        lse_cast = pypto.reshape(lse_val, [1, BLOCK])
                        pypto.assemble(lse_cast, [bh_ofs, u * BLOCK], lse_2d)
                    else:
                        oi_update[:] = o_ij
                    li_update[:] = l_ij
                    mi_update[:] = m_ij
                else:
                    mi_new = pypto.maximum(mi_update, m_ij)
                    alpha = pypto.exp(pypto.sub(mi_update, mi_new))
                    beta = pypto.exp(pypto.sub(m_ij, mi_new))
                    li_new = pypto.add(pypto.mul(alpha, li_update), pypto.mul(beta, l_ij))
                    oi_scaled = pypto.mul(oi_update, alpha)
                    o_ij_scaled = pypto.mul(o_ij, beta)
                    oi_new = pypto.add(oi_scaled, o_ij_scaled)
                    if pypto.is_loop_end(v_idx):
                        O_final = pypto.div(oi_new, li_new)
                        pypto.set_vec_tile_shapes(*vto)
                        O_cast = pypto.cast(pypto.reshape(O_final, [1, BLOCK, D]), dtype)
                        pypto.assemble(O_cast, [bh_ofs, u * BLOCK, 0], output_3d)
                        lse_val = pypto.add(mi_new, pypto.log(li_new))
                        lse_cast = pypto.reshape(lse_val, [1, BLOCK])
                        pypto.assemble(lse_cast, [bh_ofs, u * BLOCK], lse_2d)
                    else:
                        oi_update[:] = oi_new
                    li_update[:] = li_new
                    mi_update[:] = mi_new

    _fwd_sparse_cache[key] = kernel
    return kernel


# ===========================================================================
# Forward Dense Kernel (no mask overhead)
# ===========================================================================
_fwd_dense_cache = {}


def _get_dense_fwd_kernel(B, Hq, Hkv, Sq, Skv, D, numQB, numKB, cfg):
    key = ("dense_fwd", B, Hq, Hkv, Sq, Skv, D, numQB, numKB)
    if key in _fwd_dense_cache:
        return _fwd_dense_cache[key]

    softmax_scale = cfg.softmax_scale
    group = Hq // Hkv
    bx = cfg.block_shape_x
    by = cfg.block_shape_y
    ct = _CUBE_TILE_LIST
    vtl = _VEC_TILE_LOAD
    vto = _VEC_TILE_OUTPUT
    TOTAL_OUTER = B * Hq * numQB

    @pypto.frontend.jit(**_make_jit_opts(cfg, total_outer=TOTAL_OUTER))
    def kernel(
        q_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        k_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        v_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        output_3d: pypto.Tensor([B * Hq, Sq, D], pypto.DT_FP16),
        lse_2d: pypto.Tensor([B * Hq, Sq], pypto.DT_FP32),
    ):
        dtype = q_2d.dtype
        BLOCK = bx
        KV_BLOCK = by

        for outer in pypto.loop(TOTAL_OUTER, name="LOOP_fwd_d_qblk", idx_name="outer_idx"):
            u = outer % numQB
            rest = outer // numQB
            h_q_idx = rest % Hq
            b_idx = rest // Hq
            h_kv_idx = h_q_idx // group
            bh_ofs = b_idx * Hq + h_q_idx

            q_row_ofs = bh_ofs * Sq + u * BLOCK

            mi_update = pypto.tensor([BLOCK, 1], pypto.DT_FP32, "mi_update")
            li_update = pypto.tensor([BLOCK, 1], pypto.DT_FP32, "li_update")
            oi_update = pypto.tensor([BLOCK, D], pypto.DT_FP32, "oi_update")

            for v_blk in pypto.loop(numKB, name="LOOP_fwd_d_kblk", idx_name="v_blk"):
                kv_row_ofs = (b_idx * Hkv + h_kv_idx) * Skv + v_blk * KV_BLOCK

                pypto.set_vec_tile_shapes(*vtl)
                q_block = pypto.view(q_2d, [BLOCK, D], [q_row_ofs, 0])
                k_block = pypto.view(k_2d, [KV_BLOCK, D], [kv_row_ofs, 0])
                v_block = pypto.view(v_2d, [KV_BLOCK, D], [kv_row_ofs, 0])

                pypto.set_cube_tile_shapes(ct, ct, ct)
                S = pypto.matmul(q_block, k_block, pypto.DT_FP32,
                                a_trans=False, b_trans=True)
                S_scaled = pypto.mul(S, softmax_scale)

                m_ij = pypto.amax(S_scaled, dim=-1, keepdim=True)
                P_ij = pypto.exp(pypto.sub(S_scaled, m_ij))
                l_ij = pypto.sum(P_ij, dim=-1, keepdim=True)
                P_ij_fp16 = pypto.cast(P_ij, dtype)
                pypto.set_cube_tile_shapes(ct, ct, ct)
                o_ij = pypto.matmul(P_ij_fp16, v_block, pypto.DT_FP32)

                if pypto.is_loop_begin(v_blk):
                    if pypto.is_loop_end(v_blk):
                        O_final = pypto.div(o_ij, l_ij)
                        pypto.set_vec_tile_shapes(*vto)
                        O_cast = pypto.cast(pypto.reshape(O_final, [1, BLOCK, D]), dtype)
                        pypto.assemble(O_cast, [bh_ofs, u * BLOCK, 0], output_3d)
                        lse_val = pypto.add(m_ij, pypto.log(l_ij))
                        lse_cast = pypto.reshape(lse_val, [1, BLOCK])
                        pypto.assemble(lse_cast, [bh_ofs, u * BLOCK], lse_2d)
                    else:
                        oi_update[:] = o_ij
                    li_update[:] = l_ij
                    mi_update[:] = m_ij
                else:
                    mi_new = pypto.maximum(mi_update, m_ij)
                    alpha = pypto.exp(pypto.sub(mi_update, mi_new))
                    beta = pypto.exp(pypto.sub(m_ij, mi_new))
                    li_new = pypto.add(pypto.mul(alpha, li_update), pypto.mul(beta, l_ij))
                    oi_scaled = pypto.mul(oi_update, alpha)
                    o_ij_scaled = pypto.mul(o_ij, beta)
                    oi_new = pypto.add(oi_scaled, o_ij_scaled)
                    if pypto.is_loop_end(v_blk):
                        O_final = pypto.div(oi_new, li_new)
                        pypto.set_vec_tile_shapes(*vto)
                        O_cast = pypto.cast(pypto.reshape(O_final, [1, BLOCK, D]), dtype)
                        pypto.assemble(O_cast, [bh_ofs, u * BLOCK, 0], output_3d)
                        lse_val = pypto.add(mi_new, pypto.log(li_new))
                        lse_cast = pypto.reshape(lse_val, [1, BLOCK])
                        pypto.assemble(lse_cast, [bh_ofs, u * BLOCK], lse_2d)
                    else:
                        oi_update[:] = oi_new
                    li_update[:] = li_new
                    mi_update[:] = mi_new

    _fwd_dense_cache[key] = kernel
    return kernel


# ===========================================================================
# Public wrapper
# ===========================================================================
@allow_in_graph
def block_sparse_attention_forward(
    query, key, value, block_sparse_mask,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    block_shape=None, cfg=DEFAULT_CONFIG,
):
    bx = block_shape[0] if block_shape else cfg.block_shape_x
    by = block_shape[1] if block_shape else cfg.block_shape_y

    B, Hq, Sq, D = query.shape
    _, Hkv, Skv, _ = key.shape
    assert D == cfg.head_dim

    numQB = math.ceil(Sq / bx)
    numKB = math.ceil(Skv / by)
    Sq_pad = numQB * bx
    Skv_pad = numKB * by

    Q_pad, _ = _pad_to_block_aligned(query, bx)
    K_pad, _ = _pad_to_block_aligned(key, by)
    V_pad, _ = _pad_to_block_aligned(value, by)

    q_2d = Q_pad.reshape(B * Hq * Sq_pad, D)
    k_2d = K_pad.reshape(B * Hkv * Skv_pad, D)
    v_2d = V_pad.reshape(B * Hkv * Skv_pad, D)

    output_3d = torch.zeros(B * Hq, Sq_pad, D, dtype=cfg.torch_dtype, device=query.device)
    lse_2d = torch.full([B * Hq, Sq_pad], cfg.lse_init, dtype=cfg.accum_torch_dtype, device=query.device)

    is_dense = _is_dense_mask(block_sparse_mask)
    is_aligned = (Sq == Sq_pad) and (Skv == Skv_pad)

    if is_dense and is_aligned:
        kernel = _get_dense_fwd_kernel(B, Hq, Hkv, Sq_pad, Skv_pad, D, numQB, numKB, cfg)
        kernel(q_2d, k_2d, v_2d, output_3d, lse_2d)
    else:
        k_compact, v_compact, valid_mask, maxSel = _build_sparse_kv(
            block_sparse_mask, k_2d, v_2d,
            B, Hq, Hkv, Sq, Skv, Sq_pad, Skv_pad, numQB, numKB,
            bx, by, D, query.device)
        torch.npu.synchronize()
        kernel = _get_sparse_fwd_kernel(B, Hq, Hkv, Sq_pad, D, numQB, maxSel, cfg)
        kernel(q_2d, k_compact, v_compact, valid_mask, output_3d, lse_2d)

    attention_out = output_3d[:, :Sq, :].reshape(B, Hq, Sq, D)
    softmax_lse = lse_2d[:, :Sq].reshape(B, Hq, Sq)
    return attention_out, softmax_lse
