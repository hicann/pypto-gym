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
"""
"""
import pypto
import torch
import math
from ..pypto_test import TestBuilder


def _assemble_q_tensors(cur_n_tile, d_n, d_r, cur_offset, dtype, q_nope, q_rope):
    """Assemble Q (nope + rope) tensors for a head tile."""
    qn = pypto.view(q_nope, [cur_n_tile, d_n], [cur_offset, 0])
    qr = pypto.view(q_rope, [cur_n_tile, d_r], [cur_offset, 0])
    qi = pypto.tensor([cur_n_tile, d_n + d_r], dtype, "qi")
    pypto.assemble(qn, [0, 0], qi)
    pypto.assemble(qr, [0, d_n], qi)
    return qi


def _load_kv_cache_block(b_idx, bn, block_size, cur_seq, d_n, d_r, dtype,
                          block_table, k_nope_cache, k_rope_cache, v_nope_cache, is_nz_format):
    """Load one KV cache block and return assembled KJ and VJ tensors."""
    cur_s2_tile = block_size
    cur_block_idx = block_table[b_idx, bn]
    cur_block_idx.as_variable()
    kn = pypto.view(k_nope_cache, [cur_s2_tile, d_n],
                    [cur_block_idx * block_size, 0],
                    valid_shape=[(cur_seq - bn * block_size).min(block_size), d_n])
    kr = pypto.view(k_rope_cache, [cur_s2_tile, d_r],
                    [cur_block_idx * block_size, 0],
                    valid_shape=[(cur_seq - bn * block_size).min(block_size), d_r])
    kj_fmt = pypto.TileOpFormat.TILEOP_NZ if is_nz_format else pypto.TileOpFormat.TILEOP_ND
    kj = pypto.tensor([cur_s2_tile, d_n + d_r], dtype, "kj", kj_fmt)
    pypto.assemble(kn, [0, 0], kj)
    pypto.assemble(kr, [0, d_n], kj)
    kj = pypto.view(kj, [cur_s2_tile, d_n + d_r], [0, 0],
                    valid_shape=[(cur_seq - bn * block_size).min(block_size), d_r + d_n])
    vj = pypto.view(v_nope_cache, [cur_s2_tile, d_n], [cur_block_idx * block_size, 0],
                    valid_shape=[(cur_seq - bn * block_size).min(block_size), d_n])
    return kj, vj


def _compute_block_sij(qi, kj, softmax_scale, dtype, c1_tile, v1_tile):
    """MatMul Qi·Kj^T → scale → softmax → (pij_f16, lij, mij)."""
    pypto.set_semantic_label("MatMul")
    pypto.set_cube_tile_shapes(
        [c1_tile[0], c1_tile[1]], [c1_tile[2], c1_tile[3]],
        [c1_tile[4], c1_tile[5]])
    pypto.set_matrix_size([qi.shape[0], 0, kj.shape[0]])
    sij = pypto.matmul(qi, kj, pypto.DT_FP32, b_trans=True)
    pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
    pypto.set_semantic_label("SoftMax")
    sij_scale = pypto.mul(sij, float(softmax_scale))
    pypto.set_semantic_label("SoftMax")
    tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
    tsub = pypto.sub(sij_scale, tilda_mij)
    tilda_pij = pypto.exp(tsub)
    tilda_pij_f16 = pypto.cast(tilda_pij, dtype)
    tilda_lij = pypto.sum(tilda_pij, dim=-1, keepdim=True)
    return tilda_pij_f16, tilda_lij, tilda_mij


def op_page_attention(params, q_nope, k_nope_cache, v_nope_cache, q_rope, k_rope_cache,
    block_table, act_seqs, attention_out):
    block_size = params["block_size"]
    tile_config = params["tile_config"]
    max_unroll_times = params["max_unroll_times"]
    is_nz_format = params["is_nz_format"]
    dtype = q_nope.dtype
    d_n = q_nope.shape[1]
    d_r = q_rope.shape[1]
    softmax_scale = (d_n+d_r) ** -0.5
    n_tile = tile_config.head_num_q_tile
    c1_tile = tile_config.c1_tile_shape
    v1_tile = tile_config.v1_tile_shape
    c2_tile = tile_config.c2_tile_shape
    v2_tile = tile_config.v2_tile_shape

    def _bn_first_iteration_body(oi_update, li_update, mi_update, tilda_pij_f16, vj, tilda_lij, tilda_mij,
                                  c2_tile, v2_tile, bn, oi_offset, attention_out):
        pypto.set_cube_tile_shapes([c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
        pypto.set_semantic_label("b1-matmul2")
        pypto.set_matrix_size([tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], vj.shape[1]])
        oi_tmp = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
        pypto.set_semantic_label("b1-after-matmul2")
        if pypto.cond(pypto.is_loop_end(bn)):
            pypto.set_semantic_label("b1-after-matmul2")
            oi_update[:] = (pypto.div(oi_tmp, tilda_lij))
            pypto.assemble(oi_update, oi_offset, attention_out)
        else:
            oi_update[:] = (oi_tmp)
        li_update[:] = (tilda_lij)
        mi_update[:] = (tilda_mij)

    def _bn_subsequent_iteration_body(oi_update, li_update, mi_update, tilda_pij_f16, vj, tilda_mij, tilda_lij,
                                       c2_tile, v2_tile, bn, oi_offset, attention_out):
        oi = oi_update
        li = li_update
        mi = mi_update
        pypto.set_semantic_label("Softmax-acc")
        mi_new = pypto.maximum(mi, tilda_mij)
        t1 = pypto.sub(mi, mi_new)
        t2 = pypto.exp(t1)
        t3 = pypto.sub(tilda_mij, mi_new)
        t4 = pypto.exp(t3)
        t5 = pypto.mul(t4, tilda_lij)
        t6 = pypto.mul(t2, li)
        li_new = pypto.add(t6, t5)
        q3 = pypto.mul(oi, t2)
        pypto.set_semantic_label("bn-matmul2")
        pypto.set_cube_tile_shapes([c2_tile[0], c2_tile[1]], [c2_tile[2], c2_tile[3]], [c2_tile[4], c2_tile[5]])
        pypto.set_matrix_size([tilda_pij_f16.shape[0], tilda_pij_f16.shape[1], vj.shape[1]])
        q1 = pypto.matmul(tilda_pij_f16, vj, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
        pypto.set_semantic_label("bn-after-matmul2")
        q2 = pypto.mul(q1, t4)
        oi_tmp = pypto.add(q3, q2)
        if pypto.cond(pypto.is_loop_end(bn)):
            oi_update[:] = (pypto.div(oi_tmp, li_new))
            pypto.assemble(oi_update, oi_offset, attention_out)
        else:
            oi_update[:] = (oi_tmp)
        li_update[:] = (li_new)
        mi_update[:] = (mi_new)

    def inside_main_function():
        batch_size = block_table.shape[0]
        n_q = q_nope.shape[0] // batch_size
        n_loop = n_q // n_tile
        for b_idx in pypto.loop(0, batch_size, 1, name="LOOP_L0_bIdx", idx_name="b_idx"):
            def inside_b_idx_loop(b_idx):
                cur_seq = act_seqs[b_idx]
                bn_per_batch = (cur_seq + block_size - 1) // block_size
                bn_per_batch.as_variable()
                for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L1_nIdx", idx_name="n_idx"):
                    def inside_n_idx_loop(b_idx, n_idx, bn_per_batch):
                        nonlocal n_tile
                        cur_n_tile = n_tile
                        oi_update = pypto.tensor([n_tile, d_n], pypto.DT_FP32, "oi_update")
                        li_update = pypto.tensor([n_tile, 1], pypto.DT_FP32, "li_update")
                        mi_update = pypto.tensor([n_tile, 1], pypto.DT_FP32, "mi_update")
                        cur_offset = b_idx * n_q + n_idx * n_tile
                        oi_offset = [cur_offset, 0]

                        for bn in pypto.loop(0, bn_per_batch, 1, name="LOOP_L2_bn",
                                           idx_name="bn", unroll_List={max_unroll_times}):
                            def inside_bn_loop(**kwargs):
                                b_idx = kwargs.get("b_idx")
                                block_table = kwargs.get("block_table")
                                cur_seq = kwargs.get("cur_seq")
                                bn = kwargs.get("bn")
                                block_size = kwargs.get("block_size")
                                nonlocal oi_update, li_update, mi_update

                                qi = _assemble_q_tensors(cur_n_tile, d_n, d_r, cur_offset, dtype,
                                                         q_nope, q_rope)
                                kj, vj = _load_kv_cache_block(b_idx, bn, block_size, cur_seq, d_n, d_r,
                                                              dtype, block_table, k_nope_cache,
                                                              k_rope_cache, v_nope_cache, is_nz_format)
                                tilda_pij_f16, tilda_lij, tilda_mij = _compute_block_sij(
                                    qi, kj, softmax_scale, dtype, c1_tile, v1_tile)

                                if pypto.cond(pypto.is_loop_begin(bn)):
                                    _bn_first_iteration_body(oi_update, li_update, mi_update,
                                        tilda_pij_f16, vj, tilda_lij, tilda_mij, c2_tile, v2_tile, bn,
                                        oi_offset, attention_out)
                                else:
                                    _bn_subsequent_iteration_body(oi_update, li_update, mi_update,
                                        tilda_pij_f16, vj, tilda_mij, tilda_lij, c2_tile, v2_tile, bn,
                                        oi_offset, attention_out)
                            inside_bn_loop(
                                b_idx=b_idx,
                                block_table=block_table,
                                cur_seq=cur_seq,
                                bn=bn,
                                block_size=block_size,
                                bn_per_batch=bn_per_batch)
                    inside_n_idx_loop(b_idx, n_idx, bn_per_batch)
            inside_b_idx_loop(b_idx)
    inside_main_function()


def _golden_compute_per_batch(b_index, b, q_bnsd, n_tile, n_q, block_size, block_table,
                               k_cache, v_cache, d_k, d_v, scalar, block_num_per_batch, act_seqs):
    matmul_dtype = torch.float32
    cur_seq = act_seqs[b_index]
    bn_per_batch = math.ceil(cur_seq / block_size)
    n_loop = math.ceil(n_q / n_tile)
    tiled_out = []
    for n_idx in range(n_loop):
        oi_update = []
        li_update = []
        mi_update = []
        qi = q_bnsd[b_index, n_idx * n_tile: (n_idx + 1) * n_tile, :, :]
        qi = qi.reshape(-1, qi.shape[-1])
        for bn in range(block_num_per_batch[b_index]):
            cur_block_idx = block_table[b_index][bn]
            s2_tile_cur = min(block_size, cur_seq - bn * block_size)
            kj = k_cache[cur_block_idx, 0:s2_tile_cur, :]
            vj = v_cache[cur_block_idx, 0:s2_tile_cur, :]
            kj = kj.reshape(s2_tile_cur, d_k)
            vj = vj.reshape(s2_tile_cur, d_v)

            sij = torch.matmul(qi.to(matmul_dtype), kj.to(matmul_dtype).mT)
            sij_scale = sij * scalar
            tilda_mij = sij_scale.max(dim=-1, keepdim=True).values
            t_sub = sij_scale - tilda_mij
            tilda_pij = torch.exp(t_sub)
            tilda_lij = tilda_pij.sum(dim=-1, keepdim=True)

            if bn == 0:
                oi_tmp = torch.matmul(tilda_pij.to(matmul_dtype), vj.to(matmul_dtype))
                if bn_per_batch == 1:
                    oi_update = oi_tmp / tilda_lij
                else:
                    oi_update = oi_tmp
                li_update = tilda_lij
                mi_update = tilda_mij
                continue
            oi = oi_update
            li = li_update
            mi = mi_update

            mi_new = torch.maximum(mi, tilda_mij)
            t1 = mi - mi_new
            t2 = torch.exp(t1)
            t3 = tilda_mij - mi_new
            t4 = torch.exp(t3)
            t5 = t4 * tilda_lij
            t6 = t2 * li
            li_new = t6 + t5
            q3 = oi * t2
            q1 = torch.matmul(tilda_pij.to(matmul_dtype), vj.to(matmul_dtype))
            q2 = q1 * t4
            oi_tmp = q3 + q2
            if bn == block_num_per_batch[b_index] - 1:
                oi_update = oi_tmp / li_new
            else:
                oi_update = oi_tmp
            li_update = li_new
            mi_update = mi_new
        tiled_out.append(oi_update)
    return tiled_out


def op_page_attention_golden(params, q_nope, k_nope_cache, v_cache, q_rope, k_rope_cache,
    block_table, act_seqs, attention_out):
    b = params["b"]
    n_q = params["n_q"]
    s_q = params["s_q"]
    n_kv = params["n_kv"]
    kv_lora_rank = params["kv_lora_rank"]
    qk_rope_dim = params["qk_rope_dim"]
    d_q = kv_lora_rank + qk_rope_dim
    d_k = kv_lora_rank + qk_rope_dim
    d_v = kv_lora_rank
    n_tile = params["n_tile"]
    block_size = params["block_size"]
    block_num = params["block_num"]

    q_nope = q_nope.reshape(b, n_q, s_q, kv_lora_rank)
    q_rope = q_rope.reshape(b, n_q, s_q, qk_rope_dim)
    k_nope_cache = k_nope_cache.reshape(block_num, block_size, n_kv * kv_lora_rank)
    k_rope_cache = k_rope_cache.reshape(block_num, block_size, n_kv * qk_rope_dim)
    v_cache = v_cache.reshape(block_num, block_size, n_kv * d_v)
    q_bnsd = torch.cat([q_nope, q_rope], dim=-1)
    k_cache = torch.cat([k_nope_cache, k_rope_cache], dim=-1)
    scalar = d_q ** -0.5
    tiled_out = []
    block_num_per_batch = []
    for actual_seq in act_seqs:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
    for b_index in range(b):
        batch_results = _golden_compute_per_batch(b_index, b, q_bnsd, n_tile, n_q,
            block_size, block_table, k_cache, v_cache, d_k, d_v, scalar, block_num_per_batch, act_seqs)
        tiled_out.extend(batch_results)
    attent_out = torch.cat(tiled_out, dim=0)
    return (attent_out, )


class PATest(TestBuilder):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__(params, kernel, kernel_golden, tiling)

    def _generate_input_tensors(self):
        b = self.params["b"]
        n_q = self.params["n_q"]
        skv = self.params["skv"]
        block_size = self.params["block_size"]
        dtype = self.params["dtype"]
        s_q = self.params["s_q"]
        n_kv = self.params["n_kv"]
        kv_lora_rank = self.params["kv_lora_rank"]
        qk_rope_dim = self.params["qk_rope_dim"]
        d_q = kv_lora_rank + qk_rope_dim
        d_v = kv_lora_rank
        actual_seq_len = torch.full((b,), skv, dtype=torch.int32)
        s_max = max(actual_seq_len)
        shape_q = [b * n_q * s_q, d_q]
        shape_k = [b, s_max, n_kv * d_q]
        q_bnsd = self._gen_uniform_data(shape_q, -1, 1, dtype)
        k_tensor_bsh_raw = self._gen_uniform_data(shape_k, -1, 1, dtype)
        v_tensor_bsh_raw = k_tensor_bsh_raw[:, :, :n_kv * d_v]
        return actual_seq_len, s_max, q_bnsd, k_tensor_bsh_raw, v_tensor_bsh_raw, dtype

    def _build_block_table_and_caches(self, actual_seq_len, s_max, block_size, dtype,
                                       k_tensor_bsh_raw, v_tensor_bsh_raw,
                                       kv_lora_rank, qk_rope_dim):
        b = self.params["b"]
        n_kv = self.params["n_kv"]
        d_k = kv_lora_rank + qk_rope_dim
        d_v = kv_lora_rank
        block_num_per_batch = []
        block_num_min = 0
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num_min += math.ceil(actual_seq / block_size)
        block_table_shape = [b, math.ceil(s_max / block_size)]
        block_num = block_num_min
        block_idx_list = torch.arange(0, block_num, 1)
        block_idx_list = torch.randperm(len(block_idx_list), dtype=torch.int32)
        block_idx = 0
        block_table = torch.full((block_table_shape[0], block_table_shape[1]), -1, dtype=torch.int32)
        block_table_batch_idx = 0
        for idx in block_num_per_batch:
            for j in range(idx):
                block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
                block_idx += 1
            block_table_batch_idx += 1
        k_cache = torch.zeros(block_num, block_size, n_kv * d_k, dtype=dtype)
        v_cache = torch.zeros(block_num, block_size, n_kv * d_v, dtype=dtype)
        k_tensor_bsh = torch.zeros(b, block_table_shape[1] * block_size, n_kv * d_k, dtype=dtype)
        v_tensor_bsh = torch.zeros(b, block_table_shape[1] * block_size, n_kv * d_v, dtype=dtype)
        k_tensor_bsh[:, :k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
        v_tensor_bsh[:, :v_tensor_bsh_raw.shape[1], :] = v_tensor_bsh_raw[:, :, :]
        for b_idx in range(b):
            for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
                block_offset = block_i * block_size
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_cache[kv_cache_blk_id, 0:block_size, :] = k_tensor_bsh[
                                                                b_idx, block_offset:(block_offset + block_size), :]
                    v_cache[kv_cache_blk_id, 0:block_size, :] = v_tensor_bsh[
                                                                b_idx, block_offset:(block_offset + block_size), :]
        return block_table, block_num, k_cache, v_cache, k_tensor_bsh, v_tensor_bsh

    @staticmethod
    def _gen_uniform_data(data_shape, min_value, max_value, dtype):
        if min_value == 0 and max_value == 0:
            return torch.zeros(data_shape, dtype=dtype)
        if dtype == torch.bool:
            return torch.rand(data_shape) < 0.5
        return (torch.rand(data_shape) * (max_value - min_value) + min_value).to(dtype)

    @staticmethod
    def _convert_tensors_contiguous(tensor_list):
        for idx, t in enumerate(tensor_list):
            if isinstance(t, torch.Tensor):
                tensor_list[idx] = t if t.is_contiguous() else t.contiguous()
        return tensor_list

    def get_input_from_param(self):
        actual_seq_len, s_max, q_bnsd, k_tensor_bsh_raw, v_tensor_bsh_raw, dtype = self._generate_input_tensors()
        kv_lora_rank = self.params["kv_lora_rank"]
        qk_rope_dim = self.params["qk_rope_dim"]
        block_size = self.params["block_size"]
        n_kv = self.params["n_kv"]

        block_table, block_num, k_cache, v_cache, k_tensor_bsh, v_tensor_bsh = \
            self._build_block_table_and_caches(actual_seq_len, s_max, block_size, dtype,
                k_tensor_bsh_raw, v_tensor_bsh_raw, kv_lora_rank, qk_rope_dim)

        q_nope = q_bnsd[:, :kv_lora_rank]
        q_rope = q_bnsd[:, kv_lora_rank:]
        k_cache_nope_h = kv_lora_rank * n_kv
        k_cache_nope = k_cache[:, :, :k_cache_nope_h]
        k_cache_rope = k_cache[:, :, k_cache_nope_h:]
        k_cache_nope = k_cache_nope.reshape(k_cache_nope.shape[0] * k_cache_nope.shape[1], k_cache_nope.shape[-1])
        k_cache_rope = k_cache_rope.reshape(k_cache_rope.shape[0] * k_cache_rope.shape[1], k_cache_rope.shape[-1])
        v_cache = v_cache.reshape(v_cache.shape[0] * v_cache.shape[1], v_cache.shape[-1])
        kernel_inputs = [q_nope, k_cache_nope, v_cache, q_rope, k_cache_rope,
            block_table, actual_seq_len]
        kernel_inputs = self._convert_tensors_contiguous(kernel_inputs)
        self.setup_inputs(*kernel_inputs)
        self.set_tol(rtol=5e-4, atol=5e-4)
        self.params["block_num"] = block_num
        golden_inputs = [q_nope, k_cache_nope, v_cache, q_rope, k_cache_rope,
            block_table, actual_seq_len, self.params]
        return golden_inputs


class TileConfig:
    def __init__(self, head_num_q_tile, c1_tile_shape, v1_tile_shape,
    c2_tile_shape, v2_tile_shape):
        self.head_num_q_tile = head_num_q_tile
        self.c1_tile_shape = c1_tile_shape
        self.v1_tile_shape = v1_tile_shape
        self.c2_tile_shape = c2_tile_shape
        self.v2_tile_shape = v2_tile_shape


def test():
    # 目标上板函数相关超参数及golden验证函数相关超参数配置
    model_golden_params = {
        "block_size": 128,
        "tile_config": TileConfig(head_num_q_tile=32,
            c1_tile_shape=(32, 32, 64, 64, 128, 128),
            v1_tile_shape=(32, 64),
            c2_tile_shape=(32, 32, 64, 64, 128, 128),
            v2_tile_shape=(32, 64)),
        "max_unroll_times": 1,
        "is_nz_format": False,
        "b": 4,
        "n_q": 32,
        "skv": 256,
        "block_num": 64,
        "dtype": torch.float32,
        "s_q": 1,
        "n_kv": 1,
        "kv_lora_rank": 512,
        "qk_rope_dim": 64,
        "n_tile": 32
    }
    st = PATest(model_golden_params, op_page_attention, op_page_attention_golden, tiling=32)
    st()
