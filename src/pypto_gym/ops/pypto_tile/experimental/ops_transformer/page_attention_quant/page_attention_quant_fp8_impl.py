#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 Attention Module

This module implements the Attention mechanism for GLM-4.5 model, which uses
a paged memory management approach similar to operating systems to efficiently
handle variable-length sequences and dynamic batch sizes in attention computation.

Main Functions:
    - attention: Main attention function with Attention support
    - ifa_func: JIT compiled kernel implementing Flash Attention with paged KV cache
    - gen_block_table: Generate block mapping table for Attention
    - kv_cache_concat_bsnd: Convert paged KV cache to BSND format
"""
import os
from dataclasses import dataclass
from typing import Tuple
import torch
import pypto


@dataclass
class AttentionTileConfig:
    g_tile: int = 12
    s2_tile: int = 512
    c1_tile_shape: list = None
    v1_tile_shape: list = None
    c2_tile_shape: list = None
    v2_tile_shape: list = None


global_tile_config = AttentionTileConfig()


@dataclass
class AttentionConfig:
    b: int = 8
    s1: int = 1
    s2: int = 16384
    n1: int = 12
    n2: int = 1
    q_d: int = 128
    kv_d: int = 128
    block_size: int = 128
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0
    kv_layout: str = "PA_BSND"
    actual_seq: torch.Tensor = None  # 改为 torch.Tensor 类型
    block_table_batch: int = 0
    kv_num_blocks: int = 0


global_config = AttentionConfig()


def get_common_config():
    return global_config, global_tile_config


def set_qwen_common_config(b=16, s1=1, s2=8192):
    global global_tile_config
    global global_config
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    b = b
    s1 = s1
    s2 = s2
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)

    # 创建 torch tensor 类型的 actual_seq
    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale, kv_layout=kv_layout,
                                q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
                                kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor)  # 传入 tensor
    atten_cfg.max_num_blocks_per_query = (s2 + block_size - 1) // block_size
    cube_tile = 128
    m_tile = 128
    s2_tile = 1024
    tile_cfg = AttentionTileConfig(
        nq,
        s2_tile,
        [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, 1024],
        [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, 512])
    global_config = atten_cfg
    global_tile_config = tile_cfg


def dequant_dynamic(in_tensor, scale_1, scale_2):
    """
    Perform dynamic dequantization using two scale factors.

    Args:
        in_tensor: Quantized input tensor
        scale_1: First scale factor
        scale_2: Second scale factor

    Returns:
        Dequantized tensor
    """
    in_tensor_fp32 = pypto.cast(in_tensor, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_1_fp32 = pypto.cast(scale_1, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    scale_2_fp32 = pypto.cast(scale_2, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
    out_scale_2 = pypto.mul(in_tensor_fp32, scale_2_fp32)
    out = pypto.mul(out_scale_2, scale_1_fp32)
    return out


def symmetric_quantization_per_token_fp8_e4m3(input_tensor) -> Tuple:
    """
    Perform symmetric quantization per token (per row).

    Args:
        input_tensor: Input tensor to quantize

    Returns:
        Tuple of (quantized_f8_e4m3_tensor, dequantization_scale)
    """
    fp8_e4m3_max_value = 448.0
    x_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    x_abs = pypto.abs(x_fp32)
    x_max = pypto.amax(x_abs, -1, True)
    shape_0, shape_1 = x_max.shape[:2]
    x_scale = pypto.div(pypto.full([shape_0, shape_1], fp8_e4m3_max_value, pypto.DT_FP32), x_max)
    x_mul = pypto.mul(x_fp32, x_scale)
    x_fp8_e4m3 = pypto.cast(x_mul, pypto.DT_FP8E4M3)
    x_scale_quant = pypto.div(pypto.full([shape_0, shape_1], 1.0, pypto.DT_FP32), x_scale)
    return x_fp8_e4m3, x_scale_quant


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    # 当子图大小达到上界不允许与其他子图合并
    pass_options={
        # Q常驻，0代表第一组mmad，4代表4次matmul合并
        "cube_l1_reuse_setting": {0: 4},
        "vec_nbuffer_setting": {-1: 4},
        "cube_nbuffer_setting": {-1: 4}
    },
    verify_options={
        "enable_pass_verify": False,
        "pass_verify_save_tensor": False,
    },
    host_options={"compile_monitor_enable": True},
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
)
def ifa_func_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    q_scale: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    k_scale: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    v_scale: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):
    # 1. 添加支持动态的config
    pypto.experimental.set_operation_options(combine_axis=True)

    atten_cfg, tile_cfg = get_common_config()
    softmax_scale = atten_cfg.softmax_scale

    # 2. 从入参拿到输入和输出tensor
    shape_q = q.shape
    shape_k = k.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = kv_act_seqs.shape[0]

    dtype = pypto.DT_BF16
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile
    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    # 3. 得到动态tensor的shape
    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)
    q_scale_2d_shape = (b_scalar * 1 * nq, 1)
    k_scale_2d_shape = (block_num_scalar * block_size, n2_sym * 1)
    v_scale_2d_shape = (b_scalar * 1, n2_sym * dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    k_scale_2d = pypto.reshape(k_scale, k_scale_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    v_scale_2d = pypto.reshape(v_scale, v_scale_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    q_scale_2d = pypto.reshape(q_scale, q_scale_2d_shape, inplace=True)

    # 4. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        # 5. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                        qi_scale = pypto.view(q_scale_2d, [g_tile, 1], [bs_ofs * nq + n1g_ofs, 0])

                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        kj_sclae_assemble = pypto.tensor([s2_tile, 1], k_scale_2d.dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            kj_sclae_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(k_scale_2d, [block_size, 1], [block_idx_valid * block_size, 0])
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])
                        kj_sclae_assemble = pypto.view(kj_sclae_assemble, [s2_tile, 1], [0, 0], 
                                            valid_shape=[s2_tile, 1])

                        # c1
                        # 6. 下面是flash attention的计算逻辑
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij_quant = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                        # dequant
                        kj_sclae_assemble_t = pypto.transpose(kj_sclae_assemble, 0, 1)
                        sij_fp32 = dequant_dynamic(sij_quant, qi_scale, kj_sclae_assemble_t)
                        sij = pypto.view(sij_fp32, [g_tile, s2_tile], [0, 0],
                                            valid_shape=[g_tile, actual_s2_tile])
                        # v1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        if pypto.is_loop_begin(s2_idx):
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                            tsub = pypto.sub(sij_scale, tilda_mij)
                            tilda_pij = pypto.exp(tsub)
                            sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            max_update[:] = tilda_mij

                            #  quant
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            tilda_pij_fp8_e4m3, tilda_pij_scale = symmetric_quantization_per_token_fp8_e4m3(tilda_pij)
                            # c2
                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")

                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            vj_scale_assemble = pypto.tensor([1, dn], v_scale_2d.dtype, "vj_assemble")
                            vj_scale_assemble = pypto.view(v_scale_2d, [1, dn], [b_idx, 0])

                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                        [0, 0], valid_shape=[actual_s2_tile, dn])

                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp_quant = pypto.matmul(tilda_pij_fp8_e4m3, vj_assemble, pypto.DT_FP32)
                            # dequant
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            oi_tmp = dequant_dynamic(oi_tmp_quant, tilda_pij_scale, vj_scale_assemble)
                            oi_update[:] = oi_tmp
                        else:
                            pypto.set_pass_options(sg_set_scope=1)
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                            max_new = pypto.maximum(max_update, tilda_mij)
                            tsub = pypto.sub(sij_scale, max_new)
                            tilda_pij = pypto.exp(tsub)
                            sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_pass_options(sg_set_scope=2)
                            tsub2 = pypto.sub(max_update, max_new)
                            max_update[:] = max_new
                            update_mul = pypto.exp(tsub2)
                            sum_update[:] = sum_update * update_mul + sum_local
                            pypto.set_pass_options(sg_set_scope=-1)

                            # c2
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                        [0, 0], valid_shape=[actual_s2_tile, dn])
                            vj_scale_assemble = pypto.tensor([1, dn], v_scale_2d.dtype, "vj_assemble")
                            vj_scale_assemble = pypto.view(v_scale_2d, [1, dn], [b_idx, 0])
                            tilda_pij_fp8_e4m3, tilda_pij_scale = symmetric_quantization_per_token_fp8_e4m3(tilda_pij)

                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp_quant = pypto.matmul(tilda_pij_fp8_e4m3, vj_assemble, pypto.DT_FP32)
                            # dequant
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            oi_tmp = dequant_dynamic(oi_tmp_quant, tilda_pij_scale, vj_scale_assemble)
                            # v2
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_update * update_mul + oi_tmp
                        if pypto.is_loop_end(s2_idx):
                            oi_final = pypto.div(oi_update, sum_update)
                            pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                            oi_final_3d = pypto.cast(
                                pypto.reshape(oi_final, [1, g_tile, dn]),
                                dtype)
                            # 7. 将结果搬运到输出tensor上
                            pypto.assemble(oi_final_3d, oi_ofs, atten_out)