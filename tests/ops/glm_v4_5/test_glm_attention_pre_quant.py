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
import os
import logging
import torch
import torch_npu
import pytest
import numpy as np
import pypto
from numpy.testing import assert_allclose
from pypto_gym.ops.pypto_tile.glm_v4_5.glm_attention_pre_quant_impl import attention_pre_quant
from pypto_gym.ops.pypto_tile.glm_v4_5.utils.get_format import get_format


logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


def add_rms_norm_npu_golden(residual_input, x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    residual_input_fp32 = residual_input.to(torch.float32)
    x_fp32 = x_fp32 + residual_input_fp32
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16), x_fp32.to(torch.bfloat16)


def rms_norm_npu_golden(x, x_gamma, x_bias, eps):
    x_bias_fp32 = x_bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)
    x_mean_coff = 1.0 / x.shape[-1]
    x_square = x_fp32 * x_fp32
    x_mean = x_square * x_mean_coff
    x_reduce_sum = torch.sum(x_mean, dim=-1, keepdim=True) + eps
    x_reduce_sqrt = torch.sqrt(x_reduce_sum)
    x_res_div = x_fp32 / x_reduce_sqrt
    x_mul_res = x_res_div * x_gamma.to(torch.float32)
    x_add_bias = x_mul_res + x_bias_fp32

    return x_add_bias.to(torch.bfloat16)


def _apply_rotary_emb_neuron(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat((o1, o2), dim=-1)


def apply_rotary_pos_emb_v2(q, k, cos, sin):
    x_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    q_embed = _apply_rotary_emb_neuron(q, cos, sin)
    k_embed = _apply_rotary_emb_neuron(k, cos, sin)

    if x_dtype != torch.float32:
        q_embed = q_embed.to(x_dtype)
        k_embed = k_embed.to(x_dtype)
    return q_embed, k_embed


@pytest.mark.soc("950", "910")
def test_quant_attention_pre():
    bs = 8
    hidden_size = 5120
    total_head_size = 1792
    head_size = 128
    q_size = 1536
    kv_size = 128
    rotary_dim = 64
    half_rotary_dim = rotary_dim // 2
    eps = 1e-05

    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    for i in range(0, 1):
        if (i == 1):
            bs = 5
        elif (i == 2):
            bs = 11
        elif (i == 3):
            bs = 2

        np.random.seed(0)
        x = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        residual_input = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        x_gamma = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        x_bias = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        x_scale = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        x_offset = torch.rand(hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        weight = torch.randint(-128, 128, size=(hidden_size, total_head_size), dtype=torch.int8,
            device=f'npu:{device_id}')
        weight = torch_npu.npu_format_cast(weight, 29)
        quant_bias = torch.randint(-128, 128, size=(total_head_size,), dtype=torch.int32, device=f'npu:{device_id}')
        deq_scale = torch.rand(total_head_size, dtype=torch.float32, device=f'npu:{device_id}')
        q_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        q_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        k_gamma = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        k_bias = torch.rand(head_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        cos = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
        sin = torch.rand(bs, 1, half_rotary_dim, dtype=torch.bfloat16, device=f'npu:{device_id}')
        query = torch.rand(bs, q_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        key = torch.rand(bs, kv_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        value = torch.rand(bs, kv_size, dtype=torch.bfloat16, device=f'npu:{device_id}')
        residual_res = torch.rand(bs, hidden_size, dtype=torch.bfloat16, device=f'npu:{device_id}')

        inputs = [
            x,
            residual_input,
            x_gamma,
            x_bias,
            x_scale,
            x_offset,
            weight,
            quant_bias,
            deq_scale,
            q_gamma,
            q_bias,
            k_gamma,
            k_bias,
            cos,
            sin,
            query,
            key,
            value,
            residual_res
        ]

        attention_pre_quant(*inputs)

        x_g, residual_g = add_rms_norm_npu_golden(x, residual_input, x_gamma, x_bias, eps)

        x_quant = torch_npu.npu_quantize(x_g, x_scale, x_offset, torch.qint8, -1, False)
        mm_golden = torch_npu.npu_quant_matmul(x_quant, weight, deq_scale,\
                                               bias=quant_bias, output_dtype=torch.bfloat16)

        q_g, k_g, v_g = mm_golden.split([q_size, kv_size, kv_size], dim=-1)

        q_by_head = q_g.view(*q_g.shape[:-1], q_g.shape[-1] // head_size, head_size)
        q_by_head = rms_norm_npu_golden(q_by_head, q_gamma, q_bias, eps)

        k_by_head = k_g.view(*k_g.shape[:-1], k_g.shape[-1] // head_size, head_size)
        k_by_head = rms_norm_npu_golden(k_by_head, k_gamma, k_bias, eps)

        q_rot = q_by_head[..., :rotary_dim]
        q_pass = q_by_head[..., rotary_dim:]
        k_rot = k_by_head[..., :rotary_dim]
        k_pass = k_by_head[..., rotary_dim:]
        q_r, k_r = apply_rotary_pos_emb_v2(q_rot, k_rot, cos, sin)
        q_cat = torch.cat((q_r, q_pass), dim=-1)
        k_cat = torch.cat((k_r, k_pass), dim=-1)
        q_r = q_cat.view(bs, q_size)
        k_r = k_cat.view(bs, kv_size)
        assert_allclose(np.array(residual_g.cpu().flatten().tolist()), np.array(residual_res.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)
        assert_allclose(np.array(q_r.cpu().flatten().tolist()), np.array(query.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)
        assert_allclose(np.array(k_r.cpu().flatten().tolist()), np.array(key.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)
        assert_allclose(np.array(v_g.cpu().flatten().tolist()), np.array(value.cpu().flatten().tolist()),
                        rtol=0.0078125, atol=0.0001)
        logging.info("PASS")


def main():
    test_quant_attention_pre()


if __name__ == "__main__":
    main()
