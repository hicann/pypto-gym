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
import os
import math
import logging
import torch
import torch_npu
import pytest
import pypto
from torch._subclasses.fake_tensor import FakeTensor
try:
    from torch._dynamo import allow_in_graph
except Exception:
    def allow_in_graph(fn):
        return fn

from lightning_indexer_prolog_quant_hif8_impl import lightning_indexer_prolog_quant
from utils.compare_2_1 import precision_compare_triple


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("lightning_indexer_prolog_quant_hif8(Tensor x, Tensor q_norm, Tensor q_norm_scale, \
    Tensor w_qb, Tensor w_qb_scale, Tensor wk, Tensor w_proj, Tensor gamma_k, \
    Tensor cos_idx_rope, Tensor sin_idx_rope, Tensor hadamard_q, Tensor hadamard_k, Tensor k_cache, \
    Tensor k_scale_cache, Tensor k_cache_index, Tensor k_scale_cache_index) -> \
    (Tensor q_hif8, Tensor q_scale, Tensor k_hif8, Tensor k_scale, Tensor weights)")


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, *args):
        x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj, gamma_k, cos_idx_rope, \
        sin_idx_rope, hadamard_q, hadamard_k, k_cache, k_scale_cache, k_cache_index, k_scale_cache_index = args
        q_hif8, q_scale, k_hif8, k_scale, weights = torch.ops.pypto.lightning_indexer_prolog_quant_hif8(
            x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj, gamma_k, cos_idx_rope,
            sin_idx_rope, hadamard_q, hadamard_k, k_cache, k_scale_cache, k_cache_index, k_scale_cache_index)
        return q_hif8, q_scale, k_hif8, k_scale, weights


def gen_dims(params):
    dims = {}
    dims["s2"] = params["s2"]
    dims["b"] = params["b"]
    dims["t"] = params["b"] * params["s1"]
    dims["h"] = 2560
    dims["q_lora_rank"] = 1024
    dims["idx_head_dim"] = 128
    dims["idx_n_heads"] = 24
    dims["rope_head_dim"] = 64
    dims["block_size"] = 128
    dims["block_num"] = dims["b"] * dims["s2"] // dims["block_size"]
    dims["n_kv"] = 1
    return dims


def gen_block_table(act_seq, block_size, s1):
    b = act_seq.shape[0]
    block_num = 0
    block_num_each = []
    max_kv = max(act_seq)
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(torch.int32)

    block_idx = 0
    block_table_bidx = 0
    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    cache_index = -torch.ones((b, s1), dtype=torch.int64)
    for i in range(b):
        cur_act = act_seq[i]
        for j in range(s1):
            pos = cur_act - s1 + j
            block_idx_in_seq = pos // block_size
            global_block_id = block_table[i, block_idx_in_seq]

            offset_in_block = pos % block_size
            global_index = global_block_id * block_size + offset_in_block
            cache_index[i, j] = global_index

    return block_num, block_table, cache_index


def gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size, k_cache):
    dtype = k_cache_bsnd.dtype
    b, s2, n_kv, d = k_cache_bsnd.shape
    k_cache = k_cache.view(block_num, block_size, n_kv, d)
    s2_new = ((s2 + block_size - 1) // block_size) * block_size  # ceil to block_size
    k_cache_raw = torch.zeros((b, s2_new, n_kv, d), dtype=dtype)
    k_cache_raw[:, :s2, :, :] = k_cache_bsnd

    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx == -1:
                continue
            else:
                k_cache[cache_block_idx, :, :, :] = k_cache_raw[
                    b_idx, block_offset: (block_offset + block_size), :, :
                ]

    return k_cache


def gen_inputs(dims, dtype=torch.bfloat16):
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    block_size = dims["block_size"]
    n_kv = dims["n_kv"]
    s2 = dims["s2"]
    rope_head_dim = dims["rope_head_dim"]

    x = torch.empty((b, s, h), dtype=dtype).uniform_(-1, 1).npu()
    q_norm_fp32 = torch.empty((b, s, q_lora_rank), dtype=torch.float32).uniform_(-10, 10).npu()
    q_norm = torch_npu.npu_dtype_cast(q_norm_fp32, torch_npu.hifloat8)
    q_norm_scale = torch.empty((b, s, 1), dtype=torch.float32).uniform_(-1, 1).npu()

    w_idx_qb_fp32 = torch.empty((q_lora_rank, n * d), dtype=torch.float32).uniform_(-10, 10).npu()
    w_idx_qb = torch_npu.npu_dtype_cast(w_idx_qb_fp32, torch_npu.hifloat8)
    w_idx_qb_scale = torch.empty((1, n * d), dtype=torch.float32).uniform_(-1, 1).npu()

    w_idx_k = torch.empty((h, d), dtype=dtype).uniform_(-1, 1).npu()
    w_idx_proj = torch.empty((h, n), dtype=dtype).uniform_(-1, 1).npu()
    gamma = torch.ones((d,), dtype=dtype).npu()

    random_angles = (torch.rand(b, s, rope_head_dim, dtype=torch.float32) * 2 * torch.pi)
    cos = torch.cos(random_angles).to(dtype).npu()
    sin = torch.sin(random_angles).to(dtype).npu()

    hadamard_q = torch.empty((d, d), dtype=dtype).uniform_(-1, 1).npu()  # (128, 128)
    hadamard_k = torch.empty((d, d), dtype=dtype).uniform_(-1, 1).npu()

    act_seq = torch.tensor([s2] * b)  # (b,)
    k_cache_bsnd = torch.randint(0, 256, (b, s2, n_kv, d), dtype=torch.uint8)
    k_scale_cache_bsnd = torch.empty((b, s2, n_kv, 1), dtype=torch.float32).uniform_(-1, 1)
    block_num, block_table, k_cache_index = gen_block_table(act_seq, block_size, s)

    pg_cache = torch.randint(0, 256, (block_num, block_size * n_kv * d * 41), dtype=torch.uint8).npu()
    k_cache = pg_cache[:, block_size * n_kv * d: block_size * n_kv * 2 * d]
    k_fp32_offset = block_size * n_kv * 2 * d // 4
    k_scale_cache = pg_cache.view(torch.float32)[:, k_fp32_offset: k_fp32_offset + block_size * n_kv * 1]

    k_cache = gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size, k_cache)
    k_scale_cache = gen_cache_tensor(k_scale_cache_bsnd, block_table, block_num, block_size, k_scale_cache)

    return {
        "token_x": x,  # bf16
        "q_norm": q_norm,  # hif8
        "q_norm_scale": q_norm_scale,  # fp32
        "w_idx_qb": w_idx_qb,  # hif8
        "w_idx_qb_scale": w_idx_qb_scale,  # fp32
        "w_idx_k": w_idx_k,  # bf16
        "w_idx_proj": w_idx_proj,  # bf16
        "rms_norm_gamma": gamma,  # bf16
        "cos_idx_rope": cos,  # bf16
        "sin_idx_rope": sin,  # bf16
        "hadamard_q": hadamard_q,  # bf16
        "hadamard_k": hadamard_k,  # bf16
        "idx_k_cache": k_cache,  # hif8 (block_num, block_size, n_kv, d)
        "idx_k_scale_cache": k_scale_cache,  # fp32 (block_num, block_size, n_kv, 1)
        "idx_k_cache_index": k_cache_index.npu(),  # int64 (b, s)/（t,)
        "idx_block_table": block_table,  # int32 (b, ceil(s2, block_size))
        "act_seq": act_seq,  # int32
    }


def scatter_update_pa_bsnd(cache, k_bsnd, cache_index, axis):
    block_number, block_size, n_kv, d = cache.shape
    res = cache.reshape(block_number * block_size * n_kv, d)
    b, s1 = cache_index.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = cache_index[b_i][s1_i]
                res[index_value, :] = k_bsnd[b_i, s1_i, :, :]

    return res.reshape(block_number, block_size, n_kv, d)


def quant_hif8(x: torch.Tensor):
    # pertoken
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 32768.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_hif8 = torch_npu.npu_dtype_cast(y_fp32, torch_npu.hifloat8)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) hif8, (b, s, n, 1) fp32
    return y_hif8, scale_dequant


def indexer_prolog(inputs_initial: dict, dims: dict, precision: str = "same"):
    # input
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b

    if precision == "high":
        inputs = {k: v.to(torch.float32) if v.dtype in [torch.bfloat16, torch.float16] else v.clone()
                    for k, v in inputs_initial.items()}
    elif precision == "same":
        inputs = inputs_initial

    rope_head_dim = dims["rope_head_dim"]
    q_lora_rank = dims["q_lora_rank"]
    x = inputs["token_x"]  # (b, s, h)
    q_norm = inputs["q_norm"]  # (b, s, q_lora_rank), hif8
    q_norm_scale = inputs["q_norm_scale"]  # (b, s, 1), fp32
    w_idx_qb = inputs["w_idx_qb"]  # (q_lora_rank, n * d), hif8
    w_idx_qb_scale = inputs["w_idx_qb_scale"]  # (n * d, 1), fp32
    w_idx_k = inputs["w_idx_k"]  # (h, d)
    w_idx_proj = inputs["w_idx_proj"]  # (h, n)
    rms_norm_gamma = inputs["rms_norm_gamma"]  # (d,)
    cos = inputs["cos_idx_rope"]  # (b, s, rope_head_dim)
    sin = inputs["sin_idx_rope"]  # (b, s, rope_head_dim)
    hadamard_q = inputs["hadamard_q"]  # (d, d)
    hadamard_k = inputs["hadamard_k"]  # (d, d)
    idx_k_cache = inputs["idx_k_cache"]  # hif8
    idx_k_scale_cache = inputs["idx_k_scale_cache"]  # fp16
    cache_index = inputs["idx_k_cache_index"]  # (b, s), int32
    x_dtype = x.dtype

    cos = cos.view(-1, 1, 1, rope_head_dim)
    sin = sin.view(-1, 1, 1, rope_head_dim)

    # q quant matmul
    q_proj = torch_npu.npu_quant_matmul(q_norm.view(t, q_lora_rank), w_idx_qb.view(q_lora_rank, n * d),
        w_idx_qb_scale.view(n * d), pertoken_scale=q_norm_scale.view(t), x1_dtype=torch_npu.hifloat8,
        x2_dtype=torch_npu.hifloat8, output_dtype=x_dtype).view(b, s, n, d)

    # q rope
    q_rope, q_nope = torch.split(q_proj, [rope_head_dim, d - rope_head_dim], dim=-1)
    q_rope = q_rope.view(-1, n, 1, rope_head_dim)
    q_rope = torch_npu.npu_rotary_mul(q_rope, cos, sin).view(b, s, n, rope_head_dim)
    q_cat = torch.cat([q_rope, q_nope], dim=-1)
    # q hadamard
    q_hadamard = torch.matmul(q_cat, hadamard_q)
    # q quant
    if precision == "high":
        q_hif8, q_scale = quant_hif8(q_hadamard)
    elif precision == "same":
        q_hif8, q_scale = torch_npu.npu_dynamic_quant(q_hadamard, dst_type=torch_npu.hifloat8)

    # k linear
    k_proj = torch.matmul(x, w_idx_k)
    # k rms norm
    k_rms_norm = torch_npu.npu_rms_norm(k_proj, rms_norm_gamma, epsilon=1e-6)[0]
    # k rope
    k_rope, k_nope = torch.split(k_rms_norm, [rope_head_dim, d - rope_head_dim], dim=-1)
    k_rope = k_rope.view(-1, 1, 1, rope_head_dim)
    k_rope = torch_npu.npu_rotary_mul(k_rope, cos, sin).view(b, s, rope_head_dim)
    k_cat = torch.cat([k_rope, k_nope], dim=-1)
    # k hadamard
    k_hadamard = torch.matmul(k_cat, hadamard_k)
    # k quant
    if precision == "high":
        k_hif8, k_scale = quant_hif8(k_hadamard)
    elif precision == "same":
        k_hif8, k_scale = torch_npu.npu_dynamic_quant(k_hadamard, dst_type=torch_npu.hifloat8)
    # k cache update
    k_cache = idx_k_cache.clone()
    k_scale_cache = idx_k_scale_cache.clone()
    scatter_update_pa_bsnd(k_cache, k_hif8.reshape(b, s, 1, d), cache_index, -2)
    scatter_update_pa_bsnd(k_scale_cache, k_scale.reshape(b, s, 1, 1), cache_index, -2)

    # w linear
    weights = torch.matmul(x, w_idx_proj)
    weights = weights * (n ** -0.5) * (d ** -0.5)

    # output dtype: hif8, fp32, hif8, fp32, bf16
    outputs = {"query": q_hif8, "query_scale": q_scale, "idx_k_cache_out": k_cache,
               "idx_k_scale_cache_out": k_scale_cache, "weights": weights}
    return outputs


def gen_data(case_name):
    if case_name.startswith("QuantLightningIndexerPrologSTest.b1_s1_8k_s2_8k_acl"):
        params = {"b": 1, "s1": 1024 * 8, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b1_s1_8k_s2_8k"):
        params = {"b": 1, "s1": 1024 * 8, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b2_s1_8k_s2_8k"):
        params = {"b": 2, "s1": 1024 * 8, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b4_s1_8k_s2_8k"):
        params = {"b": 4, "s1": 1024 * 8, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b1_s1_4_s2_8k"):
        params = {"b": 1, "s1": 4, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b64_s1_2_s2_8k"):
        params = {"b": 64, "s1": 2, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b192_s1_1_s2_8k"):
        params = {"b": 192, "s1": 1, "s2": 1024 * 8}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b4_s1_32k_s2_32k"):
        params = {"b": 4, "s1": 1024 * 32, "s2": 1024 * 32}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b4_s1_64k_s2_64k"):
        params = {"b": 4, "s1": 1024 * 64, "s2": 1024 * 64}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b4_s1_128k_s2_128k"):
        params = {"b": 4, "s1": 1024 * 128, "s2": 1024 * 128}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b1_s1_8k_333_s2_8k_333"):
        params = {"b": 1, "s1": 1024 * 8 + 333, "s2": 1024 * 8 + 333}
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b111_s1_1_s2_8k"):
        params = {"b": 111, "s1": 1, "s2": 1024 * 8}
    else:
        raise Exception(f"Can't get func to gen golden, Case({case_name})")

    seed = 0
    torch.manual_seed(seed)
    dims = gen_dims(params)
    inputs = gen_inputs(dims, torch.bfloat16)
    outputs_high_precision = indexer_prolog(inputs, dims, "high")
    outputs_same_precision = indexer_prolog(inputs, dims, "same")

    return dims, inputs, outputs_high_precision, outputs_same_precision


def ascend_operator_accuracy_standard_version_2_1(pypto_out, npu_out, golden_out, out_name):
    result, mare, mere, rmse, small_err = precision_compare_triple(pypto_out, npu_out, golden_out)
    assert result != "FAILED", f"{out_name} FAILED: result={result}, \
        mare={mare:.6f}, mere={mere:.6f}, rmse={rmse:.6f}, small_errors={small_err}"
    logging.info(f"{out_name} PASSED - \
        mare={mare:.6f}, mere={mere:.6f}, rmse={rmse:.6f}, small_errors={small_err}")


@torch.library.impl(pyptolib, "lightning_indexer_prolog_quant_hif8", "Meta")
def lightning_indexer_prolog_quant_hif8_meta(x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj,
                                           gamma_k, cos_idx_rope, sin_idx_rope, hadamard_q, hadamard_k,
                                           k_cache, k_scale_cache, k_cache_index, k_scale_cache_index):
    t = x.shape[0]
    head_num = w_proj.shape[1]
    block_num, block_size, n_kv, head_dim = k_cache.shape
    q_hif8 = torch.empty((t, head_num, head_dim), device="meta", dtype=torch.uint8)
    q_scale = torch.empty((t, head_num, 1), device="meta", dtype=torch.float32)
    k_hif8 = k_cache
    k_scale = k_scale_cache
    weights = torch.empty((t, head_num), device="meta", dtype=torch.bfloat16)

    return q_hif8, q_scale, k_hif8, k_scale, weights


def lightning_indexer_prolog_quant_hif8_pypto(x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj,
                                           gamma_k, cos_idx_rope, sin_idx_rope, hadamard_q, hadamard_k,
                                           k_cache, k_scale_cache, k_cache_index, k_scale_cache_index):
    t = x.shape[0]
    head_num = w_proj.shape[1]
    block_num, block_size, n_kv, head_dim = k_cache.shape

    k_storage_offset = k_cache.storage_offset()
    k_scale_storage_offset = k_scale_cache.storage_offset()

    if not k_cache.is_contiguous():
        page_size = k_cache.stride()[0] // (n_kv * block_size)
        pg_cache = torch.as_strided(
            k_cache,
            size=(block_num, block_size, n_kv, page_size),
            stride=(block_size * n_kv * page_size, n_kv * page_size, page_size, 1),
            storage_offset=0
        )
        k_cache = pg_cache
        k_scale_cache = pg_cache.view(torch.float32)
        pg_cache_index = k_cache_index // block_size * (k_cache.shape[-1] // head_dim) \
            * block_size + k_cache_index % block_size + k_storage_offset // head_dim
        pg_scale_cache_index = k_scale_cache_index // block_size * k_scale_cache.shape[-1] \
            * block_size + k_scale_cache_index % block_size + k_scale_storage_offset

    k_cache = k_cache.view(block_num, block_size * (k_cache.shape[-1] // head_dim), n_kv, head_dim)
    k_scale_cache = k_scale_cache.view(block_num, block_size * k_scale_cache.shape[-1], n_kv, 1)
    k_cache_index = pg_cache_index.reshape(t, 1)
    k_scale_cache_index = pg_scale_cache_index.reshape(t, 1)

    device = x.device
    q_hif8 = torch.empty((t * head_num, head_dim), device=device, dtype=torch.uint8)
    q_scale = torch.empty((t * head_num, 1), device=device, dtype=torch.float32)
    k_hif8 = k_cache
    k_scale = k_scale_cache
    weights = torch.empty((t, head_num), device=device, dtype=torch.bfloat16)

    if isinstance(x, FakeTensor):
        return q_hif8, q_scale, k_hif8, k_scale, weights

    lightning_indexer_prolog_quant(x, q_norm, q_norm_scale, w_qb, w_qb_scale, wk, w_proj, gamma_k, cos_idx_rope,
        sin_idx_rope, hadamard_q, hadamard_k, k_cache, k_scale_cache, k_cache_index, k_scale_cache_index,
        q_hif8, q_scale, k_hif8, k_scale, weights)

    k_hif8 = k_hif8.view(block_num, -1)[:, k_storage_offset:
        k_storage_offset + block_size * n_kv * head_dim].view(block_num, block_size, n_kv, head_dim)
    k_scale = k_scale.view(block_num, -1)[:, k_scale_storage_offset:
        k_scale_storage_offset + block_size * n_kv * 1].view(block_num, block_size, n_kv, 1)

    q_hif8 = q_hif8.view(t, head_num, head_dim)
    q_scale = q_scale.view(t, head_num, 1)

    return q_hif8, q_scale, k_hif8, k_scale, weights


try:
    lightning_indexer_prolog_quant_hif8_pypto = allow_in_graph(lightning_indexer_prolog_quant_hif8_pypto)
    torch.library.impl(pyptolib, "lightning_indexer_prolog_quant_hif8", "NPU")(
        lightning_indexer_prolog_quant_hif8_pypto
    )
except Exception as e:
    if "could not parse dispatch key: NPU" in str(e):
        logging.warning(
            "Skip: torchair not installed, skip NPU registration for operator 'lightning_indexer_prolog_quant_hif8'")
    else:
        logging.warning(f"Skip: Unexpected error : {e}")


def do_test_lightning_indexer_prolog_quant(case_name, is_acl=False):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch_npu.npu.set_device(device_id)

    logging.info(f"=== run test case: {case_name} ===")

    dims, inputs_data, golden_data, npu_data = gen_data(case_name)

    t = dims["t"]
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    idx_head_dim = dims["idx_head_dim"]
    head_num = dims["idx_n_heads"]
    rope_head_dim = dims["rope_head_dim"]

    torch_npu.npu.config.allow_internal_format = True

    x = inputs_data["token_x"].reshape(t, h)
    q_norm = inputs_data["q_norm"].reshape(t, q_lora_rank)
    q_norm_scale = inputs_data["q_norm_scale"].reshape(t, 1)
    w_qb = inputs_data["w_idx_qb"]
    w_qb_scale = inputs_data["w_idx_qb_scale"].reshape(1, head_num * idx_head_dim)
    wk = inputs_data["w_idx_k"]
    w_proj = inputs_data["w_idx_proj"]
    gamma_k = inputs_data["rms_norm_gamma"].reshape(1, idx_head_dim)
    cos_idx_rope = inputs_data["cos_idx_rope"].reshape(t, rope_head_dim)
    sin_idx_rope = inputs_data["sin_idx_rope"].reshape(t, rope_head_dim)
    hadamard_q = inputs_data["hadamard_q"]
    hadamard_k = inputs_data["hadamard_k"]
    k_cache = inputs_data["idx_k_cache"]
    k_scale_cache = inputs_data["idx_k_scale_cache"]
    k_cache_index = inputs_data["idx_k_cache_index"]
    k_scale_cache_index = inputs_data["idx_k_cache_index"]

    q_hif8_golden = golden_data["query"].reshape(t, head_num, idx_head_dim)
    q_scale_golden = golden_data["query_scale"].reshape(t, head_num, 1)
    k_cache_golden = golden_data["idx_k_cache_out"]
    k_scale_cache_golden = golden_data["idx_k_scale_cache_out"]
    weights_golden = golden_data["weights"].reshape(t, head_num)

    q_hif8_npu = npu_data["query"].reshape(t, head_num, idx_head_dim)
    q_scale_npu = npu_data["query_scale"].reshape(t, head_num, 1)
    k_cache_npu = npu_data["idx_k_cache_out"]
    k_scale_cache_npu = npu_data["idx_k_scale_cache_out"]
    weights_npu = npu_data["weights"].reshape(t, head_num)

    logging.info("==================finish torch==================")

    if is_acl:
        model = Model()
        compile_forward = torch.compile(model, fullgraph=True, backend="npugraph_ex", dynamic=False)
        q_hif8, q_scale, k_hif8, k_scale, weights = compile_forward(x, q_norm, q_norm_scale,
            w_qb, w_qb_scale, wk, w_proj, gamma_k, cos_idx_rope, sin_idx_rope, hadamard_q, hadamard_k,
            k_cache, k_scale_cache, k_cache_index, k_scale_cache_index)
    else:
        q_hif8, q_scale, k_hif8, k_scale, weights = lightning_indexer_prolog_quant_hif8_pypto(x, q_norm, q_norm_scale,
            w_qb, w_qb_scale, wk, w_proj, gamma_k, cos_idx_rope, sin_idx_rope, hadamard_q, hadamard_k,
            k_cache, k_scale_cache, k_cache_index, k_scale_cache_index)

    logging.info("==================finish pypto==================")
    ascend_operator_accuracy_standard_version_2_1(q_hif8, q_hif8_npu, q_hif8_golden, "q_hif8")
    ascend_operator_accuracy_standard_version_2_1(q_scale, q_scale_npu, q_scale_golden, "q_scale")
    ascend_operator_accuracy_standard_version_2_1(k_hif8, k_cache_npu, k_cache_golden, "k_hif8")
    ascend_operator_accuracy_standard_version_2_1(k_scale, k_scale_cache_npu, k_scale_cache_golden, "k_scale")
    ascend_operator_accuracy_standard_version_2_1(weights, weights_npu, weights_golden, "weights")

    logging.info(f"=== {case_name}: PASS ===")


@pytest.mark.soc("950")
def test_b1_s1_8k_s2_8k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b1_s1_8k_s2_8k", is_acl=False)


@pytest.mark.skip(reason="large test case")
def test_b2_s1_8k_s2_8k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b2_s1_8k_s2_8k", is_acl=False)


@pytest.mark.skip(reason="large test case")
def test_b4_s1_8k_s2_8k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b4_s1_8k_s2_8k", is_acl=False)


@pytest.mark.soc("950")
def test_b1_s1_4_s2_8k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b1_s1_4_s2_8k", is_acl=False)


@pytest.mark.skip(reason="small test case")
def test_b64_s1_2_s2_8k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b64_s1_2_s2_8k", is_acl=False)


@pytest.mark.skip(reason="small test case")
def test_b192_s1_1_s2_8k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b192_s1_1_s2_8k", is_acl=False)


@pytest.mark.skip(reason="large test case")
def test_b4_s1_32k_s2_32k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b4_s1_32k_s2_32k", is_acl=False)


@pytest.mark.skip(reason="large test case")
def test_b4_s1_64k_s2_64k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b4_s1_64k_s2_64k", is_acl=False)


@pytest.mark.skip(reason="large test case")
def test_b4_s1_128k_s2_128k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b4_s1_128k_s2_128k", is_acl=False)


@pytest.mark.soc("950")
def test_b1_s1_8k_333_s2_8k_333():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b1_s1_8k_333_s2_8k_333", is_acl=False)


@pytest.mark.soc("950")
def test_b111_s1_1_s2_8k():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b111_s1_1_s2_8k", is_acl=False)


@pytest.mark.soc("950")
def test_acl():
    do_test_lightning_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b1_s1_8k_s2_8k_acl", is_acl=True)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_b1_s1_8k_s2_8k()