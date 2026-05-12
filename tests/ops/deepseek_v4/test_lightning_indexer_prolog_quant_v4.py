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
""" """
import torch
import torch_npu
import pypto
import os
import pytest
import logging

import sys, os; _p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')): _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src')); sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

from deepseek_v4.lightning_indexer_prolog_quant_v4_impl import *

from utils.golden.common_func import (
    apply_rotary_pos_emb,
    gen_uniform_data,
    quant_golden,
)
from utils.compare import compare


def compute_quant_lightning_indexer_prolog(inputs, params):
    qr = inputs[0]
    idx_wq_b = inputs[1]
    x = inputs[2]
    weights_proj = inputs[3]
    cos = inputs[4]
    sin = inputs[5]
    hadamard = inputs[6]
    qr_scale = inputs[7]
    idx_wq_b_scale = inputs[8]

    t = params.get("t")
    idx_nq = params.get("idx_nq")
    head_dim = params.get("head_dim")
    rope_dim = params.get("rope_dim")
    calc_dtype = torch.bfloat16

    # q_linear & q_rope
    q_fp32 = torch.matmul(qr.to(torch.int32), idx_wq_b.to(torch.int32)).to(
        torch.float32
    )  # (t, idx_nq * head_dim)
    q_fp32 = q_fp32 * qr_scale
    q_fp32 = q_fp32 * idx_wq_b_scale.reshape(1, idx_nq * head_dim)
    q_re = q_fp32.to(calc_dtype).reshape(t, idx_nq, head_dim)
    q_nope, q_rope = torch.split(q_re, [head_dim - rope_dim, rope_dim], dim=-1)
    q_roped = apply_rotary_pos_emb(q_rope, cos, sin)
    q = torch.cat([q_nope, q_roped], dim=-1)

    # hadamard
    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    # (t, idx_nq, head_dim) @ (1, head_dim, head_dim) -> (t, idx_nq, head_dim)
    q_hadamard = torch.matmul(
        q.to(torch.float32), hadamard.reshape(1, head_dim, head_dim).to(torch.float32)
    ).to(calc_dtype)
    # (t, idx_nq, head_dim), (t, idx_nq, 1)
    q_int8, q_scale = quant_golden(q_hadamard)
    q_scale = q_scale.to(torch.float16).reshape(t, idx_nq)

    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    weights = (
        torch.matmul(x.to(torch.float32), weights_proj.to(torch.float32))
        .to(calc_dtype)
        .to(torch.float32)
    )
    weights = weights * (idx_nq**-0.5) * (head_dim**-0.5)
    weights = weights.to(torch.float16)

    return q_int8, weights, q_scale


def gen_quant_attention_post_golden(params):
    torch.manual_seed(42)
    t = params.get("t")
    idx_nq = params.get("idx_nq")
    head_dim = params.get("head_dim")
    rope_dim = params.get("rope_dim")
    q_lora_rank = params.get("q_lora_rank")
    h = params.get("h")

    # construct inputs
    qr_ori = gen_uniform_data([t, q_lora_rank], -1, 1, torch.int8)
    idx_wq_b_ori = gen_uniform_data([q_lora_rank, idx_nq * head_dim], -1, 1, torch.int8)
    x = gen_uniform_data([t, h], -1, 1, torch.bfloat16)
    weights_proj = gen_uniform_data([h, idx_nq], -0.1, 0.1, torch.bfloat16)
    cos = gen_uniform_data([t, rope_dim], -1, 1, torch.bfloat16)
    sin = gen_uniform_data([t, rope_dim], -1, 1, torch.bfloat16)
    hadamard = gen_uniform_data([head_dim, head_dim], -1, 1, torch.bfloat16)
    qr, qr_scale = quant_golden(qr_ori)
    idx_wq_b, idx_wq_b_scale = quant_golden(idx_wq_b_ori.t())
    idx_wq_b = idx_wq_b.t()

    # generate golden outputs
    inputs = [
        qr,
        idx_wq_b,
        x,
        weights_proj,
        cos,
        sin,
        hadamard,
        qr_scale,
        idx_wq_b_scale,
    ]
    q_golden, weights_golden, q_scale_golden = compute_quant_lightning_indexer_prolog(
        inputs, params
    )
    return inputs, q_golden, weights_golden, q_scale_golden


class QuantLightningIndexerProlog(torch.nn.Module):
    def forward(
        self,
        qr,
        idx_wq_b,
        x,
        weights_proj,
        cos,
        sin,
        hadamard,
        qr_scale,
        idx_wq_b_scale,
    ):
        for _ in range(20):
            torch.add(qr_scale, 0)
        return torch.ops.pypto.quant_lightning_indexer_prolog(
            qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
        )


def do_indexer_prolog_quant_func(inputs, params, golden_list):
    """
    group       name           dtype     shape                               format
    INPUT 0	    qr	           DT_INT8	 (t, q_lora_rank)                    ND
    INPUT 1	    idx_wq_b	   DT_INT8	 (q_lora_rank, idx_nq * head_dim)	 ND
    INPUT 2	    x	           DT_BF16	 (t, h)	                             ND
    INPUT 3	    weights_proj   DT_BF16	 (h, idx_nq)	                     ND
    INPUT 4	    cos	           DT_BF16	 (t, rope_dim)                       ND
    INPUT 5	    sin	           DT_BF16	 (t, rope_dim)                       ND
    INPUT 6     hadamard       DT_BF16   (head_dim, head_dim)                ND
    INPUT 7     qr_scale       DT_FP32   (t, 1)                              ND
    INPUT 8     idx_wq_b_scale DT_FP32   (idx_nq * head_dim, 1)              ND
    OUTPUT 0	q              DT_INT8	 (t, idx_nq * head_dim)	             ND
    OUTPUT 1    weights        DT_FP16   (t, idx_nq)                         ND
    OUTPUT 2    q_scale        DT_FP16   (t, idx_nq)                         ND
    CONFIGS     tile_config    /          /                                  /
    """
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    qr = inputs[0].npu()
    idx_wq_b = inputs[1].npu()
    idx_wq_b_nz = torch_npu.npu_format_cast(idx_wq_b, torch_npu.Format.FRACTAL_NZ)
    x = inputs[2].npu()
    weights_proj = inputs[3].npu()
    weights_proj_nz = torch_npu.npu_format_cast(weights_proj, torch_npu.Format.FRACTAL_NZ)
    cos = inputs[4].npu()
    sin = inputs[5].npu()
    hadamard = inputs[6].npu()
    qr_scale = inputs[7].npu()
    idx_wq_b_scale = inputs[8].npu()

    t = params.get("t")
    idx_nq = params.get("idx_nq")
    head_dim = params.get("head_dim")

    # define npu outputs
    q = torch.zeros([t, idx_nq, head_dim]).to(torch.int8).npu()
    weights = torch.zeros([t, idx_nq]).to(torch.float16).npu()
    q_scale = torch.zeros([t, idx_nq]).to(torch.float16).npu()

    inputs = [qr, idx_wq_b_nz, x, weights_proj_nz, cos, sin, hadamard, qr_scale, idx_wq_b_scale, q, weights, q_scale]
    tile_config = IndexerPrologQuantConfig(unroll_list=[128, 64, 32, 16, 8, 1])

    # call main function
    quant_lightning_indexer_prolog_kernel(*inputs, tile_config)

    pypto.runtime._device_synchronize()
    compare(q.cpu(), golden_list[0], "q", atol=0.0001, rtol=0.005)
    compare(weights.cpu(), golden_list[1], "weights", atol=0.0001, rtol=0.007825)
    compare(q_scale.cpu(), golden_list[2], "q_scale", atol=0.0001, rtol=0.007825)


def do_indexer_prolog_quant_torch_graph(inputs, golden_list):
    """
    group       name           dtype     shape                               format
    INPUT 0	    qr	           DT_INT8	 (t, q_lora_rank)                    ND
    INPUT 1	    idx_wq_b	   DT_INT8	 (q_lora_rank, idx_nq * head_dim)	 ND
    INPUT 2	    x	           DT_BF16	 (t, h)	                             ND
    INPUT 3	    weights_proj   DT_BF16	 (h, idx_nq)	                     ND
    INPUT 4	    cos	           DT_BF16	 (t, rope_dim)                       ND
    INPUT 5	    sin	           DT_BF16	 (t, rope_dim)                       ND
    INPUT 6     hadamard       DT_BF16   (head_dim, head_dim)                ND
    INPUT 7     qr_scale       DT_FP32   (t, 1)                              ND
    INPUT 8     idx_wq_b_scale DT_FP32   (idx_nq * head_dim, 1)              ND
    OUTPUT 0	q              DT_INT8	 (t, idx_nq * head_dim)	             ND
    OUTPUT 1    weights        DT_FP16   (t, idx_nq)                         ND
    OUTPUT 2    q_scale        DT_FP16   (t, idx_nq)                         ND
    CONFIGS     tile_config    /          /                                  /
    """
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    # define npu inputs
    qr = inputs[0].npu()
    idx_wq_b = inputs[1].npu()
    idx_wq_b_nz = torch_npu.npu_format_cast(idx_wq_b, torch_npu.Format.FRACTAL_NZ)
    x = inputs[2].npu()
    weights_proj = inputs[3].npu()
    weights_proj_nz = torch_npu.npu_format_cast(
        weights_proj, torch_npu.Format.FRACTAL_NZ
    )
    cos = inputs[4].npu()
    sin = inputs[5].npu()
    hadamard = inputs[6].npu()
    qr_scale = inputs[7].npu()
    idx_wq_b_scale = inputs[8].npu()

    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    model = torch.compile(
        QuantLightningIndexerProlog(),
        dynamic=False,
        fullgraph=True,
        backend=npu_backend,
    )

    q, weights, q_scale = model(
        qr,
        idx_wq_b_nz,
        x,
        weights_proj_nz,
        cos,
        sin,
        hadamard,
        qr_scale,
        idx_wq_b_scale,
    )

    pypto.runtime._device_synchronize()
    compare(q.cpu(), golden_list[0], "q", atol=0.0001, rtol=0.005)
    compare(weights.cpu(), golden_list[1], "weights", atol=0.0001, rtol=0.007825)
    compare(q_scale.cpu(), golden_list[2], "q_scale", atol=0.0001, rtol=0.007825)


def get_indexer_prolog_quant_config(case_name: str):
    test_case_config = {
        "test_indexer_prolog_quant_b64_s1": 64,
        "test_indexer_prolog_quant_b64_s2": 128,
        "test_indexer_prolog_quant_b64_s4": 256,
        "test_indexer_prolog_quant_graph": 511,
    }
    return test_case_config.get(case_name)


def do_indexer_prolog_quant_entry(case_name: str, is_torch_graph: bool = False):
    params = {
        "idx_nq": 64,
        "head_dim": 128,
        "rope_dim": 64,
        "q_lora_rank": 1024,
        "h": 4096,
    }
    params["t"] = get_indexer_prolog_quant_config(case_name)
    if not params:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    inputs, q_golden, weights_golden, q_scale_golden = gen_quant_attention_post_golden(
        params
    )

    if is_torch_graph:
        print("\n =============== torch graph ====================")
        do_indexer_prolog_quant_torch_graph(
            inputs, [q_golden, weights_golden, q_scale_golden]
        )
    else:
        print("\n =============== st ====================")
        do_indexer_prolog_quant_func(
            inputs, params, [q_golden, weights_golden, q_scale_golden]
        )

    return True


@pytest.mark.skip(reason="large test case")
def test_indexer_prolog_quant_b64_s1():
    """
    lightning_indexer_prolog quant decode mtp=0 case
    """
    do_indexer_prolog_quant_entry(
        "test_indexer_prolog_quant_b64_s1", is_torch_graph=False
    )


@pytest.mark.skip(reason="large test case")
def test_indexer_prolog_quant_b64_s2():
    """
    lightning_indexer_prolog quant decode mtp=1 case
    """
    do_indexer_prolog_quant_entry(
        "test_indexer_prolog_quant_b64_s2", is_torch_graph=False
    )


@pytest.mark.skip(reason="large test case")
def test_indexer_prolog_quant_b64_s4():
    """
    lightning_indexer_prolog quant decode mtp=3 case
    """
    do_indexer_prolog_quant_entry(
        "test_indexer_prolog_quant_b64_s4", is_torch_graph=False
    )


@pytest.mark.skip(reason="large test case")
def test_indexer_prolog_quant_graph():
    """
    lightning_indexer_prolog quant graph typical case
    """
    do_indexer_prolog_quant_entry(
        "test_indexer_prolog_quant_graph", is_torch_graph=True
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    test_indexer_prolog_quant_b64_s1()
