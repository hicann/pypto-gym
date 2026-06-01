# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-expert BF16 SwiGLU FFN, JIT'd by PyPTO.

This is the same Gemma 4 MLP fusion kernel (single matmul against [H, 2I]
weight + fused SwiGLU + matmul against [I, H]), specialised for LLaDA2's
silu activation. It's invoked once per active expert during routed dispatch.

Caller responsibility:
  * Bucket-sort tokens by topk_ids.argsort()
  * Slice sorted_x by per-expert offsets and pass each slice in
  * Scatter outputs back into the K-replicated buffer
"""

import os
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph

import pypto


ND = pypto.TileOpFormat.TILEOP_ND


def _swiglu_silu(gate_up):
    """SiLU(gate) * up.  SiLU(x) = x / (1 + exp(-x))."""
    half = gate_up.shape[1] // 2
    gate = pypto.view(gate_up, [gate_up.shape[0], half], [0, 0])
    up = pypto.view(gate_up, [gate_up.shape[0], half], [0, half])
    neg = pypto.mul(gate, -1.0)
    e = pypto.exp(neg)
    denom = pypto.add(e, 1.0)
    silu = pypto.div(gate, denom)
    return pypto.mul(silu, up)


_CUBE_NBUF = int(os.environ.get("PYPTO_CUBE_NBUFFER", "2"))
_VEC_NBUF = int(os.environ.get("PYPTO_VEC_NBUFFER", "2"))
_MM1_M = int(os.environ.get("PYPTO_MM1_M", "8"))
_MM1_K = int(os.environ.get("PYPTO_MM1_K", "128"))
_MM1_N = int(os.environ.get("PYPTO_MM1_N", "256"))
_MM2_M = int(os.environ.get("PYPTO_MM2_M", "8"))
_MM2_K = int(os.environ.get("PYPTO_MM2_K", "128"))
_MM2_N = int(os.environ.get("PYPTO_MM2_N", "256"))
_VEC_FIRST = int(os.environ.get("PYPTO_VEC_FIRST", "13"))


@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1,
                     "stitch_function_max_num": 64},
    pass_options={
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: _CUBE_NBUF},
        "vec_nbuffer_setting": {-1: _VEC_NBUF},
    },
)
def llada2_expert_ffn_kernel(
    hidden_states: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    w13: pypto.Tensor([], pypto.DT_BF16, format=ND),       # [H, 2I]
    w2:  pypto.Tensor([], pypto.DT_BF16, format=ND),       # [I, H]
    ffn_res: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
):
    bs = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    two_i = w13.shape[1]
    intermediate_size = two_i // 2

    pypto.experimental.set_operation_options(combine_axis=True)

    mm1_cube = (_MM1_M, _MM1_K, _MM1_N)
    mm2_cube = (_MM2_M, _MM2_K, _MM2_N)
    vec_first = _VEC_FIRST

    for bs_idx, tile_batch in pypto.loop_unroll(
        0, bs, 1,
        name="LOOP_LLADA2_EXPERT_L0",
        idx_name="bs_idx",
        unroll_list=[1, 2, 4, 8, 16, 32, 64],
    ):
        tile_x = hidden_states[bs_idx:bs_idx + tile_batch, :]

        pypto.set_cube_tile_shapes(
            [tile_batch, tile_batch],
            [mm1_cube[1], mm1_cube[1] * 2],
            [mm1_cube[2], mm1_cube[2]],
            True,
        )
        gate_up = pypto.matmul(tile_x, w13, pypto.DT_FP32)

        # Activation in FP32: skip 2I-wide cast, process in FP32,
        # then cast only the I-wide result to BF16 for mm2.
        pypto.set_vec_tile_shapes(min(tile_batch, vec_first),
                                   intermediate_size)
        sw_fp32 = _swiglu_silu(gate_up)
        sw = pypto.cast(sw_fp32, pypto.DT_BF16)

        pypto.set_cube_tile_shapes(
            [tile_batch, tile_batch],
            [mm2_cube[1], mm2_cube[1] * 2],
            [mm2_cube[2], mm2_cube[2]],
            False,
        )
        down = pypto.matmul(sw, w2, pypto.DT_FP32)
        pypto.set_vec_tile_shapes(min(tile_batch, vec_first), hidden_size)
        out = pypto.cast(down, hidden_states.dtype)
        pypto.assemble(out, [bs_idx, 0], ffn_res)


def _check(hidden_states, w13, w2, ffn_res):
    assert hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16
    assert w13.dim() == 2 and w13.dtype == torch.bfloat16
    assert w2.dim() == 2 and w2.dtype == torch.bfloat16
    H = hidden_states.shape[1]
    two_I = w13.shape[1]
    assert w13.shape[0] == H
    assert w2.shape == (two_I // 2, H)
    assert ffn_res.shape == hidden_states.shape


@allow_in_graph
def llada2_expert_ffn(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    ffn_res: torch.Tensor,
) -> None:
    if isinstance(hidden_states, FakeTensor):
        return
    _check(hidden_states, w13, w2, ffn_res)
    llada2_expert_ffn_kernel(hidden_states, w13, w2, ffn_res)
