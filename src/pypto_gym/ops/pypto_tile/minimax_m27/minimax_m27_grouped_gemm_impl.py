# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouped GEMM for MiniMax M2.7 MoE — Ascend 910B (dav-c220) adapted kernel.

Same algorithm as llada2_moe's grouped GEMM (all experts in one kernel call,
pypto.loop over experts + pypto.loop_unroll over tokens), but **self-contained
and tiled for the 910B**: the per-tile SwiGLU/cast vector width is capped to a
UB-fitting value (``PYPTO_VEC_TILE``, default 128) instead of the full
intermediate/hidden width.

Why the cap (910B hardware): the chip has split CUBE/VECTOR cores with a
**192 KB UB**. A FP32 vector tile of [tile, W] with double buffering needs
``tile*W*4*2`` bytes; at W = intermediate_size (1536) or hidden_size (3072) this
far exceeds UB, and the ``PVC2_OOO`` scheduler fails the ``OoOSchedule`` pass.
A width-128 vector tile ([128,128] FP32 x2buf = 128 KB < 192 KB) schedules and
matches the reference (BF16 max_err = 0 measured). See ai/910b-hardware.md.

MiniMax M2.7 stores expert weights in F.linear convention:
    gate_up_proj  [E, 2*I, H]   (F.linear computes x @ W^T)
    down_proj     [E, H,   I]
``convert_minimax_weights`` transposes them to the kernel's direct-matmul layout
    w13_flat      [E*H, 2*I]    (kernel computes x @ W)
    w2_flat       [E*I, H]
"""

import os
from typing import NamedTuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph

import pypto


class MoeDims(NamedTuple):
    """MoE shape triple passed to the grouped-GEMM kernel: (E, H, I)."""
    num_experts: int
    hidden_size: int
    intermediate_size: int


ND = pypto.TileOpFormat.TILEOP_ND

# Vector tile width (UB-fitting). 128 verified on 910B; raise only if it still
# fits UB and schedules. Tunable for perf without touching the kernel body.
_VEC_TILE = int(os.environ.get("PYPTO_VEC_TILE", "128"))
_CUBE_NBUF = int(os.environ.get("PYPTO_CUBE_NBUFFER", "2"))
_VEC_NBUF = int(os.environ.get("PYPTO_VEC_NBUFFER", "2"))
# Cube tile shapes (K, N) for the two matmuls — all tile params env-exposed for tuning
# without touching the kernel body, matching the llada2_moe convention.
_MM1_K = int(os.environ.get("PYPTO_MM1_K", "128"))
_MM1_N = int(os.environ.get("PYPTO_MM1_N", "256"))
_MM2_K = int(os.environ.get("PYPTO_MM2_K", "128"))
_MM2_N = int(os.environ.get("PYPTO_MM2_N", "256"))


def convert_minimax_weights(gate_up_proj, down_proj):
    """Convert MiniMax M2.7 expert weights to grouped GEMM kernel format.

    Parameters
    ----------
    gate_up_proj : [E, 2*I, H] BF16 — MiniMax gate||up weights (F.linear fmt)
    down_proj    : [E, H,   I] BF16 — MiniMax down weights (F.linear fmt)

    Returns
    -------
    w13_flat : [E*H, 2*I] BF16 — flattened gate||up (direct matmul fmt)
    w2_flat  : [E*I, H]   BF16 — flattened down (direct matmul fmt)
    """
    num_experts = gate_up_proj.shape[0]
    two_intermediate = gate_up_proj.shape[1]
    hidden_size = gate_up_proj.shape[2]
    intermediate_size = down_proj.shape[2]

    w13_flat = gate_up_proj.transpose(1, 2).reshape(num_experts * hidden_size, two_intermediate).contiguous()
    w2_flat = down_proj.transpose(1, 2).reshape(num_experts * intermediate_size, hidden_size).contiguous()
    return w13_flat, w2_flat


def _swiglu_silu(gate_up):
    """Return SiLU(gate) * up."""
    half = gate_up.shape[1] // 2
    gate = pypto.view(gate_up, [gate_up.shape[0], half], [0, 0])
    up = pypto.view(gate_up, [gate_up.shape[0], half], [0, half])
    neg = pypto.mul(gate, -1.0)
    exp_neg = pypto.exp(neg)
    denom = pypto.add(exp_neg, 1.0)
    silu = pypto.div(gate, denom)
    return pypto.mul(silu, up)


@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1, "stitch_function_max_num": 64},
    pass_options={
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: _CUBE_NBUF},
        "vec_nbuffer_setting": {-1: _VEC_NBUF},
    },
)
def minimax_m27_grouped_gemm_kernel(
    sorted_tokens: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    w13_flat: pypto.Tensor([], pypto.DT_BF16, format=ND),
    w2_flat: pypto.Tensor([], pypto.DT_BF16, format=ND),
    expert_cumsum: pypto.Tensor([], pypto.DT_INT32, format=ND),
    result: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
):
    # Shapes are derived from the tensors (no separate dim params): sorted_tokens [N, H],
    # w13_flat [E*H, 2I], expert_cumsum [E+1]. Mirrors llada2_expert_ffn_kernel's pattern.
    num_experts = expert_cumsum.shape[0] - 1
    hidden_size = sorted_tokens.shape[1]
    two_i = w13_flat.shape[1]
    intermediate_size = two_i // 2
    pypto.experimental.set_operation_options(combine_axis=True)
    vt = _VEC_TILE

    for e_idx in pypto.loop(0, num_experts, 1, name="EXPERT_LOOP", idx_name="e_idx"):
        e_start = expert_cumsum[e_idx]
        e_end = expert_cumsum[e_idx + 1]
        n_e = e_end - e_start

        for tok_idx, tile_batch in pypto.loop_unroll(
            0, n_e, 1,
            name="LOOP_TOKEN",
            idx_name="tok_idx",
            unroll_list=[1, 2, 4, 8, 16, 32, 64],
        ):
            w13_e = pypto.view(w13_flat, [hidden_size, two_i], [e_idx * hidden_size, 0])
            w2_e = pypto.view(w2_flat, [intermediate_size, hidden_size], [e_idx * intermediate_size, 0])
            tile_x = pypto.view(sorted_tokens, [tile_batch, hidden_size], [e_start + tok_idx, 0])

            # mm1 -> gate_up [tile_batch, 2*I] FP32
            pypto.set_cube_tile_shapes([tile_batch, tile_batch], [_MM1_K, _MM1_K * 2], [_MM1_N, _MM1_N], True)
            gate_up = pypto.matmul(tile_x, w13_e, pypto.DT_FP32)

            # SwiGLU in FP32 -> BF16, vector tile width capped to UB-fitting `vt`
            pypto.set_vec_tile_shapes(min(tile_batch, vt), vt)
            sw = pypto.cast(_swiglu_silu(gate_up), pypto.DT_BF16)

            # mm2 -> down [tile_batch, H] FP32
            pypto.set_cube_tile_shapes([tile_batch, tile_batch], [_MM2_K, _MM2_K * 2], [_MM2_N, _MM2_N], False)
            down = pypto.matmul(sw, w2_e, pypto.DT_FP32)
            pypto.set_vec_tile_shapes(min(tile_batch, vt), vt)
            out = pypto.cast(down, pypto.DT_BF16)

            pypto.assemble(out, [e_start + tok_idx, 0], result)


def _check(sorted_tokens, weights, expert_cumsum, result, dims):
    w13_flat, w2_flat = weights
    num_experts, hidden_size, intermediate_size = dims
    if sorted_tokens.dim() != 2 or sorted_tokens.dtype != torch.bfloat16:
        raise ValueError("sorted_tokens must be 2-D bfloat16")
    if w13_flat.dim() != 2 or w13_flat.dtype != torch.bfloat16:
        raise ValueError("w13_flat must be 2-D bfloat16")
    if w2_flat.dim() != 2 or w2_flat.dtype != torch.bfloat16:
        raise ValueError("w2_flat must be 2-D bfloat16")
    if expert_cumsum.dim() != 1 or expert_cumsum.dtype != torch.int32:
        raise ValueError("expert_cumsum must be 1-D int32")
    if sorted_tokens.shape[1] != hidden_size:
        raise ValueError(f"sorted_tokens hidden {sorted_tokens.shape[1]} != {hidden_size}")
    if tuple(w13_flat.shape) != (num_experts * hidden_size, 2 * intermediate_size):
        raise ValueError("w13_flat shape mismatch")
    if tuple(w2_flat.shape) != (num_experts * intermediate_size, hidden_size):
        raise ValueError("w2_flat shape mismatch")
    if expert_cumsum.shape[0] != num_experts + 1:
        raise ValueError("expert_cumsum length must be num_experts + 1")
    if result.shape != sorted_tokens.shape:
        raise ValueError("result shape must equal sorted_tokens shape")


@allow_in_graph
def minimax_m27_moe_grouped_gemm(
    sorted_tokens: torch.Tensor,
    weights: "tuple[torch.Tensor, torch.Tensor]",
    expert_cumsum: torch.Tensor,
    result: torch.Tensor,
    dims: "MoeDims",
) -> None:
    """Grouped GEMM for MiniMax M2.7 MoE (all experts, single kernel call).

    910B-adapted: uses a UB-fitting vector tile (PYPTO_VEC_TILE, default 128).

    Parameters
    ----------
    sorted_tokens : [N_total, H] BF16 — tokens pre-sorted by expert
    weights : (w13_flat [E*H, 2*I], w2_flat [E*I, H]) BF16 — converted gate||up / down weights
    expert_cumsum : [E+1] INT32 — cumulative token counts per expert
    result : [N_total, H] BF16 — output buffer
    dims : MoeDims(num_experts, hidden_size, intermediate_size) — e.g. (256, 3072, 1536) for M2.7
    """
    if isinstance(sorted_tokens, FakeTensor):
        return
    w13_flat, w2_flat = weights
    _check(sorted_tokens, weights, expert_cumsum, result, dims)
    minimax_m27_grouped_gemm_kernel(
        sorted_tokens, w13_flat, w2_flat, expert_cumsum, result,
    )
