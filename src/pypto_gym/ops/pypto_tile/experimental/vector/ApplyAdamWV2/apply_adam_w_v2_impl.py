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
apply_adam_w_v2 PyPTO implementation.

Single-step AdamW update with bias correction.
  m_t   = beta1 * m + (1 - beta1) * g
  v_t   = beta2 * v + (1 - beta2) * g * g
  m_hat = m_t / (1 - beta1**t)
  v_hat = v_t / (1 - beta2**t)
  update = m_hat / (sqrt(v_hat) + eps) + weight_decay * w
  w_new  = w - lr * update

Two jit kernels (fp32 / bf16 weight+grad path); m, v are always fp32.
K-axis loop with view + valid_shape + assemble for tail-block handling.
"""
from typing import Tuple

import pypto
import torch

M_DIM = 7168
N_TILE = 2048
VEC_TILE_M = 16
VEC_TILE_N = 1024

# ---------------------------------------------------------------------------
# fp32 path: weight/grad are fp32
# ---------------------------------------------------------------------------
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU},
                    debug_options={"runtime_debug_mode": 0})
def apply_adam_w_v2_kernel_fp32(
    weight: pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    grad:   pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    m:      pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    v:      pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    weight_out: pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    m_out:      pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    v_out:      pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    beta1: float,
    one_m_b1: float,
    beta2: float,
    one_m_b2: float,
    bc1: float,
    bc2: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(VEC_TILE_M, VEC_TILE_N)

    K = weight.shape[1]
    k_loops = (K + N_TILE - 1) // N_TILE

    for k_idx in pypto.loop(k_loops):
        k_off = k_idx * N_TILE
        valid_k = (K - k_off).min(N_TILE)
        valid_shape = [M_DIM, valid_k]

        w_tile = pypto.view(weight, [M_DIM, N_TILE], [0, k_off], valid_shape=valid_shape)
        g_tile = pypto.view(grad,   [M_DIM, N_TILE], [0, k_off], valid_shape=valid_shape)
        m_tile = pypto.view(m,      [M_DIM, N_TILE], [0, k_off], valid_shape=valid_shape)
        v_tile = pypto.view(v,      [M_DIM, N_TILE], [0, k_off], valid_shape=valid_shape)

        beta1_m = pypto.mul(m_tile, beta1)
        one_b1_g = pypto.mul(g_tile, one_m_b1)
        m_new = pypto.add(beta1_m, one_b1_g)
        grad_sq = pypto.mul(g_tile, g_tile)
        beta2_v = pypto.mul(v_tile, beta2)
        one_b2_gs = pypto.mul(grad_sq, one_m_b2)
        v_new = pypto.add(beta2_v, one_b2_gs)
        m_hat = pypto.div(m_new, bc1)
        v_hat = pypto.div(v_new, bc2)
        sqrt_v = pypto.sqrt(v_hat)
        denom = pypto.add(sqrt_v, eps)
        term1 = pypto.div(m_hat, denom)
        term2 = pypto.mul(w_tile, weight_decay)
        update = pypto.add(term1, term2)
        scaled = pypto.mul(update, lr)
        w_new = pypto.sub(w_tile, scaled)

        pypto.assemble(w_new, [0, k_off], weight_out)
        pypto.assemble(m_new, [0, k_off], m_out)
        pypto.assemble(v_new, [0, k_off], v_out)


# ---------------------------------------------------------------------------
# bf16 path: weight/grad are bf16; m, v stay fp32; intermediate math fp32.
# ---------------------------------------------------------------------------
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU},
                    debug_options={"runtime_debug_mode": 0})
def apply_adam_w_v2_kernel_bf16(
    weight: pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_BF16),
    grad:   pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_BF16),
    m:      pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    v:      pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    weight_out: pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_BF16),
    m_out:      pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    v_out:      pypto.Tensor([M_DIM, pypto.DYNAMIC], pypto.DT_FP32),
    beta1: float,
    one_m_b1: float,
    beta2: float,
    one_m_b2: float,
    bc1: float,
    bc2: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    pypto.experimental.set_operation_options(combine_axis=True)
    pypto.set_vec_tile_shapes(VEC_TILE_M, VEC_TILE_N)

    K = weight.shape[1]
    k_loops = (K + N_TILE - 1) // N_TILE

    for k_idx in pypto.loop(k_loops):
        k_off = k_idx * N_TILE
        valid_k = (K - k_off).min(N_TILE)
        valid_shape = [M_DIM, valid_k]

        w_tile = pypto.view(weight, [M_DIM, N_TILE], [0, k_off], valid_shape=valid_shape)
        g_tile = pypto.view(grad,   [M_DIM, N_TILE], [0, k_off], valid_shape=valid_shape)
        m_tile = pypto.view(m,      [M_DIM, N_TILE], [0, k_off], valid_shape=valid_shape)
        v_tile = pypto.view(v,      [M_DIM, N_TILE], [0, k_off], valid_shape=valid_shape)

        w_f32 = pypto.cast(w_tile, pypto.DT_FP32)
        g_f32 = pypto.cast(g_tile, pypto.DT_FP32)
        beta1_m = pypto.mul(m_tile, beta1)
        one_b1_g = pypto.mul(g_f32, one_m_b1)
        m_new = pypto.add(beta1_m, one_b1_g)
        grad_sq = pypto.mul(g_f32, g_f32)
        beta2_v = pypto.mul(v_tile, beta2)
        one_b2_gs = pypto.mul(grad_sq, one_m_b2)
        v_new = pypto.add(beta2_v, one_b2_gs)
        m_hat = pypto.div(m_new, bc1)
        v_hat = pypto.div(v_new, bc2)
        sqrt_v = pypto.sqrt(v_hat)
        denom = pypto.add(sqrt_v, eps)
        term1 = pypto.div(m_hat, denom)
        term2 = pypto.mul(w_f32, weight_decay)
        update = pypto.add(term1, term2)
        scaled = pypto.mul(update, lr)
        w_new_f32 = pypto.sub(w_f32, scaled)
        w_new_bf16 = pypto.cast(w_new_f32, pypto.DT_BF16)

        pypto.assemble(w_new_bf16, [0, k_off], weight_out)
        pypto.assemble(m_new, [0, k_off], m_out)
        pypto.assemble(v_new, [0, k_off], v_out)


# ---------------------------------------------------------------------------
# Host wrapper: precompute scalars, dispatch by dtype, allocate outputs.
# ---------------------------------------------------------------------------
def apply_adam_w_v2_wrapper(
    weight: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    step: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 
    assert weight.shape == grad.shape == m.shape == v.shape, "shape mismatch"
    assert weight.dtype == grad.dtype, "weight/grad dtype must match"
    assert m.dtype == torch.float32 and v.dtype == torch.float32
    assert isinstance(step, int) and step >= 1

    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    one_m_b1 = 1.0 - beta1
    one_m_b2 = 1.0 - beta2

    weight_out = torch.empty_like(weight)
    m_out = torch.empty_like(m)
    v_out = torch.empty_like(v)

    if weight.dtype == torch.bfloat16:
        kernel = apply_adam_w_v2_kernel_bf16
    elif weight.dtype == torch.float32:
        kernel = apply_adam_w_v2_kernel_fp32
    else:
        raise TypeError(
            f"unsupported weight dtype {weight.dtype}; expected bf16 or fp32"
        )

    kernel(
        weight, grad, m, v,
        weight_out, m_out, v_out,
        float(beta1), float(one_m_b1),
        float(beta2), float(one_m_b2),
        float(bc1), float(bc2),
        float(lr), float(weight_decay), float(eps),
    )

    return weight_out, m_out, v_out
