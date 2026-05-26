#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.

"""
PyPTO RMSNorm 实现
Wrapper: compute RMS coefficient in torch, apply scaling via PyPTO kernel.
Performance note: per-row kernel calls are suboptimal; batch processing is future work.
"""

import torch
from torch._dynamo import allow_in_graph
import pypto.language as pl


@pl.jit
def rms_norm_kernel(
    hidden_states: pl.Tensor,
    output: pl.Out[pl.Tensor],
    coeff: pl.Scalar[pl.FP32],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        M, D = hidden_states.shape
        tile_h = pl.load(hidden_states, [0, 0], [M, D])
        normed = pl.mul(tile_h, coeff)
        pl.store(normed, [0, 0], output)
    return output


@allow_in_graph
def rms_norm_impl(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    PyPTO RMSNorm wrapper.
    
    Computing the RMS coefficient in torch (NPU-accelerated),
    then applying it element-wise via PyPTO kernel.
    Weight multiplication is done in torch for numerical accuracy.
    """
    input_dtype = hidden_states.dtype
    h_f32 = hidden_states.float()
    var = h_f32.pow(2).mean(-1, keepdim=True)
    coeff = torch.rsqrt(var + eps).to(input_dtype)

    orig_shape = hidden_states.shape
    D = orig_shape[-1]
    hidden_2d = hidden_states.reshape(-1, D).cpu()
    coeff_2d = coeff.reshape(-1, 1).cpu()
    N = hidden_2d.shape[0]

    output_cpu = torch.empty(N, D, dtype=hidden_2d.dtype, device="cpu")

    for i in range(N):
        row_in = hidden_2d[i:i+1, :]
        row_out = output_cpu[i:i+1, :]
        c = coeff_2d[i].item()
        rms_norm_kernel(row_in, row_out, c)

    output_normed = output_cpu.reshape(orig_shape).to(hidden_states.device)
    return output_normed * weight
