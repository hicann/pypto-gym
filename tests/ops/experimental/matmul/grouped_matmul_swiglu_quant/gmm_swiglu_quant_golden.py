#!/usr/bin/env python3
# coding: utf-8
#
# PyPTO grouped_matmul_swiglu_quant golden reference (pure torch).
# 类型 `TransposeConfig` / `GroupedMatmulInputs` 见 gmm_swiglu_quant_impl.py。

import math
from dataclasses import dataclass

import torch

from experimental.matmul.grouped_matmul_swiglu_quant.gmm_swiglu_quant_impl import (
    GroupedMatmulInputs,
    TransposeConfig,
)


@dataclass(frozen=True)
class GoldenCaseInputs:
    """Inputs required to compute a single grouped golden result."""

    x: torch.Tensor
    weight: torch.Tensor
    scaled_x_golden: torch.Tensor
    scaled_weight_golden: torch.Tensor
    transpose: TransposeConfig


def swiglu(gmm_out):
    """Apply SwiGLU activation function."""
    value, gate = gmm_out.chunk(2, dim=-1)
    silu_value = value * torch.sigmoid(value)
    return silu_value * gate


def quant_pertoken(swiglu_out):
    """Quantize each token independently to int8 and return the dequant scale."""
    input_abs = torch.abs(swiglu_out)
    input_max = torch.amax(input_abs, dim=-1, keepdim=True)
    scale_max = torch.tensor(127.0, dtype=torch.float32, device=swiglu_out.device)
    scale = input_max / scale_max
    input_scaled = swiglu_out / scale
    round_data = torch.round(input_scaled)
    round_data_clamped = torch.clamp(round_data, min=-127, max=127)
    round_data_int8 = round_data_clamped.to(dtype=torch.int8)
    scale_tensor = scale.squeeze(-1).to(dtype=torch.float32)

    return round_data_int8, scale_tensor


def compute_golden_result(golden_case: GoldenCaseInputs):
    """Compute the reference result for grouped scaled matmul + SwiGLU + int8 quant."""
    x = golden_case.x
    weight = golden_case.weight
    scaled_x_golden = golden_case.scaled_x_golden
    scaled_weight_golden = golden_case.scaled_weight_golden
    a_trans = golden_case.transpose.a_trans
    b_trans = golden_case.transpose.b_trans

    if a_trans:
        x = torch.swapaxes(x, -1, -2)
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
        if scaled_x_golden.ndim == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0] * scaled_x_golden.shape[1],
                scaled_x_golden.shape[2],
            )
        scaled_x_golden = torch.swapaxes(scaled_x_golden, -1, -2)
    else:
        if scaled_x_golden.ndim == 3:
            scaled_x_golden = scaled_x_golden.reshape(
                scaled_x_golden.shape[0],
                scaled_x_golden.shape[1] * scaled_x_golden.shape[2],
            )

    if b_trans:
        weight = torch.swapaxes(weight, -1, -2)
        if scaled_weight_golden.ndim == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0],
                scaled_weight_golden.shape[1] * scaled_weight_golden.shape[2],
            )
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
    else:
        scaled_weight_golden = torch.swapaxes(scaled_weight_golden, -1, -2)
        if scaled_weight_golden.ndim == 3:
            scaled_weight_golden = scaled_weight_golden.reshape(
                scaled_weight_golden.shape[0] * scaled_weight_golden.shape[1],
                scaled_weight_golden.shape[2],
            )

    k_dim = x.shape[-1]
    if math.ceil(k_dim / 32) % 2 != 0:
        scaled_x_golden = scaled_x_golden[:, :-1]
        scaled_weight_golden = scaled_weight_golden[:-1, :]

    scaled_x_golden_broadcast = torch.repeat_interleave(
        scaled_x_golden, repeats=32, dim=-1
    )
    scaled_weight_golden_broadcast = torch.repeat_interleave(
        scaled_weight_golden, repeats=32, dim=-2
    )
    x1_dims = x.ndim
    x2_dims = weight.ndim
    x1_pad_len = scaled_x_golden_broadcast.shape[-1] - x.shape[-1]
    x2_pad_len = scaled_weight_golden_broadcast.shape[-2] - weight.shape[-2]
    x1_pad = [0, x1_pad_len]
    for _ in range(x1_dims - 1):
        x1_pad += [0, 0]
    x1_golden = torch.nn.functional.pad(x, x1_pad, mode="constant", value=0)

    weight_pad = [0, 0]
    weight_pad += [0, x2_pad_len]
    for _ in range(x2_dims - 2):
        weight_pad += [0, 0]
    weight_golden = torch.nn.functional.pad(weight, weight_pad, mode="constant", value=0)

    x_fp32 = x.to(torch.float32)
    scaled_x_golden_broadcast_fp32 = scaled_x_golden_broadcast.to(torch.float32)
    x1_golden = x_fp32 * scaled_x_golden_broadcast_fp32

    weight_fp32 = weight.to(torch.float32)
    scaled_weight_golden_broadcast_fp32 = scaled_weight_golden_broadcast.to(torch.float32)
    weight_golden = weight_fp32 * scaled_weight_golden_broadcast_fp32

    gmm_out = torch.matmul(x1_golden, weight_golden)

    swiglu_out = swiglu(gmm_out)
    swiglu_out = swiglu_out.to(torch.bfloat16)
    swiglu_out = swiglu_out.to(torch.float32)

    quant_output, quant_scale_output = quant_pertoken(swiglu_out)

    return quant_output, quant_scale_output


def gen_golden(inputs: GroupedMatmulInputs, transpose: TransposeConfig):
    """Generate grouped reference outputs for all experts."""
    num_groups = inputs.b.shape[0]
    gmmswigluquant_output = []
    gmmswigluquant_output_scale = []
    begin = 0
    end = 0

    for i in range(num_groups):
        if inputs.group_list[i] <= 0:
            continue
        begin = end
        end = end + inputs.group_list[i]
        if transpose.a_trans:
            x = inputs.a[:, begin:end]
        else:
            x = inputs.a[begin:end, :]
        weight = inputs.b[i]
        if transpose.a_trans:
            scaled_x_golden = inputs.scaled_a[:, begin:end, :]
        else:
            scaled_x_golden = inputs.scaled_a[begin:end, :, :]
        scaled_weight_golden = inputs.scaled_b[i]

        golden_case = GoldenCaseInputs(
            x=x,
            weight=weight,
            scaled_x_golden=scaled_x_golden,
            scaled_weight_golden=scaled_weight_golden,
            transpose=transpose,
        )
        golden_temp, golden_temp_quant = compute_golden_result(golden_case)
        gmmswigluquant_output.append(golden_temp)
        gmmswigluquant_output_scale.append(golden_temp_quant)

    gmmswigluquant_output = torch.cat(gmmswigluquant_output, dim=0)
    gmmswigluquant_output_scale = torch.cat(gmmswigluquant_output_scale, dim=0)

    return gmmswigluquant_output, gmmswigluquant_output_scale
