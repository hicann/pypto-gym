#!/usr/bin/env python3
# coding: utf-8
#
# PyPTO grouped_matmul_finalize_routing golden reference implementation.
# 配置类型定义见 gmm_finalize_routing_impl.py。

import math

import torch

from experimental.matmul.grouped_matmul_finalize_routing.gmm_finalize_routing_impl import (
    FinalizeRoutingConfig,
    FinalizeRoutingGoldenInputs,
)


def _compute_mxfp8_matmul_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    pertoken_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    transpose_x1: bool,
    transpose_x2: bool,
) -> torch.Tensor:
    """Golden for one MXFP8 matmul group."""

    x_golden = x
    weight_golden = weight
    x_scale = pertoken_scale
    w_scale = weight_scale

    if transpose_x1:
        x_golden = torch.swapaxes(x_golden, -1, -2)
        x_scale = torch.swapaxes(x_scale, -1, -2)
        if len(x_scale.shape) == 3:
            x_scale = x_scale.reshape(x_scale.shape[0] * x_scale.shape[1], x_scale.shape[2])
        x_scale = torch.swapaxes(x_scale, -1, -2)
    else:
        if len(x_scale.shape) == 3:
            x_scale = x_scale.reshape(x_scale.shape[0], x_scale.shape[1] * x_scale.shape[2])

    if transpose_x2:
        weight_golden = torch.swapaxes(weight_golden, -1, -2)
        if len(w_scale.shape) == 3:
            w_scale = w_scale.reshape(w_scale.shape[0] * w_scale.shape[1], w_scale.shape[2])
        w_scale = torch.swapaxes(w_scale, -1, -2)
    else:
        w_scale = torch.swapaxes(w_scale, -1, -2)
        if len(w_scale.shape) == 3:
            w_scale = w_scale.reshape(w_scale.shape[0] * w_scale.shape[1], w_scale.shape[2])

    k_dim = x_golden.shape[-1]
    if math.ceil(k_dim / 32) % 2 != 0:
        x_scale = x_scale[:, :-1]
        w_scale = w_scale[:-1, :]

    x_scale = torch.repeat_interleave(x_scale, repeats=32, dim=-1).to(torch.float32)
    w_scale = torch.repeat_interleave(w_scale, repeats=32, dim=-2).to(torch.float32)
    return torch.matmul(x_golden.to(torch.float32) * x_scale, weight_golden.to(torch.float32) * w_scale)


def _expert_range(group_list: torch.Tensor, expert_idx: int, group_list_type: int) -> tuple[int, int]:
    """根据 group_list 解析第 expert_idx 个 expert 的 token 区间 [start, end)。"""
    if group_list_type == 0:
        end = int(group_list[expert_idx].item())
        start = 0 if expert_idx == 0 else int(group_list[expert_idx - 1].item())
        return start, end

    start = int(group_list[:expert_idx].sum().item())
    end = start + int(group_list[expert_idx].item())
    return start, end


def gen_golden(inputs: FinalizeRoutingGoldenInputs) -> torch.Tensor:
    """Golden implementation for GMM + finalize routing."""

    cfg = inputs.config
    if cfg.transpose_x1:
        raise ValueError("aclnnGroupedMatmulFinalizeRoutingV3 only supports transposeX1=False.")

    golden = inputs.out.clone()
    for expert_idx in range(cfg.num_experts):
        start, end = _expert_range(inputs.group_list, expert_idx, cfg.group_list_type)
        if end <= start:
            continue

        x = inputs.x1[start:end, :]
        weight = inputs.x2[expert_idx]
        scale = inputs.scale
        pertoken_scale = inputs.pertoken_scale[start:end, :]

        if cfg.transpose_x2:
            weight = weight.transpose(-1, -2).contiguous()
            scale = scale.transpose(0, 1).contiguous()

        mm_result = _compute_mxfp8_matmul_golden(
            x,
            weight,
            pertoken_scale,
            scale,
            transpose_x1=False,
            transpose_x2=False,
        )
        if cfg.has_logit:
            mm_result = mm_result * inputs.logit[start:end].to(torch.float32).unsqueeze(-1)

        golden.index_add_(0, inputs.row_index[start:end].to(torch.int64), mm_result)

    if cfg.has_shared_input:
        shared_start = cfg.shared_input_offset
        shared_end = shared_start + inputs.shared_input.shape[0]
        golden[shared_start:shared_end, :] = (
            golden[shared_start:shared_end, :]
            + inputs.shared_input.to(torch.float32) * cfg.shared_input_weight
        )

    return golden


def make_group_list(m: int, num_experts: int, group_list_type: int) -> torch.Tensor:
    """构造均匀分组的 group_list（支持前缀和/计数两种格式）。"""
    base = m // num_experts
    counts = torch.full((num_experts,), base, dtype=torch.int64)
    counts[: m - base * num_experts] += 1
    if group_list_type == 0:
        return torch.cumsum(counts, dim=0)
    return counts
