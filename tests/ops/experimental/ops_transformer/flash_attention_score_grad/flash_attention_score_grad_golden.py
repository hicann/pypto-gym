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
FlashAttentionScoreGrad Golden 参考实现

公式:
  前向: Y = Softmax(Q @ K^T / sqrt(D)) @ V
  反向:
    p  = exp(Q @ K^T * scale - softmax_max) / softmax_sum   (online softmax 重算)
    d  = sum(dY * attention_out, dim=-1, keepdim=True)
    dp = dY @ V^T
    ds = p * (dp - d)
    dv = p^T @ dY
    dq = ds @ K * scale
    dk = ds^T @ Q * scale

使用分块流式 softmax 计算，dtype 与 PyPTO kernel (S_TILE=128) 保持一致。

置信度: ⭐⭐⭐⭐ (标准 Flash Attention backward 公式)
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

S_TILE = 128


@dataclass
class ForwardDataConfig:
    """Configuration for generating forward data."""

    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
    device: str = 'cpu'


@dataclass
class ForwardDataResult:
    """Result from generate_forward_data."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dy: torch.Tensor
    softmax_max: torch.Tensor
    softmax_sum: torch.Tensor
    attention_out: torch.Tensor
    scale: float


class AttentionGradInputs:
    """Container for attention gradient inputs."""

    def __init__(self, query, key, value, dy,
                 softmax_max, softmax_sum, attention_out, scale_value):
        self.query = query
        self.key = key
        self.value = value
        self.dy = dy
        self.softmax_max = softmax_max
        self.softmax_sum = softmax_sum
        self.attention_out = attention_out
        self.scale_value = scale_value


def flash_attention_score_grad_golden(
        inputs: AttentionGradInputs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FlashAttentionScoreGrad 参考实现 (分块流式 softmax)

    与 PyPTO kernel (S_TILE=128) 计算流完全一致:
      - Q/K/V/dY/attention_out: BF16 输入
      - softmax_max/sum: FP32 输入
      - d_i: dy_i * ao_i (BF16*BF16→BF16) → cast(FP32) → sum(FP32)
      - compute_tile:
        s_ij = matmul(Q_BF16, K_BF16^T, FP32) * scale → FP32
        p_ij = exp(s_ij - smax_i) / ssum_i → FP32
        dp_ij = matmul(dY_BF16, V_BF16^T, FP32) → FP32
        ds_ij = p_ij * (dp_ij - d_i) → FP32
      - ds/p: cast(BF16) 后再 matmul(FP32 累加输出)
      - 累加器 (dq_acc, dk_acc, dv_acc): FP32
      - 最终输出: * scale → cast(BF16), 与 kernel 逐 tile cast 一致

    分块策略 (与 kernel 完全一致):
      - 趟1: 沿 s2 (KV) tile 累积 dQ → dQ = sum(ds @ K) * scale
      - 趟2: 沿 s1 (Q) tile 累积 dK, dV → dK = sum(ds^T @ Q) * scale, dV = sum(p^T @ dY)
    """
    q = inputs.query
    k = inputs.key
    v = inputs.value
    dy = inputs.dy
    attn_out = inputs.attention_out
    scale = inputs.scale_value

    s_max = inputs.softmax_max[:, :, :, 0:1]
    s_sum = inputs.softmax_sum[:, :, :, 0:1]

    B, N, S, D = q.shape

    dq_out = torch.zeros(B, N, S, D, dtype=torch.bfloat16, device=q.device)
    dk_out = torch.zeros(B, N, S, D, dtype=torch.bfloat16, device=q.device)
    dv_out = torch.zeros(B, N, S, D, dtype=torch.bfloat16, device=q.device)

    for b in range(B):
        for n in range(N):
            q_bn = q[b, n]            # BF16
            k_bn = k[b, n]            # BF16
            v_bn = v[b, n]            # BF16
            dy_bn = dy[b, n]          # BF16
            attn_bn = attn_out[b, n]  # BF16
            smax_bn = s_max[b, n]     # FP32
            ssum_bn = s_sum[b, n]     # FP32

            # ===== 趟1: 计算 dQ =====
            for s1_start in range(0, S, S_TILE):
                s1_end = min(s1_start + S_TILE, S)

                q_i = q_bn[s1_start:s1_end]      # BF16
                dy_i = dy_bn[s1_start:s1_end]    # BF16
                attn_i = attn_bn[s1_start:s1_end]  # BF16
                smax_i = smax_bn[s1_start:s1_end]  # FP32
                ssum_i = ssum_bn[s1_start:s1_end]  # FP32

                # kernel: mul(BF16,BF16)→BF16 → cast→FP32 → sum→FP32
                d_i = (dy_i * attn_i).float().sum(
                    dim=-1, keepdim=True)

                dq_acc = None
                for s2_start in range(0, S, S_TILE):
                    s2_end = min(s2_start + S_TILE, S)

                    k_j = k_bn[s2_start:s2_end]  # BF16
                    v_j = v_bn[s2_start:s2_end]  # BF16

                    # compute_tile: matmul(BF16,BF16)→FP32
                    scores = (q_i.float() @ k_j.float().T) * scale
                    p_ij = torch.exp(scores - smax_i) / ssum_i

                    dp_ij = dy_i.float() @ v_j.float().T
                    ds_ij = p_ij * (dp_ij - d_i)

                    # kernel: ds(FP32)→BF16 → matmul: BF16 值 + FP32 累加
                    ds_bf16 = ds_ij.to(torch.bfloat16)
                    dq_tile = ds_bf16.float() @ k_j.float()

                    if dq_acc is None:
                        dq_acc = dq_tile
                    else:
                        dq_acc += dq_tile

                dq_out[b, n, s1_start:s1_end] = (dq_acc * scale).to(torch.bfloat16)

            # ===== 趟2: 计算 dK, dV =====
            for s2_start in range(0, S, S_TILE):
                s2_end = min(s2_start + S_TILE, S)

                k_j = k_bn[s2_start:s2_end]  # BF16
                v_j = v_bn[s2_start:s2_end]  # BF16

                dk_acc = None
                dv_acc = None
                for s1_start in range(0, S, S_TILE):
                    s1_end = min(s1_start + S_TILE, S)

                    q_i = q_bn[s1_start:s1_end]    # BF16
                    dy_i = dy_bn[s1_start:s1_end]  # BF16
                    attn_i = attn_bn[s1_start:s1_end]  # BF16
                    smax_i = smax_bn[s1_start:s1_end]  # FP32
                    ssum_i = ssum_bn[s1_start:s1_end]  # FP32

                    d_i = (dy_i * attn_i).float().sum(
                        dim=-1, keepdim=True)

                    scores = (q_i.float() @ k_j.float().T) * scale
                    p_ij = torch.exp(scores - smax_i) / ssum_i

                    dp_ij = dy_i.float() @ v_j.float().T
                    ds_ij = p_ij * (dp_ij - d_i)

                    # kernel: ds/p(FP32)→BF16 → matmul: BF16 值 + FP32 累加
                    ds_bf16 = ds_ij.to(torch.bfloat16)
                    p_bf16 = p_ij.to(torch.bfloat16)

                    dk_tile = ds_bf16.float().T @ q_i.float()
                    dv_tile = p_bf16.float().T @ dy_i.float()

                    if dk_acc is None:
                        dk_acc = dk_tile
                        dv_acc = dv_tile
                    else:
                        dk_acc += dk_tile
                        dv_acc += dv_tile

                dk_out[b, n, s2_start:s2_end] = (dk_acc * scale).to(torch.bfloat16)
                dv_out[b, n, s2_start:s2_end] = dv_acc.to(torch.bfloat16)

    return dq_out, dk_out, dv_out


def _generate_tensors(cfg: ForwardDataConfig):
    """Generate random tensors for forward computation."""
    q = torch.randn(
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
        dtype=cfg.dtype, device=cfg.device)
    k = torch.randn(
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
        dtype=cfg.dtype, device=cfg.device)
    v = torch.randn(
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
        dtype=cfg.dtype, device=cfg.device)
    dy = torch.randn(
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
        dtype=cfg.dtype, device=cfg.device)
    return q, k, v, dy


BLOCK_SIZE_Q = 32
BLOCK_SIZE_KV = 64


def _compute_forward_outputs(q, k, v, scale, cfg: ForwardDataConfig):
    """分块流式 softmax 前向计算，计算流与 flash_attention_score_kernel_with_mask_origin 完全一致。

    与 kernel 一致的参数:
      - BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=64
      - 输入 BF16, matmul 输出 FP32, softmax 在 FP32 进行
      - 单趟集成计算: 在线 softmax 合并 Q×K→P×V 在同一 KV 循环中完成
      - 最终 attention_out 为 BF16, softmax_max/sum 为 FP32 (padding 到 last_dim=8)
    """
    B, N, S, D = cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim

    softmax_max = torch.zeros(B, N, S, 8, dtype=torch.float32, device=cfg.device)
    softmax_sum = torch.zeros(B, N, S, 8, dtype=torch.float32, device=cfg.device)
    attention_out = torch.zeros(B, N, S, D, dtype=torch.bfloat16, device=cfg.device)

    num_blocks_kv = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    num_blocks_q = (S + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q

    for b in range(B):
        for n in range(N):
            q_bn = q[b, n]    # BF16
            k_bn = k[b, n]    # BF16
            v_bn = v[b, n]    # BF16

            for q_block_idx in range(num_blocks_q):
                q_start = q_block_idx * BLOCK_SIZE_Q
                cur_q_size = min(BLOCK_SIZE_Q, S - q_start)

                q_block_2d = q_bn[q_start:q_start + cur_q_size].reshape(cur_q_size, D)

                mi_update = torch.full((cur_q_size, 1), float('-inf'), dtype=torch.float32, device=cfg.device)
                li_update = torch.zeros(cur_q_size, 1, dtype=torch.float32, device=cfg.device)
                oi_update = torch.zeros(cur_q_size, D, dtype=torch.float32, device=cfg.device)

                for kv_block_idx in range(num_blocks_kv):
                    kv_start = kv_block_idx * BLOCK_SIZE_KV
                    cur_block_size = min(BLOCK_SIZE_KV, S - kv_start)

                    k_block_2d = k_bn[kv_start:kv_start + cur_block_size].reshape(cur_block_size, D)

                    scores = torch.matmul(q_block_2d.float(), k_block_2d.float().T)
                    scores_scaled = scores * scale

                    m_ij = torch.amax(scores_scaled, dim=-1, keepdim=True)
                    s_ij_sub_m = scores_scaled - m_ij
                    p_ij = torch.exp(s_ij_sub_m)
                    l_ij = torch.sum(p_ij, dim=-1, keepdim=True)

                    v_block_2d = v_bn[kv_start:kv_start + cur_block_size].reshape(cur_block_size, D).float()
                    o_ij = torch.matmul(p_ij, v_block_2d)

                    if kv_block_idx == 0:
                        mi_update = m_ij
                        li_update = l_ij
                        oi_update = o_ij
                    else:
                        mi_new = torch.maximum(mi_update, m_ij)
                        alpha = torch.exp(mi_update - mi_new)
                        beta = torch.exp(m_ij - mi_new)
                        li_update = alpha * li_update + beta * l_ij
                        oi_update = alpha * oi_update + beta * o_ij
                        mi_update = mi_new

                o_final = oi_update / li_update
                attention_out[b, n, q_start:q_start + cur_q_size] = o_final.to(torch.bfloat16)
                softmax_max[b, n, q_start:q_start + cur_q_size, 0] = mi_update.squeeze(-1)
                softmax_sum[b, n, q_start:q_start + cur_q_size, 0] = li_update.squeeze(-1)

    return attention_out, softmax_max, softmax_sum


def generate_forward_data(
        cfg: ForwardDataConfig) -> ForwardDataResult:
    """生成前向数据和中间结果，供反向测试使用。"""
    torch.manual_seed(42)
    scale = 1.0 / (cfg.head_dim ** 0.5)

    q, k, v, dy = _generate_tensors(cfg)
    attention_out, softmax_max, softmax_sum = _compute_forward_outputs(
        q, k, v, scale, cfg)

    return ForwardDataResult(
        q, k, v, dy, softmax_max, softmax_sum, attention_out, scale)


def _run_single_test(tc):
    """Run a single test case and return True if passed."""
    logger.info(
        "\n--- %s (B=%d, N=%d, S=%d, D=%d) ---",
        tc['name'], tc['B'], tc['N'], tc['S'], tc['D'])
    cfg = ForwardDataConfig(tc['B'], tc['N'], tc['S'], tc['D'])
    result = generate_forward_data(cfg)
    inputs = AttentionGradInputs(
        result.q, result.k, result.v, result.dy,
        result.softmax_max, result.softmax_sum,
        result.attention_out, result.scale)
    dq_out, dk_out, dv_out = flash_attention_score_grad_golden(inputs)

    logger.info("  dq_out shape: %s, dtype: %s", dq_out.shape, dq_out.dtype)
    logger.info("  dk_out shape: %s, dtype: %s", dk_out.shape, dk_out.dtype)
    logger.info("  dv_out shape: %s, dtype: %s", dv_out.shape, dv_out.dtype)

    if dq_out.shape != result.q.shape:
        raise ValueError(
            f"dq_out shape {dq_out.shape} != q shape {result.q.shape}")
    if dk_out.shape != result.k.shape:
        raise ValueError(
            f"dk_out shape {dk_out.shape} != k shape {result.k.shape}")
    if dv_out.shape != result.v.shape:
        raise ValueError(
            f"dv_out shape {dv_out.shape} != v shape {result.v.shape}")
    for name, t_val in [("dq_out", dq_out), ("dk_out", dk_out),
                        ("dv_out", dv_out)]:
        if torch.isnan(t_val).any():
            raise ValueError(f"{name} contains NaN")
        if torch.isinf(t_val).any():
            raise ValueError(f"{name} contains Inf")
        t_f = t_val.float()
        logger.info(
            "  %s range: [%.4f, %.4f]", name, t_f.min().item(),
            t_f.max().item())

    logger.info("  ✓ Passed")
    return True


def _run_autograd_validation():
    """Run autograd cross-validation."""
    logger.info("\n--- 交叉验证: PyTorch autograd ---")
    batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 32
    scale = 1.0 / (head_dim ** 0.5)
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float64, requires_grad=True)
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float64, requires_grad=True)
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim,
        dtype=torch.float64, requires_grad=True)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    p_mat = torch.softmax(scores, dim=-1)
    y_out = torch.matmul(p_mat, v)

    dy = torch.randn_like(y_out)
    y_out.backward(dy)

    with torch.no_grad():
        scores_det = torch.matmul(q, k.transpose(-2, -1)) * scale
        row_max = scores_det.amax(dim=-1, keepdim=True)
        row_sum = torch.exp(scores_det - row_max).sum(dim=-1, keepdim=True)

    sm = torch.zeros(
        batch_size, num_heads, seq_len, 8, dtype=torch.float64)
    sm[:, :, :, 0:1] = row_max
    ss = torch.zeros(
        batch_size, num_heads, seq_len, 8, dtype=torch.float64)
    ss[:, :, :, 0:1] = row_sum

    inputs = AttentionGradInputs(
        q.detach().to(torch.bfloat16),
        k.detach().to(torch.bfloat16),
        v.detach().to(torch.bfloat16),
        dy.detach().to(torch.bfloat16),
        sm.float(), ss.float(),
        y_out.detach().to(torch.bfloat16),
        scale)
    dq_g, dk_g, dv_g = flash_attention_score_grad_golden(inputs)

    for g_name, grad_auto, grad_golden in [
            ("dq", q.grad, dq_g), ("dk", k.grad, dk_g),
            ("dv", v.grad, dv_g)]:
        diff = (
            grad_auto.float() - grad_golden.float()
        ).abs().max().item()
        logger.info("  %s max diff vs autograd: %.6e", g_name, diff)
        if diff >= 5e-2:
            raise ValueError(f"{g_name} max diff {diff} >= 5e-2")

    logger.info("  ✓ Autograd cross-validation passed")


def _validate():
    """自动生成的验证函数"""
    logger.info("=" * 60)
    logger.info("flash_attention_score_grad_golden 验证报告")
    logger.info("=" * 60)

    test_cases = [
        {"name": "Level 0: 最小", "B": 1, "N": 1, "S": 16, "D": 64},
        {"name": "Level 1: 典型", "B": 2, "N": 8, "S": 64, "D": 64},
        {"name": "Level 2: 中等", "B": 2, "N": 8, "S": 128, "D": 128},
    ]

    for tc in test_cases:
        _run_single_test(tc)

    _run_autograd_validation()

    logger.info("\n" + "=" * 60)
    logger.info("验证完成 - 所有测试通过")
    logger.info("=" * 60)


if __name__ == "__main__":
    _validate()
