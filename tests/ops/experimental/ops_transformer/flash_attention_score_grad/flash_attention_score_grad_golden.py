# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
"""
FlashAttentionScoreGrad Golden 参考实现

反向公式:
  p  = exp(Q @ K^T * scale - softmax_max) / softmax_sum   (online softmax 重算)
  d  = sum(dY * attention_out, dim=-1, keepdim=True)
  dp = dY @ V^T
  ds = p * (dp - d)
  dq = ds @ K * scale
  dk = ds^T @ Q * scale
  dv = p^T @ dY

与 PyPTO kernel (S_TILE=128) 计算流完全对齐。


========================================================
## 正向数据生成 (测试辅助)
========================================================
"""

from dataclasses import dataclass
from typing import Tuple
import torch

S_TILE = 128


@dataclass
class ForwardDataConfig:
    """前向数据生成参数。"""

    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
    device: str = 'cpu'


@dataclass
class ForwardDataResult:
    """前向数据生成结果。"""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dy: torch.Tensor
    softmax_max: torch.Tensor
    softmax_sum: torch.Tensor
    attention_out: torch.Tensor
    scale: float


_BLOCK_Q = 32
_BLOCK_KV = 64


def _compute_forward(q, k, v, scale, cfg: ForwardDataConfig):
    """在线 softmax 前向计算，与 flash_attention_score kernel 对齐。"""
    B, N, S, D = cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim

    softmax_max = torch.zeros(B, N, S, 8, dtype=torch.float32, device=cfg.device)
    softmax_sum = torch.zeros(B, N, S, 8, dtype=torch.float32, device=cfg.device)
    attention_out = torch.zeros(B, N, S, D, dtype=torch.bfloat16, device=cfg.device)

    n_kv = (S + _BLOCK_KV - 1) // _BLOCK_KV
    n_q = (S + _BLOCK_Q - 1) // _BLOCK_Q

    for b in range(B):
        for n in range(N):
            q_bn, k_bn, v_bn = q[b, n], k[b, n], v[b, n]
            for qi in range(n_q):
                qs = qi * _BLOCK_Q
                qn = min(_BLOCK_Q, S - qs)
                q2d = q_bn[qs:qs + qn].reshape(qn, D)
                mi = torch.full((qn, 1), float('-inf'), dtype=torch.float32, device=cfg.device)
                li = torch.zeros(qn, 1, dtype=torch.float32, device=cfg.device)
                oi = torch.zeros(qn, D, dtype=torch.float32, device=cfg.device)
                for ki in range(n_kv):
                    ks = ki * _BLOCK_KV
                    kn = min(_BLOCK_KV, S - ks)
                    k2d = k_bn[ks:ks + kn].reshape(kn, D)
                    scores = (q2d.float() @ k2d.float().T) * scale
                    mij = scores.amax(dim=-1, keepdim=True)
                    pij = torch.exp(scores - mij)
                    lij = pij.sum(dim=-1, keepdim=True)
                    v2d = v_bn[ks:ks + kn].reshape(kn, D).float()
                    oij = pij @ v2d
                    if ki == 0:
                        mi, li, oi = mij, lij, oij
                    else:
                        mnew = torch.maximum(mi, mij)
                        alpha = torch.exp(mi - mnew)
                        beta = torch.exp(mij - mnew)
                        li = alpha * li + beta * lij
                        oi = alpha * oi + beta * oij
                        mi = mnew
                attention_out[b, n, qs:qs + qn] = (oi / li).to(torch.bfloat16)
                softmax_max[b, n, qs:qs + qn, 0] = mi.squeeze(-1)
                softmax_sum[b, n, qs:qs + qn, 0] = li.squeeze(-1)

    return attention_out, softmax_max, softmax_sum


def generate_forward_data(cfg: ForwardDataConfig) -> ForwardDataResult:
    """生成随机输入 + 前向中间结果，供反向测试。"""
    torch.manual_seed(42)
    scale = 1.0 / (cfg.head_dim ** 0.5)

    q = torch.randn(cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
                    dtype=cfg.dtype, device=cfg.device)
    k = torch.randn(cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
                    dtype=cfg.dtype, device=cfg.device)
    v = torch.randn(cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
                    dtype=cfg.dtype, device=cfg.device)
    dy = torch.randn(cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim,
                     dtype=cfg.dtype, device=cfg.device)

    attention_out, softmax_max, softmax_sum = _compute_forward(q, k, v, scale, cfg)

    return ForwardDataResult(q, k, v, dy, softmax_max, softmax_sum, attention_out, scale)


# ========================================================
# 反向 Golden
# ========================================================


class AttentionGradInputs:
    """反向 golden 输入容器。"""

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
    FlashAttentionScoreGrad 参考实现 (分块流式 softmax, 反向)

    与 PyPTO kernel 计算流完全对齐，S_TILE=128 分块。

    输入: Q, K, V, dY (BF16), softmax_max/sum (FP32), attention_out (BF16)
    输出: dQ, dK, dV (BF16)
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
            q_bn = q[b, n]
            k_bn = k[b, n]
            v_bn = v[b, n]
            dy_bn = dy[b, n]
            attn_bn = attn_out[b, n]
            smax_bn = s_max[b, n]
            ssum_bn = s_sum[b, n]

            # ===== 趟1: 计算 dQ =====
            for s1_start in range(0, S, S_TILE):
                s1_end = min(s1_start + S_TILE, S)

                q_i = q_bn[s1_start:s1_end]
                dy_i = dy_bn[s1_start:s1_end]
                attn_i = attn_bn[s1_start:s1_end]
                smax_i = smax_bn[s1_start:s1_end]
                ssum_i = ssum_bn[s1_start:s1_end]

                d_i = (dy_i * attn_i).float().sum(dim=-1, keepdim=True)

                dq_acc = None
                for s2_start in range(0, S, S_TILE):
                    s2_end = min(s2_start + S_TILE, S)

                    k_j = k_bn[s2_start:s2_end]
                    v_j = v_bn[s2_start:s2_end]

                    scores = (q_i.float() @ k_j.float().T) * scale
                    p_ij = torch.exp(scores - smax_i) / ssum_i
                    dp_ij = dy_i.float() @ v_j.float().T
                    ds_ij = p_ij * (dp_ij - d_i)

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

                k_j = k_bn[s2_start:s2_end]
                v_j = v_bn[s2_start:s2_end]

                dk_acc = None
                dv_acc = None
                for s1_start in range(0, S, S_TILE):
                    s1_end = min(s1_start + S_TILE, S)

                    q_i = q_bn[s1_start:s1_end]
                    dy_i = dy_bn[s1_start:s1_end]
                    attn_i = attn_bn[s1_start:s1_end]
                    smax_i = smax_bn[s1_start:s1_end]
                    ssum_i = ssum_bn[s1_start:s1_end]

                    d_i = (dy_i * attn_i).float().sum(dim=-1, keepdim=True)

                    scores = (q_i.float() @ k_j.float().T) * scale
                    p_ij = torch.exp(scores - smax_i) / ssum_i
                    dp_ij = dy_i.float() @ v_j.float().T
                    ds_ij = p_ij * (dp_ij - d_i)

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
