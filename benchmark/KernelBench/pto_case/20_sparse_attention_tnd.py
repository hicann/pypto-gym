#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: Sparse Flash Attention Forward TND.

Pure PyTorch golden for aclnnSparseFlashAttention (TND format).
DeepSeek V3 MLA architecture: Q/K split into nope+rope parts,
sparse topk_indices select KV subset, GQA with group=N1/N2.
"""

import math
import torch
import torch.nn as nn

FORMULA = "O[t, h, d] = softmax((Q_nope[t, h] @ KV[topk[t]]^T + Q_pe[t, h] @ K_pe[topk[t]]^T) * scale) @ KV[topk[t]]"
DYNAMIC_AXIS = ["T1", "T2", "B"]


class Model(nn.Module):
    def __init__(
        self,
        nq: int = 2,
        n_kv: int = 1,
        kv_lora_rank: int = 512,
        qk_rope_dim: int = 64,
        sparse_size: int = 2048,
        scale: float | None = None,
    ):
        super().__init__()
        self.nq = nq
        self.n_kv = n_kv
        self.kv_lora_rank = kv_lora_rank      # D
        self.qk_rope_dim = qk_rope_dim         # D_ROPE
        self.sparse_size = sparse_size         # K
        self.group = nq // n_kv
        if scale is None:
            self.scale = 1.0 / math.sqrt(kv_lora_rank + qk_rope_dim)
        else:
            self.scale = scale

    def forward(
        self,
        q_nope: torch.Tensor,             # [T1, N1, D] bf16
        compressed_kv_norm: torch.Tensor, # [T2, N2, D] bf16  (acts as K and V)
        topk_indices: torch.Tensor,       # [T1, N2, K] int32
        q_pe: torch.Tensor,                # [T1, N1, D_ROPE] bf16
        k_pe: torch.Tensor,                # [T2, N2, D_ROPE] bf16
        npu_actual_q_len: torch.Tensor,   # [B] int32 (prefix-sum)
        npu_actual_kv_len: torch.Tensor,  # [B] int32 (prefix-sum)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T1, N1, D = q_nope.shape
        T2, N2, _ = compressed_kv_norm.shape
        D_ROPE = q_pe.shape[-1]
        K = topk_indices.shape[-1]
        group = self.group
        device = q_nope.device

        core_attn_out = torch.zeros(T1, N1, D, dtype=torch.bfloat16, device=device)
        softmax_max = torch.full(
            (N2, T1, group), float("-inf"), dtype=torch.float32, device=device
        )
        softmax_sum = torch.zeros(N2, T1, group, dtype=torch.float32, device=device)

        B = npu_actual_q_len.shape[0]
        for b in range(B):
            q_offset = 0 if b == 0 else npu_actual_q_len[b - 1].item()
            kv_offset = 0 if b == 0 else npu_actual_kv_len[b - 1].item()
            s_count = int(npu_actual_q_len[b].item() - q_offset)
            kv_len_for_batch = int(npu_actual_kv_len[b].item() - kv_offset)

            for s in range(s_count):
                t_idx = q_offset + s
                eff_topk = min(kv_len_for_batch - s_count + 1 + s, K)
                if eff_topk <= 0:
                    continue

                for kv_h in range(N2):
                    indices = topk_indices[t_idx, kv_h, :eff_topk].to(torch.long)
                    # Gather KV for sparse indices: [eff_topk, D] / [eff_topk, D_ROPE]
                    kv_sel = compressed_kv_norm[indices, kv_h, :].to(torch.float32)
                    k_pe_sel = k_pe[indices, kv_h, :].to(torch.float32)

                    for g in range(group):
                        q_h = kv_h * group + g
                        # Q slices: [1, D] / [1, D_ROPE]
                        q_n = q_nope[t_idx, q_h, :].unsqueeze(0).to(torch.float32)
                        q_r = q_pe[t_idx, q_h, :].unsqueeze(0).to(torch.float32)

                        # S = (Q_nope @ KV^T + Q_pe @ K_pe^T) * scale  → [1, eff_topk]
                        s_nope = torch.matmul(q_n, kv_sel.T)
                        s_rope = torch.matmul(q_r, k_pe_sel.T)
                        s_score = (s_nope + s_rope) * self.scale

                        # Softmax
                        s_max = s_score.max(dim=-1, keepdim=True).values
                        s_exp = torch.exp(s_score - s_max)
                        s_sum = s_exp.sum(dim=-1, keepdim=True)
                        p = s_exp / s_sum  # [1, eff_topk]

                        # Output: p @ kv_sel  → [1, D]
                        out_val = torch.matmul(p, kv_sel).squeeze(0).to(torch.bfloat16)
                        core_attn_out[t_idx, q_h, :] = out_val
                        softmax_max[kv_h, t_idx, g] = s_max.squeeze()
                        softmax_sum[kv_h, t_idx, g] = s_sum.squeeze()

        return core_attn_out, softmax_max, softmax_sum


def get_inputs():
    B = 1
    N1 = 2   # nq
    N2 = 1   # n_kv
    D = 512  # kv_lora_rank
    DR = 64  # qk_rope_dim
    K = 2048 # sparse_size

    q_lens = [2]
    kv_lens = [4096]

    T1 = sum(q_lens)
    T2 = sum(kv_lens)

    q_nope = torch.randn(T1, N1, D, dtype=torch.bfloat16) / math.sqrt(D)
    compressed_kv_norm = torch.randn(T2, N2, D, dtype=torch.bfloat16) / math.sqrt(D)
    topk_indices = torch.randint(0, T2, (T1, N2, K), dtype=torch.int32)
    q_pe = torch.randn(T1, N1, DR, dtype=torch.bfloat16) / math.sqrt(DR)
    k_pe = torch.randn(T2, N2, DR, dtype=torch.bfloat16) / math.sqrt(DR)
    npu_actual_q_len = (
        torch.tensor(q_lens, dtype=torch.int32).cumsum(dim=0).to(torch.int32)
    )
    npu_actual_kv_len = (
        torch.tensor(kv_lens, dtype=torch.int32).cumsum(dim=0).to(torch.int32)
    )

    return [
        q_nope,
        compressed_kv_norm,
        topk_indices,
        q_pe,
        k_pe,
        npu_actual_q_len,
        npu_actual_kv_len,
    ]


def get_init_inputs():
    return [2, 1, 512, 64, 2048]
