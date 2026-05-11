#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: Sparse Flash Attention Gradient TND.

Pure PyTorch golden for aclnnSparseFlashAttentionGrad (TND format).
Implements the backward pass: given forward intermediate results
(sm_max, sm_sum, out) and output gradient d_out, computes dQ, dK, dV.
"""

import math
import torch
import torch.nn as nn

FORMULA = "dQ, dK, dV = backward of SparseFlashAttention(Q_nope, Q_pe, K_nope, K_pe, V, topk_indices) w.r.t. loss"
DYNAMIC_AXIS = ["T1", "T2", "B"]


class Model(nn.Module):
    def __init__(
        self,
        n_1: int = 2,
        n_2: int = 1,
        d: int = 512,
        dr: int = 64,
        k: int = 2048,
        scale_value: float | None = None,
    ):
        super().__init__()
        self.n_1 = n_1
        self.n_2 = n_2
        self.d = d
        self.dr = dr
        self.k = k
        self.group = n_1 // n_2
        self.d_full = d + dr
        if scale_value is None:
            self.scale = 1.0 / math.sqrt(d + dr)
        else:
            self.scale = scale_value

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        sparse_idx: torch.Tensor,
        d_out: torch.Tensor,
        out: torch.Tensor,
        sm_max: torch.Tensor,
        sm_sum: torch.Tensor,
        actual_seq_qlen: torch.Tensor,
        actual_seq_kvlen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        T1, N1, D = q_nope.shape
        T2, N2, _ = k_nope.shape
        DR = q_pe.shape[-1]
        K = sparse_idx.shape[-1]
        group = self.group
        d_full = self.d_full
        scale = self.scale
        device = q_nope.device

        dq_nope = torch.zeros(T1, N1, D, dtype=torch.float32, device=device)
        dq_pe = torch.zeros(T1, N1, DR, dtype=torch.float32, device=device)
        dk_nope = torch.zeros(T2, N2, D, dtype=torch.float32, device=device)
        dk_pe = torch.zeros(T2, N2, DR, dtype=torch.float32, device=device)
        dv = torch.zeros(T2, N2, D, dtype=torch.float32, device=device)

        B = actual_seq_qlen.shape[0]
        for b in range(B):
            q_offset = 0 if b == 0 else actual_seq_qlen[b - 1].item()
            kv_offset = 0 if b == 0 else actual_seq_kvlen[b - 1].item()
            s_count = int(actual_seq_qlen[b].item() - q_offset)
            kv_len = int(actual_seq_kvlen[b].item() - kv_offset)

            for s in range(s_count):
                t_idx = q_offset + s
                eff_topk = min(kv_len - s_count + 1 + s, K)
                if eff_topk <= 0:
                    continue

                for kv_h in range(N2):
                    indices = sparse_idx[t_idx, kv_h, :eff_topk].to(torch.long)

                    sel_k_nope = k_nope[indices, kv_h, :].to(torch.float32)
                    sel_k_pe = k_pe[indices, kv_h, :].to(torch.float32)
                    sel_v = value[indices, kv_h, :].to(torch.float32)
                    sel_k = torch.cat([sel_k_nope, sel_k_pe], dim=-1)

                    for g in range(group):
                        q_h = kv_h * group + g

                        q_n = q_nope[t_idx, q_h, :].unsqueeze(0).to(torch.float32)
                        q_r = q_pe[t_idx, q_h, :].unsqueeze(0).to(torch.float32)
                        q_full = torch.cat([q_n, q_r], dim=-1)

                        do_g = d_out[t_idx, q_h, :].unsqueeze(0).to(torch.float32)
                        out_g = out[t_idx, q_h, :].unsqueeze(0).to(torch.float32)

                        mi = sm_max[kv_h, t_idx, g]
                        li = sm_sum[kv_h, t_idx, g]

                        # S = Q_full @ sel_k^T * scale
                        s_scores = torch.matmul(q_full, sel_k.T) * scale
                        s_valid = s_scores[0, :eff_topk]
                        p_mat = torch.exp(s_valid - mi) / li

                        # dP = dO @ V^T
                        dp_mat = torch.matmul(do_g, sel_v.T).squeeze(0)
                        # dV_local = P^T @ dO
                        dv_local = torch.matmul(p_mat.unsqueeze(0).T, do_g)

                        # D_val = rowsum(dO * O)
                        d_val = torch.sum(do_g * out_g, dim=-1, keepdim=True)
                        # dS = P ⊙ (dP - D_val)
                        ds_mat = p_mat * (dp_mat - d_val.squeeze())

                        # dQ_local = dS @ sel_k * scale
                        dq_local = torch.matmul(ds_mat.unsqueeze(0), sel_k) * scale
                        # dK_local = dS^T @ Q_full * scale
                        dk_local = torch.matmul(ds_mat.unsqueeze(0).T, q_full) * scale

                        dq_nope[t_idx, q_h, :] += dq_local[0, :D]
                        dq_pe[t_idx, q_h, :] += dq_local[0, D:]
                        dk_nope[indices, kv_h, :] += dk_local[:, :D] + dv_local
                        dk_pe[indices, kv_h, :] += dk_local[:, D:]
                        dv[indices, kv_h, :] += dv_local

        return (
            dq_nope.to(torch.bfloat16),
            dq_pe.to(torch.bfloat16),
            dk_nope.to(torch.bfloat16),
            dk_pe.to(torch.bfloat16),
            dv.to(torch.bfloat16),
        )


def get_inputs():
    B = 1
    N1 = 2
    N2 = 1
    D = 512
    DR = 64
    K = 2048

    actual_q_lens = [2]
    actual_kv_lens = [4096]
    T1 = sum(actual_q_lens)
    T2 = sum(actual_kv_lens)
    G = N1 // N2

    q_nope = torch.randn(T1, N1, D, dtype=torch.bfloat16) / math.sqrt(D)
    q_pe = torch.randn(T1, N1, DR, dtype=torch.bfloat16) / math.sqrt(DR)
    k_nope = torch.randn(T2, N2, D, dtype=torch.bfloat16) / math.sqrt(D)
    k_pe = torch.randn(T2, N2, DR, dtype=torch.bfloat16) / math.sqrt(DR)
    value = k_nope.clone()
    sparse_idx = torch.randint(0, T2, (T1, N2, K), dtype=torch.int32)
    d_out = torch.randn(T1, N1, D, dtype=torch.bfloat16) / math.sqrt(D)
    out = torch.randn(T1, N1, D, dtype=torch.bfloat16) / math.sqrt(D)
    sm_max = torch.randn(N2, T1, G, dtype=torch.float32)
    sm_sum = torch.rand(N2, T1, G, dtype=torch.float32) + 1e-6  # positive
    actual_seq_qlen = (
        torch.tensor(actual_q_lens, dtype=torch.int32).cumsum(dim=0).to(torch.int32)
    )
    actual_seq_kvlen = (
        torch.tensor(actual_kv_lens, dtype=torch.int32).cumsum(dim=0).to(torch.int32)
    )

    return [
        q_nope, q_pe, k_nope, k_pe, value, sparse_idx,
        d_out, out, sm_max, sm_sum,
        actual_seq_qlen, actual_seq_kvlen,
    ]


def get_init_inputs():
    return [2, 1, 512, 64, 2048]
