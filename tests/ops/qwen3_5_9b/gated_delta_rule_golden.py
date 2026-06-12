#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Pure-torch reference for the gated_delta_rule fused kernel.

Mirrors the chunk_gated_delta_rule algorithm used by Qwen3.5-9B during
prefill. Imported by ``test_gated_delta_rule.py`` for precision comparison.
"""
import torch
import torch.nn.functional as F


def _compute_iterative_inverse(A0, chunk_size):
    """Compute A = (I - A0)^-1 via column-wise update."""
    for i in range(1, chunk_size):
        row = A0[..., i, :i].clone()
        sub = A0[..., :i, :i].clone()
        A0[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return A0 + torch.eye(chunk_size, dtype=A0.dtype, device=A0.device)


def _init_state(initial_state, B, N, D_k, D_v, value):
    """Initialize or reuse the recurrence state tensor."""
    if initial_state is None:
        return torch.zeros(B, N, D_k, D_v, dtype=value.dtype, device=value.device)
    return initial_state.to(value)


def _run_chunk_recurrence(query, key, v_processed, k_cumdecay, g_cum,
                           decay_mask, state, chunks, chunk_size):
    """Run the chunk-wise recurrence loop for gated delta rule attention."""
    attn_out = torch.zeros_like(key)
    attn_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    for i in range(chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], v_processed[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(attn_mask, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ state
        attn_out[:, :, i] = (q_i * g_cum[:, :, i, :, None].exp()) @ state + attn @ v_new
        state = (
            state * g_cum[:, :, i, -1, None, None].exp()
            + (k_i * (g_cum[:, :, i, -1, None] - g_cum[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    return attn_out


def _prepare_chunk_tensors(query, key, value, beta, g, chunk_size):
    """Transpose, pad, scale, reshape inputs and build decay mask + A0 matrix."""
    # [B, S, N, D] -> [B, N, S, D]
    query = query.transpose(1, 2).float()
    key = key.transpose(1, 2).float()
    value = value.transpose(1, 2).float()
    beta = beta.transpose(1, 2).float()
    g = g.transpose(1, 2).float()

    B, N, S, D_k = key.shape
    D_v = value.shape[-1]
    pad_size = (chunk_size - S % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value)]
    beta, g = [F.pad(x, (0, pad_size)) for x in (beta, g)]
    total_S = S + pad_size

    query = query * (1.0 / (D_k ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    chunks = total_S // chunk_size
    query, key, value, k_beta, v_beta = [
        x.reshape(B, N, chunks, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(B, N, chunks, chunk_size)

    g_cum = g.cumsum(dim=-1)
    decay_mask = ((g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp()).tril()

    upper_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    A0 = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(upper_mask, 0.0)

    return (query, key, value, k_beta, v_beta, g_cum, decay_mask, A0,
            B, N, S, D_k, D_v, chunks, total_S)


def chunk_gated_delta_rule_golden(
    query, key, value, *, g, beta,
    chunk_size=128,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    """Torch reference for the chunk gated delta rule.

    Inputs:
      query, key, value: [B, S, Nv, D]
      g:                 [B, S, Nv]   (pre-cumsum, float32)
      beta:              [B, S, Nv]
      initial_state:     [B, Nv, D, D] or None
    Returns:
      core_attn_out: [B, S, Nv, D]
      last_state:    [B, Nv, D, D] or None
    """
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

    (query, key, value, k_beta, v_beta, g_cum, decay_mask, A0,
     B, N, S, D_k, D_v, chunks, total_S) = \
        _prepare_chunk_tensors(query, key, value, beta, g, chunk_size)

    A = _compute_iterative_inverse(A0, chunk_size)
    v_processed = A @ v_beta
    k_cumdecay = A @ (k_beta * g_cum.exp().unsqueeze(-1))

    state = _init_state(initial_state, B, N, D_k, D_v, value)
    attn_out = _run_chunk_recurrence(
        query, key, v_processed, k_cumdecay, g_cum, decay_mask, state, chunks, chunk_size)

    attn_out = attn_out.reshape(B, N, total_S, D_v)
    attn_out = attn_out[:, :, :S].transpose(1, 2).contiguous()

    if not output_final_state:
        return attn_out, None
    return attn_out, state


__all__ = ["chunk_gated_delta_rule_golden"]
