#!/usr/bin/env python3
"""Iter 1b precision check."""
import pytest
import torch

from pypto_gym.ops.pypto_tile.qwen3_1_7b.qwen3_iter1b_kernel import qwen3_pre_attn_iter1b, H, Nq, Nkv, D, EPS


def _rms_norm_torch(x, w, eps=EPS):
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w.to(torch.float32) * xf * torch.rsqrt(var + eps)).to(in_dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _make_cos_sin(S, dtype, device):
    half = D // 2
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(-1)
    freqs = pos * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _golden(x, cos, sin, w_in_norm, Wq, Wk, Wv, w_q_norm, w_k_norm):
    S = x.shape[0]
    n1 = _rms_norm_torch(x, w_in_norm)
    q = (n1 @ Wq).view(S, Nq, D)
    k = (n1 @ Wk).view(S, Nkv, D)
    v = n1 @ Wv
    q = _rms_norm_torch(q, w_q_norm)
    k = _rms_norm_torch(k, w_k_norm)
    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    q = (q * cos_b) + (_rotate_half(q) * sin_b)
    k = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q.view(S, Nq * D), k.view(S, Nkv * D), v


@pytest.mark.parametrize("S", [32, 128])
def test_iter1b(npu_device, S):
    torch.manual_seed(0)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=npu_device)
    cos, sin = _make_cos_sin(S, torch.bfloat16, npu_device)
    w_in_norm = torch.randn(H, dtype=torch.bfloat16, device=npu_device)
    Wq = torch.randn(H, Nq * D, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wk = torch.randn(H, Nkv * D, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wv = torch.randn(H, Nkv * D, dtype=torch.bfloat16, device=npu_device) * 0.02
    w_q_norm = torch.randn(D, dtype=torch.bfloat16, device=npu_device)
    w_k_norm = torch.randn(D, dtype=torch.bfloat16, device=npu_device)

    q_out = torch.empty(S, Nq * D, dtype=torch.bfloat16, device=npu_device)
    k_out = torch.empty(S, Nkv * D, dtype=torch.bfloat16, device=npu_device)
    v_out = torch.empty(S, Nkv * D, dtype=torch.bfloat16, device=npu_device)
    qwen3_pre_attn_iter1b(x, cos, sin, w_in_norm, Wq, Wk, Wv,
                          w_q_norm, w_k_norm, q_out, k_out, v_out)

    q_g, k_g, v_g = _golden(x, cos, sin, w_in_norm, Wq, Wk, Wv, w_q_norm, w_k_norm)

    for name, out, gold in [("q", q_out, q_g), ("k", k_out, k_g), ("v", v_out, v_g)]:
        max_d = (out.float() - gold.float()).abs().max().item()
        assert max_d < 1.0, f"{name} max_abs={max_d}"
