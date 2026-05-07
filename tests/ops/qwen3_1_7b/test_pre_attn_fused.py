#!/usr/bin/env python3
"""Test for fused pre-attention kernel (RMSNorm + QKV + Q/K-norm + RoPE)."""
import pytest
import torch

from pypto_gym.ops.pypto_tile.qwen3_1_7b.qwen3_pre_attn_fused import qwen3_pre_attn_fused, H, Nq, Nkv, D, EPS


def _rms_norm_torch(x, w, eps=EPS):
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w.to(torch.float32) * xf * torch.rsqrt(var + eps)).to(in_dtype)


def _rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)


def _make_cs(S, device):
    half = D // 2
    inv = 1.0 / (1000000.0 ** (torch.arange(half, dtype=torch.float32, device=device) / half))
    pos = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(-1)
    emb = torch.cat([pos * inv, pos * inv], dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def _golden(x, cos, sin, w_in, Wq, Wk, Wv, w_qn, w_kn):
    S = x.shape[0]
    n1 = _rms_norm_torch(x, w_in)
    q = torch.nn.functional.linear(n1, Wq).view(S, Nq, D)
    k = torch.nn.functional.linear(n1, Wk).view(S, Nkv, D)
    v = torch.nn.functional.linear(n1, Wv).view(S, Nkv, D)
    q = _rms_norm_torch(q, w_qn)
    k = _rms_norm_torch(k, w_kn)
    cb, sb = cos.unsqueeze(1), sin.unsqueeze(1)
    q = q * cb + _rotate_half(q) * sb
    k = k * cb + _rotate_half(k) * sb
    return q, k, v


@pytest.mark.parametrize("S", [1, 4, 8, 32, 128])
def test_pre_attn_fused(npu_device, S):
    torch.manual_seed(S)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=npu_device)
    cos, sin = _make_cs(S, npu_device)
    w_in = torch.randn(H, dtype=torch.bfloat16, device=npu_device)
    Wq = torch.randn(Nq * D, H, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wk = torch.randn(Nkv * D, H, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wv = torch.randn(Nkv * D, H, dtype=torch.bfloat16, device=npu_device) * 0.02
    w_qn = torch.randn(D, dtype=torch.bfloat16, device=npu_device)
    w_kn = torch.randn(D, dtype=torch.bfloat16, device=npu_device)
    q_out = torch.empty(S, Nq, D, dtype=torch.bfloat16, device=npu_device)
    k_out = torch.empty(S, Nkv, D, dtype=torch.bfloat16, device=npu_device)
    v_out = torch.empty(S, Nkv, D, dtype=torch.bfloat16, device=npu_device)
    qwen3_pre_attn_fused(x, cos, sin, w_in, Wq, Wk, Wv, w_qn, w_kn, q_out, k_out, v_out)

    qg, kg, vg = _golden(x, cos, sin, w_in, Wq, Wk, Wv, w_qn, w_kn)
    for n, o, g in [("q", q_out, qg), ("k", k_out, kg), ("v", v_out, vg)]:
        max_d = (o.float() - g.float()).abs().max().item()
        assert max_d < 5.0, f"{n} max_abs={max_d}"
