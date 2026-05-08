#!/usr/bin/env python3
"""K2 only - Q variant."""

import sys, os; _p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')): _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src')); sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import pytest
import torch
import torch_npu  # noqa: F401

from qwen3_1_7b.qwen3_k2_qk_rope import qwen3_qk_rope_q, D, EPS

Nq = 16


def _rms_norm_torch(x, w, eps=EPS):
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w.to(torch.float32) * xf * torch.rsqrt(var + eps)).to(in_dtype)


def _rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)


def _make_cos_sin(S, dtype, device):
    half = D // 2
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(-1)
    freqs = pos * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


@pytest.mark.parametrize("S", [1, 4, 16, 128])
def test_k2_q(npu_device, S):
    torch.manual_seed(S)
    q = torch.randn(S, Nq, D, dtype=torch.bfloat16, device=npu_device)
    cos, sin = _make_cos_sin(S, torch.bfloat16, npu_device)
    w = torch.randn(D, dtype=torch.bfloat16, device=npu_device)
    out = torch.empty(S, Nq, D, dtype=torch.bfloat16, device=npu_device)
    qwen3_qk_rope_q(q, cos, sin, w, out)

    n = _rms_norm_torch(q, w)
    g = (n * cos.unsqueeze(1)) + (_rotate_half(n) * sin.unsqueeze(1))
    max_d = (out.float() - g.float()).abs().max().item()
    assert max_d < 0.5, f"max_abs={max_d}"
