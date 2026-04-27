#!/usr/bin/env python3
"""K3 precision check: post-attention fused block."""
import pytest
import torch

from qwen3_k3_post_attn import qwen3_post_attn_k3, H, INT_SIZE, EPS


def _rms_norm_torch(x, w, eps=EPS):
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w.to(torch.float32) * xf * torch.rsqrt(var + eps)).to(in_dtype)


def _golden(attn_in, x_res, Wo, w_post_norm, Wgate, Wup, Wdown):
    o = attn_in @ Wo
    h1 = x_res + o
    n2 = _rms_norm_torch(h1, w_post_norm)
    gate = n2 @ Wgate
    up = n2 @ Wup
    silu_g = gate * torch.sigmoid(gate)
    mlp_h = silu_g * up
    down = mlp_h @ Wdown
    return h1 + down


@pytest.mark.parametrize("S", [32, 128])
def test_k3(npu_device, S):
    torch.manual_seed(0)
    attn_in = torch.randn(S, H, dtype=torch.bfloat16, device=npu_device) * 0.5
    x_res = torch.randn(S, H, dtype=torch.bfloat16, device=npu_device) * 0.5
    Wo = torch.randn(H, H, dtype=torch.bfloat16, device=npu_device) * 0.02
    w_post_norm = torch.randn(H, dtype=torch.bfloat16, device=npu_device)
    Wgate = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wup = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wdown = torch.randn(INT_SIZE, H, dtype=torch.bfloat16, device=npu_device) * 0.02

    y = torch.empty(S, H, dtype=torch.bfloat16, device=npu_device)
    qwen3_post_attn_k3(attn_in, x_res, Wo, w_post_norm, Wgate, Wup, Wdown, y)

    g = _golden(attn_in, x_res, Wo, w_post_norm, Wgate, Wup, Wdown)
    max_d = (y.float() - g.float()).abs().max().item()
    assert max_d < 1.0, f"max_abs={max_d}"
