#!/usr/bin/env python3
"""K3 only - varying S."""
import sys

import pytest
import torch

sys.path.insert(0, "/data/z00885570/models/Qwen3-1.7B")
from qwen3_pto_kernels.k3_post_attn import qwen3_post_attn_k3, H, INT_SIZE, EPS  # noqa: E402


def _rms_norm_torch(x, w, eps=EPS):
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w.to(torch.float32) * xf * torch.rsqrt(var + eps)).to(in_dtype)


@pytest.mark.parametrize("S", [1, 4, 16, 128])
def test_k3_only(npu_device, S):
    torch.manual_seed(S + 1000)
    attn_in = torch.randn(S, H, dtype=torch.bfloat16, device=npu_device) * 0.5
    x_res = torch.randn(S, H, dtype=torch.bfloat16, device=npu_device) * 0.5
    Wo = torch.randn(H, H, dtype=torch.bfloat16, device=npu_device) * 0.02
    w_pn = torch.randn(H, dtype=torch.bfloat16, device=npu_device)
    Wgate = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wup = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wdown = torch.randn(INT_SIZE, H, dtype=torch.bfloat16, device=npu_device) * 0.02
    y = torch.empty(S, H, dtype=torch.bfloat16, device=npu_device)
    qwen3_post_attn_k3(attn_in, x_res, Wo, w_pn, Wgate, Wup, Wdown, y)

    o = attn_in @ Wo
    h1 = x_res + o
    n2 = _rms_norm_torch(h1, w_pn)
    silu_g = (n2 @ Wgate) * torch.sigmoid(n2 @ Wgate)
    mlp_h = silu_g * (n2 @ Wup)
    g = h1 + (mlp_h @ Wdown)
    max_d = (y.float() - g.float()).abs().max().item()
    assert max_d < 5.0, f"max_abs={max_d}"
