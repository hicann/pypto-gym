#!/usr/bin/env python3
"""Single-kernel isolated test — only K1 (RMSNorm + QKV proj)."""
import pytest
import torch

from pypto_gym.ops.pypto_tile.qwen3_1_7b.qwen3_iter1a_kernel import qwen3_pre_qkv_iter1a, H, Nq, Nkv, D, EPS


def _rms_norm_torch(x, w, eps=EPS):
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w.to(torch.float32) * xf * torch.rsqrt(var + eps)).to(in_dtype)


@pytest.mark.parametrize("S", [1, 4, 16, 128])
def test_k1(npu_device, S):
    torch.manual_seed(S)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=npu_device)
    w_in_norm = torch.randn(H, dtype=torch.bfloat16, device=npu_device)
    Wq = torch.randn(Nq * D, H, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wk = torch.randn(Nkv * D, H, dtype=torch.bfloat16, device=npu_device) * 0.02
    Wv = torch.randn(Nkv * D, H, dtype=torch.bfloat16, device=npu_device) * 0.02
    q = torch.empty(S, Nq * D, dtype=torch.bfloat16, device=npu_device)
    k = torch.empty(S, Nkv * D, dtype=torch.bfloat16, device=npu_device)
    v = torch.empty(S, Nkv * D, dtype=torch.bfloat16, device=npu_device)
    qwen3_pre_qkv_iter1a(x, w_in_norm, Wq, Wk, Wv, q, k, v)

    n1 = _rms_norm_torch(x, w_in_norm)
    qg = torch.nn.functional.linear(n1, Wq)
    max_d = (q.float() - qg.float()).abs().max().item()
    assert max_d < 5.0, f"q max_abs={max_d}"
