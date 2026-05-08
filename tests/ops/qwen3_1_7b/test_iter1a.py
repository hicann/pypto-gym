#!/usr/bin/env python3
"""Iter 1a precision check: kernel vs torch golden."""

import sys, os; _p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')): _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src')); sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import pytest
import torch
import torch_npu  # noqa: F401

from qwen3_1_7b.qwen3_iter1a_kernel import qwen3_pre_qkv_iter1a, H, Nq, Nkv, D, EPS


def _rms_norm_torch(x, w, eps=EPS):
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w.to(torch.float32) * xf * torch.rsqrt(var + eps)).to(in_dtype)


@pytest.mark.parametrize("S", [32, 128])
def test_iter1a(npu_device, S):
    torch.manual_seed(0)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=npu_device)
    w_in_norm = torch.randn(H, dtype=torch.bfloat16, device=npu_device)
    Wq = torch.randn(Nq * D, H, dtype=torch.bfloat16, device=npu_device)
    Wk = torch.randn(Nkv * D, H, dtype=torch.bfloat16, device=npu_device)
    Wv = torch.randn(Nkv * D, H, dtype=torch.bfloat16, device=npu_device)
    q_out = torch.empty(S, Nq * D, dtype=torch.bfloat16, device=npu_device)
    k_out = torch.empty(S, Nkv * D, dtype=torch.bfloat16, device=npu_device)
    v_out = torch.empty(S, Nkv * D, dtype=torch.bfloat16, device=npu_device)

    qwen3_pre_qkv_iter1a(x, w_in_norm, Wq, Wk, Wv, q_out, k_out, v_out)

    n1 = _rms_norm_torch(x, w_in_norm)
    q_g = torch.nn.functional.linear(n1, Wq)
    k_g = torch.nn.functional.linear(n1, Wk)
    v_g = torch.nn.functional.linear(n1, Wv)

    for name, out, gold in [("q", q_out, q_g), ("k", k_out, k_g), ("v", v_out, v_g)]:
        max_d = (out.float() - gold.float()).abs().max().item()
        assert max_d < 5.0, f"{name} max_abs={max_d}"
