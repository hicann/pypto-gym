#!/usr/bin/env python3
"""Test decode attention kernel."""

import sys, os; _p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')): _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src')); sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import math
import pytest
import torch
import torch_npu  # noqa: F401

from qwen3_1_7b.qwen3_decode_attn import qwen3_decode_attn, Nq, D, SCALE


def _golden(q, k, v):
    Nkv = k.shape[0]
    GROUPS = Nq // Nkv
    out = torch.zeros(Nq, D, dtype=torch.bfloat16, device=q.device)
    for h in range(Nq):
        kv_h = h // GROUPS
        scores = (q[h:h + 1].float() @ k[kv_h].float().t()) * SCALE
        p = torch.softmax(scores, dim=-1)
        out[h:h + 1] = (p @ v[kv_h].float()).to(torch.bfloat16)
    return out


@pytest.mark.parametrize("Skv", [1, 2, 8, 16, 32, 33, 64, 128, 256])
def test_decode_attn(npu_device, Skv):
    torch.manual_seed(Skv)
    Nkv = 8
    S2_TILE = 64
    Skv_p = ((Skv + S2_TILE - 1) // S2_TILE) * S2_TILE
    q = torch.randn(Nq, D, dtype=torch.bfloat16, device=npu_device) * 0.5
    k_pad = torch.zeros(Nkv, Skv_p, D, dtype=torch.bfloat16, device=npu_device)
    v_pad = torch.zeros(Nkv, Skv_p, D, dtype=torch.bfloat16, device=npu_device)
    k = torch.randn(Nkv, Skv, D, dtype=torch.bfloat16, device=npu_device) * 0.5
    v = torch.randn(Nkv, Skv, D, dtype=torch.bfloat16, device=npu_device) * 0.5
    k_pad[:, :Skv] = k
    v_pad[:, :Skv] = v
    mask = torch.zeros(Skv_p, dtype=torch.float32, device=npu_device)
    mask[Skv:] = -1e30
    GROUPS = Nq // Nkv
    mask_3d = mask.view(1, 1, Skv_p).expand(Nkv, GROUPS, Skv_p).contiguous()
    out = torch.empty(Nq, D, dtype=torch.bfloat16, device=npu_device)
    qwen3_decode_attn(q, k_pad, v_pad, mask_3d, out)

    g = _golden(q, k, v)
    diff = (out.float() - g.float()).abs()
    assert diff.max().item() < 5.0, f"max_abs={diff.max().item()}"
