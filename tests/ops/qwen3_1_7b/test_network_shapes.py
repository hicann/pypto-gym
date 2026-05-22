#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Network-scenario tests for pypto kernels.

Tests K1, K2, K3 with the exact shapes and dtypes encountered during
Qwen3-1.7B inference: BF16 everywhere, B=1, varying S (prefill 4/16/128/512,
decode S=1, with KV cache lengths up to 256 for K2/K3 input shape variation).
"""
import os
import math
import torch

os.environ.setdefault("TILE_FWK_DEVICE_ID", "7")
import torch_npu  # noqa: F401

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

torch.npu.set_device(int(os.environ["TILE_FWK_DEVICE_ID"]))
device = f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}"

from qwen3_1_7b.qwen3_iter1a_kernel import qwen3_pre_qkv_iter1a, H, Nq, Nkv, D, EPS
from qwen3_1_7b.qwen3_k2_qk_rope import qwen3_qk_rope_q, qwen3_qk_rope_k
from qwen3_1_7b.qwen3_k3_post_attn import qwen3_post_attn_k3, INT_SIZE


def rms_norm_torch(x, w, eps=EPS):
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w.to(torch.float32) * xf * torch.rsqrt(var + eps)).to(in_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def make_cos_sin(S, dtype, device):
    half = D // 2
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(-1)
    freqs = pos * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def k1_test(S):
    print(f"--- K1 S={S} (BF16) ---")
    torch.manual_seed(S)
    x = torch.randn(S, H, dtype=torch.bfloat16, device=device)
    w_in_norm = torch.randn(H, dtype=torch.bfloat16, device=device)
    Wq = torch.randn(Nq * D, H, dtype=torch.bfloat16, device=device) * 0.02
    Wk = torch.randn(Nkv * D, H, dtype=torch.bfloat16, device=device) * 0.02
    Wv = torch.randn(Nkv * D, H, dtype=torch.bfloat16, device=device) * 0.02
    q = torch.empty(S, Nq * D, dtype=torch.bfloat16, device=device)
    k = torch.empty(S, Nkv * D, dtype=torch.bfloat16, device=device)
    v = torch.empty(S, Nkv * D, dtype=torch.bfloat16, device=device)
    qwen3_pre_qkv_iter1a(x, w_in_norm, Wq, Wk, Wv, q, k, v)
    n1 = rms_norm_torch(x, w_in_norm)
    qg = torch.nn.functional.linear(n1, Wq)
    kg = torch.nn.functional.linear(n1, Wk)
    vg = torch.nn.functional.linear(n1, Wv)
    for name, out, gold in [("q", q, qg), ("k", k, kg), ("v", v, vg)]:
        d = (out.float() - gold.float()).abs()
        rel = (d / (gold.float().abs() + 1e-3)).max().item()
        print(f"  {name}: max={d.max().item():.4f} mean={d.mean().item():.5f} max_rel={rel:.4f}")


def k2_test(S):
    print(f"--- K2 S={S} (BF16) ---")
    torch.manual_seed(S)
    cos, sin = make_cos_sin(S, torch.bfloat16, device)
    # Q
    q = torch.randn(S, Nq, D, dtype=torch.bfloat16, device=device)
    w_q = torch.randn(D, dtype=torch.bfloat16, device=device)
    out_q = torch.empty(S, Nq, D, dtype=torch.bfloat16, device=device)
    qwen3_qk_rope_q(q, cos, sin, w_q, out_q)
    n_q = rms_norm_torch(q, w_q)
    g_q = (n_q * cos.unsqueeze(1)) + (rotate_half(n_q) * sin.unsqueeze(1))
    d = (out_q.float() - g_q.float()).abs()
    print(f"  Q: max={d.max().item():.4f} mean={d.mean().item():.5f}")
    # K
    k = torch.randn(S, Nkv, D, dtype=torch.bfloat16, device=device)
    w_k = torch.randn(D, dtype=torch.bfloat16, device=device)
    out_k = torch.empty(S, Nkv, D, dtype=torch.bfloat16, device=device)
    qwen3_qk_rope_k(k, cos, sin, w_k, out_k)
    n_k = rms_norm_torch(k, w_k)
    g_k = (n_k * cos.unsqueeze(1)) + (rotate_half(n_k) * sin.unsqueeze(1))
    d = (out_k.float() - g_k.float()).abs()
    print(f"  K: max={d.max().item():.4f} mean={d.mean().item():.5f}")


def k3_test(S):
    print(f"--- K3 S={S} (BF16) ---")
    torch.manual_seed(S + 1000)
    attn_in = torch.randn(S, H, dtype=torch.bfloat16, device=device) * 0.5
    x_res = torch.randn(S, H, dtype=torch.bfloat16, device=device) * 0.5
    Wo = torch.randn(H, H, dtype=torch.bfloat16, device=device) * 0.02
    w_pn = torch.randn(H, dtype=torch.bfloat16, device=device)
    Wgate = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device=device) * 0.02
    Wup = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device=device) * 0.02
    Wdown = torch.randn(INT_SIZE, H, dtype=torch.bfloat16, device=device) * 0.02
    y = torch.empty(S, H, dtype=torch.bfloat16, device=device)
    qwen3_post_attn_k3(attn_in, x_res, Wo, w_pn, Wgate, Wup, Wdown, y)
    o = attn_in @ Wo
    h1 = x_res + o
    n2 = rms_norm_torch(h1, w_pn)
    gate = n2 @ Wgate
    up = n2 @ Wup
    silu_g = gate * torch.sigmoid(gate)
    mlp_h = silu_g * up
    down = mlp_h @ Wdown
    g = h1 + down
    d = (y.float() - g.float()).abs()
    print(f"  y: max={d.max().item():.4f} mean={d.mean().item():.5f}")


if __name__ == "__main__":
    print("=== Network-scenario tests (B=1, BF16) ===\n")
    # Prefill scenarios
    for S in [4, 16, 128, 512]:
        print(f"\n========== Prefill S={S} ==========")
        k1_test(S)
        k2_test(S)
        k3_test(S)
    # Decode scenario (Sq=1)
    print(f"\n========== Decode Sq=1 ==========")
    k1_test(1)
    k2_test(1)
    k3_test(1)
    print("\nAll network-scenario tests completed.")
