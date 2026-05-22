#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Single-kernel isolated test — only K1 (RMSNorm + QKV proj)."""

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import pytest
import torch
import torch_npu  # noqa: F401

from qwen3_1_7b.qwen3_iter1a_kernel import qwen3_pre_qkv_iter1a, H, Nq, Nkv, D, EPS


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
