#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E test for engram_forward — integrated JIT kernel vs golden (FP32).

单头 (HC_MULT=1) Engram 前向。精度容差为 FP32 融合算子的初始值，可在首次 NPU
跑通后按 ops-precision-standard 收紧。
"""

import logging
import os
import sys
import math
from typing import Tuple

_proj_root = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_proj_root, 'src')):
    _proj_root = os.path.dirname(_proj_root)
sys.path.insert(0, os.path.join(_proj_root, 'src'))
sys.path.insert(0, os.path.join(_proj_root, 'src', 'pypto_gym', 'ops', 'pypto_tensor'))

import torch
import torch_npu

from experimental.ops_transformer.engram.engram_forward_impl import (
    pypto_engram_forward,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# (atol, rtol) per output — FP32 初始容差，跑通后可收紧
_OUT_TOL = {
    "value_out":  (5e-2, 1e-2),
    "score_back": (5e-2, 1e-2),
    "key_back":   (5e-2, 1e-2),
    "value_back": (5e-2, 1e-2),
    "gate_back":  (1e-3, 1e-3),
}
_OUT_NAMES = ("value_out", "score_back", "key_back", "value_back", "gate_back")


def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


def linear_torch(tensor, weight, bias):
    """y = tensor @ weight.T + bias"""
    return torch.matmul(tensor, weight.t()) + bias


def rms_norm_torch(x, gamma, epsilon=1e-6):
    # 与 kernel 内 pypto.rms_norm 对齐的 torch 参考实现：
    #   rms = sqrt(mean(x^2)) + eps ;  y = (x / rms) * gamma
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) + epsilon
    return (x / rms) * gamma


def engram_forward_golden(
    hidden_states,        # [B, L, HC_MULT, D]
    embeddings,           # [B, L, D]
    key_proj_weights,     # [HC_MULT, D, D]
    key_proj_bias,        # [HC_MULT, D]
    value_proj_weights,   # [D, D]
    value_proj_bias,      # [D]
    key_gamma,            # [HC_MULT, D]
    sqrt_eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_heads, hidden_dim = hidden_states.shape
    sqrt_hidden_dim = math.sqrt(hidden_dim)

    gates, scores, key_back = [], [], []
    for hc_idx in range(num_heads):
        key = linear_torch(embeddings, key_proj_weights[hc_idx], key_proj_bias[hc_idx])  # [B, L, D]
        key_back.append(key)

        normed_key = rms_norm_torch(key, gamma=key_gamma[hc_idx])                         # [B, L, D]
        normed_query = hidden_states[:, :, hc_idx, :]                                     # [B, L, D]

        score = (normed_key * normed_query).sum(dim=-1) / sqrt_hidden_dim                   # [B, L]
        gate = torch.sqrt(score.abs() + sqrt_eps) * score.sign()                          # sign-sqrt, [B, L]
        gate = gate.sigmoid().unsqueeze(-1)                                               # [B, L, 1]

        gates.append(gate)
        scores.append(score.unsqueeze(-1))                                                # [B, L, 1]

    key_back = torch.stack(key_back, dim=2)                       # [B, L, HC_MULT, D]
    gates = torch.stack(gates, dim=2)                             # [B, L, HC_MULT, 1]
    score_back = torch.cat(scores, dim=2).unsqueeze(-1)           # [B, L, HC_MULT, 1]

    value = linear_torch(embeddings, value_proj_weights, value_proj_bias)  # [B, L, D]
    value_out = gates * value.unsqueeze(2)                                 # [B, L, HC_MULT, D]

    return value_out, score_back, key_back, value, gates


def _make_inputs(device, seed, batch_size, seq_len, num_heads, hidden_dim):
    torch.manual_seed(seed)
    return dict(
        hidden_states=torch.rand(
            [batch_size, seq_len, num_heads, hidden_dim], dtype=torch.float32, device=device),
        embeddings=torch.rand(
            [batch_size, seq_len, hidden_dim], dtype=torch.float32, device=device),
        key_proj_weights=torch.rand(
            [num_heads, hidden_dim, hidden_dim], dtype=torch.float32, device=device),
        key_proj_bias=torch.rand(
            [num_heads, hidden_dim], dtype=torch.float32, device=device),
        value_proj_weights=torch.rand(
            [hidden_dim, hidden_dim], dtype=torch.float32, device=device),
        value_proj_bias=torch.rand(
            [hidden_dim], dtype=torch.float32, device=device),
        key_gamma=torch.rand(
            [num_heads, hidden_dim], dtype=torch.float32, device=device),
    )


def _precision_verify(impl_out, gold_out, tag):
    all_ok = True
    for impl, gold, name in zip(impl_out, gold_out, _OUT_NAMES):
        atol, rtol = _OUT_TOL[name]
        gold_tensor = gold.cpu().float()
        impl_tensor = impl.cpu().float()
        abs_err = (impl_tensor - gold_tensor).abs()
        n_fail = (abs_err > (atol + rtol * gold_tensor.abs())).sum().item()
        max_diff = abs_err.max().item()
        passed = (n_fail == 0)
        logging.info(f"  {tag}_{name}: n={gold_tensor.numel()} fail={n_fail} "
                     f"max_diff={max_diff:.2e} {'PASS' if passed else 'FAIL'} "
                     f"(atol={atol}, rtol={rtol})")
        if not passed:
            all_ok = False
    return all_ok


def _run_case(device, seed, batch_size, seq_len, num_heads, hidden_dim, tag):
    logging.info("=" * 60)
    logging.info(f"{tag} — batch_size={batch_size} seq_len={seq_len} "
                 f"num_heads={num_heads} hidden_dim={hidden_dim}")
    inputs = _make_inputs(device, seed, batch_size, seq_len, num_heads, hidden_dim)

    impl_out = pypto_engram_forward(**inputs)
    gold_out = engram_forward_golden(**inputs)

    ok = _precision_verify(impl_out, gold_out, tag)
    if not ok:
        raise RuntimeError(f"{tag}: precision check FAILED")
    logging.info(f"  {tag} ALL outputs PASS")


def test_engram_forward(device_id):
    # 目标 shape: B=1, L=1024, D=1024（256 对齐，单头 HC_MULT=1）
    torch.npu.set_device(device_id)
    _run_case(f"npu:{device_id}", seed=42, batch_size=1, seq_len=1024,
              num_heads=1, hidden_dim=1024, tag="CASE")


def main():
    device_id = get_device_id()
    torch.npu.set_device(device_id)

    failed = False
    try:
        test_engram_forward(device_id)
        logging.info("\N{white heavy check mark} CASE PASSED")
    except Exception as exc:
        logging.error(f"\N{cross mark} CASE FAILED: {exc}")
        failed = True

    if failed:
        logging.error("[PRECISION_FAIL]")
        sys.exit(1)
    else:
        logging.info("=" * 60)
        logging.info("\N{white heavy check mark} ALL E2E TESTS PASSED")
        logging.info("=" * 60)
        logging.info("[PRECISION_PASS]")


if __name__ == "__main__":
    main()
