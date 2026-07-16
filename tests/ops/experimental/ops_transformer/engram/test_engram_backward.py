#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Engram Backward Operator Test
"""
import logging
import math
import os
import sys

import torch
import torch_npu

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tensor'))

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pypto

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_device_id():
    return int(os.environ.get('TILE_FWK_DEVICE_ID', 0))


# ---------------------------------------------------------------------------
# Helper: RMS-norm + linear backward reference implementations
# ---------------------------------------------------------------------------

def _rms_norm_torch(x, gamma, epsilon=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
    x_norm = x / rms
    if gamma is not None:
        x_norm = x_norm * gamma
    return x_norm


def _rms_norm_backward_golden(dy, x, gamma, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x_hat = x / rms
    d_gamma = (dy * x_hat).sum(dim=(0, 1))
    dx = (gamma / rms) * (dy - x_hat * (dy * x_hat).mean(-1, keepdim=True))
    return dx, d_gamma


def _linear_backward_golden(dy, x, weight):
    dx = dy @ weight.T
    batch_size, seq_len, dim_in = x.shape
    dim_out = dy.shape[2]
    x_flat = x.reshape(batch_size * seq_len, dim_in)
    dy_flat = dy.reshape(batch_size * seq_len, dim_out)
    d_weight = x_flat.T @ dy_flat
    db = dy.sum(dim=(0, 1))
    return dx, d_weight, db


# ---------------------------------------------------------------------------
# Golden reference for the full engram backward pass
# ---------------------------------------------------------------------------

# Note: engram_backward_golden has 11 parameters and 7 return values to match
# the Engram operator's tensor interface; grouping is not applicable (G.FNM.03/05 waived).
def engram_backward_golden(
    grad_out,            # [batch_size, seq_len, num_heads, hidden_dim]
    hidden_states,       # [batch_size, seq_len, num_heads, hidden_dim]
    embeddings,          # [batch_size, seq_len, hidden_dim]
    key_proj_weights,    # [num_heads, hidden_dim, hidden_dim]
    value_proj_weights,  # [hidden_dim, hidden_dim]
    key_gamma,           # [num_heads, hidden_dim]
    query_gamma,         # [num_heads, hidden_dim]
    key_lineared,        # [batch_size, seq_len, num_heads, hidden_dim]
    value_lineared,      # [batch_size, seq_len, hidden_dim]
    gate_back,           # [batch_size, seq_len, num_heads, 1]
    score_output,        # [batch_size, seq_len, num_heads, 1]
):
    batch_size, seq_len, num_heads, hidden_dim = hidden_states.shape
    sqrt_hidden_dim = math.sqrt(hidden_dim)
    sqrt_eps = 1e-4

    d_hidden = torch.zeros_like(hidden_states)
    d_embeddings = torch.zeros_like(embeddings)
    d_key_w = torch.zeros_like(key_proj_weights)
    d_key_b = torch.zeros(num_heads, hidden_dim, device=embeddings.device)
    d_val_w = torch.zeros_like(value_proj_weights)
    d_val_b = torch.zeros(hidden_dim, device=embeddings.device)
    d_key_gamma = torch.zeros_like(key_gamma)

    gates = gate_back                    # [batch_size, seq_len, num_heads, 1]
    scores = score_output.squeeze(-1)   # [batch_size, seq_len, num_heads]

    # Step 1: value / gate split
    d_value = (grad_out * gates).sum(dim=2)          # [batch_size, seq_len, hidden_dim]
    d_gates = grad_out * value_lineared.unsqueeze(2)  # [batch_size, seq_len, num_heads, hidden_dim]

    dx_val, d_weight_val, db_val = _linear_backward_golden(d_value, embeddings, value_proj_weights)
    d_embeddings += dx_val
    d_val_w += d_weight_val
    d_val_b += db_val

    # Step 2: per-head backward
    for hc_idx in range(num_heads):
        key = key_lineared[:, :, hc_idx, :]     # [batch_size, seq_len, hidden_dim]
        query = hidden_states[:, :, hc_idx, :]  # [batch_size, seq_len, hidden_dim]
        gate = gates[:, :, hc_idx, :]           # [batch_size, seq_len, 1]
        score = scores[:, :, hc_idx]            # [batch_size, seq_len]

        d_gate = d_gates[:, :, hc_idx, :]       # [batch_size, seq_len, hidden_dim]
        dz = d_gate * gate * (1 - gate)
        dz_sum = dz.sum(dim=-1)                 # [batch_size, seq_len]
        abs_score = score.abs() + sqrt_eps
        df_ds = 0.5 / torch.sqrt(abs_score)
        d_score = dz_sum * df_ds                # [batch_size, seq_len]

        normed_key = _rms_norm_torch(key, gamma=key_gamma[hc_idx])
        normed_query = _rms_norm_torch(query, gamma=query_gamma[hc_idx])

        d_query_normed = d_score.unsqueeze(-1) * normed_key / sqrt_hidden_dim
        d_key_normed = d_score.unsqueeze(-1) * normed_query / sqrt_hidden_dim

        d_key, dk_gamma = _rms_norm_backward_golden(d_key_normed, key, key_gamma[hc_idx])
        d_hidden[:, :, hc_idx, :] += d_query_normed
        d_key_gamma[hc_idx] += dk_gamma

        dx_key, d_weight_key, db_key = _linear_backward_golden(
            d_key, embeddings, key_proj_weights[hc_idx])
        d_embeddings += dx_key
        d_key_w[hc_idx] += d_weight_key
        d_key_b[hc_idx] += db_key

    return d_hidden, d_embeddings, d_key_w, d_key_b, d_val_w, d_val_b, d_key_gamma


# ---------------------------------------------------------------------------
# Precision comparison utility
# ---------------------------------------------------------------------------

def _log_outlier_rows(idx, v1, v2, od, ord_, n_show):
    logging.info("-" * 80)
    logging.info(f"{'Index':<20} {'Actual':<15} {'Golden':<15} {'AbsDiff':<12} {'RelDiff':<12}")
    logging.info("-" * 80)
    for i in range(n_show):
        idx_str = str(tuple(idx[j][i].item() for j in range(len(idx))))
        logging.info(
            f"{idx_str:<20} {v1[i].item():<15.6f} {v2[i].item():<15.6f} "
            f"{od[i].item():<12.6f} {ord_[i].item():<12.6f}"
        )


# Note: detailed_tensor_compare has 6 parameters for full control over comparison behavior;
# grouping tolerance params into a dataclass is not warranted here (G.FNM.03 waived).
def detailed_tensor_compare(tensor1, tensor2, rtol=1e-3, atol=1e-3,
                             verbose=True, max_outliers_display=20):
    t1, t2 = tensor1.cpu().float(), tensor2.cpu().float()
    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)
    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_mask = ~tolerance_mask

    total = t1.numel()
    n_out = out_mask.sum().item()
    ratio = n_out / total

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()

    if n_out > 0:
        out_diff = diff[out_mask]
        max_out = out_diff.max().item()
        mean_out = out_diff.mean().item()
        idx = torch.nonzero(out_mask, as_tuple=True)
        v1 = t1[out_mask]
        v2 = t2[out_mask]
        od = diff[out_mask]
        ord_ = relative_diff[out_mask]
        sort_idx = torch.argsort(od, descending=True)
        idx = tuple(i[sort_idx] for i in idx)
        v1, v2, od, ord_ = v1[sort_idx], v2[sort_idx], od[sort_idx], ord_[sort_idx]
    else:
        max_out = mean_out = 0.0
        idx = v1 = v2 = od = ord_ = None

    all_close = n_out == 0

    if verbose:
        logging.info("\n" + "=" * 60)
        logging.info("Detailed tensor comparison report")
        logging.info("=" * 60)
        logging.info(f"Total elements: {total:,}")
        logging.info(f"Out of tolerance: {n_out:,} ({ratio * 100:.4f}%)")
        logging.info(f"Max diff: {max_diff:.6f}  Mean diff: {mean_diff:.6f}  Std: {std_diff:.6f}")
        logging.info(f"rtol={rtol}, atol={atol}")
        if n_out > 0:
            logging.info(f"Max out-of-tol diff: {max_out:.6f}  Mean: {mean_out:.6f}")
            n_show = min(max_outliers_display, n_out)
            logging.info(f"\nTop-{n_show} outliers:")
            _log_outlier_rows(idx, v1, v2, od, ord_, n_show)
            if n_out > max_outliers_display:
                logging.info(f"... {n_out - max_outliers_display} more not shown.")
        logging.info(f"\nTensor match: {all_close}")
        logging.info("=" * 60)

    return all_close


# ---------------------------------------------------------------------------
# Test function
# ---------------------------------------------------------------------------

def test_engram_backward(device_id, batch_size=1, seq_len=8192, hidden_dim=1024, num_heads=1):
    from experimental.ops_transformer.engram.engram_backward_impl import engram_backward_pto

    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    torch.manual_seed(42)
    np.random.seed(42)

    logging.info(
        f"\n=== Engram Backward Test batch_size={batch_size} seq_len={seq_len} "
        f"hidden_dim={hidden_dim} num_heads={num_heads} ==="
    )

    grad_out = torch.randn(batch_size, seq_len, num_heads, hidden_dim,
                           dtype=torch.float32, device=device)
    hidden_states = torch.randn(batch_size, seq_len, num_heads, hidden_dim,
                                dtype=torch.float32, device=device)
    embeddings = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
    key_w = torch.rand(num_heads, hidden_dim, hidden_dim, dtype=torch.float32, device=device)
    value_w = torch.rand(hidden_dim, hidden_dim, dtype=torch.float32, device=device)
    key_gamma = torch.rand(num_heads, hidden_dim, dtype=torch.float32, device=device)
    query_gamma = torch.rand(num_heads, hidden_dim, dtype=torch.float32, device=device)
    key_lineared = torch.randn(batch_size, seq_len, num_heads, hidden_dim,
                               dtype=torch.float32, device=device)
    value_lineared = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=device)
    gate_back = torch.rand(batch_size, seq_len, num_heads, 1, dtype=torch.float32, device=device)
    score_output = torch.randn(batch_size, seq_len, num_heads, 1,
                               dtype=torch.float32, device=device)

    inputs = [
        grad_out, hidden_states, embeddings,
        key_w, value_w, key_gamma, query_gamma,
        key_lineared, value_lineared, gate_back, score_output,
    ]
    inputs_cloned = [t.clone() for t in inputs]

    golden_outputs = engram_backward_golden(*inputs_cloned)
    logging.info("golden done")

    goldens_cpu = [None] * len(inputs) + [g.cpu() for g in golden_outputs]
    pypto.set_verify_golden_data(goldens=goldens_cpu)

    pto_outputs = engram_backward_pto(*inputs)

    output_names = [
        "d_hidden", "d_embeddings", "d_key_w", "d_key_b",
        "d_value_w", "d_value_b", "d_key_gamma",
    ]
    all_pass = True
    for name, golden, pto in zip(output_names, golden_outputs, pto_outputs):
        logging.info(f"--- {name}")
        ok = detailed_tensor_compare(pto, golden, rtol=1e-3, atol=1e-3)
        assert_allclose(
            pto.cpu().float().numpy(),
            golden.cpu().float().numpy(),
            rtol=1e-3, atol=1e-3,
            err_msg=f"Mismatch in {name}",
        )
        if not ok:
            all_pass = False

    assert all_pass, "One or more outputs failed precision check"
    logging.info("=== PASSED ===")


@pytest.mark.parametrize("batch_size,seq_len,hidden_dim,num_heads", [
    (1, 8192, 1024, 1),
    (1, 1024, 512, 1),
    (1, 2048, 768, 1),
])
def test_engram_backward_pytest(batch_size, seq_len, hidden_dim, num_heads):
    device_id = get_device_id()
    test_engram_backward(device_id,
                         batch_size=batch_size,
                         seq_len=seq_len,
                         hidden_dim=hidden_dim,
                         num_heads=num_heads)


def main():
    device_id = get_device_id()
    torch.npu.set_device(device_id)
    test_engram_backward(device_id, batch_size=1, seq_len=8192, hidden_dim=1024, num_heads=1)


if __name__ == "__main__":
    main()
