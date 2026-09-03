# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical test for the kimi_linear_48b_a3b KDA decode PyPTO kernel.

Compares the fused decode kernel against the composed torch golden from
``kimi_fla_compat`` (ShortConvolution -> fused_kda_gate -> _naive_recurrent_kda
-> FusedRMSNormGated -> o_proj).
"""

import logging
import os

import numpy as np
import torch

from _kda_test_common import _HAS_NPU, torch_npu, HEAD_DIM, RTOL, ATOL

from kimi_fla_compat import (
    FusedRMSNormGated,
    ShortConvolution,
    _naive_recurrent_kda,
    fused_kda_gate,
)
from kda_fused_decode_impl import (
    CONV_K,
    kda_fused_decode,
    make_fused_buffers,
    prepare_kda_fused_weights,
    seed_fused_buffers,
    KdaBufferParams,
)
from kda_decode_graph import pack_conv_state


HIDDEN = 2304
NUM_HEADS = 32
PROJ = NUM_HEADS * HEAD_DIM
RMS_EPS = 1e-5
DTYPE = torch.bfloat16


class _RefLayer(torch.nn.Module):
    """Minimal stand-in for KimiDeltaAttention holding just the KDA weights."""

    def __init__(self, device):
        super().__init__()
        self.hidden_size = HIDDEN
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        self.conv_size = CONV_K

        def lin(i, o):
            return torch.nn.Linear(i, o, bias=False)

        self.q_proj, self.k_proj, self.v_proj = (lin(HIDDEN, PROJ),
                                                 lin(HIDDEN, PROJ),
                                                 lin(HIDDEN, PROJ))
        self.f_a_proj, self.f_b_proj = lin(HIDDEN, HEAD_DIM), lin(HEAD_DIM, PROJ)
        self.g_a_proj, self.g_b_proj = lin(HIDDEN, HEAD_DIM), lin(HEAD_DIM, PROJ)
        self.b_proj = lin(HIDDEN, NUM_HEADS)
        self.o_proj = lin(PROJ, HIDDEN)

        self.q_conv1d = ShortConvolution(PROJ, CONV_K, activation="silu")
        self.k_conv1d = ShortConvolution(PROJ, CONV_K, activation="silu")
        self.v_conv1d = ShortConvolution(PROJ, CONV_K, activation="silu")
        for c in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            torch.nn.init.normal_(c.weight, std=0.3)

        setattr(self, "A_log", torch.nn.Parameter(
            torch.log(torch.empty(NUM_HEADS).uniform_(1, 16)).view(1, 1, -1, 1)))
        self.dt_bias = torch.nn.Parameter(torch.randn(PROJ) * 0.1)
        self.o_norm = FusedRMSNormGated(HEAD_DIM, eps=RMS_EPS,
                                        activation="sigmoid")
        with torch.no_grad():
            self.o_norm.weight.copy_(1.0 + 0.1 * torch.randn(HEAD_DIM))
        self.to(device=device, dtype=DTYPE)
        self.o_norm.weight.data = self.o_norm.weight.data.float()
        getattr(self, "A_log").data = getattr(self, "A_log").data.float()
        self.dt_bias.data = self.dt_bias.data.float()


@torch.no_grad()
def _golden_step(layer, hs, conv, state):
    """One reference decode step. ``conv`` is a (q, k, v) tuple of [B, P, K-1]."""
    x = hs.unsqueeze(1) if hs.dim() == 2 else hs
    q, cq = layer.q_conv1d(layer.q_proj(x), cache=conv[0], output_final_state=True)
    k, ck = layer.k_conv1d(layer.k_proj(x), cache=conv[1], output_final_state=True)
    v, cv = layer.v_conv1d(layer.v_proj(x), cache=conv[2], output_final_state=True)

    g = fused_kda_gate(layer.f_b_proj(layer.f_a_proj(x)), layer.A_log,
                       layer.head_dim, g_bias=layer.dt_bias)
    beta = layer.b_proj(x).float().sigmoid()

    shape = (*x.shape[:2], NUM_HEADS, HEAD_DIM)
    o, new_state = _naive_recurrent_kda(
        q.view(shape), k.view(shape), v.view(shape), g, beta,
        initial_state=state, output_final_state=True,
        use_qk_l2norm_in_kernel=True)

    gate = layer.g_b_proj(layer.g_a_proj(x)).view(*o.shape)
    o = layer.o_norm(o, gate).reshape(*x.shape[:2], PROJ)
    return layer.o_proj(o), (cq, ck, cv), new_state


def _make_case(device, batch, seed=0):
    """Create one test case with random weights and initial state."""
    torch.manual_seed(seed)
    layer = _RefLayer(device)
    w = prepare_kda_fused_weights(layer)
    params = KdaBufferParams(0, batch, NUM_HEADS, HEAD_DIM, HIDDEN, DTYPE, device)
    buf = make_fused_buffers(params)

    conv = tuple(torch.randn(batch, PROJ, CONV_K - 1, device=device,
                             dtype=DTYPE) * 0.3 for _ in range(3))
    state = torch.randn(batch, NUM_HEADS, HEAD_DIM, HEAD_DIM, device=device,
                        dtype=torch.float32) * 0.1
    seed_fused_buffers(buf, conv, state)
    return layer, w, buf, conv, state


def run_single_step(device, batch=1, seed=0):
    """Run one fused decode step and assert it matches the torch golden."""
    layer, w, buf, conv, state = _make_case(device, batch, seed)

    hs = torch.randn(batch, HIDDEN, device=device, dtype=DTYPE) * 0.5
    ref_out, ref_conv, ref_state = _golden_step(
        layer, hs, tuple(c.clone() for c in conv), state.clone())

    out = kda_fused_decode(hs, w, buf)

    # Squeeze middle dimension if present (hs was unsqueezed in golden)
    if ref_out.dim() == 3 and ref_out.shape[1] == 1:
        ref_out = ref_out.squeeze(1)
    
    ref_out = ref_out.float()
    out_diff = (out.float() - ref_out).abs().max().item()
    st_diff = (buf.state.float() - ref_state.reshape(buf.state.shape)).abs().max().item()
    
    name = f"B{batch}_seed{seed}"
    logging.info(f"[{name}] out.max_diff={out_diff:.3e} state.max_diff={st_diff:.3e}")

    np.testing.assert_allclose(out.cpu().float().numpy(), ref_out.cpu().float().numpy(),
                               rtol=RTOL, atol=ATOL, err_msg=f"{name}: output mismatch")
    np.testing.assert_allclose(buf.state.cpu().float().numpy(),
                               ref_state.reshape(buf.state.shape).cpu().float().numpy(),
                               rtol=RTOL, atol=ATOL, err_msg=f"{name}: state mismatch")
    np.testing.assert_allclose(buf.conv_state.cpu().float().numpy(),
                               pack_conv_state(*ref_conv).cpu().float().numpy(),
                               rtol=RTOL, atol=ATOL, err_msg=f"{name}: conv history mismatch")
    return max(out_diff, st_diff)


def run_two_steps(device, batch=1, seed=3):
    """Run two consecutive fused decode steps and assert they match two golden steps.

    A single-step test passes even if the buffers are not carried forward
    correctly; this one fails if the state or conv history is dropped, reset or
    frozen between tokens.
    """
    layer, w, buf, conv, state = _make_case(device, batch, seed)

    hs1 = torch.randn(batch, HIDDEN, device=device, dtype=DTYPE) * 0.5
    hs2 = torch.randn(batch, HIDDEN, device=device, dtype=DTYPE) * 0.5

    r1, conv, state = _golden_step(layer, hs1, tuple(c.clone() for c in conv),
                                   state.clone())
    r2, _, _ = _golden_step(layer, hs2, conv, state)

    kda_fused_decode(hs1, w, buf)
    out2 = kda_fused_decode(hs2, w, buf)

    # Squeeze middle dimension if present
    if r2.dim() == 3 and r2.shape[1] == 1:
        r2 = r2.squeeze(1)
    
    ref = r2.float()
    out_diff = (out2.float() - ref).abs().max().item()
    
    name = f"B{batch}_seed{seed}_twostep"
    logging.info(f"[{name}] out.max_diff={out_diff:.3e}")

    np.testing.assert_allclose(out2.cpu().float().numpy(), ref.cpu().float().numpy(),
                               rtol=RTOL, atol=ATOL,
                               err_msg=f"{name}: second-step mismatch (state not carried?)")
    return out_diff


def test_kda_fused_decode():
    """Test the fused decode kernel against torch golden."""
    if not _HAS_NPU:
        return
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    dev = f"npu:{device_id}"

    max_err = 0.0
    max_err = max(max_err, run_single_step(dev, batch=1, seed=0))
    max_err = max(max_err, run_two_steps(dev, batch=1, seed=3))

    logging.info(f"\n[summary] max abs error across all cases = {max_err:.3e}")


def main():
    test_kda_fused_decode()
    logging.info("[PRECISION_PASS]")
    logging.info("All tests passed!")


if __name__ == "__main__":
    if _HAS_NPU:
        main()
    else:
        logging.info("skip: no NPU")
