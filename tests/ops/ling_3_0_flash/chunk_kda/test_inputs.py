# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# test_inputs.py — chunk_kda adversarial input generator (verifier-owned)
#
# Single source of truth for input construction consumed by adversarial_runner.py
# (and, later, per-module tests). All tensors are created DIRECTLY on the fixed
# NPU device (npu:<TILE_FWK_DEVICE_ID>, default 0) — never CPU-then-.npu(), so no
# mixed-device window exists. Mirrors module_interfaces.yaml primary_inputs order.
#
# Stability HARD CONSTRAINT (SPEC.md §9, MEMORY composition_verification): the
# chunk_kda forward-subst inverse NaNs on raw randn + near-1 gates. Internal
# precision MUST use:  q/k/v = randn*0.1 ; g = logsigmoid(randn) <= 0 ; beta =
# sigmoid(randn) ; scale = K**-0.5 ; T % 64 == 0. Do NOT change these.
# =============================================================================
import os

import torch

# Fixed device — TILE_FWK_DEVICE_ID=0, no card switch (per dispatch + golden).
DEVICE = torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")

# Golden positional tensor order (must match module_interfaces.yaml primary_inputs).
PRIMARY_INPUT_ORDER = ["q", "k", "v", "g", "beta"]

_DTYPE = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16,   "fp16": torch.float16,
    "float32": torch.float32,   "fp32": torch.float32,
}


def _dtype_from_str(s: str) -> torch.dtype:
    return _DTYPE[str(s).lower()]


def _shape(case: dict) -> dict:
    sh = case["shape"]
    B, T, H, K = int(sh["B"]), int(sh["T"]), int(sh["H"]), int(sh["K"])
    V = int(sh.get("V", K))
    assert T % 128 == 0, f"T={T} must be a multiple of chunk_size 128"
    return {"B": B, "T": T, "H": H, "K": K, "V": V}


def make_inputs(case: dict) -> dict:
    """Build the 5 primary tensors {q,k,v,g,beta} on DEVICE under stable scaling.

    Returns only the 5 PRIMARY_INPUT_ORDER tensors. Scalar / optional knobs
    (scale, initial_state, output_final_state, chunk_size) are resolved by
    make_call_kwargs() so positional dispatch stays clean.
    """
    s = _shape(case)
    B, T, H, K, V = s["B"], s["T"], s["H"], s["K"], s["V"]
    dt = _dtype_from_str(case.get("dtype", {}).get("default", "bfloat16"))
    seed = int(case.get("seed", 42))
    torch.manual_seed(seed)
    g0 = torch.randn(B, T, H, K, device=DEVICE)
    inputs = {
        "q":    (torch.randn(B, T, H, K, device=DEVICE, dtype=dt) * 0.1),
        "k":    (torch.randn(B, T, H, K, device=DEVICE, dtype=dt) * 0.1),
        "v":    (torch.randn(B, T, H, V, device=DEVICE, dtype=dt) * 0.1),
        "g":    (torch.nn.functional.logsigmoid(g0).float()),      # log-domain <= 0; fp32 (kernel ABI, wrapper no longer casts)
        "beta": (torch.sigmoid(torch.randn(B, T, H, device=DEVICE)).float()),  # fp32 (Task3 kernel ABI)
    }
    if "cancellation_stress" in case:
        inputs = apply_cancellation_stress(case, inputs)
    return inputs


def make_call_kwargs(case: dict, inputs: dict) -> dict:
    """Resolve the golden's scalar/optional args from the case dict."""
    s = _shape(case)
    K = s["K"]
    scale = case.get("scale", None)
    if scale is None:
        scale = K ** -0.5
    init_mode = case.get("initial_state", "none")
    if init_mode in (None, "none", "zero"):
        initial_state = None
    elif init_mode == "rand":
        torch.manual_seed(int(case.get("seed", 42)) + 7)
        initial_state = torch.randn(
            s["B"], s["H"], K, s["V"], device=DEVICE, dtype=torch.float32) * 0.1
    else:
        raise ValueError(f"unknown initial_state mode {init_mode!r}")
    return {
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": bool(case.get("output_final_state", False)),
        "chunk_size": int(case.get("chunk_size", 64)),
    }


# -----------------------------------------------------------------------------
# cancellation_stress (SPEC.md §9: v = u - w@S subtractive accumulation in M3)
# Engineers q/k/v scale so the (u, w@S) pair stresses catastrophic cancellation.
# Deterministic under cfg.seed; degrades gracefully (best-found) if budget spent.
# -----------------------------------------------------------------------------
_KNOB_SCALES = [0.01, 0.03, 0.1, 0.3, 1.0]


def apply_cancellation_stress(case: dict, inputs: dict) -> dict:
    cfg = case["cancellation_stress"]
    target = float(cfg.get("relative_gap", 1e-5))
    seed = int(cfg.get("seed", case.get("seed", 42)))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best_gap, best = float("inf"), inputs
    for _ in range(64):
        i = torch.randint(0, len(_KNOB_SCALES), (3,), generator=gen).tolist()
        sq, sk, sv = (_KNOB_SCALES[j] for j in i)
        trial = dict(inputs)
        trial["q"] = inputs["q"] * sq
        trial["k"] = inputs["k"] * sk
        trial["v"] = inputs["v"] * sv
        try:
            A, Bp = _probe_v_minus_wS(trial)
        except Exception:
            continue
        denom = torch.maximum(A.abs(), Bp.abs()).clamp_min(1e-30)
        gap = float((A - Bp).abs().div(denom).max().item())
        if gap < best_gap:
            best_gap, best = gap, trial
        if gap <= target:
            return trial
    return best


def _probe_v_minus_wS(inputs: dict):
    """Return (u, w@S) operands of v=u-w@S for the last chunk — pure torch."""
    q, k, v, g = inputs["q"].float(), inputs["k"].float(), inputs["v"].float(), inputs["g"].float()
    B, T, H, K = q.shape
    V, BT = v.shape[-1], 64
    NT = T // BT
    idx = torch.arange(BT, device=q.device)
    tri = (idx.reshape(BT, 1) >= idx.reshape(1, BT)).float()
    kc = k.reshape(B, NT, BT, H, K).transpose(1, 3).transpose(2, 3)[:, :, -1]   # last chunk
    vc = v.reshape(B, NT, BT, H, V).transpose(1, 3).transpose(2, 3)[:, :, -1]
    gc = torch.matmul(tri, g.reshape(B, NT, BT, H, K).transpose(1, 3).transpose(2, 3)[:, :, -1])
    A = (kc.unsqueeze(-2) * kc.unsqueeze(-3) * torch.exp(gc.unsqueeze(-3) - gc.unsqueeze(-2))).sum(-1)
    u = torch.matmul(A * tri, vc)
    S = torch.matmul((torch.exp(gc) * kc).transpose(-2, -1), vc)
    wS = torch.matmul(A * tri, S.mean(-1, keepdim=True).transpose(-2, -1).expand_as(vc))
    return u, wS
