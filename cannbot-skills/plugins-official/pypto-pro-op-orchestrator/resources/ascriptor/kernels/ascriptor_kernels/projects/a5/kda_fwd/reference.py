# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference for the KDA forward: the token-by-token recurrence the
public outputs answer to, and the staged checkpoints each of the five kernels answers to."""

import torch

# ----------------------------------------------------------------------------------------------------
# oracle.py
# End-to-end PyTorch oracle for KDA forward.
#
# This mirrors `fla/ops/kda/naive.py::naive_recurrent_kda` for the fixed-length
# path. Chunked helpers default to `chunk_size=64`, matching the upstream KDA
# default, but accept other chunk lengths for reference checks. Inputs use the
# upstream public layout:
#
#   q, k:  [B, T, H, K]
# Input shape for v: [B, T, HV, V]
#   g:     [B, T, HV, K]  natural-log gate increments
# Input shape for beta: [B, T, HV]
#
# The returned output is cast back to `v.dtype`; the final state remains fp32,
# matching the optimized chunk path.
# ----------------------------------------------------------------------------------------------------

def oracle(q, k, v, g, beta, initial_state=None):
    """Return the original fixed-length PyTorch result for the plan inputs."""
    return oracle_full(q, k, v, g, beta, initial_state=initial_state)


def oracle_full(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=True):
    dtype = v.dtype
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    qf = q.float().repeat_interleave(G, dim=2) * scale
    kf = k.float().repeat_interleave(G, dim=2)
    vf = v.float()
    gf = g.float()
    betaf = beta.float()

    state = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        state = state + initial_state.float()

    out = torch.zeros_like(vf)
    for t in range(T):
        q_t = qf[:, t]
        k_t = kf[:, t]
        v_t = vf[:, t]
        g_t = gf[:, t]
        beta_t = betaf[:, t]

        state = state * torch.exp(g_t)[..., None]
        residual = v_t - (k_t[..., None] * state).sum(-2)
        state = state + torch.einsum("bhk,bhv->bhkv", beta_t[..., None] * k_t, residual)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q_t, state)

    final_state = state if output_final_state else None
    return out.to(dtype), final_state


def to_chunked_inputs(q, k, v, g, beta, chunk_size=64):
    """Convert upstream `[B, T, ...]` inputs into KDA chunked refs."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    C = T // chunk_size
    if T % chunk_size != 0:
        raise ValueError(f"T={T} must be divisible by chunk_size={chunk_size}")

    q_c = q.reshape(B, C, chunk_size, H, K).permute(0, 3, 1, 2, 4).contiguous()
    k_c = k.reshape(B, C, chunk_size, H, K).permute(0, 3, 1, 2, 4).contiguous()
    v_c = v.reshape(B, C, chunk_size, HV, V).permute(0, 3, 1, 2, 4).contiguous()
    g_c = g.reshape(B, C, chunk_size, HV, K).permute(0, 3, 1, 2, 4).contiguous()
    beta_c = beta.reshape(B, C, chunk_size, HV).permute(0, 3, 1, 2).contiguous()
    return q_c, k_c, v_c, g_c, beta_c


def sample_inputs(B=1, H=2, HV=2, C=2, L=64, K=64, V=64, dtype=torch.bfloat16, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    T = C * L
    q = torch.randn(B, T, H, K, generator=generator).mul(0.04).to(dtype)
    k = torch.randn(B, T, H, K, generator=generator).mul(0.04).to(dtype)
    v = torch.randn(B, T, HV, V, generator=generator).mul(0.04).to(dtype)
    # Small mostly-negative log-space increments keep recurrent decay bounded.
    g = -torch.rand(B, T, HV, K, generator=generator).mul(0.03)
    beta = torch.rand(B, T, HV, generator=generator).mul(0.45).add(0.05)
    initial_state = torch.randn(B, HV, K, V, generator=generator).mul(0.01)
    return q, k, v, g, beta, initial_state

# ----------------------------------------------------------------------------------------------------
# gate.py
# Sub-kernel reference: sub1_gate - chunk-local linear gate cumsum.
#
# I/O contract
# ------------
# Inputs:
#   - g_raw: torch.float32 [B, HV, C, L, K]
#
# Outputs:
#   - g:     torch.float32 [B, HV, C, L, K] exp(cumsum(g_raw))
#
# `g_raw` is in natural-log units. This EasyASC plan materializes cumulative
# gates in linear space so later stages can consume `exp(cumsum(g_raw))`
# directly.
#
# Primitives used: [P1]
# ----------------------------------------------------------------------------------------------------

def gate(g_raw):
    return torch.exp(torch.cumsum(g_raw.float(), dim=-2))

# ----------------------------------------------------------------------------------------------------
# stages.py
# Independent Torch checkpoints with the actual forward precision boundaries.
# ----------------------------------------------------------------------------------------------------

def reference_stages(inputs):
    """Every checkpoint the five kernels hand each other, at the precision they hand it at.

    The bf16 rounding of `a_beta`, `k_exp` and the per-chunk `h` is not cosmetic: those are
    the cube operand boundaries, and an fp32-throughout reference would not be the thing the
    WY and recurrent stages are trying to match.
    """
    q, k, v, raw, beta = to_chunked_inputs(*(inputs[name] for name in ("q", "k", "v", "g_raw", "beta")), chunk_size=64)
    b, _, c, length, width = q.shape
    hv = v.shape[1]
    q = q.float().repeat_interleave(hv // q.shape[1], dim=1)
    k = k.float().repeat_interleave(hv // k.shape[1], dim=1)
    g = gate(raw)
    score = torch.matmul(q * g * (128 ** -0.5), (k / g).transpose(-1, -2))
    strict = torch.tril(torch.matmul(-(k * g * beta[..., None]), (k / g).transpose(-1, -2)), diagonal=-1)
    aqk = torch.tril(score).to(torch.bfloat16)
    eye = torch.eye(length, dtype=torch.float32).expand(b, hv, c, length, length)
    akk = torch.linalg.solve_triangular(eye - strict, eye, upper=False, unitriangular=True).to(torch.bfloat16)
    a_beta = (akk.float() * beta[..., None, :]).to(torch.bfloat16).float()
    k_exp = (k * g).to(torch.bfloat16).float()
    w = torch.matmul(a_beta, k_exp).to(torch.bfloat16)
    u = torch.matmul(a_beta, v.float()).to(torch.bfloat16)
    qg = (q * g).to(torch.bfloat16)
    kg = (k * (g[..., -1:, :] / g)).to(torch.bfloat16)
    state = inputs["initial_state"].clone()
    out = torch.empty_like(v)
    for chunk in range(c):
        h = state.to(torch.bfloat16).float()
        new = (u[:, :, chunk].float() - w[:, :, chunk].float() @ h).to(torch.bfloat16).float()
        qscaled = (q[:, :, chunk] * g[:, :, chunk] * (width ** -0.5)).to(torch.bfloat16).float()
        out[:, :, chunk] = (qscaled @ h + aqk[:, :, chunk].float() @ new).to(torch.bfloat16)
        state = state * g[:, :, chunk, -1, :, None] + kg[:, :, chunk].float().transpose(-1, -2) @ new
    return {"gate.g": g, "score.Aqk": aqk, "score.strict": strict, "inverse.Akk": akk, "inverse.strict_lower": torch.tril(akk, diagonal=-1), "wy.w": w, "wy.u": u, "wy.qg": qg, "wy.kg": kg, "recurrent.o": out, "recurrent.final_state": state}

# ----------------------------------------------------------------------------------------------------
# inputs.py
# Deterministic public inputs.
# ----------------------------------------------------------------------------------------------------

def make_inputs(case):
    p = case["parameters"]
    if p.get("L") != 64 or p.get("K") != 128 or p.get("V") != 128:
        raise ValueError("KDA forward kernels require L=64 and K=V=128")
    if any(isinstance(p.get(name), bool) or not isinstance(p.get(name), int) or p[name] <= 0 for name in ("B", "H", "HV", "C")) or p["HV"] % p["H"]:
        raise ValueError("B/H/HV/C must be positive integers and HV must be divisible by H")
    if p.get("initial_state", "random") not in ("random", "zero"):
        raise ValueError("initial_state must be random or zero")
    values = sample_inputs(B=p["B"], H=p["H"], HV=p["HV"], C=p["C"], L=64, K=128, V=128, seed=case["seed"])
    result = dict(zip(("q", "k", "v", "g_raw", "beta", "initial_state"), values, strict=True))
    if p.get("initial_state", "random") == "zero":
        result["initial_state"].zero_()
    return result


def reference(inputs):
    """The two public outputs, from the token-by-token recurrence rather than the chunked one.

    `reference_stages` follows the kernels' own decomposition; this does not, which is the
    point: it is the independent answer the whole pipeline is measured against.
    """
    out, state = oracle(*(inputs[name] for name in ("q", "k", "v", "g_raw", "beta", "initial_state")))
    return {"o": out, "final_state": state}
