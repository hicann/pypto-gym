# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference for the KDA backward: the saved forward caches the backward
consumes, an autograd oracle for the public gradients, and a physical per-stage model."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------
# types.py
# Independent KDA backward types formulas; migrated from reviewed source.
# ----------------------------------------------------------------------------------------------------

RCP_LN2 = 1.0 / math.log(2.0)

IO_DTYPE = torch.bfloat16

ATOL = RTOL = 1e-3

INPUT_SCALE = 0.01

INPUT_K_SCALE = 0.05

REPRESENTATIVE_SHAPE = {"B": 1, "T": 1024, "H": 32, "HV": 32, "K": 128, "V": 128, "chunk_size": 64}

@dataclass(frozen=True)
class KDAOracleInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    initial_state: torch.Tensor
    do: torch.Tensor
    dht: torch.Tensor
    chunk_size: int

@dataclass
class KDAOracleOutputs:
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    dbeta: torch.Tensor
    dg: torch.Tensor
    dh0: torch.Tensor

@dataclass
class KDAForwardCaches:
    g_cumsum: torch.Tensor
    Aqk: torch.Tensor
    Akk: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    v_new: torch.Tensor
    h: torch.Tensor

@dataclass
class KDAScanBwdMids:
    dAqk: torch.Tensor
    dh: torch.Tensor
    dv: torch.Tensor
    dh0: torch.Tensor

@dataclass
class KDALocalBwdMids:
    dq_hv: torch.Tensor
    dk_hv: torch.Tensor
    dv: torch.Tensor
    dbeta: torch.Tensor
    dg_core: torch.Tensor
    dAkk: torch.Tensor

# ----------------------------------------------------------------------------------------------------
# numeric.py
# Independent KDA backward numeric formulas; migrated from reviewed source.
# ----------------------------------------------------------------------------------------------------

def _exp2(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(x) if hasattr(torch, "exp2") else torch.pow(2.0, x)

def _to_io(x: torch.Tensor) -> torch.Tensor:
    return x.to(IO_DTYPE)

def _chunk_bounds(total_length: int, chunk_size: int):
    return [(start, min(total_length, start + chunk_size)) for start in range(0, total_length, chunk_size)]

def _chunk_local_cumsum(x: torch.Tensor, chunk_size: int, reverse: bool = False) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float32)
    for start, end in _chunk_bounds(x.shape[1], chunk_size):
        block = x[:, start:end].float()
        if reverse:
            block = torch.flip(torch.cumsum(torch.flip(block, dims=(1,)), dim=1), dims=(1,))
        else:
            block = torch.cumsum(block, dim=1)
        out[:, start:end] = block
    return out

def _pairwise_decay(g_blk: torch.Tensor) -> torch.Tensor:
    return _exp2(g_blk[:, None, :] - g_blk[None, :, :])

def _chunk_local_cumsum_diff(x: torch.Tensor, chunk_size: int, reverse: bool = False) -> torch.Tensor:
    blocks = []
    for start, end in _chunk_bounds(x.shape[1], chunk_size):
        block = x[:, start:end].float()
        if reverse:
            block = torch.flip(torch.cumsum(torch.flip(block, dims=(1,)), dim=1), dims=(1,))
        else:
            block = torch.cumsum(block, dim=1)
        blocks.append(block)
    return torch.cat(blocks, dim=1)

# ----------------------------------------------------------------------------------------------------
# inputs.py
# Independent KDA backward inputs formulas; migrated from reviewed source.
# ----------------------------------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def _validate_contract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
    do: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
) -> tuple[int, int, int, int, int, int, int, float]:
    _require(q.ndim == 4, "q must have shape [B, T, H, K]")
    _require(k.shape == q.shape, "k must match q shape [B, T, H, K]")
    _require(v.ndim == 4, "v must have shape [B, T, HV, V]")
    _require(g.ndim == 4, "g must have shape [B, T, HV, K]")
    _require(beta.ndim == 3, "beta must have shape [B, T, HV]")
    _require(initial_state.ndim == 4, "initial_state must have shape [B, HV, K, V]")
    _require(chunk_size > 0, "chunk_size must be positive")

    B, T, H, Kdim = q.shape
    Bv, Tv, HV, Vdim = v.shape
    Bg, Tg, HVg, Kg = g.shape
    Bb, Tb, HVb = beta.shape
    Bs, HVs, Ks, Vs = initial_state.shape

    _require(min(B, T, H, HV, Kdim, Vdim) > 0, "all tensor dimensions must be positive")
    _require((Bv, Tv) == (B, T), "v must match q batch/time dimensions")
    _require((Bg, Tg, Kg) == (B, T, Kdim), "g must match q batch/time/K dimensions")
    _require((Bb, Tb) == (B, T), "beta must match q batch/time dimensions")
    _require(HVg == HV == HVb == HVs, "v, g, beta, initial_state must agree on HV")
    _require((Bs, Ks, Vs) == (B, Kdim, Vdim), "initial_state must have shape [B, HV, K, V]")
    _require(HV % H == 0, "HV must be an integer multiple of H")

    tensors = [q, k, v, g, beta, initial_state]
    if do is not None:
        _require(do.shape == v.shape, "do must match v shape [B, T, HV, V]")
        tensors.append(do)
    if dht is not None:
        _require(dht.shape == initial_state.shape, "dht must match initial_state shape [B, HV, K, V]")
        tensors.append(dht)

    devices = {tensor.device for tensor in tensors}
    _require(len(devices) == 1, "all tensors must be on the same device")
    _require(all(tensor.dtype == IO_DTYPE for tensor in tensors), "all public tensors must be bfloat16")

    group = HV // H
    scale = Kdim ** -0.5
    return B, T, H, HV, Kdim, Vdim, group, scale

def validate_inputs(inputs: KDAOracleInputs) -> tuple[int, int, int, int, int, int, int, float]:
    return _validate_contract(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.initial_state,
        inputs.chunk_size,
        do=inputs.do,
        dht=inputs.dht,
    )

def generate_inputs(
    B: int = 1,
    H: int = 32,
    HV: int = 32,
    T: int | None = None,
    C: int | None = None,
    K: int = 128,
    V: int = 128,
    chunk_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = IO_DTYPE,
    scale: float = INPUT_SCALE,
    k_scale: float = INPUT_K_SCALE,
) -> KDAOracleInputs:
    if T is None and C is None:
        C = 16
    if T is None:
        _require(C is not None, "either T or C must be provided")
        T = C * chunk_size
    elif C is not None:
        _require(C * chunk_size == T, "when both T and C are provided, they must satisfy T == C * chunk_size")

    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    q = (randn(B, T, H, K) * scale).to(dtype)
    k = (randn(B, T, H, K) * k_scale).to(dtype)
    v = (randn(B, T, HV, V) * scale).to(dtype)
    g = (-F.softplus(randn(B, T, HV, K))).to(dtype)
    beta = torch.sigmoid(randn(B, T, HV)).to(dtype)
    initial_state = (randn(B, HV, K, V) * scale).to(dtype)
    do = (randn(B, T, HV, V) * scale).to(dtype)
    dht = (randn(B, HV, K, V) * scale).to(dtype)
    return KDAOracleInputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        do=do,
        dht=dht,
        chunk_size=chunk_size,
    )

def make_representative_inputs(
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = IO_DTYPE,
    scale: float = INPUT_SCALE,
    k_scale: float = INPUT_K_SCALE,
) -> KDAOracleInputs:
    return generate_inputs(
        B=REPRESENTATIVE_SHAPE["B"],
        H=REPRESENTATIVE_SHAPE["H"],
        HV=REPRESENTATIVE_SHAPE["HV"],
        T=REPRESENTATIVE_SHAPE["T"],
        K=REPRESENTATIVE_SHAPE["K"],
        V=REPRESENTATIVE_SHAPE["V"],
        chunk_size=REPRESENTATIVE_SHAPE["chunk_size"],
        seed=seed,
        device=device,
        dtype=dtype,
        scale=scale,
        k_scale=k_scale,
    )

# ----------------------------------------------------------------------------------------------------
# forward.py
# Independent KDA backward forward formulas; migrated from reviewed source.
# ----------------------------------------------------------------------------------------------------

def forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 inputs -> fp32 outputs (internal building block for the oracle)."""
    _validate_contract(q, k, v, g, beta, initial_state, chunk_size)
    return _forward_reference_f32(
        q.float(),
        k.float(),
        v.float(),
        g.float(),
        beta.float(),
        initial_state.float(),
        chunk_size,
    )

def build_saved_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
) -> KDAForwardCaches:
    """bf16 inputs -> bf16 saved caches (computed in fp32, downcast on store)."""
    B, T, H, HV, Kdim, Vdim, group, scale = _validate_contract(q, k, v, g, beta, initial_state, chunk_size)
    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = _chunk_local_cumsum(g.float(), chunk_size) * RCP_LN2
    betaf = beta.float()
    initf = initial_state.float()
    chunk_bounds = _chunk_bounds(T, chunk_size)
    num_chunks = len(chunk_bounds)

    Aqk = torch.zeros(B, T, HV, chunk_size, dtype=IO_DTYPE, device=q.device)
    Akk = torch.zeros(B, T, HV, chunk_size, dtype=IO_DTYPE, device=q.device)
    w = torch.empty(B, T, HV, Kdim, dtype=IO_DTYPE, device=q.device)
    u = torch.empty(B, T, HV, Vdim, dtype=IO_DTYPE, device=q.device)
    qg = torch.empty(B, T, HV, Kdim, dtype=IO_DTYPE, device=q.device)
    kg = torch.empty(B, T, HV, Kdim, dtype=IO_DTYPE, device=q.device)
    v_new = torch.empty(B, T, HV, Vdim, dtype=IO_DTYPE, device=q.device)
    h = torch.empty(B, num_chunks, HV, Kdim, Vdim, dtype=IO_DTYPE, device=q.device)
    states = initf.clone()

    for b in range(B):
        for chunk_idx, (start, end) in enumerate(chunk_bounds):
            valid = end - start
            for hv in range(HV):
                hq = hv // group
                q_blk = qf[b, start:end, hq]
                k_blk = kf[b, start:end, hq]
                v_blk = vf[b, start:end, hv]
                g_blk = gf[b, start:end, hv]
                beta_blk = betaf[b, start:end, hv]

                decay = _pairwise_decay(g_blk)
                score = (q_blk[:, None, :] * k_blk[None, :, :] * decay).sum(dim=-1)
                score = torch.tril(score) * scale
                kk = (k_blk[:, None, :] * k_blk[None, :, :] * decay).sum(dim=-1)
                kk_lower = torch.tril(kk * beta_blk[:, None], diagonal=-1)
                A_blk = torch.linalg.inv(torch.eye(valid, device=q.device, dtype=torch.float32) + kk_lower)

                exp_g = _exp2(g_blk)
                qg_blk = q_blk * exp_g
                w_blk = A_blk @ (k_blk * beta_blk[:, None] * exp_g)
                u_blk = A_blk @ (v_blk * beta_blk[:, None])
                g_last = g_blk[valid - 1]
                kg_blk = k_blk * _exp2(g_last[None, :] - g_blk)

                state = states[b, hv]
                v_new_blk = u_blk - w_blk @ state
                next_state = state * _exp2(g_last)[:, None] + kg_blk.transpose(-1, -2) @ v_new_blk

                Aqk[b, start:end, hv, :valid] = _to_io(score)
                Akk[b, start:end, hv, :valid] = _to_io(A_blk)
                w[b, start:end, hv] = _to_io(w_blk)
                u[b, start:end, hv] = _to_io(u_blk)
                qg[b, start:end, hv] = _to_io(qg_blk)
                kg[b, start:end, hv] = _to_io(kg_blk)
                v_new[b, start:end, hv] = _to_io(v_new_blk)
                h[b, chunk_idx, hv] = _to_io(state)
                states[b, hv] = next_state

    return KDAForwardCaches(
        g_cumsum=_to_io(gf),
        Aqk=Aqk,
        Akk=Akk,
        w=w,
        u=u,
        qg=qg,
        kg=kg,
        v_new=v_new,
        h=h,
    )

def replay_forward(caches: KDAForwardCaches, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 caches -> fp32 replayed output/final state."""
    B, T, HV, Kdim = caches.g_cumsum.shape
    Vdim = caches.v_new.shape[-1]
    scale = Kdim ** -0.5
    chunk_bounds = _chunk_bounds(T, chunk_size)
    num_chunks = len(chunk_bounds)
    _require(caches.h.shape == (B, num_chunks, HV, Kdim, Vdim), "h shape mismatch")

    outputs = torch.empty(B, T, HV, Vdim, dtype=torch.float32, device=caches.g_cumsum.device)
    final_state = torch.empty(B, HV, Kdim, Vdim, dtype=torch.float32, device=caches.g_cumsum.device)
    for b in range(B):
        for hv in range(HV):
            state = None
            for chunk_idx, (start, end) in enumerate(chunk_bounds):
                valid = end - start
                Aqk_blk = torch.tril(caches.Aqk[b, start:end, hv, :valid].float())
                qg_blk = caches.qg[b, start:end, hv].float()
                kg_blk = caches.kg[b, start:end, hv].float()
                h_blk = caches.h[b, chunk_idx, hv].float()
                v_new_blk = caches.v_new[b, start:end, hv].float()
                outputs[b, start:end, hv] = Aqk_blk @ v_new_blk + scale * (qg_blk @ h_blk)

                g_last = caches.g_cumsum[b, end - 1, hv].float()
                state = h_blk * _exp2(g_last)[:, None] + kg_blk.transpose(-1, -2) @ v_new_blk

            final_state[b, hv] = state

    return outputs, final_state

def _forward_reference_f32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable fp32 forward used by the oracle (inputs already fp32)."""
    B, T, H, Kdim = q.shape
    HV = v.shape[2]
    group = HV // H
    scale = Kdim ** -0.5
    gf = _chunk_local_cumsum_diff(g, chunk_size) * RCP_LN2
    chunk_bounds = _chunk_bounds(T, chunk_size)

    batch_outputs = []
    batch_final_states = []
    for b in range(B):
        hv_outputs = []
        hv_states = []
        for hv in range(HV):
            hq = hv // group
            state = initial_state[b, hv]
            chunk_outputs = []
            for start, end in chunk_bounds:
                q_blk = q[b, start:end, hq]
                k_blk = k[b, start:end, hq]
                v_blk = v[b, start:end, hv]
                g_blk = gf[b, start:end, hv]
                beta_blk = beta[b, start:end, hv]

                decay = _pairwise_decay(g_blk)
                score = (q_blk[:, None, :] * k_blk[None, :, :] * decay).sum(dim=-1)
                score = torch.tril(score)
                kk = (k_blk[:, None, :] * k_blk[None, :, :] * decay).sum(dim=-1)
                kk_lower = torch.tril(kk * beta_blk[:, None], diagonal=-1)
                A_blk = torch.linalg.inv(torch.eye(end - start, device=q.device, dtype=torch.float32) + kk_lower)

                exp_g = _exp2(g_blk)
                qg_blk = q_blk * exp_g
                w_blk = A_blk @ (k_blk * beta_blk[:, None] * exp_g)
                u_blk = A_blk @ (v_blk * beta_blk[:, None])
                g_last = g_blk[end - start - 1]
                kg_blk = k_blk * _exp2(g_last[None, :] - g_blk)
                v_new_blk = u_blk - w_blk @ state

                chunk_outputs.append(torch.tril(score * scale) @ v_new_blk + scale * (qg_blk @ state))
                state = state * _exp2(g_last)[:, None] + kg_blk.transpose(-1, -2) @ v_new_blk

            hv_outputs.append(torch.cat(chunk_outputs, dim=0))
            hv_states.append(state)

        batch_outputs.append(torch.stack(hv_outputs, dim=1))
        batch_final_states.append(torch.stack(hv_states, dim=0))

    return torch.stack(batch_outputs, dim=0), torch.stack(batch_final_states, dim=0)

# ----------------------------------------------------------------------------------------------------
# oracle.py
# Independent KDA backward oracle formulas; migrated from reviewed source.
# ----------------------------------------------------------------------------------------------------

def oracle(inputs: KDAOracleInputs) -> KDAOracleOutputs:
    """bf16 inputs -> bf16 grads; autograd through the fp32 forward formula."""
    validate_inputs(inputs)

    q = inputs.q.detach().clone().float().requires_grad_(True)
    k = inputs.k.detach().clone().float().requires_grad_(True)
    v = inputs.v.detach().clone().float().requires_grad_(True)
    g = inputs.g.detach().clone().float().requires_grad_(True)
    beta = inputs.beta.detach().clone().float().requires_grad_(True)
    initial_state = inputs.initial_state.detach().clone().float().requires_grad_(True)
    do = inputs.do.detach().clone().float()
    dht = inputs.dht.detach().clone().float()

    output, final_state = _forward_reference_f32(
        q, k, v, g, beta, initial_state, inputs.chunk_size
    )
    loss = (output * do).sum() + (final_state * dht).sum()
    dq, dk, dv, dg, dbeta, dh0 = torch.autograd.grad(loss, (q, k, v, g, beta, initial_state))
    return KDAOracleOutputs(
        dq=_to_io(dq.detach()),
        dk=_to_io(dk.detach()),
        dv=_to_io(dv.detach()),
        dbeta=_to_io(dbeta.detach()),
        dg=_to_io(dg.detach()),
        dh0=_to_io(dh0.detach()),
    )

# ----------------------------------------------------------------------------------------------------
# stages.py
# Independent Torch checkpoints for each preserved physical kernel ABI.
#
# The end-to-end oracle separately differentiates the unrounded forward formula.
# These checkpoints express actual bf16 storage seams, the fp32 beta-gradient
# seam, dense scan scores, and fp32 accumulation before the final head reduction.
# ----------------------------------------------------------------------------------------------------

def pack(value):
    b, t, heads, width = value.shape
    return value.reshape(b, t // 64, 64, heads, width).permute(0, 3, 1, 2, 4).contiguous()


def unpack(value):
    b, heads, chunks, length, width = value.shape
    return value.permute(0, 2, 3, 1, 4).reshape(b, chunks * length, heads, width).contiguous()


def bf(value):
    return value.to(torch.bfloat16)


def transpose(value):
    return value.transpose(-1, -2)


def exp2(value):
    """A5 expresses exp2 as float32 multiply by ln(2), then natural exp."""
    return torch.exp(value * math.log(2.0))


def checkpoints(inputs):
    saved = inputs["saved"]
    b, t, heads, width = inputs["q"].shape
    hv, chunks = inputs["v"].shape[2], t // 64
    group, scale = hv // heads, width ** -0.5
    q = pack(inputs["q"].repeat_interleave(group, dim=2)).float()
    k = pack(inputs["k"].repeat_interleave(group, dim=2)).float()
    v, grad = pack(inputs["v"]).float(), pack(inputs["do"]).float()
    beta = inputs["beta"].reshape(b, chunks, 64, hv).permute(0, 3, 1, 2).float()
    gate = pack(saved["g_cumsum"]).float()
    aqk, akk = pack(saved["Aqk"]).float(), pack(saved["Akk"]).float()
    kg, qg, w = (pack(saved[name]).float() for name in ("kg", "qg", "w"))
    new = pack(saved["v_new"]).float()
    h = saved["h"].permute(0, 2, 1, 3, 4).float()
    daqk = bf((grad @ transpose(new)) * scale)
    dv0 = transpose(aqk) @ grad
    dv = torch.empty_like(new, dtype=torch.bfloat16)
    dh = torch.empty_like(h, dtype=torch.bfloat16)
    state = inputs["dht"].float().clone()
    for chunk in range(chunks - 1, -1, -1):
        dh[:, :, chunk] = bf(state)
        current = bf(dv0[:, :, chunk] + kg[:, :, chunk] @ bf(state).float())
        dv[:, :, chunk] = current
        state = state * exp2(gate[:, :, chunk, -1, :, None]) + (transpose(qg[:, :, chunk]) @ grad[:, :, chunk]) * scale - transpose(w[:, :, chunk]) @ current.float()
    dh0 = bf(state)

    dqg = bf(grad @ transpose(h))
    dkg = bf(new @ transpose(dh.float()))
    dw = bf(-(dv.float() @ transpose(h)))
    dvbeta = bf(transpose(akk) @ dv.float())
    dkbetag = bf(transpose(akk) @ dw.float())

    eg = exp2(gate)
    glast = gate[..., -1:, :]
    elast = exp2(glast - gate)
    dq = dqg.float() * eg * scale
    dkkg = dkg.float() * elast
    kexp = k * eg
    dvout = bf(dvbeta.float() * beta[..., None])
    dbeta = (dvbeta.float() * v).sum(-1) + (dkbetag.float() * kexp).sum(-1)
    dk = bf(dkkg + dkbetag.float() * beta[..., None] * eg)
    dglast = (h * dh.float()).sum(-1) * exp2(gate[..., -1, :]) + (k * dkkg).sum(-2)
    dgcore = bf(q * dq - k * dkkg + dkbetag.float() * kexp * beta[..., None])
    dgcore[..., -1, :] = bf(dgcore[..., -1, :].float() + dglast)
    dq = bf(dq)
    kexp = bf(kexp)

    dainv = dv.float() @ transpose(v) + dw.float() @ transpose(kexp.float())
    dtri = bf(torch.tril(dainv * beta[..., None, :], diagonal=-1))
    temp = bf(dtri.float() @ transpose(akk))
    dakk = bf(torch.tril(-(transpose(akk) @ temp.float()), diagonal=-1))

    rowscale, colscale = exp2(gate - glast), exp2(glast - gate)
    qscaled, kscaled, scaled_k = bf(q * rowscale), bf(k * rowscale), bf(k * colscale)
    mqk, mbase = torch.tril(daqk), torch.tril(dakk, diagonal=-1)
    mbeta = bf(mbase.float() * beta[..., :, None])
    qkl = bf(mqk.float() @ scaled_k.float())
    qkr = bf(transpose(mqk.float()) @ qscaled.float())
    sbase = bf(mbase.float() @ scaled_k.float())
    tbeta = bf(transpose(mbeta.float()) @ kscaled.float())

    dqpair, dkpair = rowscale * qkl.float(), colscale * qkr.float()
    row_contrib, col_contrib = rowscale * sbase.float(), colscale * tbeta.float()
    dqpost = dq.float() + dqpair
    dkpost = dk.float() + dkpair + (row_contrib * beta[..., None] + col_contrib)
    dbpost = bf(dbeta + (k * row_contrib).sum(-1))
    dg = dgcore.float() + (q * dqpair - k * dkpair) + (k * row_contrib * beta[..., None] - k * col_contrib)
    dg = bf(torch.flip(torch.cumsum(torch.flip(dg, dims=(-2,)), dim=-2), dims=(-2,)))
    dqfinal = bf(unpack(dqpost).reshape(b, t, heads, group, width).sum(3))
    dkfinal = bf(unpack(dkpost).reshape(b, t, heads, group, width).sum(3))
    def scalar_public(x):
        return x.permute(0, 2, 3, 1).reshape(b, t, hv).contiguous()

    def token_native(x):
        return x.reshape(b, hv, t, x.shape[-1])

    return {"scan.dAqk": unpack(daqk), "scan.dh": dh.permute(0, 2, 1, 3, 4).contiguous(), "scan.dv": unpack(dv), "scan.dh0": dh0,
            "inverse_mm.d_qg": dqg, "inverse_mm.d_kg": dkg, "inverse_mm.d_w": dw, "inverse_mm.d_v_beta": dvbeta, "inverse_mm.d_k_beta_g": dkbetag,
            "inverse_epilogue.dq_hv": unpack(dq), "inverse_epilogue.dk_hv": unpack(dk), "inverse_epilogue.dv": unpack(dvout), "inverse_epilogue.dbeta": scalar_public(dbeta), "inverse_epilogue.dg_core": unpack(dgcore), "inverse_epilogue.k_exp": kexp,
            "inverse_dainv.D_tri": dtri, "inverse_dakk.dAkk": unpack(dakk),
            "finalize_pre.q_scaled": qscaled, "finalize_pre.k_scaled": kscaled, "finalize_pre.kg": scaled_k, "finalize_pre.M_qk": mqk, "finalize_pre.M_base": mbase, "finalize_pre.M_beta": mbeta,
            "finalize_pair.qk_left": token_native(qkl), "finalize_pair.qk_right": token_native(qkr), "finalize_pair.s_base": token_native(sbase), "finalize_pair.t_beta": token_native(tbeta),
            "finalize_post.dq_hv": dqpost, "finalize_post.dk_hv": dkpost, "finalize_post.dbeta": scalar_public(dbpost), "finalize_post.dg": unpack(dg), "finalize_reduce.dq": dqfinal, "finalize_reduce.dk": dkfinal}

# ----------------------------------------------------------------------------------------------------
# comparison.py
# Unquantized mathematical targets for the original finalize-pre numeric gate.
#
# Physical BF16 checkpoints still feed every later stage. M10-058 restores the
# standalone source test's FP32 exp2 products only at the comparison boundary.
# ----------------------------------------------------------------------------------------------------

SCALED_TARGETS = (
    "finalize_pre.q_scaled",
    "finalize_pre.k_scaled",
    "finalize_pre.kg",
)


def comparison_targets(inputs):
    result = checkpoints(inputs)
    group = inputs["v"].shape[2] // inputs["q"].shape[2]
    q = pack(inputs["q"].repeat_interleave(group, dim=2)).float()
    k = pack(inputs["k"].repeat_interleave(group, dim=2)).float()
    gate = pack(inputs["saved"]["g_cumsum"]).float()
    last = gate[..., -1:, :]
    row_scale = torch.exp2(gate - last)
    col_scale = torch.exp2(last - gate)
    result.update(zip(SCALED_TARGETS, (q * row_scale, k * row_scale, k * col_scale)))
    return result

# ----------------------------------------------------------------------------------------------------
# public.py
# The three entry points main.py calls: deterministic inputs, the autograd oracle, and the
# physical per-stage checkpoints.
#
# `make_inputs` owns the saved forward caches too. KDA's backward consumes nine of them
# (g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h) and none of them are recomputed on device, so a
# case is only reproducible if the same seed rebuilds the same caches: `build_saved_forward`
# runs the fp32 forward and downcasts on store, exactly where the real forward would.
# ----------------------------------------------------------------------------------------------------

def make_inputs(case: dict) -> dict:
    p = case["parameters"]
    source = generate_inputs(B=p["B"], H=p["H"], HV=p["HV"], C=p["C"], K=p["K"], V=p["V"],
                             chunk_size=p["L"], seed=case["seed"])
    values = dict(vars(source))
    # The gate is stored as bf16 like every other public tensor, so a gentler decay is a
    # rescale of g before the cumulative sum, not a different generator.
    if p.get("gate_multiplier", 1) != 1:
        values["g"] = (values["g"].float() * p["gate_multiplier"]).to(torch.bfloat16)
    if p.get("initial_state", "random") == "zero":
        values["initial_state"].zero_()
    saved = build_saved_forward(*(values[name] for name in
                                  ("q", "k", "v", "g", "beta", "initial_state")), p["L"])
    values["saved"] = dict(vars(saved))
    return values


def reference(inputs: dict) -> dict:
    """The six public gradients, differentiated through the unrounded fp32 forward."""
    names = KDAOracleInputs.__dataclass_fields__
    return dict(vars(oracle(KDAOracleInputs(**{name: inputs[name] for name in names}))))


def reference_stages(inputs: dict) -> dict:
    """Every stage's own output, modelled at the bf16/fp32 seams the kernels actually use.

    This is a different model from `reference`: it rounds where the hardware rounds, so a
    stage comparison measures the kernel and not the precision of the algorithm. The three
    finalize_pre scaled products are the exception — `comparison_targets` replaces them with
    unquantized fp32 exp2 mathematics.
    """
    return comparison_targets(inputs)
