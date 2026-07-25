# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure-torch / NPU-compatible replacement for the ``fla`` (flash-linear-attention)
symbols imported by ``modeling_kimi.py``.

``fla`` is CUDA/Triton-only, so on Ascend NPU we provide drop-in torch
implementations of exactly the symbols ``modeling_kimi.py`` needs:

    FusedRMSNormGated         (fla.modules)
    ShortConvolution          (fla.modules)
    fused_kda_gate            (fla.ops.kda.gate)
    chunk_kda                 (fla.ops.kda)        -> vec_chunk_kda (torch chunked)
    fused_recurrent_kda       (fla.ops.kda)        -> _naive_recurrent_kda
    prepare_cu_seqlens_from_mask / prepare_lens_from_mask  (fla.ops.utils.index)
    tensor_cache              (fla.utils)

``modeling_kimi.py`` imports these directly from this module (a guarded
``try: from fla... except ImportError: from .kimi_fla_compat import ...``). No
fake ``fla`` namespace is registered into ``sys.modules``.

``chunk_kda``/``fused_recurrent_kda`` here are the **torch reference KDA ops**
(correct, NPU-runnable). The PyPTO fused kernel replaces ``chunk_kda`` at
runtime via the ``kimi_linear_48b_a3b_pto_kernels`` sys.modules injection inside
``KimiDeltaAttention.forward``.
"""
__all__ = [
    "ShortConvolution",
    "FusedRMSNormGated",
    "fused_kda_gate",
    "chunk_kda",
    "fused_recurrent_kda",
    "prepare_cu_seqlens_from_mask",
    "prepare_lens_from_mask",
    "tensor_cache",
]

import functools
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# index / cache helpers (fla.ops.utils.index, fla.utils)
# ---------------------------------------------------------------------------

def tensor_cache(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def prepare_lens_from_mask(mask):
    return mask.sum(dim=-1, dtype=torch.int32)


def prepare_cu_seqlens_from_mask(mask, dtype=torch.int32):
    lens = prepare_lens_from_mask(mask)
    return F.pad(lens.cumsum(0, dtype=dtype), (1, 0))


# ---------------------------------------------------------------------------
# fla.modules.ShortConvolution
# ---------------------------------------------------------------------------

class ShortConvolution(nn.Module):
    """Causal depthwise conv1d (kernel K) + optional activation, with cache.
    Signature matches fla usage in modeling_kimi.py:
        out, new_cache = conv(x, cache=..., output_final_state=..., cu_seqlens=...)
    x: [B, T, D]; cache: [B, D, K-1] previous pre-conv inputs.
    """

    def __init__(self, hidden_size, kernel_size, activation=None, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        b, t, d = x.shape
        xc = x.transpose(1, 2)  # [B, D, T]
        k = self.kernel_size
        if cache is not None:
            xc_in = torch.cat([cache.to(xc.dtype), xc], dim=-1)
        else:
            xc_in = F.pad(xc, (k - 1, 0))
        new_cache = None
        if output_final_state:
            if k > 1:
                new_cache = xc_in[..., -(k - 1):].contiguous()
            else:
                new_cache = xc_in.new_zeros(b, d, 0)
        out = F.conv1d(xc_in, self.weight, self.bias, groups=d)
        out = out[..., -t:]
        out = out.transpose(1, 2)
        out = self._act(out)
        return out, new_cache

    def _act(self, x):
        if self.activation is None:
            return x
        if self.activation in ("silu", "swish"):
            return F.silu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        raise NotImplementedError(self.activation)


# ---------------------------------------------------------------------------
# fla.modules.FusedRMSNormGated
# ---------------------------------------------------------------------------

class FusedRMSNormGated(nn.Module):
    """RMSNorm(x) * act(gate). KDA uses activation='sigmoid'. Norm over last dim."""

    def __init__(self, hidden_size, eps=1e-5, activation="sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.activation = activation

    def forward(self, x, gate):
        dt = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xn = xf * torch.rsqrt(var + self.eps)
        xn = xn * self.weight.float()
        out = xn * self._gate_act(gate.float())
        return out.to(dt)

    def _gate_act(self, g):
        if self.activation == "sigmoid":
            return torch.sigmoid(g)
        if self.activation in ("silu", "swish"):
            return F.silu(g)
        raise NotImplementedError(self.activation)


# ---------------------------------------------------------------------------
# fla.ops.kda.gate.fused_kda_gate
# ---------------------------------------------------------------------------

def fused_kda_gate(g, a_log, head_dim, g_bias=None, lower_bound=-5.0):
    """g_raw: [B,T,H*head_dim]; a_log: [1,1,H,1]; g_bias(dt_bias): [H*head_dim].
    Returns g (log-space gate): [B,T,H,head_dim].
    The log-space gate is lower_bound times sigmoid of exp(a_log) applied to
    (g_raw + dt_bias); alpha is exp of that gate.
    """
    b, t, pd = g.shape
    h = pd // head_dim
    gf = g.float()
    if g_bias is not None:
        gf = gf + g_bias.float()
    gf = gf.view(b, t, h, head_dim)
    a_gate = torch.exp(a_log.float())  # [1,1,h,1]
    out = lower_bound * torch.sigmoid(a_gate * gf)
    return out


# ---------------------------------------------------------------------------
# shared KDA prep: float casts plus optional in-kernel L2 norm of q and k
# ---------------------------------------------------------------------------

class Qkvgb(NamedTuple):
    """A (q, k, v, g, beta) tensor bundle in FLA layout (internal helper type)."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor


def _prep_qkvgb(qkvgb, use_qk_l2norm_in_kernel):
    """Cast q,k,v,g,beta to float and, when requested, L2-normalize q and k over
    the last dim (the exact in-kernel L2 norm both reference ops share).
    Returns a float ``Qkvgb`` (q and k optionally L2-normalized).
    """
    qf = qkvgb.q.float()
    kf = qkvgb.k.float()
    vf = qkvgb.v.float()
    gf = qkvgb.g.float()
    bf = qkvgb.beta.float()
    if use_qk_l2norm_in_kernel:
        qf = qf / (qf.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
        kf = kf / (kf.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
    return Qkvgb(qf, kf, vf, gf, bf)


# ---------------------------------------------------------------------------
# torch reference for the FLA fused_recurrent_kda op (naive recurrent)
# ---------------------------------------------------------------------------

def _naive_recurrent_kda(q, k, v, g, beta, initial_state=None,
                         output_final_state=True, use_qk_l2norm_in_kernel=True,
                         scale=None, **kwargs):
    """Naive recurrent KDA (delta-rule, per-channel gate). Torch reference op.
    Layout: q,k,g [B,T,H,K]; v [B,T,H,V]; beta [B,T,H]; state [B,H,V,K]."""
    batch, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    dt = q.dtype
    if scale is None:
        scale = head_k_dim ** -0.5
    qf, kf, vf, gf, bf = _prep_qkvgb(Qkvgb(q, k, v, g, beta), use_qk_l2norm_in_kernel)
    if initial_state is not None:
        state = initial_state.float().clone()  # [B,H,V,K]
    else:
        state = torch.zeros(batch, num_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=q.device)
    out = torch.zeros(batch, seq_len, num_heads, head_v_dim, dtype=torch.float32, device=q.device)
    for t in range(seq_len):
        qt = qf[:, t]          # [B,H,K]
        kt = kf[:, t]
        vt = vf[:, t]          # [B,H,V]
        gt = gf[:, t]          # log-space gate
        bt = bf[:, t]          # [B,H]
        alpha = torch.exp(gt)
        state = state * alpha.unsqueeze(2)               # Diag(alpha) on K
        pred = (state * kt.unsqueeze(2)).sum(-1)         # [B,H,V]
        delta = bt.unsqueeze(-1) * (vt - pred)
        state = state + delta.unsqueeze(-1) * kt.unsqueeze(2)
        ot = (state * qt.unsqueeze(2)).sum(-1) * scale
        out[:, t] = ot
    final_state = state if output_final_state else None
    return out.to(dt), final_state


fused_recurrent_kda = _naive_recurrent_kda


# ---------------------------------------------------------------------------
# torch reference for the FLA chunk_kda op (vectorized chunked)
# ---------------------------------------------------------------------------

_CHUNK = 16  # keep exp(-G) in fp32 range for g in [-5,0]: 16*5=80, exp(80)<3.4e38


def _chunk_step(chunk, state, scale):
    """One subchunk of the blocked delta-rule recurrence.

    chunk: ``Qkvgb`` of [B,H,sub,*] slices; state [B,H,V,K] fp32.
    Returns (o_chunk [B,H,sub,V], new_state [B,H,V,K]).
    """
    qc, kc, vc, gc, bc = chunk
    batch, num_heads, sub_len, _ = qc.shape
    device = qc.device
    g_chunk = gc.cumsum(dim=2)
    exp_g = torch.exp(g_chunk)
    exp_neg_g = torch.exp(-g_chunk)
    qd = qc * exp_g
    kd = kc * exp_neg_g
    idx = torch.arange(sub_len, device=device)
    le = (idx[:, None] >= idx[None, :]).float()
    lt = (idx[:, None] > idx[None, :]).float()
    # a_qk: scaled qd·kdᵀ masked by tril_le
    a_qk = scale * torch.einsum('bhik,bhjk->bhij', qd, kd) * le
    # a_kk: strictly-lower kg·kdᵀ scaled by beta (nilpotent decay matrix)
    a_kk = torch.einsum('bhik,bhjk->bhij', kc * exp_g, kd)
    a_kk = (bc.unsqueeze(-1) * a_kk) * lt
    eye = torch.eye(sub_len, device=device).expand(batch, num_heads, sub_len, sub_len)
    t_mat = torch.linalg.solve_triangular(eye + a_kk, eye, upper=False, unitriangular=True)
    bv = bc.unsqueeze(-1) * vc
    bgk = bc.unsqueeze(-1) * exp_g * kc
    # u and w: apply the triangular inverse to bv and bgk
    u = torch.einsum('bhij,bhjv->bhiv', t_mat, bv)
    w = torch.einsum('bhij,bhjk->bhik', t_mat, bgk)
    w_s = torch.einsum('bhik,bhvk->bhiv', w, state)
    v_new = u - w_s
    o_inter = scale * torch.einsum('bhik,bhvk->bhiv', qd, state)
    o_intra = torch.einsum('bhij,bhjv->bhiv', a_qk, v_new)
    o_chunk = o_inter + o_intra
    g_last = g_chunk[:, :, sub_len - 1]
    kg = torch.exp(g_last.unsqueeze(2) - g_chunk) * kc
    new_state = torch.exp(g_last).unsqueeze(2) * state + torch.einsum('bhjk,bhjv->bhvk', kg, v_new)
    return o_chunk, new_state


def vec_chunk_kda(q, k, v, g, beta, initial_state=None, output_final_state=True,
                  use_qk_l2norm_in_kernel=True, cu_seqlens=None, scale=None, **kwargs):
    """Vectorized chunked-torch KDA (torch reference op).

    Same blocked delta-rule algorithm that fla.ops.kda.chunk_kda fuses on CUDA,
    expressed in eager PyTorch (batched einsum / triangular-solve per chunk).
    NPU-runnable and numerically stable (subchunk=16). Matches
    ``_naive_recurrent_kda`` to <= 6e-5.

    FLA layout: q,k,g [B,T,H,K]; v [B,T,H,V]; beta [B,T,H]; state [B,H,V,K] fp32.
    Returns (out [B,T,H,V] same dtype as q, final_state [B,H,V,K] fp32).
    """
    batch, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    dt = q.dtype
    if scale is None:
        scale = head_k_dim ** -0.5
    qf, kf, vf, gf, bf = _prep_qkvgb(Qkvgb(q, k, v, g, beta), use_qk_l2norm_in_kernel)
    qf = qf.transpose(1, 2)
    kf = kf.transpose(1, 2)
    vf = vf.transpose(1, 2)
    gf = gf.transpose(1, 2)
    bf = bf.transpose(1, 2)  # [B,H,T,*], bf [B,H,T]

    if initial_state is None:
        state = torch.zeros(batch, num_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.float().clone()
    out = torch.zeros(batch, num_heads, seq_len, head_v_dim, dtype=torch.float32, device=q.device)

    chunk = _CHUNK
    num_chunks = (seq_len + chunk - 1) // chunk
    for c in range(num_chunks):
        s0 = c * chunk
        s1 = min(s0 + chunk, seq_len)
        chunk_in = Qkvgb(qf[:, :, s0:s1], kf[:, :, s0:s1], vf[:, :, s0:s1],
                         gf[:, :, s0:s1], bf[:, :, s0:s1])
        o_chunk, state = _chunk_step(chunk_in, state, scale)
        out[:, :, s0:s1] = o_chunk

    out = out.transpose(1, 2).contiguous().to(dt)  # [B,T,H,V]
    return out, (state if output_final_state else None)


chunk_kda = vec_chunk_kda
