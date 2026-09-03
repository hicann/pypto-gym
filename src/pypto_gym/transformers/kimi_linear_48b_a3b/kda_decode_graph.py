# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Graph-capturable decode stage for Kimi-Linear-48B-A3B.

Provides:
    StaticDecodeCache      Preallocated KV cache for MLA + KDA buffers (no torch.cat)
    make_decode_stage      Per-token closure over one rank's layers
    make_decode_stage_run  Warmup, capture, and determinism verification
    seed_from_prefill      Copy finished prefill into static cache

The static MoE route is installed from bench_kimi_multinpu.py before capture.
One captured graph is valid while the KV window layout holds.
"""
__all__ = [
    "StaticDecodeCache",
    "prepare_decode_weights",
    "make_decode_stage",
    "make_decode_stage_run",
    "seed_from_prefill",
    "DecodeStageParams",
    "DecodeStageRunParams",
]

import logging
from dataclasses import dataclass

import torch

log = logging.info


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

@dataclass
class DecodeStageParams:
    """Parameters for creating a decode stage."""
    model: object
    layer_ids: list
    cache: object
    prepared: dict
    kda_impl: object
    dtype: torch.dtype
    use_fused: bool = True


@dataclass
class DecodeStageRunParams:
    """Parameters for running a decode stage."""
    stage: object
    cache: object
    batch: int
    hidden_size: int
    device: torch.device
    dtype: torch.dtype
    warmup: int = 3


@dataclass
class LayerProcessArgs:
    """Arguments for processing one attention layer."""
    hidden: torch.Tensor
    layer: object
    idx: int
    cache: object
    prepared: dict
    kda_impl: object
    dtype: torch.dtype
    mask: torch.Tensor | None
    use_fused: bool


# ---------------------------------------------------------------------------
# static cache
# ---------------------------------------------------------------------------

class StaticDecodeCache:
    """Fixed-size decode cache for graph capture.

    Every tensor allocated once and mutated in place.
    MLA layers: [B, H, max_len, D] key/value.
    KDA layers: one KdaDecodeBuffers per layer.
    """

    def __init__(self, cfg, layers, batch, max_len, dtype, device,
                 kda_impl):
        self.cfg = cfg
        self.batch = batch
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.kda_buffers = {}
        self.key_cache = {}
        self.value_cache = {}
        self.pos = torch.zeros((), dtype=torch.int32, device=device)

        for idx, layer in layers.items():
            sa = layer.self_attn
            if getattr(layer, "is_linear_attn", False):
                params = kda_impl.KdaBufferParams(
                    idx, batch, sa.num_heads, sa.head_dim,
                    self.cfg.hidden_size, dtype, device)
                self.kda_buffers[idx] = kda_impl.make_fused_buffers(params)
            else:
                q_head = sa.qk_nope_head_dim + sa.qk_rope_head_dim
                self.key_cache[idx] = torch.zeros(
                    batch, sa.num_heads, max_len, q_head, dtype=dtype,
                    device=device)
                self.value_cache[idx] = torch.zeros(
                    batch, sa.num_heads, max_len, sa.v_head_dim, dtype=dtype,
                    device=device)

    def write_kv(self, layer_idx, key, value):
        """Scatter one token into cache at device cursor."""
        k_buf = self.key_cache[layer_idx]
        v_buf = self.value_cache[layer_idx]
        idx = self.pos.reshape(1).long()
        k_buf.index_copy_(2, idx, key.to(k_buf.dtype))
        v_buf.index_copy_(2, idx, value.to(v_buf.dtype))
        return k_buf, v_buf

    def causal_mask(self, dtype):
        """Additive mask valid up to current cursor.

        MUST be called inside stage: reads pos which changes every step.
        """
        valid = torch.arange(self.max_len, device=self.device)
        keep = valid.view(1, 1, 1, -1) <= self.pos.long()
        return torch.where(
            keep,
            torch.zeros((), dtype=dtype, device=self.device),
            torch.full((), torch.finfo(dtype).min, dtype=dtype,
                       device=self.device))

    def advance(self):
        """Bump cursor. Runs inside graph."""
        self.pos.add_(1)

    def snapshot(self):
        return (
            self.pos.clone(),
            {i: (k.clone(), self.value_cache[i].clone())
             for i, k in self.key_cache.items()},
            {i: (b.conv_state.clone(), b.state.clone())
             for i, b in self.kda_buffers.items()},
        )

    def restore(self, snap):
        pos, kv, kda = snap
        self.pos.copy_(pos)
        for i, (k, v) in kv.items():
            self.key_cache[i].copy_(k)
            self.value_cache[i].copy_(v)
        for i, (cs, st) in kda.items():
            b = self.kda_buffers[i]
            b.conv_state.copy_(cs)
            b.state.copy_(st)

    def reset(self):
        self.pos.zero_()
        for t in list(self.key_cache.values()) + list(self.value_cache.values()):
            t.zero_()
        for b in self.kda_buffers.values():
            b.conv_state.zero_()
            b.state.zero_()


def seed_from_prefill(cache, dyn_cache, seed_fused_buffers):
    """Copy finished prefill (KimiDynamicCache) into static cache.

    Runs once at prefill -> decode handoff, outside any captured region.
    """
    seq = None
    for idx in cache.key_cache:
        k = dyn_cache.key_cache[idx]
        v = dyn_cache.value_cache[idx]
        seq = k.shape[2]
        cache.key_cache[idx][:, :, :seq].copy_(k)
        cache.value_cache[idx][:, :, :seq].copy_(v)
    for idx, buf in cache.kda_buffers.items():
        seed_fused_buffers(buf, dyn_cache.conv_states[idx],
                            dyn_cache.recurrent_states[idx])
    if seq is not None:
        cache.pos.fill_(seq)
    return cache


# ---------------------------------------------------------------------------
# decode forwards
# ---------------------------------------------------------------------------

def _mla_decode_forward(self, hidden_states, cache, causal_mask):
    """MLA layer, one token, against preallocated KV."""
    b, t, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(b, t, -1, self.q_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(
        compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(
        b, t, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
    k_pass, value = torch.split(
        k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(b, 1, t, self.qk_rope_head_dim).expand(
        *k_pass.shape[:-1], -1)

    key = torch.cat((k_pass, k_rot), dim=-1)
    query = torch.cat((q_pass, q_rot), dim=-1)

    k_all, v_all = cache.write_kv(self.layer_idx, key, value)

    attn = torch.matmul(query, k_all.transpose(2, 3)) * self.scaling + causal_mask
    attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(
        query.dtype)
    out = torch.matmul(attn, v_all).transpose(1, 2).reshape(b, t, -1)
    return self.o_proj(out)


def prepare_decode_weights(model, layer_ids, kda_impl, gate_lower_bound=-5.0):
    """Pack fused-op weights for this rank's KDA layers.

    Call once before capture.
    """
    prepared = {}
    for idx in layer_ids:
        layer = model.model.layers[idx]
        if getattr(layer, "is_linear_attn", False):
            prepared[idx] = kda_impl.prepare_kda_fused_weights(
                layer.self_attn, lower_bound=gate_lower_bound)
    log(f"[decode] fused KDA weights prepared for {len(prepared)} layers")
    return prepared


def make_decode_stage(params):
    """Comm-free decode closure over this rank's layers."""
    model, layer_ids, cache, prepared, kda_impl, dtype, use_fused = (
        params.model, params.layer_ids, params.cache, params.prepared,
        params.kda_impl, params.dtype, params.use_fused)

    def stage(hidden):
        mask = cache.causal_mask(dtype)
        for idx in layer_ids:
            layer = model.model.layers[idx]
            residual = hidden
            h = layer.input_layernorm(hidden)
            args = LayerProcessArgs(h, layer, idx, cache, prepared, kda_impl, dtype, mask, use_fused)
            h = _process_layer(args)
            hidden = residual + h

            residual = hidden
            h = layer.post_attention_layernorm(hidden)
            h = _run_mlp_or_moe(h, layer)
            hidden = residual + h
        cache.advance()
        return hidden

    return stage


def _process_layer(args):
    """Process one attention layer (KDA or MLA)."""
    if getattr(args.layer, "is_linear_attn", False):
        if args.use_fused:
            if args.hidden.dtype != args.dtype:
                args.hidden = args.hidden.to(args.dtype)
            return args.kda_impl.kda_fused_decode_step(
                args.hidden, args.prepared[args.idx], args.cache.kda_buffers[args.idx])
        return _torch_kda_decode(args.layer.self_attn, args.hidden,
                                args.cache.kda_buffers[args.idx], args.kda_impl)
    return _mla_decode_forward(args.layer.self_attn, args.hidden, args.cache, args.mask)


def _run_mlp_or_moe(hidden, layer):
    """Run MLP or MoE block."""
    return (layer.block_sparse_moe(hidden) if hasattr(layer, "block_sparse_moe")
            else layer.mlp(hidden))


def pack_conv_state(q_cache: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    """Pack (q, k, v) conv histories [B, P, K-1] into [B, K-1, 3P] format."""
    return torch.cat([q_cache, k_cache, v_cache], dim=1).transpose(1, 2).contiguous()


def unpack_conv_state(packed: torch.Tensor, proj_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack [B, K-1, 3P] into (q, k, v) conv histories [B, P, K-1]."""
    unpacked = packed.transpose(1, 2)
    q, k, v = unpacked.chunk(3, dim=1)
    return q, k, v


def _torch_kda_decode(sa, hidden, buf, kda_impl):
    """Baseline KDA decode over same persistent buffers for benchmark comparison."""
    from pypto_gym.transformers.kimi_linear_48b_a3b.kimi_fla_compat import (
        fused_kda_gate, fused_recurrent_kda)

    x = hidden
    proj_dim = sa.num_heads * sa.head_dim
    cq, ck, cv = unpack_conv_state(buf.conv_state, proj_dim)
    q, cq = sa.q_conv1d(sa.q_proj(x), cache=cq, output_final_state=True)
    k, ck = sa.k_conv1d(sa.k_proj(x), cache=ck, output_final_state=True)
    v, cv = sa.v_conv1d(sa.v_proj(x), cache=cv, output_final_state=True)
    buf.conv_state.copy_(pack_conv_state(cq, ck, cv))

    g = fused_kda_gate(sa.f_b_proj(sa.f_a_proj(x)), sa.A_log, sa.head_dim,
                       g_bias=sa.dt_bias)
    beta = sa.b_proj(x).float().sigmoid()
    num_heads, head_dim = sa.num_heads, sa.head_dim
    shape = (*x.shape[:2], num_heads, head_dim)
    o, new_state = fused_recurrent_kda(
        q.view(shape), k.view(shape), v.view(shape), g, beta,
        initial_state=buf.state.view(x.shape[0], num_heads, head_dim, head_dim),
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    buf.state.copy_(new_state.reshape(buf.state.shape).float())

    gate = sa.g_b_proj(sa.g_a_proj(x)).view(*o.shape)
    o = sa.o_norm(o, gate).reshape(*x.shape[:2], proj_dim)
    return sa.o_proj(o)


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

def make_decode_stage_run(params):
    """Warm up, capture, and verify replay determinism with state restore."""
    stage, cache, batch, hidden_size, device, dtype, warmup = (
        params.stage, params.cache, params.batch, params.hidden_size,
        params.device, params.dtype, params.warmup)
    hin = torch.zeros(batch, 1, hidden_size, dtype=dtype, device=device)
    base = cache.snapshot()
    with torch.no_grad():
        for _ in range(warmup):
            stage(hin)
    cache.restore(base)
    torch.npu.synchronize()

    gobj = torch.npu.NPUGraph()
    with torch.no_grad(), torch.npu.graph(gobj):
        hout = stage(hin)
    torch.npu.synchronize()
    log("[decode-graph] capture OK")

    def stage_run(hidden):
        hin.copy_(hidden)
        gobj.replay()
        return hout

    probe = torch.randn(batch, 1, hidden_size, dtype=dtype, device=device)
    cache.restore(base)
    o1 = stage_run(probe).clone()
    cache.restore(base)
    o2 = stage_run(probe).clone()
    if not torch.equal(o1, o2):
        raise RuntimeError(
            "[decode-graph] replay differs from identical state -- captured graph "
            "is non-deterministic.")
    log("[decode-graph] determinism OK (state restored between probes)")
    cache.restore(base)
    return stage_run
