#!/usr/bin/env python3
"""Torch-native golden reference for Qwen3-1.7B DecoderLayer fused op.

Provides two entry points:
  - qwen3_decoder_layer_prefill_golden(...)
  - qwen3_decoder_layer_decode_golden(...)

Each replicates Qwen3DecoderLayer.forward exactly and returns (y, k_cache_out, v_cache_out).
Used both as target for kernel correctness check and as standalone sanity check against
the original transformers modeling_qwen3 implementation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import torch


# ---- config constants (Qwen3-1.7B) ----
H = 2048
Nq = 16
Nkv = 8
D = 128
INT_SIZE = 6144
GROUPS = Nq // Nkv  # 2
EPS = 1e-6
SCALE = 1.0 / math.sqrt(D)


@dataclass
class LayerWeights:
    w_in_norm: torch.Tensor      # [H]
    Wq: torch.Tensor             # [Nq*D, H]  (torch nn.Linear weight is [out, in])
    Wk: torch.Tensor             # [Nkv*D, H]
    Wv: torch.Tensor             # [Nkv*D, H]
    w_q_norm: torch.Tensor       # [D]
    w_k_norm: torch.Tensor       # [D]
    Wo: torch.Tensor             # [H, Nq*D]
    w_post_norm: torch.Tensor    # [H]
    Wgate: torch.Tensor          # [INT_SIZE, H]
    Wup: torch.Tensor            # [INT_SIZE, H]
    Wdown: torch.Tensor          # [H, INT_SIZE]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight.to(torch.float32) * x).to(in_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # cos/sin: [B, S, D]; q/k: [B, N, S, D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def _attention_block(
    x: torch.Tensor,                 # [B, S, H] BF16
    cos: torch.Tensor, sin: torch.Tensor,
    W: LayerWeights,
    k_cache_prev: Optional[torch.Tensor],  # [B, Nkv, Sprev, D] or None
    v_cache_prev: Optional[torch.Tensor],
):
    B, Sq, _ = x.shape
    # Pre-attn RMSNorm
    n1 = rms_norm(x, W.w_in_norm)                                             # [B, Sq, H] BF16
    # Q/K/V proj
    q = torch.nn.functional.linear(n1, W.Wq)                                  # [B, Sq, Nq*D]
    k = torch.nn.functional.linear(n1, W.Wk)                                  # [B, Sq, Nkv*D]
    v = torch.nn.functional.linear(n1, W.Wv)                                  # [B, Sq, Nkv*D]
    q = q.view(B, Sq, Nq, D)
    k = k.view(B, Sq, Nkv, D)
    v = v.view(B, Sq, Nkv, D)
    # per-head RMSNorm on Q/K
    q = rms_norm(q, W.w_q_norm)                                               # normalises along D
    k = rms_norm(k, W.w_k_norm)
    # transpose to [B, N, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # RoPE
    q, k = apply_rope(q, k, cos, sin)
    # Concat with previous KV cache
    if k_cache_prev is not None:
        k_full = torch.cat([k_cache_prev, k], dim=2)
        v_full = torch.cat([v_cache_prev, v], dim=2)
    else:
        k_full = k
        v_full = v
    Skv = k_full.shape[2]
    # repeat_kv to Nq heads
    k_rep = repeat_kv(k_full, GROUPS)
    v_rep = repeat_kv(v_full, GROUPS)
    # Attention
    scores = torch.matmul(q.to(torch.float32), k_rep.to(torch.float32).transpose(-1, -2)) * SCALE  # [B, Nq, Sq, Skv]
    # causal mask: j <= i + (Skv - Sq)
    mask = torch.ones(Sq, Skv, dtype=torch.bool, device=x.device)
    offset = Skv - Sq
    for i in range(Sq):
        mask[i, : offset + i + 1] = False  # keep
    scores = scores.masked_fill(mask, float('-inf'))
    p = torch.softmax(scores, dim=-1).to(x.dtype)
    o = torch.matmul(p, v_rep.to(x.dtype))                                    # [B, Nq, Sq, D]
    o = o.transpose(1, 2).contiguous().view(B, Sq, Nq * D)
    o = torch.nn.functional.linear(o, W.Wo)                                   # [B, Sq, H]
    return x + o, k_full, v_full


def _mlp_block(h: torch.Tensor, W: LayerWeights) -> torch.Tensor:
    n2 = rms_norm(h, W.w_post_norm)
    gate = torch.nn.functional.linear(n2, W.Wgate)   # [B, S, INT]
    up = torch.nn.functional.linear(n2, W.Wup)       # [B, S, INT]
    silu_gate = gate * torch.sigmoid(gate)           # SiLU
    mlp_hidden = silu_gate * up                      # [B, S, INT]
    down = torch.nn.functional.linear(mlp_hidden, W.Wdown)  # [B, S, H]
    return h + down


def qwen3_decoder_layer_prefill_golden(
    x: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    W: LayerWeights,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill: no prior KV cache. Returns (y, k_cache_out, v_cache_out)."""
    h1, k_full, v_full = _attention_block(x, cos, sin, W, None, None)
    y = _mlp_block(h1, W)
    return y, k_full, v_full


def qwen3_decoder_layer_decode_golden(
    x: torch.Tensor,                  # [B, 1, H]
    cos: torch.Tensor, sin: torch.Tensor,   # [B, 1, D]
    k_cache_in: torch.Tensor,         # [B, Nkv, Sprev, D]
    v_cache_in: torch.Tensor,
    W: LayerWeights,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode: Sq=1. Returns (y, k_cache_out=[B,Nkv,Sprev+1,D], v_cache_out)."""
    assert x.shape[1] == 1
    h1, k_full, v_full = _attention_block(x, cos, sin, W, k_cache_in, v_cache_in)
    y = _mlp_block(h1, W)
    return y, k_full, v_full


# ===========================================================================
# Sanity check: golden == transformers Qwen3DecoderLayer
# ===========================================================================

def _sanity_check():
    import sys
    import os
    model_dir = os.environ.get("QWEN3_MODEL_DIR", "/data/z00885570/models/Qwen3-1.7B")
    sys.path.insert(0, model_dir)
    from core.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
    from core.configuration_qwen3 import Qwen3Config
    import json

    cfg_path = os.path.join(model_dir, 'config.json')
    cfg_dict = json.load(open(cfg_path))
    # Drop unknown keys so Qwen3Config ctor is stable across transformers versions
    for drop in ('auto_map',):
        cfg_dict.pop(drop, None)
    config = Qwen3Config(**{k: v for k, v in cfg_dict.items() if k not in ('auto_map',)})
    config._attn_implementation = "eager"
    # layer_types is required by the modeling code
    if not hasattr(config, 'layer_types') or config.layer_types is None:
        config.layer_types = ["full_attention"] * config.num_hidden_layers

    torch.manual_seed(0)
    layer = Qwen3DecoderLayer(config, layer_idx=0).to(torch.bfloat16).eval()
    rope = Qwen3RotaryEmbedding(config=config).to(torch.bfloat16).eval()

    # Assemble LayerWeights from the torch module
    W = LayerWeights(
        w_in_norm=layer.input_layernorm.weight.detach(),
        Wq=layer.self_attn.q_proj.weight.detach(),
        Wk=layer.self_attn.k_proj.weight.detach(),
        Wv=layer.self_attn.v_proj.weight.detach(),
        w_q_norm=layer.self_attn.q_norm.weight.detach(),
        w_k_norm=layer.self_attn.k_norm.weight.detach(),
        Wo=layer.self_attn.o_proj.weight.detach(),
        w_post_norm=layer.post_attention_layernorm.weight.detach(),
        Wgate=layer.mlp.gate_proj.weight.detach(),
        Wup=layer.mlp.up_proj.weight.detach(),
        Wdown=layer.mlp.down_proj.weight.detach(),
    )

    # Prefill
    B, S = 1, 16
    x = torch.randn(B, S, H, dtype=torch.bfloat16)
    pos_ids = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rope(x, pos_ids)
        # Build causal mask for transformers: additive [B, 1, Sq, Skv] of -inf on masked
        causal_mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
        for i in range(S):
            causal_mask[:, :, i, i+1:] = float('-inf')
        ref = layer(x, position_embeddings=(cos, sin), attention_mask=causal_mask)
        if isinstance(ref, tuple): ref = ref[0]
        y_gold, k_cache, v_cache = qwen3_decoder_layer_prefill_golden(x, cos, sin, W)

    diff = (ref.float() - y_gold.float()).abs()
    print(f"[prefill] max_diff={diff.max().item():.6f} mean_diff={diff.mean().item():.6f} shape={y_gold.shape}")
    assert diff.max().item() < 0.02, "golden diverges from transformers"

    # Decode
    with torch.no_grad():
        x1 = torch.randn(B, 1, H, dtype=torch.bfloat16)
        pos_ids1 = torch.tensor([[S]])
        cos1, sin1 = rope(x1, pos_ids1)
        y_d, k_cache2, v_cache2 = qwen3_decoder_layer_decode_golden(x1, cos1, sin1, k_cache, v_cache, W)
        print(f"[decode] y_shape={y_d.shape} k_cache_shape={k_cache2.shape}")
        assert k_cache2.shape == (B, Nkv, S+1, D)

    print("Golden sanity check PASSED.")


if __name__ == "__main__":
    _sanity_check()
