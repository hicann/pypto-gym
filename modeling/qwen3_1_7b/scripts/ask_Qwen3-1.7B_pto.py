#!/usr/bin/env python3
"""Integrate pypto fused kernels into Qwen3-1.7B via monkey-patch.

Usage:
    python3 ask_Qwen3-1.7B_pto.py --device 7 [--prompt ...] [--no-pto]

By default this imports the pypto kernel adapters and patches every
Qwen3DecoderLayer instance so its forward calls our fused kernels.
Attention itself continues to use torch's eager_attention_forward path.
"""
import os
import sys
from pathlib import Path
import argparse
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer, AutoModelForCausalLM


def patch_qwen3_with_pypto(model, adapter, force_torch_attn=False):
    """Monkey-patch every Qwen3DecoderLayer.forward to use pypto fused kernels."""
    from typing import Optional
    from types import MethodType

    # Find a Qwen3DecoderLayer class instance to access original forward
    any_layer = None
    for m in model.modules():
        if m.__class__.__name__ == "Qwen3DecoderLayer":
            any_layer = m
            break
    if any_layer is None:
        raise RuntimeError("No Qwen3DecoderLayer found")

    # Import original eager_attention_forward from the loaded modeling module
    modeling_module = sys.modules[any_layer.__module__]
    eager_attention_forward = modeling_module.eager_attention_forward

    # Prepare weights per-layer (once, at patch time).
    # All matmul kernels use b_trans=True, so we keep natural Linear.weight layout [out, in].
    def _prep_weights(layer):
        att = layer.self_attn
        mlp = layer.mlp
        layer._pto_Wq = att.q_proj.weight.contiguous()
        layer._pto_Wk = att.k_proj.weight.contiguous()
        layer._pto_Wv = att.v_proj.weight.contiguous()
        layer._pto_Wo = att.o_proj.weight.contiguous()
        layer._pto_Wgate = mlp.gate_proj.weight.contiguous()
        layer._pto_Wup = mlp.up_proj.weight.contiguous()
        layer._pto_Wdown = mlp.down_proj.weight.contiguous()
        layer._pto_w_in_norm = layer.input_layernorm.weight
        layer._pto_w_post_norm = layer.post_attention_layernorm.weight
        layer._pto_w_q_norm = att.q_norm.weight
        layer._pto_w_k_norm = att.k_norm.weight

    # New forward for Qwen3DecoderLayer
    def pto_layer_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        # shape [B, S, H]
        B, S, H = hidden_states.shape
        att = self.self_attn
        Nq, Nkv, D = att.config.num_attention_heads, att.config.num_key_value_heads, att.head_dim

        x_flat = hidden_states.reshape(B * S, H).contiguous()
        residual_flat = x_flat  # keep residual1 input

        # --- Fused pre-attention: RMSNorm + QKV proj + Q/K-norm + RoPE ---
        cos, sin = position_embeddings
        cos_flat = cos.reshape(B * S, D).contiguous()
        sin_flat = sin.reshape(B * S, D).contiguous()
        q3d_roped, k3d_roped, v3d = adapter.pre_attn_fused(
            x_flat, cos_flat, sin_flat,
            self._pto_w_in_norm,
            self._pto_Wq, self._pto_Wk, self._pto_Wv,
            self._pto_w_q_norm, self._pto_w_k_norm,
            Nq=Nq, Nkv=Nkv, D=D,
        )

        # reshape to [B, N, S, D] (match attention interface)
        query_states = q3d_roped.view(B, S, Nq, D).transpose(1, 2)
        key_states   = k3d_roped.view(B, S, Nkv, D).transpose(1, 2)
        value_states = v3d.view(B, S, Nkv, D).transpose(1, 2)

        # --- KV cache update ---
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, att.layer_idx, cache_kwargs)

        # --- Attention ---
        Sq_now = query_states.shape[2]
        if Sq_now == 1 and B == 1 and not force_torch_attn:
            S2_TILE = 64
            q_2d = query_states.reshape(Nq, D)
            # K/V: [1, Nkv, Skv, D] -> [Nkv, Skv, D]   (no repeat: kernel handles GQA)
            k_full = key_states.squeeze(0).contiguous()
            v_full = value_states.squeeze(0).contiguous()
            cur_skv = k_full.shape[1]
            Skv_p = ((cur_skv + S2_TILE - 1) // S2_TILE) * S2_TILE
            if Skv_p != cur_skv:
                pad = torch.zeros(Nkv, Skv_p - cur_skv, D, device=k_full.device, dtype=k_full.dtype)
                k_full = torch.cat([k_full, pad], dim=1)
                v_full = torch.cat([v_full, pad], dim=1)
            o_2d = adapter.decode_attn(q_2d, k_full, v_full, cur_skv)
            attn_output = o_2d.view(B, Sq_now, Nq, D)
        else:
            attn_output, _ = eager_attention_forward(
                att,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0,
                scaling=att.scaling,
                sliding_window=att.sliding_window,
                **kwargs,
            )
        # attn_output: [B, S, Nq, D]
        attn_output_flat = attn_output.reshape(B * Sq_now, Nq * D).contiguous()

        # --- K3: O-proj + residual1 + post-RMSNorm + SwiGLU MLP + residual2 ---
        y_flat = adapter.post_attn(
            attn_output_flat,
            residual_flat,
            self._pto_Wo,
            self._pto_w_post_norm,
            self._pto_Wgate, self._pto_Wup, self._pto_Wdown,
        )

        y = y_flat.view(B, Sq_now, H)
        return y

    # Apply to every layer
    n_patched = 0
    for m in model.modules():
        if m.__class__.__name__ == "Qwen3DecoderLayer":
            _prep_weights(m)
            m.forward = MethodType(pto_layer_forward, m)
            n_patched += 1
    print(f"[pto] Patched {n_patched} Qwen3DecoderLayer instances.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="你好")
    parser.add_argument("--no-pto", action="store_true", help="Disable pypto patch (torch baseline)")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--model-path", type=str, default="/data/z00885570/models/Qwen3-1.7B")
    parser.add_argument("--pto-attn", action="store_true", help="Enable pypto fused decode attention (default: use torch eager attention which is faster)")
    parser.add_argument("--step", action="store_true", help="Manual decode loop with progress prints")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    print(f"[pto] device={device} pto={'OFF' if args.no_pto else 'ON'}")

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device).eval()

    if not args.no_pto:
        # adapter sits next to this script, in ../qwen3_pto_kernels
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import qwen3_pto_kernels as adapter
        patch_qwen3_with_pypto(model, adapter, force_torch_attn=not args.pto_attn)

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    if args.step:
        # Manual decode loop with per-token timing.
        # Warmup: run a short generation first so the JIT cache + dynamic-shape
        # specializations are in place; then time the second run as steady state.
        def _gen_loop(input_ids, n_new, label):
            ids = input_ids
            past = None
            all_ids = ids
            t0 = time.perf_counter()
            with torch.no_grad():
                for i in range(n_new):
                    t_step = time.perf_counter()
                    out = model(input_ids=ids, past_key_values=past, use_cache=True)
                    logits = out.logits[:, -1, :]
                    next_id = logits.argmax(-1, keepdim=True)
                    past = out.past_key_values
                    all_ids = torch.cat([all_ids, next_id], dim=1)
                    ids = next_id
                    dt = time.perf_counter() - t_step
                    tok_text = tok.decode(next_id[0], skip_special_tokens=True)
                    print(f"[{label} step {i+1}] {dt*1000:.0f}ms  '{tok_text}'", flush=True)
            return all_ids, time.perf_counter() - t0

        print("=== Warmup run (JIT compile) ===", flush=True)
        _gen_loop(inputs.input_ids, args.max_new, "warmup")
        print("=== Steady-state run ===", flush=True)
        all_ids, dt_total = _gen_loop(inputs.input_ids, args.max_new, "steady")
        text = tok.decode(all_ids[0], skip_special_tokens=True)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        dt = time.perf_counter() - t0
        text = tok.decode(out[0], skip_special_tokens=True)
        print(f"\n[pto] generated {out.shape[1] - inputs.input_ids.shape[1]} tokens in {dt:.2f}s")
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
