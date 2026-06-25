# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Reusable capture-safe NPUGraph helpers for HF models on Ascend NPU.

Generalizes the per-model harnesses (gemma4 decode capture, llada2 diffusion-block capture,
minimax_m27 static-route grouped GEMM) into model-agnostic utilities. See
references/npu-run-and-measure.md for the error->cause->fix table this encodes.
"""
import sys
import time

import torch

# error code -> (cause, fix) ; see references/npu-run-and-measure.md
CAPTURE_GOTCHAS = {
    "107025": ("default SDPA fused kernel on a side stream, OR a host sync (mask/rope)",
               "load attn_implementation='eager'; prebuilt mask; precompute cos/sin"),
    "107027": ("an op uses a copy/side stream under capture - most often MoE argsort(int64) "
               "on AICPU, but any side-stream op (sliding-window mask build, qk-norm) qualifies",
               "MoE -> static routing (fixed assignment, precomputed cumsum); else bisect"),
    "107030": ("H2D copy during capture (DynamicCache host ops / mask prep)",
               "capture-safe KV cache (fixed-slot index_copy_); prebuilt mask"),
    "507015": ("PyPTO grouped_gemm fed a growing/dynamic shape",
               "fix the shape before capture (single block / bucket-pad n)"),
}


def explain_capture_error(err):
    """Map an ACL error string to the matching capture gotcha and its fix."""
    text = str(err)
    for code, (cause, fix) in CAPTURE_GOTCHAS.items():
        if code in text:
            return "[capture {}] cause: {}\n            fix: {}".format(code, cause, fix)
    return "[capture FAILED] {}: {}".format(type(err).__name__, text[:200])


def find_body(model):
    """The decoder stack: has embed + layers + final norm."""
    for module in model.modules():
        has_layers = hasattr(module, "layers")
        has_norm = hasattr(module, "norm") or hasattr(module, "final_layernorm")
        has_embed = hasattr(module, "embed_tokens") or hasattr(module, "word_embeddings")
        if has_layers and has_norm and has_embed:
            return module
    return None


def find_rotary(model):
    """Locate the rotary-embedding module (has inv_freq + rope_init_fn)."""
    return next(
        (module for module in model.modules()
         if hasattr(module, "inv_freq") and hasattr(module, "rope_init_fn")),
        None,
    )


def apply_capture_mitigations(model, window, device, dtype=torch.bfloat16):
    """Neutralize the host-sync sources that abort capture, for a FIXED ``window``.

    Returns (cos, sin, zero_mask, position_ids) for a manual decoder-stack forward. Assumes
    the model was loaded with attn_implementation='eager' (the side-stream fix this cannot
    do retroactively).
    """
    position_ids = torch.arange(window, device=device).unsqueeze(0)

    # 1) mask prep host sync -> return the prebuilt additive mask verbatim
    modeling = sys.modules.get(model.__class__.__module__)
    if modeling is not None and hasattr(modeling, "_prepare_4d_causal_attention_mask_for_sdpa"):
        def passthrough_mask(attention_mask, *args, **kwargs):
            return attention_mask
        setattr(modeling, "_prepare_4d_causal_attention_mask_for_sdpa", passthrough_mask)

    # 2) rotary autocast + dynamic_rope_update host checks -> cache cos/sin (fixed positions)
    cos = None
    sin = None
    rotary = find_rotary(model)
    if rotary is not None:
        probe = torch.zeros(1, window, 1, device=device, dtype=dtype)
        with torch.no_grad():
            cos, sin = rotary.forward(probe, position_ids)
        cos = cos.contiguous()
        sin = sin.contiguous()

        def cached_rotary(hidden_states, position_ids=None):
            return cos, sin
        rotary.forward = cached_rotary

    zero_mask = torch.zeros(1, 1, window, window, device=device, dtype=dtype)
    return cos, sin, zero_mask, position_ids


class CaptureCache:
    """Duck-typed KV cache with fixed-slot writes and no host ops (capture-stable).

    Per-layer head_dim / n_kv supported (e.g. gemma sliding vs global). Advance the
    ``write_pos`` tensor IN PLACE between replays to walk the decode positions.
    """

    def __init__(self, hd_list, nkv_list, maxlen, device, dtype=torch.bfloat16, batch=1):
        num_layers = len(hd_list)
        self.keys = [
            torch.zeros(batch, nkv_list[i], maxlen, hd_list[i], device=device, dtype=dtype)
            for i in range(num_layers)
        ]
        self.values = [
            torch.zeros(batch, nkv_list[i], maxlen, hd_list[i], device=device, dtype=dtype)
            for i in range(num_layers)
        ]
        self.write_pos = None      # device LongTensor, advanced in place
        self.length = 0

    def update(self, key, value, layer_idx, *args, **kwargs):
        self.keys[layer_idx].index_copy_(2, self.write_pos, key)
        self.values[layer_idx].index_copy_(2, self.write_pos, value)
        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.length

    def get_mask_sizes(self, q_length, layer_idx=0):
        return self.keys[0].shape[2], 0


def capture_and_replay(forward_fn, n_replays, warmup_replays=2):
    """Warmup on a side stream, capture ``forward_fn()``, return (graph, output, best_replay_s).

    Re-raises the original capture error (use explain_capture_error to interpret).
    """
    side_stream = torch.npu.Stream()
    side_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side_stream):
        for _ in range(3):
            forward_fn()
    torch.npu.current_stream().wait_stream(side_stream)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = forward_fn()

    for _ in range(warmup_replays):
        graph.replay()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(n_replays):
        graph.replay()
    torch.npu.synchronize()
    return graph, output, (time.perf_counter() - start) / n_replays


def bisect_capture(model, body, layout, input_ids):
    """Localize a capture-hostile op by capturing embed+norm, then +attention, then +full layer.

    ``layout`` carries the prebuilt tensors as a dict: cos, sin, mask, position_ids. Returns
    (first_failing_stage, detail). Diagnostic aid (see the capture-safe recipe).
    """
    cos, sin = layout["cos"], layout["sin"]
    mask, position_ids = layout["mask"], layout["position_ids"]
    final_norm = body.norm if hasattr(body, "norm") else body.final_layernorm
    embed = body.embed_tokens if hasattr(body, "embed_tokens") else body.word_embeddings

    def make_forward(stage):
        def forward_fn():
            with torch.no_grad():
                hidden = embed(input_ids)
                if stage == "embed":
                    return final_norm(hidden)
                for layer in body.layers:
                    if stage == "attn":
                        attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attention
                        attended = attn(
                            hidden_states=layer.input_layernorm(hidden), attention_mask=mask,
                            position_ids=position_ids, position_embeddings=(cos, sin), use_cache=False,
                        )[0]
                        hidden = hidden + attended
                    else:
                        hidden = layer(
                            hidden, attention_mask=mask, position_ids=position_ids,
                            position_embeddings=(cos, sin), use_cache=False,
                        )[0]
                return final_norm(hidden)
        return forward_fn

    for stage in ("embed", "attn", "full"):
        try:
            capture_and_replay(make_forward(stage), 2)
        except Exception as err:  # noqa: BLE001 - diagnostic: any capture failure localizes here
            return stage, explain_capture_error(err)
    return None, "all stages captured OK"
