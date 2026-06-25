# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Generalized E2E NPU benchmark for an HF causal LM - the eager / graph measurement layer.

Measures throughput in a chosen regime (see references/npu-run-and-measure.md):
  - prefill : tok/s = seq_len / single_forward   (any HF causal LM, always works)
  - decode  : tok/s = gen / generate_loop        (real autoregressive output speed)

PyPTO measurement: this harness does NOT wire PyPTO kernels - that is owned by the
pypto-fused-op-integration skill (its USE_PTO_<OP> switch + sys.modules injection, done
before transformers import). Once a kernel is wired and active, run the graph arm here and
it captures the PyPTO-accelerated forward (the "pypto+graph" number).

Usage:
  python bench_npu.py --model-path PATH --device 0 --regime decode --gen 128
  python bench_npu.py --model-path PATH --device 0 --regime prefill --seq 16 --arms eager,graph
"""
import argparse
import json
import logging
import os
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bench_npu")


def parse_args():
    parser = argparse.ArgumentParser(description="Eager / NPUGraph E2E benchmark for an HF causal LM")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--regime", choices=["prefill", "decode"], default="decode")
    parser.add_argument("--seq", type=int, default=16, help="prefill / graph forward length")
    parser.add_argument("--gen", type=int, default=128, help="decode: tokens to generate")
    parser.add_argument("--arms", default="eager,graph", help="comma list: eager,graph")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--report-file", default=None)
    return parser.parse_args()


def best_of(run_once, warmup, iters):
    """Warm up ``warmup`` times, then return the fastest of ``iters`` timed runs (seconds)."""
    for _ in range(warmup):
        run_once()
    return min(run_once() for _ in range(iters))


def time_forward(model, input_ids, attention_mask):
    """Time a single forward pass (seconds)."""
    torch.npu.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    torch.npu.synchronize()
    return time.perf_counter() - start


def time_generate(model, input_ids, gen):
    """Time a greedy generate of ``gen`` tokens (seconds)."""
    torch.npu.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=gen, do_sample=False)
    torch.npu.synchronize()
    return time.perf_counter() - start


def measure_eager(model, vocab, device, args):
    """Eager arm: prefill (seq / forward) or decode (gen / generate-loop) throughput."""
    torch.manual_seed(0)
    if args.regime == "prefill":
        input_ids = torch.randint(0, vocab, (1, args.seq), device=device)
        attention_mask = torch.ones((1, args.seq), dtype=torch.long, device=device)
        best = best_of(lambda: time_forward(model, input_ids, attention_mask), args.warmup, args.iters)
        return {"tok_s": round(args.seq / best, 2), "forward_ms": round(best * 1e3, 2)}
    input_ids = torch.randint(0, vocab, (1, 8), device=device)
    best = best_of(lambda: time_generate(model, input_ids, args.gen), args.warmup, args.iters)
    return {"tok_s": round(args.gen / best, 2), "gen_s": round(best, 3)}


def measure_graph(model, capture, vocab, device, args):
    """Graph arm: capture model.forward at a fixed shape and time the replay.

    Only the rotary host-sync + mask prep are neutralized (the model still builds its own
    correctly-shaped position embeddings); ``use_cache=False`` avoids DynamicCache host ops.
    """
    window = args.seq
    capture.apply_capture_mitigations(model, window, device)
    input_ids = torch.randint(0, vocab, (1, window), device=device)
    attention_mask = torch.ones((1, window), dtype=torch.long, device=device)

    def forward_fn():
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    try:
        _graph, _out, best = capture.capture_and_replay(forward_fn, n_replays=20)
        result = {"tok_s": round(window / best, 2), "replay_ms": round(best * 1e3, 2),
                  "capture": "OK", "note": "fixed-shape forward replay (prefill-shape)"}
        logger.info("graph: %s", result)
    except Exception as err:  # noqa: BLE001 - report any capture failure with the gotcha hint
        result = {"capture": "FAILED", "hint": capture.explain_capture_error(err)}
        logger.info("graph FAILED: %s", result["hint"])
        logger.info("  for MoE/decode capture, apply a static-route forward / capture-safe KV cache")
    return result


def main():
    args = parse_args()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]

    torch.npu.set_device(args.device)
    device = "npu:{}".format(args.device)
    os.environ["TILE_FWK_DEVICE_ID"] = str(args.device)
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import npu_capture
    from transformers import AutoModelForCausalLM

    # eager attention is the capture-safe default (SDPA's fused kernel aborts capture, 107025)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": device},
        local_files_only=True, trust_remote_code=args.trust_remote_code,
        attn_implementation="eager").eval()
    vocab = int(getattr(model.config, "vocab_size", 32000))

    result = {"model": args.model_path, "device": device, "regime": args.regime, "arms": {}}
    if "eager" in arms:
        result["arms"]["eager"] = measure_eager(model, vocab, device, args)
        logger.info("eager: %s", result["arms"]["eager"])
    if "graph" in arms:
        result["arms"]["graph"] = measure_graph(model, npu_capture, vocab, device, args)

    result["peak_mem_mb"] = round(torch.npu.max_memory_allocated() / 1024 ** 2, 1)
    logger.info("RESULT %s", json.dumps(result))
    if args.report_file:
        with open(args.report_file, "w") as report:
            json.dump(result, report, indent=2)


if __name__ == "__main__":
    main()
