# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""LLaDA2.0-mini E2E benchmark: warmup + N measurement iterations.

LLaDA2 uses block-wise masked diffusion (not autoregressive).
Warmup iterations absorb JIT compilation; measurement iterations are timed separately.

Usage:
    python modeling/transformers/llada2_moe/bench_LLaDA2-mini.py \
        --model-path /path/to/LLaDA2.0-mini [--use_pypto] [--device 14]
"""

import argparse
import json
import logging
import os
import shutil
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

# Patch: add 'default' rope type if missing (transformers >= 4.57 compat)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if "default" not in ROPE_INIT_FUNCTIONS:
    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                     .to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

logging.basicConfig(level=logging.INFO, format="%(message)s")

parser = argparse.ArgumentParser(description="LLaDA2.0-mini E2E benchmark")
parser.add_argument("--model-path", required=True, help="Path to model weights")
parser.add_argument("--device", default=14, type=int, help="NPU device ID")
parser.add_argument("--prompt", default="Explain the concept of mixture of experts in neural networks.",
                    help="Input prompt")
parser.add_argument("--output_length", default=100, type=int, help="Generation length (tokens)")
parser.add_argument("--steps", default=32, type=int, help="Diffusion steps per block")
parser.add_argument("--block_length", default=32, type=int, help="Block length for masked diffusion")
parser.add_argument("--warmup", default=3, type=int, help="Warmup iterations (absorbs JIT)")
parser.add_argument("--iters", default=10, type=int, help="Measurement iterations")
parser.add_argument("--use_pypto", action="store_true", help="Enable PyPTO fused kernels")
parser.add_argument("--report-file", default=None, help="Output JSON report path")
args = parser.parse_args()


def run_benchmark():
    mode = "pypto" if args.use_pypto else "baseline"
    print(f"\n{'='*70}")
    print(f"  LLaDA2.0-mini E2E Benchmark — {mode.upper()}")
    print(f"{'='*70}")

    torch.npu.set_device(args.device)
    os.environ["TILE_FWK_DEVICE_ID"] = str(args.device)

    # Setup PyPTO if needed
    if args.use_pypto:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        sys.path.insert(0, os.path.join(repo_root, "src"))
        patched_src = os.path.join(repo_root, "src", "pypto_gym", "transformers",
                                   "llada2_moe", "modeling_llada2_moe.py")
        patched_dst = os.path.join(args.model_path, "modeling_llada2_moe.py")
        if os.path.exists(patched_src):
            backup = patched_dst + ".orig"
            if not os.path.exists(backup):
                shutil.copy2(patched_dst, backup)
            shutil.copy2(patched_src, patched_dst)
            print(f"[PyPTO] Installed patched modeling to {patched_dst}")
        from pypto_gym.ops.pypto_tile import llada2_moe as llada2_kernels
        llada2_kernels.USE_PTO_EXPERT_FFN = True
        sys.modules["llada2_pto_kernels"] = llada2_kernels
        print("[PyPTO] Expert FFN kernel enabled")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    print(f"Loading model from {args.model_path} ...")
    torch.npu.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                               trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        device_map={"": f"npu:{args.device}"},
        local_files_only=True, trust_remote_code=True,
    )
    torch.npu.synchronize()
    model_load_s = time.perf_counter() - t0
    peak_load_mem = torch.npu.max_memory_allocated() / 1024**2
    print(f"Model loaded in {model_load_s:.1f}s (peak {peak_load_mem:.0f} MB)")
    torch.npu.reset_peak_memory_stats()

    # Tokenize — LLaDA2 generate() takes raw input_ids tensor
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(f"npu:{args.device}")
    input_len = input_ids.shape[1]
    print(f"Input tokens: {input_len}, Output length (gen_length): {args.output_length}")

    # LLaDA2 generation params (block-wise masked diffusion)
    gen_kwargs = dict(gen_length=args.output_length, steps=args.steps,
                      block_length=args.block_length, temperature=0.0)

    print(f"\nWarmup ({args.warmup} iterations)...")
    for i in range(args.warmup):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(input_ids, **gen_kwargs)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  warmup {i+1}: {elapsed:.2f}s")

    # Measurement iterations
    print(f"\nMeasurement ({args.iters} iterations)...")
    times = []
    tokens_list = []
    for i in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(input_ids, **gen_kwargs)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - t0
        n_tokens = outputs.shape[1] - input_len
        tps = n_tokens / elapsed if elapsed > 0 else 0
        times.append(elapsed)
        tokens_list.append(n_tokens)
        print(f"  iter {i+1:2d}: {elapsed:.3f}s, {n_tokens} tokens, {tps:.1f} tok/s")

    peak_gen_mem = torch.npu.max_memory_allocated() / 1024**2

    # Statistics
    tps_list = [t / e for t, e in zip(tokens_list, times)]
    result = {
        "model": "LLaDA2.0-mini",
        "mode": mode,
        "device": f"npu:{args.device}",
        "input_tokens": input_len,
        "output_length": args.output_length,
        "model_load_s": round(model_load_s, 2),
        "model_load_peak_mem_mb": round(peak_load_mem, 1),
        "generate_peak_mem_mb": round(peak_gen_mem, 1),
        "warmup_iters": args.warmup,
        "num_iters": args.iters,
        "time_mean_s": round(statistics.mean(times), 3),
        "time_std_s": round(statistics.stdev(times), 3) if len(times) > 1 else 0,
        "time_min_s": round(min(times), 3),
        "time_max_s": round(max(times), 3),
        "tps_mean": round(statistics.mean(tps_list), 1),
        "tps_std": round(statistics.stdev(tps_list), 1) if len(tps_list) > 1 else 0,
        "tps_min": round(min(tps_list), 1),
        "tps_max": round(max(tps_list), 1),
        "per_iter": [{"time_s": round(t, 3), "tokens": n, "tps": round(n/t, 1)}
                     for t, n in zip(times, tokens_list)],
    }

    print(f"\n--- {mode.upper()} Summary ---")
    print(f"  Time:       {result['time_mean_s']:.3f} +/- {result['time_std_s']:.3f}s "
          f"(min={result['time_min_s']:.3f}, max={result['time_max_s']:.3f})")
    print(f"  Throughput: {result['tps_mean']:.1f} +/- {result['tps_std']:.1f} tok/s "
          f"(min={result['tps_min']:.1f}, max={result['tps_max']:.1f})")
    print(f"  Peak mem:   {result['generate_peak_mem_mb']:.0f} MB")

    return result


if __name__ == "__main__":
    result = run_benchmark()

    if args.report_file:
        report_path = args.report_file
    else:
        tag = "pypto" if args.use_pypto else "baseline"
        report_path = os.path.join(os.path.dirname(__file__), f"bench_{tag}.json")

    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nReport saved: {report_path}")
