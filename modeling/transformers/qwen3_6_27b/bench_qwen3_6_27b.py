# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""End-to-end generation benchmark for Qwen3.5-9B / Qwen3.6-27B (Ascend 910B).

Whole-model token GENERATION (prefill + autoregressive decode), mirroring the repo's ask_*.py flow
(chat template, the same PyPTO GatedDeltaRule wiring). Greedy (deterministic) decoding so eager vs PyPTO
outputs can be compared bit-for-bit. Per the benchmarking standard, the timed measurement is ONE warmup
generate() + ONE measured generate() (the warmup absorbs JIT compile). A lightweight streamer splits the
measured call into prefill->first-token latency and the decode per-token mean.

Run BOTH input regimes: the natural short prompt and a synthetic 256-token input.

Usage:
  python3 bench_qwen3_6_27b.py --model qwen3_6_27b --device 7 \
      --model-path /path/to/Qwen3.6-27B [--use_pypto] \
      [--output_length 100] [--report-file out.json]
"""

import argparse
import importlib
import json
import logging
import math
import os
import sys
import time

import torch
import torch_npu  # noqa: F401  (registers the npu backend)

_PROMPT = "你好，请介绍一下自己。"
_MODELS = {"qwen3_5_9b": "qwen3_5_9b_pto_kernels", "qwen3_6_27b": "qwen3_6_27b_pto_kernels"}


def _parse_args():
    ap = argparse.ArgumentParser(description="Qwen3.5/3.6 E2E generation benchmark")
    ap.add_argument("--model", choices=list(_MODELS), required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    ap.add_argument("--output_length", type=int, default=100, help="max_new_tokens (greedy)")
    ap.add_argument("--long-seq", type=int, default=256, help="synthetic long-input prefill length")
    ap.add_argument("--use_pypto", action="store_true", help="enable fused PyPTO GatedDeltaRule")
    ap.add_argument("--kernel-path", default="/tmp/qwen_kernels",
                    help="dir holding the <model>_pto_kernels package (the model-dir symlink may be stale)")
    ap.add_argument("--report-file", default=None)
    return ap.parse_args()


def _wire_pypto(kernel_path, model_path, kernel_mod):
    """Inject the kernel module under the name the modeling expects (must precede model build).
    Use ONLY --kernel-path (canonical /tmp/qwen_kernels). The model-dir <kernel>_pto_kernels packages
    are unreliable — q35's is a STALE dead symlink, q36's __init__ uses a flat `from gated_delta_rule_impl
    import` that breaks unless its own subdir is on sys.path — and inserting model_path would SHADOW the
    good /tmp package (insert(0) puts the later path first). model_path is intentionally ignored here."""
    if kernel_path and kernel_path not in sys.path:
        sys.path.insert(0, kernel_path)
    pto = importlib.import_module(kernel_mod)
    sys.modules[kernel_mod] = pto
    return pto


def _patch_gated_delta_rule(model, pto):
    """Route the chunk-prefill path through the PyPTO wrapper (same hook as ask_*.py)."""
    import types
    pto.USE_PTO_GATED_DELTA_RULE = True
    patched = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen3_5GatedDeltaNet":
            orig_forward = module.forward

            def new_forward(self, *a, _orig=orig_forward, _pk=pto, **kw):
                orig_chunk = self.chunk_gated_delta_rule
                if getattr(_pk, "USE_PTO_GATED_DELTA_RULE", False):
                    def _patched(q, k, v, **kk):
                        try:
                            return _pk.gated_delta_rule_wrapper(q, k, v, **kk)
                        except NotImplementedError:
                            return orig_chunk(q, k, v, **kk)
                    self.chunk_gated_delta_rule = _patched
                try:
                    return _orig(*a, **kw)
                finally:
                    self.chunk_gated_delta_rule = orig_chunk
            module.forward = types.MethodType(new_forward, module)
            patched += 1
    return patched


class _TimingStreamer:
    """Records (timestamp, token-count) per streamer.put. transformers echoes the PROMPT via
    put(input_ids) BEFORE the prefill forward, then put(next_token) per generated token — so the
    first put (numel>1) is the prompt echo, the 2nd put is the first generated token (prefill done),
    and the rest are decode steps."""

    def __init__(self):
        self.t = []
        self.n = []

    def put(self, value):
        self.t.append(time.time())
        try:
            self.n.append(int(value.numel()))
        except Exception:  # noqa: BLE001
            self.n.append(1)

    def end(self):
        pass


def _gen_once(model, inputs, out_len, dev, timed):
    """One greedy generate(); returns (out_ids, prefill_s, decode_mean_s, total_s) when timed."""
    streamer = _TimingStreamer() if timed else None
    torch.npu.synchronize(dev)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=out_len, do_sample=False,
                             num_beams=1, streamer=streamer)
    torch.npu.synchronize(dev)
    total = time.time() - t0
    if not timed:
        return out, None, None, None
    ts, ns = streamer.t, streamer.n
    if len(ts) >= 2:
        # If the prompt was echoed first (numel>1), ts[0]=prompt echo (pre-prefill) and ts[1]=first
        # token (prefill done), so prefill = ts[1]-ts[0] and decode gaps start at index 2. Otherwise
        # ts[0] is already the first token and decode gaps start at index 1.
        if ns and ns[0] > 1:
            prefill, start = ts[1] - ts[0], 2
        else:
            prefill, start = ts[0] - t0, 1
        deltas = [ts[i] - ts[i - 1] for i in range(start, len(ts))]
    else:
        prefill, deltas = float("nan"), []
    decode_mean = (sum(deltas) / len(deltas)) if deltas else float("nan")
    return out, prefill, decode_mean, total


def _measure(model, inputs, out_len, dev, label):
    """1 warmup generate (untimed, absorbs JIT) + 1 measured generate. Returns a metrics dict."""
    in_len = int(inputs["input_ids"].shape[1])
    _gen_once(model, inputs, out_len, dev, timed=False)              # warmup
    torch.npu.reset_peak_memory_stats(dev)
    out, prefill, dmean, total = _gen_once(model, inputs, out_len, dev, timed=True)  # measured
    new_tok = int(out.shape[1] - in_len)
    peak = torch.npu.max_memory_allocated(dev) / 1024 / 1024
    decode_tps = (1.0 / dmean) if (dmean and dmean == dmean and dmean > 0) else float("nan")
    e2e_tps = (new_tok / total) if total else float("nan")
    gen_ids = out[0, in_len:].tolist()
    m = {"input": label, "input_len": in_len, "new_tokens": new_tok,
         "prefill_first_tok_ms": round(prefill * 1e3, 2),
         "decode_mean_ms": round(dmean * 1e3, 3), "decode_tok_s": round(decode_tps, 2),
         "e2e_infer_s": round(total, 3), "e2e_tok_s": round(e2e_tps, 2),
         "peak_mem_mb": round(peak, 1),
         # greedy is deterministic -> these let an offline diff confirm eager==pypto outputs
         "gen_ids_len": len(gen_ids), "gen_ids_head": gen_ids[:8], "gen_ids_tail": gen_ids[-8:]}
    logging.info("[%s|%s] prefill1tok=%.1fms decode_mean=%.3fms decode=%.1f tok/s "
                 "e2e=%.1f tok/s peak=%.0fMB new=%d",
                 label, "pto" if model.bench_pypto else "eager", m["prefill_first_tok_ms"],
                 m["decode_mean_ms"], m["decode_tok_s"], m["e2e_tok_s"], m["peak_mem_mb"], new_tok)
    return m


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.model_path:
        raise SystemExit("--model-path or MODEL_PATH required")

    pto = _wire_pypto(args.kernel_path, args.model_path, _MODELS[args.model]) if args.use_pypto else None

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    torch.npu.set_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    text = tok.apply_chat_template([{"role": "user", "content": _PROMPT}],
                                   tokenize=False, add_generation_prompt=True)

    t_load = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(f"npu:{args.device}").eval()
    torch.npu.synchronize(args.device)
    logging.info("model loaded in %.1fs", time.time() - t_load)

    n_patched = _patch_gated_delta_rule(model, pto) if args.use_pypto else 0
    if args.use_pypto:
        logging.info("PyPTO GatedDeltaRule patched to %d layers", n_patched)
    model.bench_pypto = bool(args.use_pypto)

    dev = args.device
    nat = tok(text, return_tensors="pt").to(f"npu:{dev}")
    # synthetic long input: tile the natural prompt's ids up to --long-seq tokens (content is
    # irrelevant for timing; only the prefill length matters)
    ids = nat["input_ids"]
    reps = math.ceil(args.long_seq / ids.shape[1])
    long_ids = ids.repeat(1, reps)[:, :args.long_seq].contiguous()
    long_inputs = {"input_ids": long_ids,
                   "attention_mask": torch.ones_like(long_ids)}

    results = [
        _measure(model, nat, args.output_length, dev, "natural"),
        _measure(model, long_inputs, args.output_length, dev, f"long{args.long_seq}"),
    ]

    report = {"model": args.model, "mode": "pto" if args.use_pypto else "eager",
              "device": dev, "output_length": args.output_length, "greedy": True,
              "warmup_gens": 1, "measured_gens": 1, "results": results,
              "command": "python3 " + " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])}
    if args.report_file:
        with open(args.report_file, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logging.info("report: %s", args.report_file)


if __name__ == "__main__":
    main()
