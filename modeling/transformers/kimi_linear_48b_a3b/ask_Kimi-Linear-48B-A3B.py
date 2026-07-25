#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Kimi-Linear-48B-A3B-Instruct 推理脚本 (Ascend NPU, 多卡)

- 在导入 transformers 之前，将 PyPTO 算子库 (kimi_linear_48b_a3b_pto_kernels)
  注入 sys.modules，使 modeling_kimi.py 的 KDA 分发钩子能取到它。
- 模型 ~96GB bf16 (>64GB/卡)，必须跨多卡：构造均衡的 device_map。
- 贪心生成，内嵌时延 + 峰值 HBM 采集。

用法:
  python3 ask_Kimi-Linear-48B-A3B.py [--prompt ...] [--model-path 路径]
        [--sentence_file 文件] [--output_length 40] [--num_npus 4]
        [--use_pypto] [--report-file out.json]
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import torch_npu  # noqa: F401  (registers npu backend)

parser = argparse.ArgumentParser(description="Kimi-Linear-48B-A3B-Instruct 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于 --sentence_file）")
parser.add_argument("--model-path", default="/data/models/Kimi-Linear-48B-A3B-Instruct",
                    help="模型权重路径")
parser.add_argument("--sentence_file", type=str, default=None, help="从文件读取提示词（取首行）")
parser.add_argument("--output_length", type=int, default=40, help="最大生成 token 数")
parser.add_argument("--num_npus", type=int, default=4, help="使用的 NPU 卡数")
parser.add_argument("--use_pypto", action="store_true", help="启用 PyPTO 融合 KDA 算子")
parser.add_argument("--report-file", default=None, help="JSON 性能报告输出路径")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(message)s')

# prompt
if args.prompt is not None:
    prompt = args.prompt
elif args.sentence_file and os.path.exists(args.sentence_file):
    with open(args.sentence_file) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    prompt = lines[0] if lines else "Explain what a large language model is in two sentences."
    logging.info(f"从文件读取提示词: {args.sentence_file}")
else:
    prompt = "Explain what a large language model is in two sentences."

# PyPTO injection must happen BEFORE transformers is imported.
# NOTE(multi-NPU coverage): this is a SINGLE-PROCESS launch. The 48B model is
# sharded across NPUs (device_map below), but PyPTO binds to ONE NPU per process,
# so only the bound NPU's KDA layers actually run on PyPTO (e.g. ~6 of 20 with
# 4-way sharding); the others fall back to torch (the kernel warns once). This is
# fine for a quick check but is NOT full-model acceleration.
# Note (future work): for full 20/20 PyPTO coverage, run one process per NPU
#       (multi-process pipeline / tensor parallel) instead of single-process device_map.
if args.use_pypto:
    sys.path.insert(0, args.model_path)
    import kimi_linear_48b_a3b_pto_kernels as pto_kernels
    sys.modules["kimi_linear_48b_a3b_pto_kernels"] = pto_kernels
    # NOTE: only register the module here (must precede the transformers import so
    # modeling_kimi can sys.modules.get it). The USE_PTO_KDA toggle is flipped
    # AFTER the model is loaded onto NPU (step-22 timing rule) — see below.

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

n = min(args.num_npus, torch.npu.device_count())
logging.info(f"NPUs available: {torch.npu.device_count()}, using {n}")
logging.info(f"模型路径: {args.model_path}")
if args.use_pypto and n > 1:
    logging.warning(
        "[PyPTO coverage] single-process + %d-way sharding: PyPTO binds to ONE NPU, "
        "so only that NPU's KDA layers are accelerated (partial, ~6/20 with 4 NPUs) "
        "and the rest fall back to torch. This is NOT full-model PyPTO. For full "
        "20/20 coverage run one process per NPU (multi-process pipeline).", n)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path, trust_remote_code=True, local_files_only=True)

# ---- build a balanced device_map over n NPUs ----
# Model is ~96GB bf16 (>64GB/card) so it MUST span >=2 cards. Distribute the
# KimiDecoderLayer blocks; keep embed_tokens on npu:0, norm + lm_head on the
# last card used.
cfg = AutoConfig.from_pretrained(
    args.model_path, trust_remote_code=True, local_files_only=True)
nlayers = cfg.num_hidden_layers

device_map = {}
device_map["model.embed_tokens"] = "npu:0"
per = (nlayers + n - 1) // n
for i in range(nlayers):
    device_map[f"model.layers.{i}"] = f"npu:{min(i // per, n - 1)}"
last = f"npu:{n - 1}"
device_map["model.norm"] = last
device_map["lm_head"] = last

logging.info("loading model (sharded ~96GB across NPUs)...")
t_load = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map=device_map,
).eval()
logging.info(f"model loaded in {time.perf_counter() - t_load:.1f}s")

# Flip the PyPTO toggle only now that the model is on NPU (step-22 timing rule:
# the JIT kernel must never receive a CPU tensor; modeling reads USE_PTO_KDA
# per-forward via sys.modules, so enabling it here takes effect for generation).
if args.use_pypto:
    pto_kernels.USE_PTO_KDA = True
    logging.info("PyPTO mode enabled: KDA chunk fused (USE_PTO_KDA=True)")

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to("npu:0")
attention_mask = inputs.get("attention_mask")
if attention_mask is not None:
    attention_mask = attention_mask.to("npu:0")
logging.info(f"输入 token 数: {input_ids.shape[1]}")
if args.use_pypto and input_ids.shape[1] <= 64:
    logging.warning(
        "[PyPTO] prompt is %d tokens (<=64): KDA takes the fused_recurrent (decode) "
        "path, NOT the chunk path, so the PyPTO CHUNK kernel will NOT fire this "
        "prefill. Use a prompt >64 tokens to exercise the chunk kernel.",
        input_ids.shape[1])

for d in range(n):
    torch.npu.reset_peak_memory_stats(d)

torch.npu.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.output_length,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
torch.npu.synchronize()
elapsed = time.perf_counter() - t0

new_tokens = gen.shape[1] - input_ids.shape[1]
text = tokenizer.decode(gen[0], skip_special_tokens=True)
completion = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)

peak = {d: torch.npu.max_memory_allocated(d) / (1024**2) for d in range(n)}
tok_per_s = new_tokens / elapsed if elapsed > 0 else 0.0

logging.info("\n================ PROMPT ================")
logging.info(prompt)
logging.info("================ COMPLETION ================")
logging.info(completion)
logging.info("============================================")
logging.info(f"[metrics] new_tokens={new_tokens} elapsed={elapsed:.3f}s "
             f"throughput={tok_per_s:.2f} tok/s")
for d in range(n):
    logging.info(f"[metrics] npu:{d} peak HBM = {peak[d]:.1f} MB")

if args.report_file:
    rep = {
        "mode": "pypto" if args.use_pypto else "baseline",
        "prompt": prompt,
        "completion": completion,
        "new_tokens": new_tokens,
        "elapsed_s": elapsed,
        "throughput_tok_s": tok_per_s,
        "peak_hbm_mb": peak,
        "num_npus": n,
    }
    with open(args.report_file, "w") as f:
        json.dump(rep, f, indent=2)
    logging.info(f"[info] wrote report to {args.report_file}")
