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
Qwen3-1.7B 推理脚本 (with benchmark instrumentation)
用法: python3 ask_Qwen3-1.7B.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
       [--sentence_file 提示词文件] [--output_length 长度] [--use-pto]
       [--report-file 报告文件路径]
"""

import argparse
import sys
import json
import time
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

parser = argparse.ArgumentParser(description="Qwen3-1.7B 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于--sentence_file）")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default="/data/h00520348/optimize0524/models/Qwen3-1.7B", help="模型权重路径")
parser.add_argument("--sentence_file", type=str, default=None, help="从文件读取提示词（多行以换行拼接）")
parser.add_argument("--output_length", type=int, default=100, help="最大生成token数")
parser.add_argument("--use-pto", action="store_true", help="启用PyPTO融合算子")
parser.add_argument("--report-file", type=str, default=None, help="性能报告输出文件（JSON）")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(message)s')

metrics = {}

# ---- PyPTO setup (sys.modules injection) ----
if args.use_pto:
    sys.path.insert(0, args.model_path)
    import pto_kernels
    sys.modules["pto_kernels"] = pto_kernels
    pto_kernels.rope.USE_PTO_ROPE = True
    logging.info("PyPTO RoPE mode enabled")

metrics = {}

# ---- Prompt ----
if args.prompt:
    prompt = args.prompt
elif args.sentence_file:
    with open(args.sentence_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    prompt = "\n".join(lines)
    logging.info(f"从文件读取提示词: {args.sentence_file} ({len(prompt)} 字符)")
else:
    prompt = "你好"

logging.info(f"使用设备: npu:{args.device}")
logging.info(f"模型路径: {args.model_path}")

torch.npu.set_device(args.device)

# ---- Tokenizer ----
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
metrics["tokenizer_load_s"] = round(time.perf_counter() - t0, 3)

# ---- Model ----
torch.npu.reset_peak_memory_stats()
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.float16,
    device_map={"": f"npu:{args.device}"}, local_files_only=True, trust_remote_code=True
)
torch.npu.synchronize()
metrics["model_load_s"] = round(time.perf_counter() - t0, 3)
metrics["model_load_peak_mem_mb"] = round(torch.npu.max_memory_allocated() / 1024**2, 1)
torch.npu.reset_peak_memory_stats()

# ---- Tokenize ----
t0 = time.perf_counter()
inputs = tokenizer(prompt, return_tensors="pt").to(f"npu:{args.device}")
metrics["tokenize_s"] = round(time.perf_counter() - t0, 3)
input_len = inputs.input_ids.shape[1]
metrics["input_tokens"] = input_len
logging.info(f"输入token数: {input_len}")

# ---- Generate ----
torch.npu.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=args.output_length, temperature=0.7, do_sample=True)
torch.npu.synchronize()
metrics["generate_s"] = round(time.perf_counter() - t0, 3)

generated_tokens = outputs.shape[1] - input_len
metrics["generated_tokens"] = generated_tokens
metrics["tokens_per_second"] = round(generated_tokens / metrics["generate_s"], 1) if metrics["generate_s"] > 0 else 0
metrics["generate_peak_mem_mb"] = round(torch.npu.max_memory_allocated() / 1024**2, 1)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logging.info(response)

# ---- Report ----
metrics["mode"] = "pypto" if args.use_pto else "baseline"
metrics["model"] = "Qwen3-1.7B"

logging.info(f"\n--- Performance ---")
logging.info(f"  模式:           {metrics['mode']}")
logging.info(f"  模型加载:       {metrics['model_load_s']}s (峰值显存 {metrics['model_load_peak_mem_mb']}MB)")
logging.info(f"  推理耗时:       {metrics['generate_s']}s")
logging.info(f"  生成token数:    {metrics['generated_tokens']}")
logging.info(f"  吞吐量:         {metrics['tokens_per_second']} tokens/s")
logging.info(f"  推理峰值显存:   {metrics['generate_peak_mem_mb']}MB")

if args.report_file:
    with open(args.report_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"  报告已写入:     {args.report_file}")
