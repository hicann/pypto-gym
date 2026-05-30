#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ---------------------------------------------------------------------------
"""
Generate ask script with timing/memory instrumentation, --report-file, per-step-timing,
and ACL Graph (torch.compile) support.

Usage: python3 generate_ask_script.py --model-name "ModelName" --script-dir "/scripts/dir"
       [--default-model-dir "/data/models"] [--pto-package "pto_kernels_pkg"]
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')

parser = argparse.ArgumentParser(description="Generate LLM inference script with perf + aclgraph support")
parser.add_argument("--model-name", required=True, help="Model name (e.g. Qwen2-7B)")
parser.add_argument("--script-dir", required=True, help="Script output directory (absolute path)")
parser.add_argument("--default-model-dir", default="/data/models", help="Default model storage directory")
parser.add_argument("--pto-package", default=None, help="PyPTO kernel package name (auto-derived if omitted)")
args = parser.parse_args()

default_model_path = f"{args.default_model_dir}/{args.model_name}"

# Derive pto_package name: "Qwen3-1.7B" → "qwen3_pto_kernels"
if args.pto_package:
    pkg_name = args.pto_package
else:
    base = args.model_name.split("-")[0].lower()
    pkg_name = f"{base}_pto_kernels"

content = f'''#!/usr/bin/env python3
"""
{args.model_name} 推理脚本 (with benchmark + ACL Graph instrumentation)
用法: python3 ask_{args.model_name}.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
       [--output-length 长度] [--use-pto] [--use-acl-graph] [--per-step-timing] [--report-file 路径]
"""

import argparse, os, sys, json, time, torch, torch_npu

parser = argparse.ArgumentParser(description="{args.model_name} 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于--sentence_file）")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default="{default_model_path}", help="模型权重路径")
parser.add_argument("--sentence_file", type=str, default=None, help="从文件读取提示词")
parser.add_argument("--output-length", type=int, default=100, help="最大生成token数")
parser.add_argument("--use-pto", action="store_true", help="启用 PyPTO 融合算子")
parser.add_argument("--use-acl-graph", action="store_true", help="启用 ACL Graph (torch.compile)")
parser.add_argument("--per-step-timing", action="store_true", help="打印每个 forward step 耗时")
parser.add_argument("--report-file", type=str, default=None, help="性能报告输出文件（JSON）")
args = parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

metrics = {{}}

# ---- Prompt ----
if args.prompt:
    prompt = args.prompt
elif args.sentence_file:
    with open(args.sentence_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    prompt = "\\n".join(lines)
    logging.info(f"从文件读取提示词: {{args.sentence_file}} ({{len(prompt)}} 字符)")
else:
    prompt = "你好"

# ---- ACL Graph: JIT compile mode (必须在 set_device 之前) ----
if args.use_acl_graph:
    torch.npu.set_compile_mode(jit_compile=True)
    logging.info("ACL Graph (JIT Compile) 已启用")

# ---- PTO / ACL Graph 注入 (必须在 transformers 导入之前) ----
pto_kernels = None
if args.use_pto or args.use_acl_graph:
    sys.path.insert(0, args.model_path)
    try:
        import {pkg_name} as pto_kernels
        sys.modules["{pkg_name}"] = pto_kernels
        if args.use_pto:
            pto_kernels.USE_PTO_RMS_NORM = True
            pto_kernels.USE_PTO_ATTN_PROLOG = True
            logging.info("PyPTO 融合算子已启用")
        if args.use_acl_graph:
            pto_kernels.USE_ACL_GRAPH = True
    except ImportError:
        logging.warning("PyPTO kernel 包 {pkg_name} 未找到，跳过注入")

from transformers import AutoModelForCausalLM, AutoTokenizer

torch.npu.set_device(args.device)
logging.info(f"使用设备: npu:{{args.device}}")
logging.info(f"模型路径: {{args.model_path}}")

# ---- Tokenizer ----
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
metrics["tokenizer_load_s"] = round(time.perf_counter() - t0, 3)

# ---- Model ----
torch.npu.reset_peak_memory_stats()
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.float16,
    device_map={{"": f"npu:{{args.device}}"}}, local_files_only=True, trust_remote_code=True
)
torch.npu.synchronize()
metrics["model_load_s"] = round(time.perf_counter() - t0, 3)
metrics["model_load_peak_mem_mb"] = round(torch.npu.max_memory_allocated() / 1024**2, 1)
torch.npu.reset_peak_memory_stats()

# ---- ACL Graph: torch.compile wrapping (warmup triggers first compilation) ----
if args.use_acl_graph:
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    compiler_config = CompilerConfig()
    compiler_config.experimental_config.frozen_parameter = True
    compiler_config.experimental_config.tiling_schedule_optimize = True
    compiler_config.experimental_config.enable_view_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    model = torch.compile(model, dynamic=True, fullgraph=True, backend=npu_backend)
    logging.info("ACL Graph torch.compile 已封装")

# ---- Warmup ----
for i in range(3):
    warm = tokenizer("Hello world", return_tensors="pt").to(f"npu:{{args.device}}")
    _ = model.generate(**warm, max_new_tokens=8, do_sample=False)
torch.npu.synchronize()
logging.info("Warmup 完成 (3x8 tokens)")

# ---- Per-step timing (hook model forward, 必须在 warmup 之后) ----
if args.per_step_timing:
    _orig_forward = model.forward
    def _timed_forward(*model_args, **model_kwargs):
        _step_t0 = time.perf_counter()
        _out = _orig_forward(*model_args, **model_kwargs)
        torch.npu.synchronize()
        _dur = (time.perf_counter() - _step_t0) * 1e6
        print(f"====duration==== {{_dur:.2f}}us")
        return _out
    model.forward = _timed_forward

# ---- Tokenize ----
inputs = tokenizer(prompt, return_tensors="pt").to(f"npu:{{args.device}}")
input_len = inputs.input_ids.shape[1]
metrics["input_tokens"] = input_len
logging.info(f"输入token数: {{input_len}}")

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

# ---- Decode ----
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logging.info(response)

# ---- Report ----
mode_parts = []
if args.use_acl_graph:
    mode_parts.append("aclgraph")
if args.use_pto:
    mode_parts.append("pypto")
metrics["mode"] = "+".join(mode_parts) if mode_parts else "eager"
metrics["model"] = "{args.model_name}"

logging.info(f"\\n--- Performance ---")
logging.info(f"  模式:           {{metrics['mode']}}")
logging.info(f"  模型加载:       {{metrics['model_load_s']}}s (峰值显存 {{metrics['model_load_peak_mem_mb']}}MB)")
logging.info(f"  推理耗时:       {{metrics['generate_s']}}s")
logging.info(f"  生成token数:    {{metrics['generated_tokens']}}")
logging.info(f"  吞吐量:         {{metrics['tokens_per_second']}} tokens/s")
logging.info(f"  推理峰值显存:   {{metrics['generate_peak_mem_mb']}}MB")

if args.report_file:
    with open(args.report_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"  报告已写入:     {{args.report_file}}")
'''

os.makedirs(args.script_dir, exist_ok=True)
output_file = os.path.join(args.script_dir, f"ask_{args.model_name}.py")
with open(output_file, "w") as f:
    f.write(content)
os.chmod(output_file, 0o755)

logging.info(f"脚本已生成: {output_file}")
logging.info(f"\n使用方法:")
logging.info(f"  python3 {output_file}")
logging.info(f"  python3 {output_file} --prompt '你的问题'")
logging.info(f"  python3 {output_file} --device 7")
logging.info(f"  python3 {output_file} --use-pto")
logging.info(f"  python3 {output_file} --use-acl-graph")
logging.info(f"  python3 {output_file} --use-pto --use-acl-graph")
logging.info(f"  python3 {output_file} --per-step-timing")
logging.info(f"  python3 {output_file} --report-file bench.json")
