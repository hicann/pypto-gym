#!/usr/bin/env python3
"""
spatial_ssrl_3b 推理脚本 (with benchmark instrumentation)
用法: python3 ask_spatial_ssrl_3b.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
       [--sentence_file 提示词文件] [--output_length 长度] [--use_pypto]
       [--report-file 报告文件路径]
"""

import argparse
import sys
import json
import time
import torch
import torch_npu
from transformers import AutoModel, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

parser = argparse.ArgumentParser(description="spatial_ssrl_3b 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于--sentence_file）")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default="/data/h00520348/optimize525/models/spatial_ssrl_3b", help="模型权重路径")
parser.add_argument("--sentence_file", type=str, default=None, help="从文件读取提示词（多行以换行拼接）")
parser.add_argument("--output_length", type=int, default=100, help="最大生成token数")
parser.add_argument("--use_pto", action="store_true", help="PyPTO融合算子模式")
parser.add_argument("--use_acl_graph", action="store_true", help="aclgraph图模式（torch.compile + torchair）")
parser.add_argument("--use_partial_aclgraph", action="store_true",
                    help="partial aclgraph模式（只编译MLP+RMSNorm，排除Attention）")
parser.add_argument("--report-file", type=str, default=None, help="性能报告输出文件（JSON）")
args = parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ---- PyPTO setup ----
if args.use_pto:
    import sys
    sys.path.insert(0, args.model_path)
    import spatial_ssrl_3b_pto_kernels as pto_kernels
    sys.modules["spatial_ssrl_3b_pto_kernels"] = pto_kernels
    pto_kernels.USE_PTO_RMS_NORM = True
    pto_kernels.USE_PTO_ROPE = True
    logging.info("PyPTO mode enabled: RMSNorm + RoPE kernels activated")

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

# ---- PyPTO setup (placeholder) ----
if args.use_pto:
    logging.info("PyPTO mode enabled: RMSNorm + RoPE")

logging.info(f"使用设备: npu:{args.device}")
logging.info(f"模型路径: {args.model_path}")

torch.npu.set_device(args.device)

# ---- Processor ----
t0 = time.perf_counter()
processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
metrics["processor_load_s"] = round(time.perf_counter() - t0, 3)

# ---- Model ----
torch.npu.reset_peak_memory_stats()
t0 = time.perf_counter()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.float16,
    device_map={"": f"npu:{args.device}"}, local_files_only=True
)
torch.npu.synchronize()
metrics["model_load_s"] = round(time.perf_counter() - t0, 3)
metrics["model_load_peak_mem_mb"] = round(torch.npu.max_memory_allocated() / 1024**2, 1)
torch.npu.reset_peak_memory_stats()

# ---- aclgraph setup ----
if args.use_acl_graph or args.use_partial_aclgraph:
    import torchair as tng
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
    from torchair.configs.compiler_config import CompilerConfig

    compiler_config = CompilerConfig()
    compiler_config.experimental_config.frozen_parameter = True
    compiler_config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)

    if args.use_acl_graph:
        # Full aclgraph (会失败，npu_fusion_attention不支持FakeTensor)
        model.model = torch.compile(model.model, dynamic=True, fullgraph=True, backend=npu_backend)
        logging.info("aclgraph mode enabled: torch.compile + torchair activated")
    elif args.use_partial_aclgraph:
        # Partial aclgraph: 只编译 MLP + RMSNorm，排除 Attention
        logging.info("partial aclgraph mode: compiling MLP + RMSNorm, excluding Attention")

        for _, layer in enumerate(model.model.language_model.layers):
            layer.mlp = torch.compile(layer.mlp, dynamic=True, fullgraph=False, backend=npu_backend)
            layer.input_layernorm = torch.compile(
    layer.input_layernorm,
    dynamic=True,
    fullgraph=False,
     backend=npu_backend)
            layer.post_attention_layernorm = torch.compile(
    layer.post_attention_layernorm, dynamic=True, fullgraph=False, backend=npu_backend)

        model.model.language_model.embed_tokens = torch.compile(
            model.model.language_model.embed_tokens, dynamic=True, fullgraph=False, backend=npu_backend
        )
        model.model.language_model.norm = torch.compile(
            model.model.language_model.norm, dynamic=True, fullgraph=False, backend=npu_backend
        )

        logging.info(f"  - Compiled {len(model.model.language_model.layers)} layers (MLP + RMSNorm)")
        logging.info("  - Attention remains eager mode (npu_fusion_attention unsupported)")

# ---- Prepare inputs ----
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

t0 = time.perf_counter()
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to(f"npu:{args.device}")
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

# ---- Decode ----
generated_ids = outputs[0][input_len:]
response = processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
logging.info(response)

# ---- Report ----
metrics["mode"] = "pypto" if args.use_pto else "baseline"
if args.use_acl_graph:
    metrics["mode"] += "+aclgraph(full)"
elif args.use_partial_aclgraph:
    metrics["mode"] += "+aclgraph(partial)"
metrics["model"] = "spatial_ssrl_3b"

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
