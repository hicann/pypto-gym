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
DeepSeek-V2-Lite-Chat 推理脚本 (with PyPTO + aclgraph integration)
用法: python3 ask_DeepSeek-V2-Lite-Chat.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
       [--sentence_file 提示词文件] [--output_length 长度] [--use_pto] [--use_rope] [--use_acl_graph]
       [--report-file 报告文件路径]
"""

import argparse
import sys
import json
import time
import os

# ===== PyPTO编译环境配置（必须在任何导入前） =====
os.environ.setdefault('PTO_TILE_LIB_CODE_PATH', '/path/to/pto-isa')
os.environ['ASCEND_HOME_PATH'] = '/usr/local/Ascend/cann-9.0.0'

# ===== 参数解析 =====
parser = argparse.ArgumentParser(description="DeepSeek-V2-Lite-Chat 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于--sentence_file）")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument(
    "--model-path",
    default=os.environ.get("MODEL_PATH", "/path/to/models/DeepSeek-V2-Lite-Chat"),
     help="模型权重路径")
parser.add_argument("--sentence_file", type=str, default=None, help="从文件读取提示词（多行以换行拼接）")
parser.add_argument("--output_length", type=int, default=100, help="最大生成token数")
parser.add_argument("--use_pto", action="store_true", help="启用PyPTO融合算子（RMSNorm）")
parser.add_argument("--use_rope", action="store_true", help="启用PyPTO RoPE算子")
parser.add_argument("--use_kv_fusion", action="store_true", help="启用PyPTO KV融合算子")
parser.add_argument("--use_acl_graph", action="store_true", help="启用aclgraph图编译模式")
parser.add_argument("--report-file", type=str, default=None, help="性能报告输出文件（JSON）")
args = parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== PyPTO sys.modules注入（transformers导入前） =====
pto_kernels = None
pto_enabled = args.use_pto or args.use_rope or args.use_kv_fusion or args.use_acl_graph
if pto_enabled:
    logging.info("PyPTO mode enabled")
    sys.path.insert(0, args.model_path)
    import pto_kernels as _pto_kernels
    pto_kernels = _pto_kernels
    sys.modules["deepseek_v2_lite_chat_pto_kernels"] = pto_kernels

    if args.use_pto:
        pto_kernels.USE_PTO_RMS_NORM = True
        logging.info("RMS_NORM_PTO_AVAILABLE = True")

    if args.use_rope:
        pto_kernels.USE_PTO_ROPE = True
        logging.info("ROPE_PTO_AVAILABLE = True")

    if args.use_kv_fusion:
        pto_kernels.USE_PTO_MLA_PROLOG = True
        # 设置mla_prolog模块的开关（内部wrapper使用）
        if hasattr(pto_kernels, 'mla_prolog'):
            pto_kernels.mla_prolog.USE_PTO_MLA_PROLOG = True
        logging.info("MLA_PROLOG_PTO_AVAILABLE = True")

    if args.use_acl_graph:
        pto_kernels.USE_ACL_GRAPH = True
        logging.info("ACL_GRAPH_AVAILABLE = True")

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

metrics = {}

# ---- Prompt ----
if args.prompt:
    prompt = args.prompt
elif args.sentence_file:
    with open(args.sentence_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    prompt = "\n".join(lines)
    logging.info(f"Prompt read from file: {args.sentence_file} ({len(prompt)} chars)")
else:
    prompt = "Hello"

logging.info(f"Using device: npu:{args.device}")
logging.info(f"Model path: {args.model_path}")

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

# ---- KV融合算子预热（多场景编译缓存） ----
if pto_kernels and pto_kernels.USE_PTO_MLA_PROLOG:
    logging.info("Warming up KV fusion op (multi-scenario compile cache)...")
    warmup_start = time.perf_counter()

    first_layer = model.model.layers[0].self_attn
    warmup_kv_a_weight = first_layer.kv_a_proj_with_mqa.weight.t().contiguous()
    warmup_kv_b_weight = first_layer.kv_b_proj.weight.t().contiguous()
    warmup_ln_weight = first_layer.kv_a_layernorm.weight

    # 按需预热策略：只预热常见场景（seq_len=1,2）
    # 端到端推理主要调用q_len=1和q_len=2
    warmup_seq_lens = [1, 2]  # 优化：从5个场景减少到2个，预热时间减少约60%
    warmup_errors = []

    for warmup_seq in warmup_seq_lens:
        try:
            warmup_hidden = torch.randn(1, warmup_seq, 2048, dtype=torch.float16, device=f"npu:{args.device}")
            warmup_cos = torch.randn(warmup_seq, 64, dtype=torch.float16, device=f"npu:{args.device}")
            warmup_sin = torch.randn(warmup_seq, 64, dtype=torch.float16, device=f"npu:{args.device}")
            warmup_pos_ids = torch.arange(warmup_seq, dtype=torch.long, device=f"npu:{args.device}").unsqueeze(0)

            _ = pto_kernels.mla_mla_prolog_v2(
                warmup_hidden, warmup_kv_a_weight, warmup_kv_b_weight,
                warmup_ln_weight, first_layer.kv_a_layernorm.variance_epsilon,
                warmup_cos, warmup_sin, warmup_pos_ids
            )
            logging.info(f"  ✓ seq_len={warmup_seq} warmup succeeded")
        except Exception as e:
            warmup_errors.append(f"seq_len={warmup_seq}: {str(e)[:50]}")
            logging.warning(f"  ⚠️ seq_len={warmup_seq} warmup failed: {str(e)[:50]}")

    warmup_time = time.perf_counter() - warmup_start
    metrics["warmup_s"] = round(warmup_time, 3)

    if warmup_errors:
        logging.warning(
            f"Partial warmup failure ({len(warmup_errors)}/{len(warmup_seq_lens)}), "
            f"elapsed: {warmup_time:.3f}s"
        )
    else:
        logging.info(f"✓ KV fusion op warmup complete ({len(warmup_seq_lens)} scenarios), elapsed: {warmup_time:.3f}s")

# ---- aclgraph 编译（可选） ----
if pto_kernels is not None and pto_kernels.USE_ACL_GRAPH:
    logging.info("Enabling aclgraph graph compilation mode (reduce-overhead)")
    try:
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig

        compiler_config = CompilerConfig()
        compiler_config.mode = "reduce-overhead"
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)

        t0_compile = time.perf_counter()
        model = torch.compile(model, dynamic=False, fullgraph=True, backend=npu_backend)
        compile_time = time.perf_counter() - t0_compile
        logging.info(f"aclgraph compile time: {compile_time:.2f}s (torchair)")
        metrics["aclgraph_compile_s"] = round(compile_time, 3)
        metrics["aclgraph_backend"] = "torchair"
    except ImportError:
        logging.info("torchair not installed, using native torch.compile")
        t0_compile = time.perf_counter()
        model = torch.compile(model, dynamic=False, fullgraph=False)
        compile_time = time.perf_counter() - t0_compile
        logging.info(f"torch.compile compile time: {compile_time:.2f}s (native)")
        metrics["aclgraph_compile_s"] = round(compile_time, 3)
        metrics["aclgraph_backend"] = "native"

# ---- Tokenize ----
t0 = time.perf_counter()
inputs = tokenizer(prompt, return_tensors="pt").to(f"npu:{args.device}")
metrics["tokenize_s"] = round(time.perf_counter() - t0, 3)
input_len = inputs.input_ids.shape[1]
metrics["input_tokens"] = input_len
logging.info(f"Input token count: {input_len}")

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
if args.use_acl_graph:
    metrics["mode"] = "aclgraph"
elif args.use_kv_fusion:
    metrics["mode"] = "kv_fusion"
elif args.use_pto:
    metrics["mode"] = "pypto"
else:
    metrics["mode"] = "baseline"
metrics["model"] = "DeepSeek-V2-Lite-Chat"

logging.info(f"\n--- Performance ---")
logging.info(f"  Mode:           {metrics['mode']}")
logging.info(f"  Model load:     {metrics['model_load_s']}s (peak memory {metrics['model_load_peak_mem_mb']}MB)")
logging.info(f"  Inference time: {metrics['generate_s']}s")
logging.info(f"  Generated tokens: {metrics['generated_tokens']}")
logging.info(f"  Throughput:     {metrics['tokens_per_second']} tokens/s")
logging.info(f"  Inference peak memory: {metrics['generate_peak_mem_mb']}MB")

if args.report_file:
    with open(args.report_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"  Report written to:     {args.report_file}")
