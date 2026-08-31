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
SwiGLU MLP整网集成 - torch.compile模式演示
backend=reduce-overhead模式

使用方式:
    # Baseline Eager（默认）
    python3 ask_gutenocr_3b_compile.py --prompt "你好"
    
    # torch.compile模式
    python3 ask_gutenocr_3b_compile.py --prompt "你好" --use_compile
    
    # PyPTO + torch.compile模式
    python3 ask_gutenocr_3b_compile.py --prompt "你好" --use_pto --use_compile
    
    # aclgraph模式
    python3 ask_gutenocr_3b_compile.py --prompt "你好" --use_acl_graph
"""

import logging
import argparse
import sys
import os
import time
import json
import torch
import torch_npu

_cpp = os.environ.get("CANN_POW_PATCH_PATH", "")
if _cpp:
    sys.path.insert(0, _cpp)
try:
    import cann_pow_patch
except ImportError:
    pass
from transformers import AutoModelForImageTextToText, AutoProcessor

# ===== 参数解析 =====
parser = argparse.ArgumentParser(description="SwiGLU MLP整网集成演示")
parser.add_argument("--prompt", default="你好", help="提问文本")
parser.add_argument("--device", default=1, type=int, help="NPU卡号")
parser.add_argument("--model-path", default=os.environ.get("GUTENOCR_MODEL_PATH", "."), help="模型路径")
parser.add_argument("--warmup", default=5, type=int, help="Warmup iterations")
parser.add_argument("--use_pto", action="store_true", help="启用PyPTO SwiGLU融合算子")
parser.add_argument("--use_compile", action="store_true", help="启用torch.compile模式")
parser.add_argument("--use_acl_graph", action="store_true", help="启用aclgraph模式（需torchair）")
parser.add_argument("--use_dynamic_config", action="store_true", help="启用动态配置（根据batch自动选择最优算子）")
parser.add_argument("--simple_prompt", action="store_true", help="使用简化prompt（不使用apply_chat_template，高性能）")
parser.add_argument("--backend", default="inductor", help="torch.compile backend (inductor/npugraphs/npu)")
parser.add_argument("--mode", default=None, help="torch.compile mode (reduce-overhead/max-autotune/default)")
parser.add_argument("--batch", default=1, type=int, help="Batch size")
parser.add_argument("--output_length", default=50, type=int, help="生成token数")
parser.add_argument("--report-file", type=str, default=None, help="性能报告JSON")
args = parser.parse_args()

# ===== NPU设置 =====
torch.npu.set_device(args.device)
device = f"npu:{args.device}"

logging.info("=" * 80)
logging.info("SwiGLU MLP full-network integration - torch.compile mode")
logging.info("=" * 80)
logging.info(f"Device: {device}")
logging.info(f"Batch: {args.batch}")
logging.info(f"Prompt: {args.prompt}")
logging.info(f"Output length: {args.output_length}")
logging.info(f"Mode:")
logging.info(f"  PyPTO: {args.use_pto}")
logging.info(f"  Dynamic Config: {args.use_dynamic_config}")
logging.info(f"  Simple Prompt: {args.simple_prompt}")
logging.info(f"  torch.compile: {args.use_compile}")
logging.info(f"  Backend: {args.backend}")
logging.info(f"  Mode: {args.mode if args.mode else 'default'}")
logging.info(f"  aclgraph: {args.use_acl_graph}")
logging.info("=" * 80)

metrics = {
    "config": {
        "device": device,
        "batch": args.batch,
        "output_length": args.output_length,
        "use_pto": args.use_pto,
        "use_compile": args.use_compile,
        "backend": args.backend,
        "use_acl_graph": args.use_acl_graph
    }
}

# ===== 步骤22：sys.modules注入 =====
if args.use_pto or args.use_dynamic_config:
    logging.info("\n[Step 22] sys.modules injection...")

    # 1. 添加路径
    sys.path.insert(0, args.model_path)

    # 2. 导入动态配置模块（如果启用）
    if args.use_dynamic_config:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dynamic_pto_config import DynamicPTOConfig
        logging.info("✓ Imported dynamic_pto_config successfully")

    # 3. 导入模块（注意：实际模块名是pto_kernels）
    try:
        import pto_kernels
        logging.info("✓ Imported pto_kernels successfully")
    except ImportError as e:
        logging.error(f"✗ Failed to import pto_kernels: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to import pto_kernels: {e}") from e

    # 4. 注册全局（关键：注册为gutenocr_3b_pto_kernels）
    sys.modules["gutenocr_3b_pto_kernels"] = pto_kernels
    logging.info("✓ sys.modules registration complete (registered as gutenocr_3b_pto_kernels)")

    # 5. 应用算子配置
    if args.use_dynamic_config:
        # 动态配置：根据batch自动选择最优算子
        config = DynamicPTOConfig.get_optimal_config(args.batch)

        # 自动调整aclgraph参数（如果动态配置建议使用aclgraph）
        if config['use_aclgraph'] and not args.use_acl_graph:
            args.use_acl_graph = True
            logging.info(f"✓ Dynamic config auto-enabled aclgraph: {config['mode']}")

        # 应用配置
        DynamicPTOConfig.apply_config(pto_kernels, args.batch, config['mode'])

        logging.info(f"✓ Dynamic config applied successfully (Batch={args.batch}):")
        logging.info(f"  USE_PTO_RMS_NORM = {pto_kernels.USE_PTO_RMS_NORM}")
        logging.info(f"  USE_PTO_MROPE = {pto_kernels.USE_PTO_MROPE}")
        logging.info(f"  USE_PTO_SWIGLU_MLP = {pto_kernels.USE_PTO_SWIGLU_MLP}")
        logging.info(f"  Expected throughput = {config['expected_throughput']:.2f} tokens/s")
        logging.info(f"  Config reason: {config['reason']}")

        metrics["dynamic_config"] = config
    else:
        # 手动配置：启用所有算子
        pto_kernels.USE_PTO_RMS_NORM = True
        pto_kernels.USE_PTO_MROPE = True
        pto_kernels.USE_PTO_SWIGLU_MLP = True
        logging.info(f"✓ USE_PTO_RMS_NORM = {pto_kernels.USE_PTO_RMS_NORM}")
        logging.info(f"✓ USE_PTO_MROPE = {pto_kernels.USE_PTO_MROPE}")
        logging.info(f"✓ USE_PTO_SWIGLU_MLP = {pto_kernels.USE_PTO_SWIGLU_MLP}")

    # 6. 设置compile模式开关
    if args.use_compile:
        pto_kernels.USE_COMPILE = True
        logging.info(f"✓ USE_COMPILE = {pto_kernels.USE_COMPILE}")

    # 7. 设置aclgraph模式开关
    if args.use_acl_graph:
        pto_kernels.USE_ACL_GRAPH = True
        logging.info(f"✓ USE_ACL_GRAPH = {pto_kernels.USE_ACL_GRAPH}")

    logging.info("[Step 22] ✓ sys.modules injection complete")
else:
    pto_kernels = None
    logging.info("[Step 22] PyPTO or dynamic config not enabled, skipping injection")

# ===== 加载模型 =====
logging.info("\n[Model Loading]")
start_load = time.time()


model = AutoModelForImageTextToText.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": device},
    local_files_only=True,
    trust_remote_code=True, attn_implementation="eager"
)

# 修复rope_scaling
for layer in model.model.language_model.layers:
    if hasattr(layer.self_attn, 'rope_scaling'):
        if layer.self_attn.rope_scaling is None:
            layer.self_attn.rope_scaling = {"mrope_section": [16, 24, 24], "type": "mrope"}

model.eval()
load_time = time.time() - start_load
logging.info(f"✓ Model loaded ({load_time:.2f}s)")
metrics["load_time"] = load_time

# ===== 步骤23：aclgraph适配 =====
if args.use_compile or args.use_acl_graph:
    logging.info("\n[Step 23] torch.compile configuration...")

    if args.use_acl_graph:
        # aclgraph模式（使用torchair CompilerConfig）
        try:
            import torchair as tng
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
            from torchair.configs.compiler_config import CompilerConfig

            logging.info("✓ torchair available")

            # ⭐ 按照正确范式配置CompilerConfig
            compiler_config = CompilerConfig()
            compiler_config.experimental_config.frozen_parameter = True
            compiler_config.experimental_config.tiling_schedule_optimize = True

            npu_backend = tng.get_npu_backend(compiler_config=compiler_config)

            logging.info(f"✓ CompilerConfig configured")
            logging.info(f"  frozen_parameter: True")
            logging.info(f"  tiling_schedule_optimize: True")

            # torch.compile配置（支持mode参数）
            compile_kwargs = {
                "dynamic": True,
                "fullgraph": True,
                "backend": npu_backend,
            }

            if args.mode:
                compile_kwargs["mode"] = args.mode
                logging.info(f"  Mode: {args.mode}")

            model = torch.compile(model, **compile_kwargs)
            logging.info(f"✓ torch.compile(aclgraph) done")

        except ImportError as e:
            logging.info(f"✗ torchair unavailable: {e}")
            logging.info(f"  Falling back to torch.compile(backend='{args.backend}', mode='{args.mode}')")

            # 回退配置
            compile_kwargs = {"backend": args.backend}
            if args.mode:
                compile_kwargs["mode"] = args.mode
            model = torch.compile(model, **compile_kwargs)
    else:
        # torch.compile模式（使用backend + mode）
        logging.info(f"  Backend: {args.backend}")
        if args.mode:
            logging.info(f"  Mode: {args.mode}")

        # ⭐ 支持reduce-overhead等mode参数
        compile_kwargs = {"backend": args.backend}
        if args.mode:
            compile_kwargs["mode"] = args.mode

        model = torch.compile(model, **compile_kwargs)

        mode_str = f"mode={args.mode}" if args.mode else "default mode"
        logging.info(f"✓ torch.compile(backend={args.backend}, {mode_str}) done")

    metrics["compile_mode"] = f"{args.backend}" if not args.use_acl_graph else "aclgraph"

# ===== 准备输入 =====
logging.info("\n[Input Preparation]")
processor = AutoProcessor.from_pretrained(
    args.model_path,
    local_files_only=True,
    trust_remote_code=True, attn_implementation="eager"
)

if args.simple_prompt:
    # 简化prompt模式（高性能）
    prompts = [args.prompt] * args.batch
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    logging.info(f"✓ Simple mode: input shape={inputs.input_ids.shape} (high performance)")
else:
    # 标准模式（apply_chat_template）
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if args.batch > 1:
        texts = [text] * args.batch
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    else:
        inputs = processor(text=[text], return_tensors="pt").to(device)

    logging.info(f"✓ Standard mode: input shape={inputs.input_ids.shape} (full format)")

logging.info(f"✓ Input preparation complete")

# ===== 推理执行 =====
logging.info("\n[Inference]")

# Warmup
logging.info(f"Warmup ({args.warmup} iters)...")
with torch.no_grad():
    for i in range(args.warmup):
        if args.batch > 1:
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        else:
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        torch.npu.synchronize()
        if i % 5 == 0 or i == args.warmup - 1:
            logging.info(f"  Warmup {i+1}/{args.warmup}")
logging.info("✓ Warmup complete")

# 正式推理（多次取平均）
logging.info(f"Inference (max_new_tokens={args.output_length}, average of 5 runs)...")
torch.npu.reset_peak_memory_stats()

infer_times = []
throughputs = []

for i in range(5):
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.output_length,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    torch.npu.synchronize()

    infer_time = time.time() - start_time
    tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
    throughput = tokens_generated / infer_time

    infer_times.append(infer_time)
    throughputs.append(throughput)

    if i % 2 == 0 or i == 4:
        logging.info(f"  Run {i+1}: {infer_time:.3f}s, {throughput:.2f} tokens/s")

avg_infer_time = sum(infer_times) / len(infer_times)
avg_throughput = sum(throughputs) / len(throughputs)
peak_memory = torch.npu.max_memory_allocated() / 1024**2  # MB

logging.info(f"✓ Inference complete")
logging.info(f"  Avg inference time: {avg_infer_time:.3f}s")
logging.info(f"  Avg throughput: {avg_throughput:.2f} tokens/s")
logging.info(f"  Peak memory: {peak_memory:.2f}MB")

infer_time = avg_infer_time  # 用于后续报告

metrics["infer_time"] = infer_time
metrics["peak_memory_mb"] = peak_memory
metrics["tokens_generated"] = outputs.shape[1] - inputs.input_ids.shape[1]
metrics["throughput_tokens_per_sec"] = metrics["tokens_generated"] / infer_time

# ===== 输出结果 =====
logging.info("\n[Output Results]")
generated_ids = outputs[0][inputs.input_ids.shape[1]:]
generated_text = processor.decode(generated_ids, skip_special_tokens=True)

logging.info(f"Generated text:")
logging.info(f"  {generated_text}")

# ===== 性能报告 =====
logging.info("\n" + "=" * 80)
logging.info("Performance Report")
logging.info("=" * 80)
logging.info(
    f"Mode: {'PyPTO' if args.use_pto else 'Baseline'} + "
    f"{'compile(' + args.backend + ')' if args.use_compile else 'Eager'}"
)
logging.info(f"Inference time: {infer_time:.3f}s")
logging.info(f"Generated tokens: {metrics['tokens_generated']}")
logging.info(f"Throughput: {metrics['throughput_tokens_per_sec']:.2f} tokens/s")
logging.info(f"Peak memory: {peak_memory:.2f}MB")
logging.info("=" * 80)

# ===== 保存JSON报告 =====
if args.report_file:
    with open(args.report_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"✓ Performance report saved to: {args.report_file}")

logging.info("\n[Integration Complete] SwiGLU MLP successfully integrated into the full network")
