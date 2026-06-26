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

print("=" * 80)
print("SwiGLU MLP整网集成 - torch.compile模式")
print("=" * 80)
print(f"设备: {device}")
print(f"Batch: {args.batch}")
print(f"Prompt: {args.prompt}")
print(f"Output length: {args.output_length}")
print(f"模式:")
print(f"  PyPTO: {args.use_pto}")
print(f"  Dynamic Config: {args.use_dynamic_config}")
print(f"  Simple Prompt: {args.simple_prompt}")
print(f"  torch.compile: {args.use_compile}")
print(f"  Backend: {args.backend}")
print(f"  Mode: {args.mode if args.mode else 'default'}")
print(f"  aclgraph: {args.use_acl_graph}")
print("=" * 80)

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
    print("\n[步骤22] sys.modules注入...")

    # 1. 添加路径
    sys.path.insert(0, args.model_path)

    # 2. 导入动态配置模块（如果启用）
    if args.use_dynamic_config:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dynamic_pto_config import DynamicPTOConfig
        print("✓ 导入dynamic_pto_config成功")

    # 3. 导入模块（注意：实际模块名是pto_kernels）
    try:
        import pto_kernels
        print("✓ 导入pto_kernels成功")
    except ImportError as e:
        print(f"✗ 导入pto_kernels失败: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to import pto_kernels: {e}") from e

    # 4. 注册全局（关键：注册为gutenocr_3b_pto_kernels）
    sys.modules["gutenocr_3b_pto_kernels"] = pto_kernels
    print("✓ sys.modules注册完成（注册为gutenocr_3b_pto_kernels）")

    # 5. 应用算子配置
    if args.use_dynamic_config:
        # 动态配置：根据batch自动选择最优算子
        config = DynamicPTOConfig.get_optimal_config(args.batch)

        # 自动调整aclgraph参数（如果动态配置建议使用aclgraph）
        if config['use_aclgraph'] and not args.use_acl_graph:
            args.use_acl_graph = True
            print(f"✓ 动态配置自动启用aclgraph: {config['mode']}")

        # 应用配置
        DynamicPTOConfig.apply_config(pto_kernels, args.batch, config['mode'])

        print(f"✓ 动态配置应用成功 (Batch={args.batch}):")
        print(f"  USE_PTO_RMS_NORM = {pto_kernels.USE_PTO_RMS_NORM}")
        print(f"  USE_PTO_MROPE = {pto_kernels.USE_PTO_MROPE}")
        print(f"  USE_PTO_SWIGLU_MLP = {pto_kernels.USE_PTO_SWIGLU_MLP}")
        print(f"  预期吞吐 = {config['expected_throughput']:.2f} tokens/s")
        print(f"  配置原因: {config['reason']}")

        metrics["dynamic_config"] = config
    else:
        # 手动配置：启用所有算子
        pto_kernels.USE_PTO_RMS_NORM = True
        pto_kernels.USE_PTO_MROPE = True
        pto_kernels.USE_PTO_SWIGLU_MLP = True
        print(f"✓ USE_PTO_RMS_NORM = {pto_kernels.USE_PTO_RMS_NORM}")
        print(f"✓ USE_PTO_MROPE = {pto_kernels.USE_PTO_MROPE}")
        print(f"✓ USE_PTO_SWIGLU_MLP = {pto_kernels.USE_PTO_SWIGLU_MLP}")

    # 6. 设置compile模式开关
    if args.use_compile:
        pto_kernels.USE_COMPILE = True
        print(f"✓ USE_COMPILE = {pto_kernels.USE_COMPILE}")

    # 7. 设置aclgraph模式开关
    if args.use_acl_graph:
        pto_kernels.USE_ACL_GRAPH = True
        print(f"✓ USE_ACL_GRAPH = {pto_kernels.USE_ACL_GRAPH}")

    print("[步骤22] ✓ sys.modules注入完成")
else:
    pto_kernels = None
    print("[步骤22] 未启用PyPTO或动态配置，跳过注入")

# ===== 加载模型 =====
print("\n[模型加载]")
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
print(f"✓ 模型加载完成 ({load_time:.2f}s)")
metrics["load_time"] = load_time

# ===== 步骤23：aclgraph适配 =====
if args.use_compile or args.use_acl_graph:
    print("\n[步骤23] torch.compile配置...")

    if args.use_acl_graph:
        # aclgraph模式（使用torchair CompilerConfig）
        try:
            import torchair as tng
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
            from torchair.configs.compiler_config import CompilerConfig

            print("✓ torchair可用")

            # ⭐ 按照正确范式配置CompilerConfig
            compiler_config = CompilerConfig()
            compiler_config.experimental_config.frozen_parameter = True
            compiler_config.experimental_config.tiling_schedule_optimize = True

            npu_backend = tng.get_npu_backend(compiler_config=compiler_config)

            print(f"✓ CompilerConfig配置完成")
            print(f"  frozen_parameter: True")
            print(f"  tiling_schedule_optimize: True")

            # torch.compile配置（支持mode参数）
            compile_kwargs = {
                "dynamic": True,
                "fullgraph": True,
                "backend": npu_backend,
            }

            if args.mode:
                compile_kwargs["mode"] = args.mode
                print(f"  Mode: {args.mode}")

            model = torch.compile(model, **compile_kwargs)
            print(f"✓ torch.compile(aclgraph)完成")

        except ImportError as e:
            print(f"✗ torchair不可用: {e}")
            print(f"  回退到 torch.compile(backend='{args.backend}', mode='{args.mode}')")

            # 回退配置
            compile_kwargs = {"backend": args.backend}
            if args.mode:
                compile_kwargs["mode"] = args.mode
            model = torch.compile(model, **compile_kwargs)
    else:
        # torch.compile模式（使用backend + mode）
        print(f"  Backend: {args.backend}")
        if args.mode:
            print(f"  Mode: {args.mode}")

        # ⭐ 支持reduce-overhead等mode参数
        compile_kwargs = {"backend": args.backend}
        if args.mode:
            compile_kwargs["mode"] = args.mode

        model = torch.compile(model, **compile_kwargs)

        mode_str = f"mode={args.mode}" if args.mode else "default mode"
        print(f"✓ torch.compile(backend={args.backend}, {mode_str})完成")

    metrics["compile_mode"] = f"{args.backend}" if not args.use_acl_graph else "aclgraph"

# ===== 准备输入 =====
print("\n[输入准备]")
processor = AutoProcessor.from_pretrained(
    args.model_path,
    local_files_only=True,
    trust_remote_code=True, attn_implementation="eager"
)

if args.simple_prompt:
    # 简化prompt模式（高性能）
    prompts = [args.prompt] * args.batch
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    print(f"✓ 简化模式: 输入shape={inputs.input_ids.shape} (高性能)")
else:
    # 标准模式（apply_chat_template）
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if args.batch > 1:
        texts = [text] * args.batch
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    else:
        inputs = processor(text=[text], return_tensors="pt").to(device)

    print(f"✓ 标准模式: 输入shape={inputs.input_ids.shape} (完整格式)")

print(f"✓ 输入准备完成")

# ===== 推理执行 =====
print("\n[推理执行]")

# Warmup
print(f"Warmup ({args.warmup}次)...")
with torch.no_grad():
    for i in range(args.warmup):
        if args.batch > 1:
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        else:
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        torch.npu.synchronize()
        if i % 5 == 0 or i == args.warmup - 1:
            print(f"  Warmup {i+1}/{args.warmup}")
print("✓ Warmup完成")

# 正式推理（多次取平均）
print(f"推理 (max_new_tokens={args.output_length}, 5次平均)...")
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
        print(f"  第{i+1}次: {infer_time:.3f}s, {throughput:.2f} tokens/s")

avg_infer_time = sum(infer_times) / len(infer_times)
avg_throughput = sum(throughputs) / len(throughputs)
peak_memory = torch.npu.max_memory_allocated() / 1024**2  # MB

print(f"✓ 推理完成")
print(f"  平均耗时: {avg_infer_time:.3f}s")
print(f"  平均吞吐: {avg_throughput:.2f} tokens/s")
print(f"  峰值显存: {peak_memory:.2f}MB")

infer_time = avg_infer_time  # 用于后续报告

metrics["infer_time"] = infer_time
metrics["peak_memory_mb"] = peak_memory
metrics["tokens_generated"] = outputs.shape[1] - inputs.input_ids.shape[1]
metrics["throughput_tokens_per_sec"] = metrics["tokens_generated"] / infer_time

# ===== 输出结果 =====
print("\n[输出结果]")
generated_ids = outputs[0][inputs.input_ids.shape[1]:]
generated_text = processor.decode(generated_ids, skip_special_tokens=True)

print(f"生成文本:")
print(f"  {generated_text}")

# ===== 性能报告 =====
print("\n" + "=" * 80)
print("性能报告")
print("=" * 80)
print(
    f"模式: {'PyPTO' if args.use_pto else 'Baseline'} + "
    f"{'compile(' + args.backend + ')' if args.use_compile else 'Eager'}"
)
print(f"推理耗时: {infer_time:.3f}s")
print(f"生成tokens: {metrics['tokens_generated']}")
print(f"吞吐: {metrics['throughput_tokens_per_sec']:.2f} tokens/s")
print(f"峰值显存: {peak_memory:.2f}MB")
print("=" * 80)

# ===== 保存JSON报告 =====
if args.report_file:
    with open(args.report_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ 性能报告已保存: {args.report_file}")

print("\n[集成完成] SwiGLU MLP已成功集成到整网")