#!/usr/bin/env python3
"""
Qwen3-1.7B 单次问答脚本
用法: python3 ask_Qwen3-1.7B.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
"""

import argparse, torch, torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Qwen3-1.7B 问答脚本")
parser.add_argument("--prompt", default="你好")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default="/data/z00885570/models/Qwen3-1.7B", help="模型权重路径")
args = parser.parse_args()

print(f"使用设备: npu:{args.device}")
print(f"模型路径: {args.model_path}")

torch.npu.set_device(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.float16,
    device_map={"": f"npu:{args.device}"}, local_files_only=True, trust_remote_code=True
)

inputs = tokenizer(args.prompt, return_tensors="pt").to(f"npu:{args.device}")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
