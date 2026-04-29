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
根据模板生成大语言模型问答脚本
用法: python3 generate_ask_script.py --model-name "模型名" --script-dir "脚本目录"
      [--default-model-dir "默认模型目录"]
示例: python3 generate_ask_script.py --model-name "Qwen2-7B" --script-dir "/path/to/scripts"
      --default-model-dir "/data/models"
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')

parser = argparse.ArgumentParser(description="生成大语言模型问答脚本")
parser.add_argument("--model-name", required=True, help="模型名称（如 Qwen2-7B）")
parser.add_argument("--script-dir", required=True, help="脚本输出目录（绝对路径）")
parser.add_argument(
    "--default-model-dir", default="/data/models", help="默认模型存放目录"
)
args = parser.parse_args()

content = (
    '''#!/usr/bin/env python3
"""
'''
    + args.model_name
    + """ 推理脚本
用法: python3 ask_"""
    + args.model_name
    + '''.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
       [--sentence_file 提示词文件] [--output_length 长度] [--use_pypto]
"""

import argparse, torch, torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="'''
    + args.model_name
    + ''' 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于--sentence_file）")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default="'''
    + args.default_model_dir
    + """/"""
    + args.model_name
    + """", help="模型权重路径")
parser.add_argument("--sentence_file", type=str, default=None, help="从文件读取提示词（多行以换行拼接）")
parser.add_argument("--output_length", type=int, default=100, help="最大生成token数")
parser.add_argument("--use_pypto", action="store_true", help="PyPTO融合算子模式（占位）")
args = parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

logging.info(f"使用设备: npu:{args.device}")
logging.info(f"模型路径: {args.model_path}")

# Resolve prompt: explicit --prompt > --sentence_file > default
if args.prompt:
    prompt = args.prompt
elif args.sentence_file:
    with open(args.sentence_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    prompt = "\\n".join(lines)
    logging.info(f"从文件读取提示词: {args.sentence_file} ({len(prompt)} 字符)")
else:
    prompt = "你好"

if args.use_pypto:
    logging.info("PyPTO mode enabled (placeholder)")

torch.npu.set_device(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.float16,
    device_map={"": f"npu:{args.device}"}, local_files_only=True, trust_remote_code=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(f"npu:{args.device}")
logging.info(f"输入token数: {inputs.input_ids.shape[1]}")
outputs = model.generate(**inputs, max_new_tokens=args.output_length, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logging.info(response)
"""
)

os.makedirs(args.script_dir, exist_ok=True)
output_file = os.path.join(args.script_dir, f"ask_{args.model_name}.py")
with open(output_file, "w") as f:
    f.write(content)

os.chmod(output_file, 0o755)

logging.info(f"脚本已生成: {output_file}")
logging.info(f"使用设备: npu:0（默认）")
logging.info(f"\n使用方法:")
logging.info(f"  python3 {output_file}")
logging.info(f"  python3 {output_file} --prompt '你的问题'")
logging.info(f"  python3 {output_file} --device 7")
logging.info(f"  python3 {output_file} --model-path /custom/path")
logging.info(f"  python3 {output_file} --sentence_file prompts.txt --output_length 200")
logging.info(f"  python3 {output_file} --use_pypto")
