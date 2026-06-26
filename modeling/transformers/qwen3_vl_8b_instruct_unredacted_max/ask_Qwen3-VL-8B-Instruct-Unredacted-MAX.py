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
Qwen3-VL-8B-Instruct-Unredacted-MAX 推理脚本
用法: python3 ask_Qwen3-VL-8B-Instruct-Unredacted-MAX.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
       [--sentence_file 提示词文件] [--output_length 长度] [--use_pypto]
"""

import argparse
import sys
import torch
import torch_npu
# PyPTO: add workspace root for cann_pow_patch
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _WORKSPACE_ROOT)
import cann_pow_patch
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="Qwen3-VL-8B-Instruct-Unredacted-MAX 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于--sentence_file）")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument(
    "--model-path",
    default=os.environ.get("MODEL_PATH", ""),
     help="模型权重路径")
parser.add_argument("--sentence_file", type=str, default=None, help="从文件读取提示词（多行以换行拼接）")
parser.add_argument("--output_length", type=int, default=100, help="最大生成token数")
parser.add_argument("--use_pypto", action="store_true", help="PyPTO融合算子模式")
args = parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

if args.prompt:
    prompt = args.prompt
elif args.sentence_file:
    with open(args.sentence_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    prompt = "\n".join(lines)
    logging.info(f"从文件读取提示词: {args.sentence_file} ({len(prompt)} 字符)")
else:
    prompt = "你好，请介绍一下自己。"

if args.use_pypto:
    sys.path.insert(0, args.model_path)
    import pto_kernels
    sys.modules["pto_kernels"] = pto_kernels
    pto_kernels.USE_PTO_RMS_NORM = True
    logging.info("PyPTO mode enabled: RMSNorm fused")

logging.info(f"使用设备: npu:{args.device}")
logging.info(f"模型路径: {args.model_path}")

torch.npu.set_device(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path, local_files_only=True,
    torch_dtype=torch.bfloat16, attn_implementation="eager"
).to(f"npu:{args.device}").eval()

# Monkey-patch RMSNorm for PyPTO injection
if args.use_pypto:
    import types
    patched_count = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen3VLTextRMSNorm":

            orig_forward = module.forward

            def new_forward(self, hidden_states, orig=orig_forward):
                pk = sys.modules.get("pto_kernels")
                if pk is not None and getattr(pk, "USE_PTO_RMS_NORM", False):
                    return pk.rms_norm_wrapper(hidden_states, self.weight, self.variance_epsilon)
                return orig(hidden_states)
            module.forward = types.MethodType(new_forward, module)
            patched_count += 1
    logging.info(f"PyPTO RMSNorm monkey-patched to {patched_count} layers")

inputs = tokenizer(text, return_tensors="pt").to(f"npu:{args.device}")
logging.info(f"输入token数: {inputs.input_ids.shape[1]}")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=args.output_length, temperature=0.7, do_sample=True, top_p=0.8)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logging.info(response)
