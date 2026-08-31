#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Qwen3-1.7B 单次问答脚本
用法: python3 infer.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
"""

import argparse
import os
import torch
import torch_npu

parser = argparse.ArgumentParser(description="Qwen3-1.7B 问答脚本")
parser.add_argument("--prompt", default="你好")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/path/to/models/Qwen3-1.7B"), help="模型权重路径")
parser.add_argument("--use-pto", action="store_true", help="启用PyPTO算子优化")
args = parser.parse_args()

import sys
if args.use_pto:
    sys.path.insert(0, args.model_path)
    import pto_kernels
    sys.modules["pto_kernels"] = pto_kernels
    pto_kernels.USE_PTO_RMS_NORM = True
    logging.info("* PyPTO RMSNorm enabled")

from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

logging.info(f"Using device: npu:{args.device}")
logging.info(f"Model path: {args.model_path}")

torch.npu.set_device(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.float16,
    device_map={"": f"npu:{args.device}"}, local_files_only=True, trust_remote_code=True
)

inputs = tokenizer(args.prompt, return_tensors="pt").to(f"npu:{args.device}")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logging.info(response)
