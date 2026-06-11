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
Qwen3.6-27B 推理脚本
用法: python3 ask_Qwen3.6-27B.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
       [--sentence_file 提示词文件] [--output_length 长度] [--use_pypto]
"""

import argparse
import sys
import torch
import torch_npu

parser = argparse.ArgumentParser(description="Qwen3.6-27B 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于--sentence_file）")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default="/mnt/workspace/gitCode/cann/models/pure/Qwen3.6-27B", help="模型权重路径")
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

# PyPTO injection must happen BEFORE transformers is imported.
if args.use_pypto:
    sys.path.insert(0, args.model_path)
    import qwen3_6_27b_pto_kernels as pto_kernels
    sys.modules["qwen3_6_27b_pto_kernels"] = pto_kernels
    pto_kernels.USE_PTO_GATED_DELTA_RULE = True
    logging.info("PyPTO mode enabled: GatedDeltaRule fused")

from transformers import AutoTokenizer
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

logging.info(f"使用设备: npu:{args.device}")
logging.info(f"模型路径: {args.model_path}")

torch.npu.set_device(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    args.model_path, local_files_only=True, trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(f"npu:{args.device}").eval()

# Monkey-patch Qwen3_5GatedDeltaNet to route the chunk-prefill path through
# the PyPTO wrapper. The hook is also embedded in the bundled modeling file
# (src/pypto_gym/transformers/qwen3_6_27b/modeling_qwen3_5.py); this runtime
# patch covers users who load the model with the upstream transformers code.
if args.use_pypto:
    import types
    patched_count = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen3_5GatedDeltaNet":

            orig_forward = module.forward

            def new_forward(self, *fwd_args, _orig=orig_forward, _pk=pto_kernels, **fwd_kwargs):
                _orig_chunk = self.chunk_gated_delta_rule
                if getattr(_pk, "USE_PTO_GATED_DELTA_RULE", False):
                    def _patched(q, k, v, **kw):
                        try:
                            return _pk.gated_delta_rule_wrapper(q, k, v, **kw)
                        except NotImplementedError:
                            return _orig_chunk(q, k, v, **kw)
                    self.chunk_gated_delta_rule = _patched
                try:
                    return _orig(*fwd_args, **fwd_kwargs)
                finally:
                    self.chunk_gated_delta_rule = _orig_chunk
            module.forward = types.MethodType(new_forward, module)
            patched_count += 1
    logging.info(f"PyPTO GatedDeltaRule monkey-patched to {patched_count} layers")

inputs = tokenizer(text, return_tensors="pt").to(f"npu:{args.device}")
logging.info(f"输入token数: {inputs.input_ids.shape[1]}")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=args.output_length, temperature=0.7, do_sample=True, top_p=0.8)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logging.info(response)
