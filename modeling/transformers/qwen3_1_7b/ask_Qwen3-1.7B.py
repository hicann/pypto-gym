#!/usr/bin/env python3
"""
Qwen3-1.7B 推理脚本
用法: python3 ask_Qwen3-1.7B.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
       [--sentence_file 提示词文件] [--output_length 长度] [--use_pypto]
"""

import sys
import argparse, torch, torch_npu

parser = argparse.ArgumentParser(description="Qwen3-1.7B 推理脚本")
parser.add_argument("--prompt", default=None, help="提问文本（优先级高于--sentence_file）")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default="/mnt/workspace/gitCode/cann/models/pure/Qwen3-1.7B", help="模型权重路径")
parser.add_argument("--sentence_file", type=str, default=None, help="从文件读取提示词（多行以换行拼接）")
parser.add_argument("--output_length", type=int, default=100, help="最大生成token数")
parser.add_argument("--use_pypto", action="store_true", help="PyPTO融合算子模式")
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
    prompt = "\n".join(lines)
    logging.info(f"从文件读取提示词: {args.sentence_file} ({len(prompt)} 字符)")
else:
    prompt = "你好"

if args.use_pypto:
    sys.path.insert(0, args.model_path)
    import pto_kernels
    sys.modules["pto_kernels"] = pto_kernels
    pto_kernels.USE_PTO_RMS_NORM = True
    logging.info("PyPTO mode enabled: RMSNorm fused")

from transformers import AutoModelForCausalLM, AutoTokenizer

torch.npu.set_device(args.device)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.float16,
    device_map={"": f"npu:{args.device}"}, local_files_only=True,
    trust_remote_code=True
)

if args.use_pypto:
    import types
    patched_count = 0
    for module in model.modules():
        if type(module).__name__ == "Qwen3RMSNorm":
            orig_forward = module.forward
            def new_forward(self, hidden_states, orig=orig_forward):
                pk = sys.modules.get("pto_kernels")
                if pk is not None and getattr(pk, "USE_PTO_RMS_NORM", False):
                    return pk.rms_norm_wrapper(hidden_states, self.weight, self.variance_epsilon)
                return orig(hidden_states)
            module.forward = types.MethodType(new_forward, module)
            patched_count += 1
    logging.info(f"PyPTO RMSNorm monkey-patched to {patched_count} layers")

inputs = tokenizer(prompt, return_tensors="pt").to(f"npu:{args.device}")
logging.info(f"输入token数: {inputs.input_ids.shape[1]}")
outputs = model.generate(**inputs, max_new_tokens=args.output_length, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logging.info(response)
