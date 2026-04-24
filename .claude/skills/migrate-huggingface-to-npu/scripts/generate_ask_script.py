#!/usr/bin/env python3
"""
根据模板生成大语言模型问答脚本
用法: python3 generate_ask_script.py --model-name "模型名" --script-dir "脚本目录" [--default-model-dir "默认模型目录"]
示例: python3 generate_ask_script.py --model-name "Qwen2-7B" --script-dir "/path/to/scripts" --default-model-dir "/data/models"
"""

import argparse
import os

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
    + """ 单次问答脚本
用法: python3 ask_"""
    + args.model_name
    + '''.py [--prompt "问题"] [--device 卡号] [--model-path 路径]
"""

import argparse, torch, torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="'''
    + args.model_name
    + ''' 问答脚本")
parser.add_argument("--prompt", default="你好")
parser.add_argument("--device", default=0, type=int, help="NPU卡号")
parser.add_argument("--model-path", default="'''
    + args.default_model_dir
    + """/"""
    + args.model_name
    + """", help="模型权重路径")
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
"""
)

os.makedirs(args.script_dir, exist_ok=True)
output_file = os.path.join(args.script_dir, f"ask_{args.model_name}.py")
with open(output_file, "w") as f:
    f.write(content)

os.chmod(output_file, 0o755)

print(f"脚本已生成: {output_file}")
print(f"使用设备: npu:0（默认）")
print(f"\n使用方法:")
print(f"  python3 {output_file}")
print(f"  python3 {output_file} --prompt '你的问题'")
print(f"  python3 {output_file} --device 7")
print(f"  python3 {output_file} --model-path /custom/path")
