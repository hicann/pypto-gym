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
生成两段式基准测试脚本 bench_{model_name}.sh
Phase 1: baseline (no PyPTO)
Phase 2: with PyPTO

用法: python3 generate_bench_script.py --model-name "模型名" --script-dir "脚本目录" --model-dir "模型权重目录"
示例: python3 generate_bench_script.py --model-name "Qwen2-7B" --script-dir "/data/models/Qwen2-7B/scripts"
      --model-dir "/data/models/Qwen2-7B"
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')

parser = argparse.ArgumentParser(description="生成两段式基准测试脚本")
parser.add_argument("--model-name", required=True, help="模型名称（如 Qwen2-7B）")
parser.add_argument("--script-dir", required=True, help="脚本输出目录（绝对路径）")
parser.add_argument("--model-dir", required=True, help="模型权重目录（绝对路径）")
args = parser.parse_args()

content = f'''#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

# Benchmark script for {args.model_name}
# Compares baseline vs PyPTO backend

set -e

MODEL_ID="{args.model_name}"
MODEL_PATH="{args.model_dir}"
INPUT_FILE="$(dirname $0)/sample_inputs.txt"
OUTPUT_LENGTH=100
SUMMARY_FILE="$(dirname $0)/benchmark_summary.txt"

echo "========================================"
echo "  {args.model_name} Performance Benchmark"
echo "========================================"
echo ""
echo "Model: ${{MODEL_ID}}"
echo "Model path: ${{MODEL_PATH}}"
echo "Input: ${{INPUT_FILE}}"
echo "Output length: ${{OUTPUT_LENGTH}} tokens"
echo ""

# Clean previous results
rm -f ${{SUMMARY_FILE}}

echo "Phase 1: Running baseline (no PyPTO)..."
python3 $(dirname $0)/ask_{args.model_name}.py \\
    --model-path ${{MODEL_PATH}} \\
    --sentence_file ${{INPUT_FILE}} \\
    --output_length ${{OUTPUT_LENGTH}}

echo ""
echo "Phase 2: Running with PyPTO..."
python3 $(dirname $0)/ask_{args.model_name}.py \\
    --model-path ${{MODEL_PATH}} \\
    --sentence_file ${{INPUT_FILE}} \\
    --output_length ${{OUTPUT_LENGTH}} \\
    --use_pypto

echo ""
echo "========================================"
echo "  Benchmark Complete"
echo "========================================"
'''

os.makedirs(args.script_dir, exist_ok=True)
output_file = os.path.join(args.script_dir, f"bench_{args.model_name}.sh")
with open(output_file, "w") as f:
    f.write(content)

os.chmod(output_file, 0o755)

logging.info(f"Benchmark脚本已生成: {output_file}")
logging.info(f"使用方法: bash {output_file}")
