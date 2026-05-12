#!/usr/bin/env python3
"""
测试 RMSNorm PyPTO 实现

验证步骤：
1. 使用真实 shape/dtype
2. 对比 PyPTO 和 Golden 的精度
3. 确保 diff < 2e-3
"""

import os
import sys
import torch
import torch_npu

os.environ.setdefault('TILE_FWK_DEVICE_ID', '0')

torch.npu.set_device(5)
device = 'npu:5'

from src.pypto_gym.ops.pypto_tile.qwen3_1_7b.rms_norm.rms_norm_impl import rms_norm_qwen3_wrapper
from rms_norm_golden import rms_norm_golden

print("=" * 60)
print("测试 RMSNorm PyPTO 实现")
print("=" * 60)

# 测试场景（来自 test_cases.json）
test_cases = [
    {
        "name": "input_layernorm",
        "shape": [1, 1, 2048],
        "weight_shape": [2048],
    },
    {
        "name": "q_norm",
        "shape": [1, 1, 16, 128],
        "weight_shape": [128],
    },
    {
        "name": "k_norm",
        "shape": [1, 1, 8, 128],
        "weight_shape": [128],
    },
]

torch.manual_seed(42)

for case in test_cases:
    name = case["name"]
    shape = case["shape"]
    weight_shape = case["weight_shape"]
    
    # 创建测试数据
    x = torch.randn(shape, dtype=torch.float16, device=device)
    weight = torch.ones(weight_shape, dtype=torch.float16, device=device)
    
    # PyPTO 实现
    output_pto = rms_norm_qwen3_wrapper(x, weight, eps=1e-6)
    
    # Golden 实现
    output_golden = rms_norm_golden(x.cpu(), weight.cpu(), eps=1e-6).to(device)
    
    # 精度对比
    max_diff = (output_pto - output_golden).abs().max().item()
    print(f"[{name}] shape={shape}, max_diff={max_diff:.6f}")
    
    assert max_diff < 2e-3, f"精度差异过大: {max_diff}"
    
print("\n" + "=" * 60)
print("✓ 所有 RMSNorm 测试通过")
print("=" * 60)