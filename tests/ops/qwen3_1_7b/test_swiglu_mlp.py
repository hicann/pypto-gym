#!/usr/bin/env python3
"""
SwiGLU MLP 算子测试

验证精度：PyPTO kernel vs Golden实现
"""

import os
import sys
import torch

from src.pypto_gym.ops.pypto_tile.qwen3_1_7b.swiglu_mlp.swiglu_mlp_impl import swiglu_mlp_impl
from swiglu_mlp_golden import swiglu_mlp_golden

H = 2048
INT_SIZE = 6144


def test_precision(device='cpu'):
    """精度测试"""
    print("=" * 60)
    print("SwiGLU MLP 精度测试")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    test_cases = [
        (1, H),    # 单token
        (4, H),    # 小batch
        (8, H),    # BS_TILE边界
        (16, H),   # 超过BS_TILE
        (32, H),   # 较大batch
    ]
    
    for shape in test_cases:
        print(f"\n测试 shape={shape}")
        
        # 生成随机输入
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)
        Wgate = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device=device)
        Wup = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device=device)
        Wdown = torch.randn(INT_SIZE, H, dtype=torch.bfloat16, device=device)
        
        # PyPTO kernel
        output_pto = swiglu_mlp_impl(x, Wgate, Wup, Wdown)
        
        # Golden参考
        output_golden = swiglu_mlp_golden(x, Wgate, Wup, Wdown)
        
        # 精度对比
        max_diff = (output_pto - output_golden).abs().max().item()
        mean_diff = (output_pto - output_golden).abs().mean().item()
        
        print(f"  PyPTO输出范围: [{output_pto.min().item():.4f}, {output_pto.max().item():.4f}]")
        print(f"  Golden输出范围: [{output_golden.min().item():.4f}, {output_golden.max().item():.4f}]")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print(f"  [PRECISION_PASS] ✓ 精度通过")
        else:
            print(f"  [PRECISION_FAIL] ✗ 精度超限")
            return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SwiGLU MLP 测试")
    parser.add_argument('--device', default='cpu', help='运行设备')
    args = parser.parse_args()
    
    success = test_precision(args.device)
    sys.exit(0 if success else 1)