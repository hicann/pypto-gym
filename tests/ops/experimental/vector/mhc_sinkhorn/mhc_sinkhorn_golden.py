#!/usr/bin/env python3
# coding: utf-8

"""PyPTO mhc_sinkhorn golden reference implementation.

Copyright (c) 2025 Huawei Technologies Co., Ltd.

mhc_sinkhorn 实现 Sinkhorn-Knopp 双随机矩阵迭代归一化算法。
该算子将矩阵通过交替行列归一化迭代转换为双随机矩阵（每行每列和为1的矩阵）。
用于 mHC (Manifold-Constrained Hyper-Connections) 中的残差连接矩阵约束。

"""

import torch


def mhc_sinkhorn_golden(
    x: torch.Tensor,
    eps: float = 1e-6,
    num_iters: int = 20
) -> torch.Tensor:
    """Sinkhorn-Knopp 双随机矩阵迭代归一化。

    算法流程:
        1. softmax(dim=-1) + eps
        2. 列归一化 (sum dim=-2 + eps)
        3. 循环 num_iters-1 次:
           - 行归一化 (sum dim=-1 + eps)
           - 列归一化 (sum dim=-2 + eps)

    Args:
        x: 输入 tensor, shape [B*S, N, N], dtype float32。
           B*S 为动态轴（batch和seq_len合并），N 固定为 8。
        eps: 数值稳定性参数，防止除零。默认 1e-6。
        num_iters: Sinkhorn 迭代次数。默认 20。

    Returns:
        双随机矩阵 tensor, shape [B*S, N, N], dtype float32。
        每行和每列的元素和均为 1（近似，受 eps 影响）。
    """
    # Step 1: Row-wise softmax + eps
    h_comb = torch.softmax(x, dim=-1) + eps
    
    # Step 2: Initial column normalization
    col_sum = h_comb.sum(dim=-2, keepdim=True)
    h_comb = h_comb / (col_sum + eps)
    
    # Step 3: Alternate normalization (row->col), repeated (num_iters-1) times
    for _ in range(max(num_iters - 1, 0)):
        # Row Norm
        row_sum = h_comb.sum(dim=-1, keepdim=True)
        h_comb = h_comb / (row_sum + eps)
        
        # Col Norm
        col_sum = h_comb.sum(dim=-2, keepdim=True)
        h_comb = h_comb / (col_sum + eps)
    
    return h_comb


# ==================== 自动生成的验证代码 ====================

def _validate():
    """验证 golden 函数的正确性。"""
    
    print("=" * 60)
    print("mhc_sinkhorn_golden 验证报告")
    print("=" * 60)
    
    # 参数配置
    eps = 1e-6
    num_iters = 20
    
    # 典型 case 验证（来自 spec.md §11）
    print("\n[典型 case 验证]")
    
    test_cases = [
        ("性能_P0", [4096, 8, 8]),
        ("性能_P1", [2048, 8, 8]),
        ("功能_P0", [1024, 8, 8]),
    ]
    
    for name, shape in test_cases:
        x = torch.randn(shape, dtype=torch.float32)
        y = mhc_sinkhorn_golden(x, eps, num_iters)
        
        # 检查 shape
        shape_match = y.shape == torch.Size(shape)
        
        # 检查 dtype
        dtype_match = y.dtype == torch.float32
        
        # 检查双随机矩阵性质（行列和近似为1）
        row_sum = y.sum(dim=-1)
        col_sum = y.sum(dim=-2)
        row_sum_close = torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)
        col_sum_close = torch.allclose(col_sum, torch.ones_like(col_sum), atol=1e-4)
        
        status = "✓ PASS" if (shape_match and dtype_match and row_sum_close and col_sum_close) else "✗ FAIL"
        print(f"  {name}: shape={shape} ... {status}")
        if status == "✗ FAIL":
            print(f"    shape_match={shape_match}, dtype_match={dtype_match}")
            print(f"    row_sum_close={row_sum_close}, col_sum_close={col_sum_close}")
    
    # 泛化 case 验证（动态轴边界）
    print("\n[泛化 case 验证]")
    
    dynamic_axis_values = [1024, 2048, 4096]  # B*S 取值范围
    for bs in dynamic_axis_values:
        shape = [bs, 8, 8]
        x = torch.randn(shape, dtype=torch.float32)
        y = mhc_sinkhorn_golden(x, eps, num_iters)
        
        shape_match = y.shape == torch.Size(shape)
        dtype_match = y.dtype == torch.float32
        status = "✓ PASS" if (shape_match and dtype_match) else "✗ FAIL"
        print(f"  B*S={bs}: shape={shape} ... {status}")
    
    # 数值稳定性检查
    print("\n[数值稳定性检查]")
    
    # 大值输入
    x_large = torch.randn([1024, 8, 8], dtype=torch.float32) * 100
    y_large = mhc_sinkhorn_golden(x_large, eps, num_iters)
    no_nan_inf = not (torch.isnan(y_large).any() or torch.isinf(y_large).any())
    print(f"  大值输入 (scale=100) ... {'✓ PASS' if no_nan_inf else '✗ FAIL'}")
    
    # 小值输入
    x_small = torch.randn([1024, 8, 8], dtype=torch.float32) * 0.01
    y_small = mhc_sinkhorn_golden(x_small, eps, num_iters)
    no_nan_inf_small = not (torch.isnan(y_small).any() or torch.isinf(y_small).any())
    print(f"  小值输入 (scale=0.01) ... {'✓ PASS' if no_nan_inf_small else '✗ FAIL'}")
    
    # 值域检查（输出应为正数，且行列和近似为1）
    print("\n[值域检查]")
    
    x = torch.randn([1024, 8, 8], dtype=torch.float32)
    y = mhc_sinkhorn_golden(x, eps, num_iters)
    
    # 输出应为正数（因为 softmax + eps）
    all_positive = (y > 0).all()
    print(f"  输出全为正数 ... {'✓ PASS' if all_positive else '✗ FAIL'}")
    
    # 双随机矩阵性质
    row_sum = y.sum(dim=-1)
    col_sum = y.sum(dim=-2)
    row_sum_1 = torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3)
    col_sum_1 = torch.allclose(col_sum, torch.ones_like(col_sum), atol=1e-3)
    print(f"  行和近似为1 (atol=1e-3) ... {'✓ PASS' if row_sum_1 else '✗ FAIL'}")
    print(f"  列和近似为1 (atol=1e-3) ... {'✓ PASS' if col_sum_1 else '✗ FAIL'}")
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    _validate()