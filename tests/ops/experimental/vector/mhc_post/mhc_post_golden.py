#!/usr/bin/env python3
# coding: utf-8

"""PyPTO mhc_post golden reference implementation.

算子名称: mhc_post
公式:
    h_out_fp32 = h_out.float()
    x_fp32 = x.float()
    h_post_term = h_post.unsqueeze(-1) * h_out_fp32.unsqueeze(-2)
    h_comb_term = torch.sum(h_res.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3)
    output = (h_post_term + h_comb_term).to(torch.bfloat16)
"""

import torch
from typing import Optional

# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def mhc_post_golden(
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
) -> torch.Tensor:
    """mhc_post golden reference implementation.
    
    实现 MHC (Manifold-Constrained Hyper-Connections) 系统的后处理融合计算：
    - h_post_term: 逐样本动态权重广播乘法 (h_post × h_out)
    - h_comb_term: 逐样本流间混合求和 (h_res × x + dim=-3 reduction)
    - 融合输出: h_post_term + h_comb_term
    
    数学公式:
        h_post_term = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
        h_comb_term = torch.sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)
        output = (h_post_term + h_comb_term).to(torch.bfloat16)
    
    Args:
        x: 输入 tensor，shape [B*S, N, D]，dtype bfloat16
           B*S 动态维度 ∈ {1024, 2048, 4096}，N=4 固定，D ∈ {2560, 5120}
        h_res: 流间混合权重矩阵，shape [B*S, N, N]，dtype float32
           逐样本动态权重
        h_out: 输出项数据，shape [B*S, D]，dtype bfloat16
        h_post: 后处理权重，shape [B*S, N]，dtype float32
           逐样本动态权重
    
    Returns:
        output: 融合计算结果，shape [B*S, N, D]，dtype bfloat16
    """
    # 步骤 1: 将 BF16 输入转换为 FP32，保证中间计算精度
    h_out_fp32 = h_out.float()  # [B*S, D] -> [B*S, D] float32
    x_fp32 = x.float()  # [B*S, N, D] -> [B*S, N, D] float32
    
    # 步骤 2: 计算 h_post_term (广播乘法)
    # h_post: [B*S, N] -> unsqueeze(-1) -> [B*S, N, 1]
    # h_out_fp32: [B*S, D] -> unsqueeze(-2) -> [B*S, 1, D]
    # 广播乘法: [B*S, N, 1] × [B*S, 1, D] -> [B*S, N, D]
    h_post_term = h_post.unsqueeze(-1) * h_out_fp32.unsqueeze(-2)
    
    # 步骤 3: 计算 h_comb_term (加权求和)
    # h_res: [B*S, N, N] -> unsqueeze(-1) -> [B*S, N, N, 1]
    # x_fp32: [B*S, N, D] -> unsqueeze(-2) -> [B*S, N, 1, D]
    # 广播乘法: [B*S, N, N, 1] × [B*S, N, 1, D] -> [B*S, N, N, D]
    # 沿 dim=-3 (第一个 N 维) 求和 -> [B*S, N, D]
    h_comb_term = torch.sum(h_res.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3)
    
    # 步骤 4: 融合相加并转换回 BF16
    result_fp32 = h_post_term + h_comb_term  # [B*S, N, D] float32
    output = result_fp32.to(torch.bfloat16)  # [B*S, N, D] bfloat16
    
    return output


# ==========================================
# 验证
# ==========================================

def _validate():
    """自动生成的验证函数 - 运行时动态生成验证报告"""
    
    print("=" * 60)
    print("mhc_post_golden 验证报告")
    print("=" * 60)
    
    # -- 1. 典型 case 验证（来自算子规格中的典型配置）--
    print("\n[典型 case 验证]")
    
    # P0 配置
    test_cases = [
        ("P0_标准", 1024, 4, 2560),
        ("P0_大D", 1024, 4, 5120),
        ("P0_大BS", 4096, 4, 2560),
        ("P0_最大", 4096, 4, 5120),
    ]
    
    all_passed = True
    for name, BS, N, D in test_cases:
        try:
            # 创建输入数据
            x = torch.randn(BS, N, D, dtype=torch.bfloat16)
            h_res = torch.randn(BS, N, N, dtype=torch.float32)
            h_out = torch.randn(BS, D, dtype=torch.bfloat16)
            h_post = torch.randn(BS, N, dtype=torch.float32)
            
            # 执行 golden
            output = mhc_post_golden(x, h_res, h_out, h_post)
            
            # 验证输出
            assert output.shape == (BS, N, D), f"Shape mismatch: expected {(BS, N, D)}, got {output.shape}"
            assert output.dtype == torch.bfloat16, f"Dtype mismatch: expected bfloat16, got {output.dtype}"
            
            # 验证数值合理性（无 NaN/Inf）
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            
            print(f"  {name}: BS={BS}, N={N}, D={D} ... ✓ PASS")
        except Exception as e:
            print(f"  {name}: BS={BS}, N={N}, D={D} ... ✗ FAIL: {e}")
            all_passed = False
    
    # -- 2. 泛化 case 验证（来自算子规格中的动态轴范围）--
    print("\n[泛化 case 验证]")
    
    # 动态轴 B*S 取值范围: {1024, 2048, 4096}
    # D 取值范围: {2560, 5120}
    # N 固定为 4
    dynamic_cases = [
        (2048, 4, 2560),  # 中等 BS, 小 D
        (2048, 4, 5120),  # 中等 BS, 大 D
        (1024, 4, 5120),  # 小 BS, 大 D
        (4096, 4, 2560),  # 大 BS, 小 D
    ]
    
    for BS, N, D in dynamic_cases:
        try:
            x = torch.randn(BS, N, D, dtype=torch.bfloat16)
            h_res = torch.randn(BS, N, N, dtype=torch.float32)
            h_out = torch.randn(BS, D, dtype=torch.bfloat16)
            h_post = torch.randn(BS, N, dtype=torch.float32)
            
            output = mhc_post_golden(x, h_res, h_out, h_post)
            
            assert output.shape == (BS, N, D), f"Shape mismatch"
            assert output.dtype == torch.bfloat16, f"Dtype mismatch"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            
            print(f"  BS={BS}, N={N}, D={D} ... ✓ PASS")
        except Exception as e:
            print(f"  BS={BS}, N={N}, D={D} ... ✗ FAIL: {e}")
            all_passed = False
    
    # -- 3. 值域检查（从公式推导）--
    print("\n[值域检查]")
    
    # mhc_post 输出值域取决于输入，无明显约束
    # 验证输出与输入类型的对应关系
    try:
        BS, N, D = 1024, 4, 2560
        x = torch.randn(BS, N, D, dtype=torch.bfloat16)
        h_res = torch.randn(BS, N, N, dtype=torch.float32)
        h_out = torch.randn(BS, D, dtype=torch.bfloat16)
        h_post = torch.randn(BS, N, dtype=torch.float32)
        
        output = mhc_post_golden(x, h_res, h_out, h_post)
        
        # 输出应为 bfloat16
        print(f"  输出类型验证 (期望 bfloat16) ... {'✓ PASS' if output.dtype == torch.bfloat16 else '✗ FAIL'}")
        if output.dtype != torch.bfloat16:
            all_passed = False
    except Exception as e:
        print(f"  值域检查 ... ✗ FAIL: {e}")
        all_passed = False
    
    # -- 4. 数值稳定性检查 --
    print("\n[数值稳定性检查]")
    
    # 测试极端输入值
    stability_tests = [
        ("大值输入", lambda: torch.randn(1024, 4, 2560, dtype=torch.bfloat16) * 100),
        ("小值输入", lambda: torch.randn(1024, 4, 2560, dtype=torch.bfloat16) * 0.01),
        ("零值输入", lambda: torch.zeros(1024, 4, 2560, dtype=torch.bfloat16)),
    ]
    
    for name, input_fn in stability_tests:
        try:
            BS, N, D = 1024, 4, 2560
            x = input_fn()
            h_res = torch.randn(BS, N, N, dtype=torch.float32)
            h_out = torch.randn(BS, D, dtype=torch.bfloat16)
            h_post = torch.randn(BS, N, dtype=torch.float32)
            
            output = mhc_post_golden(x, h_res, h_out, h_post)
            
            # 检查无 NaN/Inf
            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()
            
            if has_nan or has_inf:
                print(f"  {name} ... ✗ FAIL: contains NaN={has_nan}, Inf={has_inf}")
                all_passed = False
            else:
                print(f"  {name} ... ✓ PASS")
        except Exception as e:
            print(f"  {name} ... ✗ FAIL: {e}")
            all_passed = False
    
    # -- 5. 与参考实现对比 --
    print("\n[参考实现对比]")
    
    # 使用规格说明书中提供的 hc_post 函数进行对比
    def hc_post_reference(h_out, H_post, H_comb, x):
        """规格说明书中提供的参考实现"""
        h_out_fp32 = h_out.float()
        x_fp32 = x.float()
        h_post_term = H_post.unsqueeze(-1) * h_out_fp32.unsqueeze(-2)
        h_comb_term = torch.sum(H_comb.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3)
        y = h_post_term + h_comb_term
        return y.to(torch.bfloat16)
    
    try:
        BS, N, D = 1024, 4, 2560
        x = torch.randn(BS, N, D, dtype=torch.bfloat16)
        h_res = torch.randn(BS, N, N, dtype=torch.float32)
        h_out = torch.randn(BS, D, dtype=torch.bfloat16)
        h_post = torch.randn(BS, N, dtype=torch.float32)
        
        # 使用 golden 函数
        output_golden = mhc_post_golden(x, h_res, h_out, h_post)
        
        # 使用参考函数（注意参数顺序）
        output_ref = hc_post_reference(h_out, h_post, h_res, x)
        
        # 比较结果（考虑 BF16 精度）
        atol = 0.0001
        rtol = 0.0078125
        is_close = torch.allclose(output_golden, output_ref, atol=atol, rtol=rtol)
        
        if is_close:
            print(f"  与参考实现对比 (atol={atol}, rtol={rtol}) ... ✓ PASS")
        else:
            diff = torch.abs(output_golden - output_ref)
            max_diff = diff.max().item()
            print(f"  与参考实现对比 ... ✗ FAIL: max_diff={max_diff:.6f}")
            all_passed = False
    except Exception as e:
        print(f"  与参考实现对比 ... ✗ FAIL: {e}")
        all_passed = False
    
    # -- 最终结果 --
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("❌ 部分验证失败")
    print("=" * 60)
    
    return all_passed


# 验证代码仅在显式调用时执行
if __name__ == "__main__":
    # 延迟执行验证，避免导入超时
    import sys
    if "--validate" in sys.argv:
        _validate()
    else:
        print("Golden file imported successfully. Run with --validate to execute tests.")