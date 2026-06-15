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

"""PyPTO mhc_post golden reference implementation.

算子名称: mhc_post
公式:
    h_out_fp32 = h_out.float()
    x_fp32 = x.float()
    h_post_term = h_post.unsqueeze(-1) * h_out_fp32.unsqueeze(-2)
    h_comb_term = torch.sum(h_res.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3)
    output = (h_post_term + h_comb_term).to(torch.bfloat16)
"""

from typing import Optional

import torch

# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────


def mhc_post_golden(
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
) -> torch.Tensor:
    """mhc_post golden reference implementation."""
    h_out_fp32 = h_out.float()
    x_fp32 = x.float()
    h_post_term = h_post.unsqueeze(-1) * h_out_fp32.unsqueeze(-2)
    h_comb_term = torch.sum(h_res.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3)
    result_fp32 = h_post_term + h_comb_term
    output = result_fp32.to(torch.bfloat16)
    return output


# ==========================================
# 验证
# ==========================================

def _validate_typical_cases():
    """Validate typical cases from operator spec."""
    all_passed = True
    print("[典型 case 验证]")
    test_cases = [
        ("P0_标准", 1024, 4, 2560),
        ("P0_大D", 1024, 4, 5120),
        ("P0_大BS", 4096, 4, 2560),
        ("P0_最大", 4096, 4, 5120),
    ]
    for name, BS, N, D in test_cases:
        try:
            x = torch.randn(BS, N, D, dtype=torch.bfloat16)
            h_res = torch.randn(BS, N, N, dtype=torch.float32)
            h_out = torch.randn(BS, D, dtype=torch.bfloat16)
            h_post = torch.randn(BS, N, dtype=torch.float32)
            output = mhc_post_golden(x, h_res, h_out, h_post)
            assert output.shape == (BS, N, D), f"Shape mismatch: expected {(BS, N, D)}, got {output.shape}"
            assert output.dtype == torch.bfloat16, f"Dtype mismatch: expected bfloat16, got {output.dtype}"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            print(f"  {name}: BS={BS}, N={N}, D={D} ... ✓ PASS")
        except Exception as e:
            print(f"  {name}: BS={BS}, N={N}, D={D} ... ✗ FAIL: {e}")
            all_passed = False
    return all_passed


def _validate_dynamic_cases():
    """Validate dynamic axis generalization cases."""
    all_passed = True
    print("\n[泛化 case 验证]")
    dynamic_cases = [
        (2048, 4, 2560),
        (2048, 4, 5120),
        (1024, 4, 5120),
        (4096, 4, 2560),
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
    return all_passed


def _validate_range_check():
    """Validate output dtype."""
    all_passed = True
    print("\n[值域检查]")
    try:
        BS, N, D = 1024, 4, 2560
        x = torch.randn(BS, N, D, dtype=torch.bfloat16)
        h_res = torch.randn(BS, N, N, dtype=torch.float32)
        h_out = torch.randn(BS, D, dtype=torch.bfloat16)
        h_post = torch.randn(BS, N, dtype=torch.float32)
        output = mhc_post_golden(x, h_res, h_out, h_post)
        print(f"  输出类型验证 (期望 bfloat16) ... {'✓ PASS' if output.dtype == torch.bfloat16 else '✗ FAIL'}")
        if output.dtype != torch.bfloat16:
            all_passed = False
    except Exception as e:
        print(f"  值域检查 ... ✗ FAIL: {e}")
        all_passed = False
    return all_passed


def _validate_stability():
    """Validate numerical stability with extreme inputs."""
    all_passed = True
    print("\n[数值稳定性检查]")
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
    return all_passed


def _validate_ref_comparison():
    """Compare golden output with reference implementation."""
    all_passed = True
    print("\n[参考实现对比]")

    def hc_post_reference(h_out, H_post, H_comb, x):
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
        output_golden = mhc_post_golden(x, h_res, h_out, h_post)
        output_ref = hc_post_reference(h_out, h_post, h_res, x)
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
    return all_passed


def _validate():
    """自动生成的验证函数 - 运行时动态生成验证报告"""
    print("=" * 60)
    print("mhc_post_golden 验证报告")
    print("=" * 60)

    all_passed = True
    all_passed &= _validate_typical_cases()
    all_passed &= _validate_dynamic_cases()
    all_passed &= _validate_range_check()
    all_passed &= _validate_stability()
    all_passed &= _validate_ref_comparison()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("❌ 部分验证失败")
    print("=" * 60)
    return all_passed


# 验证代码仅在显式调用时执行
if __name__ == "__main__":
    import sys
    if "--validate" in sys.argv:
        _validate()
    else:
        print("Golden file imported successfully. Run with --validate to execute tests.")
